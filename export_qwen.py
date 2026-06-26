"""Convert Qwen2.5-1.5B-Instruct (HF safetensors) into qwen.bin for run.c.

The output is a self-describing v2 checkpoint:

  header (48 bytes): 11 int32 + 1 float32, little-endian
    magic, version, dim, hidden_dim, n_layers, n_heads, n_kv_heads,
    vocab_size, seq_len, head_dim, shared_classifier, rope_theta

  weights (all float32), in this order:
    token_embedding_table        (vocab, dim)
    rms_att_weight               (n_layers, dim)
    wq                           (n_layers, n_heads*head_dim, dim)
    wk                           (n_layers, n_kv_heads*head_dim, dim)
    wv                           (n_layers, n_kv_heads*head_dim, dim)
    wo                           (n_layers, dim, n_heads*head_dim)
    rms_ffn_weight               (n_layers, dim)
    w1 (gate)                    (n_layers, hidden_dim, dim)
    w2 (down)                    (n_layers, dim, hidden_dim)
    w3 (up)                      (n_layers, hidden_dim, dim)
    rms_final_weight             (dim,)
    bq                           (n_layers, n_heads*head_dim)
    bk                           (n_layers, n_kv_heads*head_dim)
    bv                           (n_layers, n_kv_heads*head_dim)
    [wcls only if not tied]      (vocab, dim)

HF stores every linear weight as (out_features, in_features) row-major, which is
exactly what run.c's matmul wants -- so nothing here is transposed.
"""
import json, struct, mmap, sys
import numpy as np

SRC_DIR = "qwen_model"
OUT = "qwen.bin"
MAGIC = 0x716E3277   # "qwn2"
VERSION = 1
SEQ_LEN = 4096       # our chat context cap (model max is 32768; bounds the KV cache)

cfg = json.load(open(f"{SRC_DIR}/config.json"))
dim       = cfg["hidden_size"]
hidden    = cfg["intermediate_size"]
n_layers  = cfg["num_hidden_layers"]
n_heads   = cfg["num_attention_heads"]
n_kv      = cfg["num_key_value_heads"]
vocab     = cfg["vocab_size"]
head_dim  = dim // n_heads
rope_theta = float(cfg["rope_theta"])
tied      = bool(cfg.get("tie_word_embeddings", False))

print(f"config: dim={dim} hidden={hidden} layers={n_layers} heads={n_heads} "
      f"kv={n_kv} vocab={vocab} head_dim={head_dim} rope_theta={rope_theta} tied={tied}")

# ---- open the safetensors file and parse its JSON header ----
f = open(f"{SRC_DIR}/model.safetensors", "rb")
hlen = struct.unpack("<Q", f.read(8))[0]
st_header = json.loads(f.read(hlen))
data_start = 8 + hlen
mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

def tensor(name):
    """Return an HF tensor as a float32 numpy array (upconverting bf16/f16)."""
    meta = st_header[name]
    s, e = meta["data_offsets"]
    raw = mm[data_start + s : data_start + e]
    dt = meta["dtype"]
    if dt == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        arr = (u16.astype(np.uint32) << 16).view(np.float32)
    elif dt == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dt == "F32":
        arr = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(f"unhandled dtype {dt} for {name}")
    return arr.reshape(meta["shape"])

def layers(fmt):
    """Concatenate a per-layer tensor across all layers, flattened."""
    return np.concatenate([tensor(fmt.format(l)).ravel() for l in range(n_layers)])

print("gathering tensors...")
groups = [
    ("token_embedding", tensor("model.embed_tokens.weight").ravel()),
    ("rms_att",   layers("model.layers.{}.input_layernorm.weight")),
    ("wq",        layers("model.layers.{}.self_attn.q_proj.weight")),
    ("wk",        layers("model.layers.{}.self_attn.k_proj.weight")),
    ("wv",        layers("model.layers.{}.self_attn.v_proj.weight")),
    ("wo",        layers("model.layers.{}.self_attn.o_proj.weight")),
    ("rms_ffn",   layers("model.layers.{}.post_attention_layernorm.weight")),
    ("w1_gate",   layers("model.layers.{}.mlp.gate_proj.weight")),
    ("w2_down",   layers("model.layers.{}.mlp.down_proj.weight")),
    ("w3_up",     layers("model.layers.{}.mlp.up_proj.weight")),
    ("rms_final", tensor("model.norm.weight").ravel()),
    ("bq",        layers("model.layers.{}.self_attn.q_proj.bias")),
    ("bk",        layers("model.layers.{}.self_attn.k_proj.bias")),
    ("bv",        layers("model.layers.{}.self_attn.v_proj.bias")),
]
if not tied:
    groups.append(("wcls", tensor("lm_head.weight").ravel()))

# ---- write header + weights ----
header = struct.pack("<11i f", MAGIC, VERSION, dim, hidden, n_layers, n_heads,
                     n_kv, vocab, SEQ_LEN, head_dim, 1 if tied else 0, rope_theta)
assert len(header) == 48, len(header)

total = 0
with open(OUT, "wb") as out:
    out.write(header)
    for name, arr in groups:
        a = np.ascontiguousarray(arr, dtype="<f4")
        a.tofile(out)
        total += a.size
        print(f"  {name:16s} {a.size:>12d} floats")

print(f"\nwrote {OUT}: header 48 B + {total} floats "
      f"({48 + total*4} bytes, {(48 + total*4)/1e9:.2f} GB)")
