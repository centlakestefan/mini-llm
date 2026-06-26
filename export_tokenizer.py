"""Flatten Qwen's tokenizer.json into qwen_tokenizer.bin for run.c.

Layout (little-endian):
  magic 'QTK1' (4 bytes)
  int32 vocab_size           (matches the model's padded vocab, 151936)
  int32 n_merges
  int32 n_special
  vocab:   vocab_size entries, each [int32 len][len bytes]   (byte-unicode UTF-8)
  merges:  n_merges  entries, each [int32 len][len bytes]    ("left right")
  special: n_special entries, each [int32 len][len bytes][int32 id]

The vocab strings live in GPT-2 "byte-level" space (every raw byte mapped to a
printable unicode char). run.c converts text<->bytes<->that space itself.
"""
import json, struct

VOCAB_SIZE = 151936   # model's padded vocab (embedding rows)

d = json.load(open("qwen_model/tokenizer.json", encoding="utf-8"))
model = d["model"]
vocab = model["vocab"]              # piece -> id
merges = model["merges"]            # list of "left right"
added = d["added_tokens"]           # special tokens with explicit ids

# id -> string table, sized to the padded vocab
table = [""] * VOCAB_SIZE
for piece, i in vocab.items():
    if 0 <= i < VOCAB_SIZE:
        table[i] = piece
specials = []
for t in added:
    if 0 <= t["id"] < VOCAB_SIZE:
        table[t["id"]] = t["content"]
        if t.get("special"):
            specials.append((t["content"], t["id"]))

def wstr(out, s):
    b = s.encode("utf-8")
    out.write(struct.pack("<i", len(b)))
    out.write(b)

with open("qwen_tokenizer.bin", "wb") as out:
    out.write(b"QTK1")
    out.write(struct.pack("<iii", VOCAB_SIZE, len(merges), len(specials)))
    for s in table:
        wstr(out, s)
    for m in merges:
        wstr(out, m if isinstance(m, str) else " ".join(m))
    for content, tid in specials:
        wstr(out, content)
        out.write(struct.pack("<i", tid))

print(f"wrote qwen_tokenizer.bin: vocab={VOCAB_SIZE} merges={len(merges)} "
      f"specials={len(specials)}")
print("specials:", sorted(i for _, i in specials))
