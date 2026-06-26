# mini-llm

A tiny [llama.cpp](https://github.com/ggerganov/llama.cpp)-style LLM **inference** engine, written from scratch in C — small enough to read in one sitting, but running a *real* trained model. It loads a 15M-parameter [TinyStories](https://huggingface.co/karpathy/tinyllamas) model (same architecture as Llama) and generates text on the CPU. *(A GPU/CUDA port is the subject of a planned follow-up.)*

Built as a learning project: every step — tokenizer (BPE), embeddings, RoPE, multi-head attention with a KV cache, the SwiGLU feed-forward network, and sampling — is plain, commented C in a single file (`run.c`).

## Get the model

The model weights aren't checked in (they're 60 MB). Download them next to the source:

```sh
curl -L -o stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

The tokenizer (`tokenizer.bin`, 424 KB) is included in the repo.

## Build & run (CPU)

**Any platform with a C compiler:**

```sh
cc -O2 -o run run.c -lm
./run "Once upon a time"
```

**Windows / MSVC** (from a Developer Command Prompt):

```sh
cl /O2 /D_CRT_SECURE_NO_WARNINGS run.c
run.exe "Once upon a time"
```

or open the included Visual Studio solution, `mini_llm.sln`.

Decoding is **greedy by default**, so a no-argument run is deterministic and reproducible. Usage:

```
run [-v] [prompt] [temperature]
  -v            verbose trace: prefill/decode markers + per-token confidence
  prompt        the text to continue (default: "Once upon a time")
  temperature   0 = greedy (default); >0 samples, e.g. 1.0 for variety
```

Try `./run -v "Once upon a time"` to watch the model think token by token.

## Run with Docker (no toolchain needed)

Prefer not to install a compiler? Build and run it in a container. The image fetches the model itself, so there's nothing else to set up:

```sh
docker build -t mini-llm .
docker run --rm mini-llm                        # default greedy story
docker run --rm mini-llm -v "Once upon a time"  # per-token trace
docker run --rm mini-llm "The dragon" 1.0       # sampled, your own prompt
```

## Chat with a real model (Qwen2.5-1.5B-Instruct)

The same engine also runs a real instruction-tuned chat model: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). `run.c` auto-detects the format, so it adds grouped-query attention, q/k/v biases, rotate-half (HF) RoPE, a byte-level BPE tokenizer, and a ChatML loop whose KV cache persists across turns. A one-time conversion turns the HuggingFace weights into the engine's format:

```sh
# 1. Python deps for the conversion (no PyTorch needed)
pip install numpy huggingface_hub tokenizers

# 2. download (~3 GB) and convert: qwen.bin (~6 GB fp32) + the tokenizer
python fetch_qwen.py          # -> qwen_model/
python export_qwen.py         # -> qwen.bin
python export_tokenizer.py    # -> qwen_tokenizer.bin

# 3. build (OpenMP gives ~3x on multi-core CPUs; it stays bit-identical)
cc -O2 -fopenmp -o run run.c -lm                  # gcc / clang
# or MSVC:  cl /O2 /openmp /D_CRT_SECURE_NO_WARNINGS run.c

# 4. chat
./run -m qwen.bin
```

It runs in **fp32 on the CPU** (~1–2 tok/s, memory-bandwidth bound — every token streams all 6 GB of weights). Quantization and a GPU path are the obvious next steps. `test_tokenizer.py` checks the C tokenizer against the reference, and `run -encode -` prints token ids for piped text.

## Credits & prior work

This engine is modeled on Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) — the single-file C inference approach this whole project follows. The model, `stories15M`, is one of Karpathy's TinyStories checkpoints, trained on the **TinyStories** dataset by **Ronen Eldan and Yuanzhi Li** (Microsoft Research). The architecture and tokenizer follow **Meta's Llama**. I rebuilt the engine from scratch to understand it — but the path was theirs first.

## License

[MIT](LICENSE).
