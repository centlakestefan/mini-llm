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

## Credits & prior work

This engine is modeled on Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) — the single-file C inference approach this whole project follows. The model, `stories15M`, is one of Karpathy's TinyStories checkpoints, trained on the **TinyStories** dataset by **Ronen Eldan and Yuanzhi Li** (Microsoft Research). The architecture and tokenizer follow **Meta's Llama**. I rebuilt the engine from scratch to understand it — but the path was theirs first.

## License

[MIT](LICENSE).
