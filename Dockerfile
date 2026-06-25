# Build and run the engine on Linux, no local toolchain needed:
#   docker build -t mini-llm .
#   docker run --rm mini-llm                      # default greedy "Once upon a time" story
#   docker run --rm mini-llm -v "Once upon a time" # per-token trace
#   docker run --rm mini-llm "The dragon" 1.0      # sampled, your own prompt
FROM gcc:14
WORKDIR /app

# Fetch the model weights (60 MB, not committed to the repo).
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && curl -L -o stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

COPY tokenizer.bin run.c ./
RUN cc -O2 -o run run.c -lm

ENTRYPOINT ["./run"]
CMD ["Once upon a time"]
