"""Download just the files we need from Qwen2.5-1.5B-Instruct into ./qwen_model."""
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    local_dir="qwen_model",
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
    ],
)
print("downloaded to:", path)
