"""Verification harness for the Qwen forward pass (before the C tokenizer exists).

  python verify_qwen.py encode "What is the capital of France?"
      -> prints a comma-separated id list for a ChatML prompt
  python verify_qwen.py decode "12,34,56"
      -> prints the text those ids decode to
"""
import sys
from tokenizers import Tokenizer

tok = Tokenizer.from_file("qwen_model/tokenizer.json")

SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

def chatml(user):
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")

if sys.argv[1] == "encode":
    ids = tok.encode(chatml(sys.argv[2])).ids
    print(",".join(str(i) for i in ids))
elif sys.argv[1] == "decode":
    ids = [int(x) for x in sys.argv[2].replace(" ", ",").split(",") if x != ""]
    print(tok.decode(ids, skip_special_tokens=False))
