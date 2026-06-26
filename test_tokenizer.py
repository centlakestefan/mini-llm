"""Compare run.exe -encode against the reference tokenizers lib."""
import subprocess
from tokenizers import Tokenizer

ref = Tokenizer.from_file("qwen_model/tokenizer.json")

tests = [
    "Hello, world!",
    "What is the capital of France?",
    "The answer is 42.",
    "2 + 2 = 4",
    "I don't think it's working, but we'll see.",
    "def add(a, b):\n    return a + b",
    "Multiple     spaces   and\ttabs.",
    "Line one.\nLine two.\n\nParagraph.",
    "café naïve résumé Москва 日本語 🚀",
    "ALL CAPS and MixedCase and snake_case and kebab-case",
    "<|im_start|>user\nHi there!<|im_end|>\n<|im_start|>assistant\n",
    "Numbers: 1234567890 and 3.14159",
    "Email me@example.com or visit https://example.com/path?q=1",
    "   leading spaces and trailing   ",
    "Quotes: \"double\" and 'single' and `code`",
]

ok = 0
for s in tests:
    expected = ref.encode(s).ids
    out = subprocess.run(["./run.exe", "-encode", "-"], input=s.encode("utf-8"),
                         capture_output=True)
    got = [int(x) for x in out.stdout.decode().strip().split(",") if x]
    match = got == expected
    ok += match
    if not match:
        print(f"MISMATCH: {s.encode('unicode_escape').decode()}")
        print(f"  expected ({len(expected)}): {expected}")
        print(f"  got      ({len(got)}): {got}")
print(f"\n{ok}/{len(tests)} match")
