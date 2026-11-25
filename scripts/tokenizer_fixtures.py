"""Encode fixed strings with the model's tokenizer.json -> ids fixture (outputs only, no vocab).

Usage: .venv/bin/python -I scripts/tokenizer_fixtures.py <model_dir> <out_json> [<prompt_fixture_json>]
"""
import json, sys, pathlib
from tokenizers import Tokenizer

model_dir, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
texts = [
    "Hello, world!", "The quick brown fox jumps over the lazy dog.", "  leading spaces", "trailing spaces   ",
    "multiple\n\nnewlines\r\nwindows", "tabs\tand\tmore", "Numbers: 1234567 and 3.14159 and -42",
    "Emoji 😀👍🏽 and flags 🇫🇷", "日本語のテキスト、中文文本。", "Café naïve résumé façade", "Привет, мир!",
    "مرحبا بالعالم", "def f(x):\n    return x ** 2  # square", "let x = [1, 2, 3].map { $0 * 2 }",
    "<html><body><p class=\"a\">Hi</p></body></html>", "URL https://example.com/a?b=c&d=e#f",
    "CamelCaseIdentifier and snake_case_name", "I'm, you're, they've, we'll, he'd, it's",
    "ALL CAPS SENTENCE HERE", "a", " ", "\n", "", "     ", "x" * 50,
    "<|message_user|><|content_text|>Hi<|end_message|>",
    "text<|content_thinking|>thought<|end_message|><|message_model|><|content_text|>answer",
    "<|content_model_end_sampling|><|endoftext|><|content_invoke_tool_json|>{\"name\":\"a\"}",
    "Mixed 123abc456 def789", "∑∫√π≈≠ math symbols",
]
if len(sys.argv) > 3:
    texts.append(json.load(open(sys.argv[3]))["expected"])
cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in texts]
out.write_text(json.dumps(cases, ensure_ascii=False, indent=0))
print(len(cases), "cases;", sum(len(c["ids"]) for c in cases), "ids")
