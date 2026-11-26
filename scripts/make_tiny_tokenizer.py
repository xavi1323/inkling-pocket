"""Write a synthetic byte-level tokenizer fixture (256 byte tokens + Inkling special markers).

Not the model's tokenizer: no merges, so every byte is one token. The end-of-sampling marker keeps the real id 200006.
    python3 -I scripts/make_tiny_tokenizer.py InklingKit/Tests/InklingEngineTests/Fixtures/tiny-tokenizer
"""
import base64, json, os, sys

out = sys.argv[1]
os.makedirs(os.path.join(out, "tiktoken"), exist_ok=True)
with open(os.path.join(out, "tiktoken", "tokenizer.model"), "w") as f:
    for i in range(256):
        f.write(f"{base64.b64encode(bytes([i])).decode()} {i}\n")
specials = ["<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>", "<|content_text|>",
            "<|content_thinking|>", "<|content_invoke_tool_json|>", "<|content_xml|>", "<|end_message|>"]
added = [{"id": 256 + i, "content": s, "special": True} for i, s in enumerate(specials)]
added.append({"id": 200006, "content": "<|content_model_end_sampling|>", "special": True})
tok = {"added_tokens": added,
       "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "\\s+|[^\\s]+"}}]}}
with open(os.path.join(out, "tokenizer.json"), "w") as f:
    json.dump(tok, f, indent=1)
