"""Render the model's chat_template.jinja for fixed cases -> JSON fixtures.

Usage: .venv/bin/python -I scripts/render_template_fixtures.py <model_dir> <out_dir>
Only reads chat_template.jinja; writes {input, expected} JSON. Never copies the template.
"""
import json, re, sys, pathlib
import jinja2

model_dir, out_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
src = (model_dir / "chat_template.jinja").read_text()
# HF-only tag: {% generation %}...{% endgeneration %} just marks the span; render its body.
src = re.sub(r"\{%-?\s*generation\s*-?%\}", "", src)
src = re.sub(r"\{%-?\s*endgeneration\s*-?%\}", "", src)

def raise_exception(msg):
    raise jinja2.exceptions.TemplateError(msg)

def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception
template = env.from_string(src)

search_tool = {"type": "function", "function": {
    "name": "web_search", "description": "Search the web.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "What to search"}},
                   "required": ["query"]}}}
calc_tool = {"type": "function", "function": {
    "name": "calculator", "description": "Evaluate math.",
    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}

cases = {
    "a_user_only": dict(messages=[{"role": "user", "content": "Hello"}], tools=None, effort="low"),
    "b_system_user": dict(messages=[{"role": "system", "content": "You are Inkling."},
                                    {"role": "user", "content": "Hi there"}], tools=None, effort="medium"),
    "c_tools_user": dict(messages=[{"role": "system", "content": "Use tools."},
                                   {"role": "user", "content": "Weather in Paris?"}],
                         tools=[search_tool, calc_tool], effort="low"),
    "d_reasoning_history": dict(messages=[
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "reasoning_content": "Simple sum.", "content": "4"},
        {"role": "user", "content": "And 3+3?"}], tools=None, effort="high"),
    "e_tool_roundtrip": dict(messages=[
        {"role": "user", "content": "Search cats"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
            "name": "web_search", "arguments": {"query": "cats", "options": {"z": 1, "a": [1.5, True, None]}}}}]},
        {"role": "tool", "name": "web_search", "content": "Cats are mammals."},
        {"role": "assistant", "content": "Cats are mammals [1]."},
        {"role": "user", "content": "Thanks"}], tools=[search_tool], effort="low"),
    "f_none_unicode": dict(messages=[{"role": "user", "content": "Café ☕ 日本 \"quoted\"\nnew line"}],
                           tools=None, effort="none"),
    "g_xhigh_error_tool": dict(messages=[
        {"role": "user", "content": "Open it"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
            "name": "open_url", "arguments": {"url": "https://x.y/a?b=c"}}}]},
        {"role": "tool", "name": "open_url", "content": "Error: unsupported content type: application/pdf"}],
        tools=None, effort="xhigh"),
}

out_dir.mkdir(parents=True, exist_ok=True)
for name, c in cases.items():
    kwargs = {"messages": c["messages"], "reasoning_effort": c["effort"]}
    if c["tools"]:
        kwargs["tools"] = c["tools"]
    expected = template.render(**kwargs)
    (out_dir / f"{name}.json").write_text(json.dumps(
        {"messages": c["messages"], "tools": c["tools"] or [], "effort": c["effort"], "expected": expected},
        ensure_ascii=False, indent=1))
    print(name, len(expected))
