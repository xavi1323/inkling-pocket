# Inkling Pocket — Spec

Status: derived from approved `Design.md` (v3), 2026-11-25
Each requirement has an ID, and "Check" says how it is proved. "iPad" = owner runs it on the iPad Air M4 in LiveContainer.

## Platform
| ID | Requirement | Check |
|---|---|---|
| P1 | iPadOS 26.0+, iPad only (`TARGETED_DEVICE_FAMILY = 2`), any window size, landscape + portrait | CI build settings |
| P2 | Project generated from `project.yml` (XcodeGen); builds on GitHub Actions `macos-26` with Xcode 26.6 | green CI run |
| P3 | CI uploads unsigned `InklingPocket.ipa` artifact on every push to `main` | artifact present |
| P4 | Swift 6 language mode, strict concurrency complete | CI build |
| P5 | Runs inside LiveContainer; no code depends on our own entitlements or bundle ID | iPad smoke test |
| P6 | Dynamic Type, VoiceOver labels on all controls, dark mode, ⌘↩ send, ⌘N new chat | UI tests + iPad |

## Model import
| ID | Requirement | Check |
|---|---|---|
| M1 | First run shows Import screen; folder picker (UIDocumentPicker, folders) for the 12GB model folder | iPad |
| M2 | Validates: `config.json`, `model.safetensors.index.json`, every shard listed in the index exists with non-zero size, tokenizer + template files present; `model_type`/architecture is `inkling2*` | unit tests on fixture folders |
| M3 | Model copied into app storage with progress + cancel, or used in place via security-scoped bookmark (setting) | iPad |
| M4 | Clear error text for each validation failure | unit tests |

## Engine
| ID | Requirement | Check |
|---|---|---|
| E1 | Reference path: MLX Swift port of `inkling2_loop` text model (24 layers, layers 6–17 ×3 with per-visit caches, MoE 128/8, sliding 1,024 + global attention on 5/11/17/23, attention sinks, short conv) | tiny-model golden test in CI |
| E2 | Loads mixed per-module quantization from `config.json` (`quantization`) incl. 2/3/4/6/8-bit affine; NVFP4 if present | loader unit test + tiny model |
| E3 | Weights memory-mapped read-only; no full copy into allocated memory on the fast path | Benchmark shows footprint < weights size |
| E4 | Memory budget read from `os_proc_available_memory()`; manual override in Settings | Benchmark shows value |
| E5 | Hot-expert residency: per-layer hit counts, hottest experts pinned within budget, prefetch hint for routed experts | Benchmark hit-rate metric |
| E6 | Fast path: fused MoE decode kernel (gather + dequant + matmul + SwiGLU), GPU sampler, compiled per-token graph | Self-test logits match reference path |
| E7 | Streaming generation API: `AsyncStream<Token>`; cancel stops within one token | unit test with fake model |
| E8 | KV cache: ring buffer 1,024 for sliding layers; growing cache for 8 global visits; context cap setting (default 16K); prefix reuse across turns in a chat | unit tests |
| E9 | Chunked prefill (default 256) | Benchmark TTFT |
| E10 | Never crashes on memory pressure: on warning, unpin cold experts and continue | iPad stress run |
| E11 | Sampling: temperature, top-p, top-k, repetition penalty; greedy when temperature = 0 | unit tests |

## Tokenizer & chat format
| ID | Requirement | Check |
|---|---|---|
| T1 | tiktoken-style BPE from the model folder files; encode/decode round-trip | fixture tests |
| T2 | Chat prompt byte-identical to `chat_template.jinja` output, with `reasoning_effort` and `tools` | test vs Python-jinja2-rendered fixtures |
| T3 | Streaming parser emits `.thinking(text)`, `.answer(text)`, `.toolCall(name, json)`, `.end` from `<|message_model|>`, `<|content_thinking|>`, `<|content_text|>`, `invoke_tool_json`, `<|end_message|>` — correct even when a marker is split across tokens | unit tests |

## Agent & tools
| ID | Requirement | Check |
|---|---|---|
| A1 | Agent loop: generate → tool call → run → append result → continue; max steps setting (default 8) | fake-model tests |
| A2 | Tools: `web_search`, `open_url`, `memory_save`, `memory_search`, `memory_forget`, `chat_search`, `current_time`, `calculator` | per-tool unit tests |
| A3 | Tool output trimmed to a token budget (default 2,000 tokens per result) | unit test |
| A4 | Default `reasoning_effort = low`; per-chat picker | UI |
| A5 | Bad tool-call JSON → error result fed back to the model, never a crash | unit test |

## Crawler
| ID | Requirement | Check |
|---|---|---|
| C1 | Hidden reused WKWebView (non-persistent store), wait for load + JS idle, timeout 15 s; URLSession fallback | iPad |
| C2 | Swift port of crawl4ai PruningContentFilter + markdown generator (links → numbered citations) | HTML fixture tests vs expected markdown |
| C3 | DuckDuckGo HTML results parsed to `[title, url, snippet]`; top N (default 3) crawled in parallel | fixture test + iPad |
| C4 | Works on Wi-Fi and cellular; generic user agent; fails softly with "search unavailable" | unit test for failure path |

## Memory
| ID | Requirement | Check |
|---|---|---|
| R1 | SwiftData models: `Chat`, `Message`, `Fact` | unit tests (in-memory container) |
| R2 | Auto facts via `memory_save`; Memory screen lists, edits, deletes facts | UI |
| R3 | Search = `NLContextualEmbedding` similarity + keyword match; top facts injected into system prompt (≤300 tokens) | unit tests |

## OpenUI
| ID | Requirement | Check |
|---|---|---|
| O1 | Streaming OpenUI Lang parser compatible with upstream syntax for v1 components; renders only validated parts; malformed fragments dropped | parser unit tests |
| O2 | Components: Card, Table, List, Chart (bar/line/pie, Swift Charts), Callout, CodeBlock, Steps, Buttons, Form, Image | snapshot-free view tests + iPad |
| O3 | Component library prompt generated from the Swift registry | unit test |
| O4 | Button tap / form submit sends a follow-up message to the agent | unit test |

## UI
| ID | Requirement | Check |
|---|---|---|
| U1 | `NavigationSplitView`: sidebar (chats, Memory, Benchmark, Settings), detail = chat | UI |
| U2 | Streaming answer, collapsible Thinking, tool chips, OpenUI blocks, markdown | UI |
| U3 | Liquid Glass: floating composer + tok/s HUD use `.glassEffect()` in a `GlassEffectContainer`; no glass on message content; Reduce Transparency respected | UI |
| U4 | Stop button cancels generation | UI |

## Self-test & Benchmark (on iPad)
| ID | Requirement | Check |
|---|---|---|
| B1 | Self-test: reference path vs fast path logits on fixed prompts (teacher-forced), max abs diff + top-1 agreement; pass/fail | iPad |
| B2 | Known-answer prompts (greedy) pass/fail | iPad |
| B3 | Benchmark: prefill tok/s, decode tok/s, TTFT, peak footprint, memory limit, expert hit rate, per-layer/per-pass routing counts, thermal state | iPad |
| B4 | Results copyable as plain text | iPad |

## Repo
| ID | Requirement | Check |
|---|---|---|
| G1 | Public GitHub repo `inkling-pocket`; no model weights, tokenizer or template files committed | `.gitignore` + review |
| G2 | `THIRD_PARTY_NOTICES.md` for mlx-vlm (MIT), crawl4ai (Apache-2.0), OpenUI (MIT); license headers kept on ported files | review |
