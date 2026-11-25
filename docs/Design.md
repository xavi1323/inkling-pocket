# Inkling Pocket — Design

Status: APPROVED by owner 2026-11-25 (v3)
Date: 2026-11-25

## 1. What it is, and why

Inkling Pocket is an iPad app: a private AI agent that runs **entirely on the iPad**.
The brain is **Inkling 2 Mini, abliterated, MLX 12GB build**
(`orcarouter/Inkling-2-Mini-abliterated-MLX`, folder `Inkling-2-Mini-abliterated-MLX-12GB`).
No existing iPad app or engine can run this model, so the project includes a
**custom inference engine** built for this one model on this one iPad.

The owner wants a general-purpose agent ("everything equally"): chat, web research,
memory about the owner, coding help and rich answers drawn as live UI.

## 2. Goals

- G1 Runs Inkling 2 Mini 12GB on an **iPad Air M4 (12 GB RAM)**, fully offline for chat.
- G2 Decode speed target **30–40 tokens/s**. Hard floor for "usable": 15 tok/s.
- G3 Agent tools: **web search + page reading** (crawl4ai rebuilt in Swift),
  **memory** (auto-saved facts + search over past chats).
- G4 Answers can render as **live UI** (cards, tables, charts, forms, lists) through a
  Swift port of **OpenUI** (thesysdev/openui, "OpenUI Lang"), drawn while tokens stream.
- G5 Liquid Glass look on navigation and controls.
- G6 Delivered as an **unsigned `.ipa`** built by GitHub Actions.

## 3. Non-goals (v1)

- Other models, other devices (iPhone, Mac), App Store / TestFlight release.
- Audio input. Image input is a stretch goal (the vision tower is in the files; untested by the publisher).
- Speculative decoding with the original DSpark drafter (not in this repo; possible v2).
- In-app model download / Hugging Face login. **Owner downloads the model and imports it.**
- Cloud fallback of any kind.

## 4. Anti-goals (would be a failure even if it "works")

- Any network call during plain chat. Network is used only when a tool needs it.
- An engine that crashes when memory runs short instead of slowing down.
- Fake speed numbers: tok/s shown in the app must be measured, not estimated.
- Safety filters or refusal layers added by the app (owner chose an abliterated model for personal use).

## 5. Facts and constraints

### Hardware
- iPad Air M4: M4 chip, 9-core GPU, 16-core Neural Engine, **12 GB unified memory, 120 GB/s bandwidth** (Apple spec page 126472).
- **Runtime host: LiveContainer.** Inkling Pocket is installed *inside* LiveContainer and runs in
  LiveContainer's process. Owner uses **iRAM+**, which adds the `increased-memory-limit` and
  `extended-virtual-addressing` entitlements to LiveContainer. So Inkling Pocket inherits them;
  its own entitlements and bundle ID are ignored by LiveContainer.
- The raised limit is **not unlimited** and its exact value on a 12 GB iPad is unknown (reports range 6–12 GB).
  Physical RAM is 12 GB and iPadOS uses ~2–3 GB, so the 11.6 GB of weights **cannot all be dirty/resident at once**.
- Key fact the engine is built around: on iOS, **clean file-backed (mmapped, read-only) pages do not count toward
  the app's memory footprint limit**; memory the app allocates and fills does. A normal MLX load copies weights
  into allocated buffers (counted). Inkling Pocket's loader instead maps the safetensors files and wraps them as
  Metal buffers without copying (`makeBuffer(bytesNoCopy:)`), so most weight bytes are reclaimable page cache, not footprint.

### Model (from the model card + mlx-vlm `models/inkling2`)
- Architecture `inkling2_loop`: 26.4B params (26.07B text, 0.32B vision), **3.4B active per token**.
- 24 distinct layers; **layers 6–17 run 3×** with shared weights → 48 layer visits per token,
  each visit with its **own attention cache and convolution cache**.
- Every layer is MoE: **128 routed experts, 8 active**, no shared expert.
- Attention: sliding window 1,024 in 20 layers; global in layers 5, 11, 17, 23 (so 8 global visits per token). Attention sinks.
- Short convolution per layer (custom kernel in mlx-vlm `kernels.py`, keeps FP32 intermediate math).
- Hidden size 2,304. Vocabulary 201,024. tiktoken-based tokenizer, `chat_template.jinja`.
- 12GB build: routed experts **2–4-bit affine**; attention/embedding/head/vision **6–8-bit affine**; norms/conv/sinks BF16.
  Per-module format is listed in `config.json` → `quantization`.
- Chat format: model opens its own turn with `<|message_model|>`, then `<|content_thinking|>…`,
  then `<|content_text|>…<|end_message|>`. Tool calls come back as `invoke_tool_json` messages.
  `reasoning_effort`: none / minimal / low / medium / high / xhigh / max.
- Publisher: **never run on Apple Silicon**; checked only on Linux (CPU/CUDA). Reference code:
  mlx-vlm commit `d998a62`, MLX 0.32.4.

### Speed math (why 30–40 is possible, and what threatens it)
- Bytes read per token ≈ active weights × bits ≈ 3.4B × ~3.3 bits ≈ **~1.4 GB** (+ KV cache).
- 120 GB/s ÷ 1.4 GB ≈ **85 tok/s ceiling** if everything is in RAM. Real engines reach 40–60% of the
  ceiling → **~35–50 tok/s** possible.
- The threat is **expert pages not in RAM**: every miss reads from flash (a few GB/s), much slower.
  The engine must keep the hot experts resident and let cold ones be dropped.

### Platform / tooling
- No Mac. All builds run on GitHub Actions `macos-26` (arm64). Newest Xcode there: **26.6**.
  → Build with Xcode 26.6, iPadOS 26 SDK. **Deployment target iPadOS 26.0** (Liquid Glass is 26.0+, so no `#available` fallbacks needed for glass).
- Project kept as text: `project.yml` + XcodeGen.
- Dev PC: Linux, 9 GB RAM. **No real-model runs on the PC, ever** (owner). Swift toolchain gets installed for
  pure-logic tests; a tiny random-weight Inkling (a few MB) is generated for engine maths tests (see §9).
- Only 4 small text files from the model folder are read on the PC: `config.json`, `model.safetensors.index.json`,
  `chat_template.jinja`, `tokenizer_config.json` (owner downloads the model to `~/Downloads`). Weights never leave the iPad flow.
- Libraries (latest at time of writing): mlx-swift **0.33.0**, mlx-swift-lm **3.33.0** (has no Inkling port; has MoE models such as Qwen3MoE to learn from).
- Public GitHub repo, created with `gh` under the owner's account. App name/bundle: "Inkling Pocket".

## 6. Approaches considered (owner picked)

### Engine
| Option | Trade-off | Picked |
|---|---|---|
| **MLX Swift port + custom Metal kernels on the hot paths** | Talking early, then tuned where measurement says it's slow | ✅ owner |
| Pure Metal from scratch | Max control, but weeks before first token, big bug surface | ✗ |
| Use an existing iOS LLM app/engine | None support `inkling2_loop` | ✗ |

### Fitting 11.6 GB into 12 GB
| Option | Trade-off | Picked |
|---|---|---|
| **Memory-mapped weights + hot-expert residency** (owner removes app cap) | Best quality; speed depends on cache hit rate | ✅ (owner: "custom patch removes the cap") |
| Custom smaller build (~7 GB) | Faster, worse answers, needs a 64 GB PC to make | ✗ |

### Web crawling ("crawl4ai in the app")
| Option | Trade-off | Picked |
|---|---|---|
| **Swift port**: hidden WKWebView renders page (JS), crawl4ai's pruning filter + HTML→Markdown rewritten in Swift | Fast, ~1 MB, works on Wi-Fi or cellular | ✅ owner |
| Embedded Python running real crawl4ai cleaning code | +40 MB, cross-compiled lxml, slower | ✗ |
| Real crawl4ai (Playwright + Chromium) | Impossible on iPadOS (no Chromium, no subprocesses) | ✗ |

### Search source
- Owner wanted crawl4ai, working on any network (Wi-Fi or cellular), no PC/server dependency.
  → Search = crawl DuckDuckGo's HTML/lite results page with the same in-app crawler, then crawl top results.

## 7. The design

### 7.1 Modules (Swift packages inside one repo)

```
InklingPocket (app target, SwiftUI)
 ├─ InklingEngine      model port, weight loader, Metal kernels, KV/conv caches, sampler
 ├─ InklingTokenizer   tiktoken BPE + chat template + stream parser (thinking / text / tool calls)
 ├─ AgentCore          agent loop, tool protocol, tool registry, context budget
 ├─ PocketCrawler      WKWebView fetcher + Swift ports of crawl4ai PruningContentFilter & markdown generator
 ├─ PocketMemory       SwiftData facts + chat history; on-device embeddings for search
 └─ OpenUIKit          OpenUI Lang streaming parser + SwiftUI component renderers
```
Pure-logic packages (`InklingTokenizer` parser parts, `AgentCore`, `OpenUIKit` parser,
crawler cleaning) have **no Apple-only imports** where possible, so they unit-test fast.

### 7.2 Engine (InklingEngine)

Phase A — **correct first**
- Port `inkling2_loop` from mlx-vlm `models/inkling2/language.py` (+ `grouped_experts.py`, `kernels.py`) to MLX Swift.
  Loop layers 6–17 ×3 with **per-visit caches**. Short-convolution kernel ported via `MLXFast.metalKernel`.
- Load mixed per-module quantization from `config.json`.
- Text first; vision tower ported later (stretch).

Phase B — **memory**
- Weights opened with **`mmap`** (file-backed, read-only), page-aligned per tensor, wrapped as no-copy
  `MTLBuffer`s and handed to MLX. Under pressure iPadOS drops pages and re-reads them instead of killing the app.
  (Needs extended virtual addressing for an 11.6 GB mapping — provided by iRAM+ via LiveContainer.)
- **Budget discovery:** at launch read `os_proc_available_memory()` (reflects the real jetsam limit inside
  LiveContainer); show it in Benchmark; Settings has a manual override. On a host without the raised limit the app
  still loads (all experts mmapped, fewer pinned) — slower, never a crash.
- Non-expert weights (attention, embeddings, head, norms, router: the always-used parts) are **wired/pinned**.
- Routed experts: **hot-expert residency** — track per-layer expert hit counts; keep the hottest
  pinned within a memory budget (set from `os_proc_available_memory()` at launch); cold experts stay
  mmapped and are allowed to be evicted. Prefetch: router of layer N is known before its experts run →
  `madvise(WILLNEED)` the chosen experts while attention runs.
- KV cache: sliding-window layers use fixed 1,024-slot ring buffers; 8 global visits grow.
  KV cache quantized to 8-bit as an option if memory forces it. Context cap set in Settings (default 16K).

Phase C — **speed** (each step only if the on-device benchmark says it helps)
- Fused MoE kernel: gather 8 experts + 2/3/4-bit dequant + matmul + SwiGLU in one pass (single-token decode path).
- Whole-token graph compiled once (`compile`) → one GPU submission per token; no per-token allocations.
- Looped layers: reuse the same weight buffers across the 3 visits (no re-binding cost).
- Sampler on GPU (top-k/top-p/temperature without copying 201K logits to CPU).
- Prefill in chunks (e.g. 256) for fast first token.
- Thermal: report throttling in the HUD; no fake numbers.

Built-in **Benchmark screen** (owner has no Mac): runs fixed prompts, reports prefill tok/s,
decode tok/s, time-to-first-token, peak memory, expert cache hit rate, and a pass/fail
**numerics self-test** against stored reference logits. Results copyable as text so the owner can paste them back.

### 7.3 Tokenizer and chat format (InklingTokenizer)
- Own tiktoken-style BPE in Swift from `tiktoken/tokenizer.model` + special tokens from `tokenizer.json` (no swift-transformers dependency).
- Chat prompt built by a native Swift formatter that reproduces `chat_template.jinja` byte-for-byte (the template uses HF-only tags such as `{% generation %}`), with `reasoning_effort` and `tools`.
- **Streaming parser** splits model output into: thinking stream, answer stream, tool calls (`invoke_tool_json`), end of message.

### 7.4 Agent (AgentCore)
- Loop: build prompt → generate → if tool call: run tool, append result, continue → until final answer.
  Max 8 tool steps per user message (Settings).
- Tools v1:
  - `web_search(query)` → top results (title, url, snippet) from DuckDuckGo via PocketCrawler.
  - `open_url(url)` → clean markdown of the page (PocketCrawler), trimmed to a token budget.
  - `memory_save(fact)`, `memory_search(query)`, `memory_forget(id)`.
  - `chat_search(query)` → snippets from past chats.
  - `current_time`, `calculator` (cheap, makes the model more reliable).
- Tool results that could be huge are **summarised by token budget**, not dumped (context is RAM).
- Default `reasoning_effort = low` (abliteration was tuned at low); changeable per chat.
- KV-cache **prefix reuse**: system prompt + tool definitions are cached once per chat, not re-prefilled each turn.

### 7.5 Crawler (PocketCrawler) — "crawl4ai, in Swift"
- Fetch: one hidden, reused `WKWebView` (non-persistent data store), waits for load + short JS idle; fallback
  to plain `URLSession` for simple pages. Works on Wi-Fi or cellular.
- Clean: Swift port of crawl4ai's **PruningContentFilter** (text density / link density / tag weight scoring)
  and **markdown generator** (with links as citations). HTML parsed with SwiftSoup.
- Search: crawl DuckDuckGo results page → parse results → optionally crawl top N in parallel (N=3 default).
- Timeouts, max page size, and robots-respectful user agent (generic, no personal info).

### 7.6 Memory (PocketMemory)
- SwiftData: `Chat`, `Message`, `Fact` (text, created, source chat, pinned).
- **Auto facts**: after each exchange the agent may call `memory_save`; facts are shown in a Memory screen
  where the owner can edit/delete.
- Search: on-device sentence embeddings (Apple `NaturalLanguage` `NLContextualEmbedding`) + keyword match.
  Relevant facts are injected into the system prompt at chat start (small budget, e.g. 300 tokens).

### 7.7 OpenUI (OpenUIKit)
- System prompt gets the OpenUI Lang component library description (generated from the Swift component registry,
  as OpenUI does from its Zod registry).
- Streaming parser renders **only validated parts**, progressively, as tokens arrive; malformed fragments dropped.
- v1 components: Card, Table, List, Chart (Swift Charts: bar/line/pie), Callout, CodeBlock, Steps, Buttons
  (tap = sends a follow-up message), simple Form (fields → sends values back to the agent), Image (URL).
- Plain markdown still works for normal answers.

### 7.8 UI
- `NavigationSplitView`: sidebar = chats + Memory + Benchmark + Settings; detail = chat.
- Chat: streaming text, collapsible "Thinking" section, tool-use chips ("Searched: …", "Read: …"), OpenUI blocks.
- Liquid Glass: standard bars/toolbars get it automatically; custom floating **composer bar** and **tok/s HUD**
  use `.glassEffect()` inside a `GlassEffectContainer`. Glass only on controls, never on message content.
  Respect Reduce Transparency / Reduce Motion.
- First run: **Import model** screen → Files folder picker for `Inkling-2-Mini-abliterated-MLX-12GB`; app checks
  every file exists and sizes match `model.safetensors.index.json`; copies into app storage (11.6 GB) or uses it in place
  via a security-scoped bookmark (builder decides after testing which is faster/safer).
- Dynamic Type, VoiceOver labels, dark mode, hardware keyboard (⌘↩ send, ⌘N new chat), pointer.

## 8. Delivery
- GitHub Actions on `macos-26`, Xcode 26.6: XcodeGen → build → tests → unsigned `InklingPocket.ipa` artifact.
- First commit: empty app that builds green in CI. Then features.
- Owner downloads the `.ipa` from the run and **imports it into LiveContainer** (LiveContainer signs it with its own
  certificate; no signing needed on our side). No entitlements in our `project.yml` matter at runtime.
- LiveContainer specifics to verify on device early (M0 smoke build): Metal device available, WKWebView works,
  Files folder picker + security-scoped bookmark work, SwiftData container path is writable.
- Public repo contents: code only. **No model files, tokenizer or chat template are committed** — they are read from
  the owner's imported model folder. Fixtures come from a tiny random-weight model we generate. Ports of mlx-vlm (MIT),
  crawl4ai (Apache-2.0) and OpenUI (MIT) keep their license headers + a `THIRD_PARTY_NOTICES.md`.

### What "v1 done" means (builder's call, owner can change)
1. Engine matches reference numerics (tiny model in CI + full-model self-test on device).
2. Chat with thinking, tools (search, open_url, memory, chat_search), OpenUI v1 components, Liquid Glass UI.
3. Decode speed is **reported honestly**. Features ship even if speed is 15–29 tok/s; speed work continues after,
   driven by the owner's Benchmark results. Below 15 tok/s, speed work is prioritised before new features.

## 9. Testing (no Mac, no device in the loop)
- **Golden numerics**: a tiny random-weight `inkling2_loop` config (same structure: looped layers, MoE, sinks,
  conv, sliding + global attention, mixed quant) generated on the Linux PC with Python MLX (CPU) + mlx-vlm `d998a62`;
  reference logits saved as fixtures. Swift engine must match within tolerance — run in CI on the macOS runner.
- Unit tests (Swift Testing): tokenizer round-trips vs Python tiktoken fixtures, chat template vs Python render,
  stream parser, OpenUI parser, crawler cleaning on saved HTML fixtures, agent loop with a scripted fake model.
- **Real-model tests run on the iPad only** (owner). The app ships two engine paths:
  - **Reference path:** plain MLX Swift port, no custom kernels, no residency tricks — slow, easy to trust.
  - **Fast path:** custom Metal kernels + mmap/residency.
  The on-device **Self-test** runs both on fixed prompts and checks logits match within tolerance (teacher-forced).
  Plus **known-answer prompts** (greedy, e.g. "capital of France?" → contains "Paris").
- On-device **Benchmark** also records **per-layer, per-pass expert routing counts** → decides residency policy and
  shows whether the 3 loop passes pick different experts (real bytes/token).
- Local fast loop: Swift toolchain on the Linux PC (owner approved) for pure-logic packages; MLX/Metal tests run in CI.
- On-device: Benchmark screen + numerics self-test (§7.2). **Nothing is called "fast" until the owner's iPad says so.**

## 10. Risks
- R1 Speed depends on expert cache hit rate; if routing is spread evenly, flash reads dominate → below 30 tok/s.
- R1b If "3.4B active" counts the looped layers' weights only once and each pass picks different experts, real bytes
  per token are up to ~2× → ceiling ~43 tok/s, realistic **~17–26 tok/s**. Measured by the on-device routing stats (§9) before
  promising numbers.
- R1c MLX Swift may not accept externally-owned no-copy buffers for quantized weights cleanly; fallback is our own
  Metal kernels reading the mmapped buffers directly for the MoE path.
- R2 MLX Swift on iPadOS may not expose wiring/residency controls; may need direct Metal buffers (`MTLHeap`, residency sets).
- R3 Publisher never ran it on Apple GPUs; Metal-specific numeric issues possible.
- R4 GitHub macOS runners have limited Metal; GPU tests may need to fall back to MLX CPU on CI.
- R5 DuckDuckGo may change or block its HTML page; crawler must fail softly ("search unavailable").
- R6 iRAM+'s raised limit on a 12 GB iPad is unmeasured; Benchmark reports it on first run.
- R7 LiveContainer quirks (WKWebView, file pickers, background suspension during long generations) — checked in the M0 smoke build.

## 11. Left to the builder
- Exact expert-residency policy and budgets, kernel tiling, prefill chunk size, compile boundaries.
- Copy model into app storage vs use in place.
- Exact OpenUI Lang subset (keep compatible with upstream syntax for the v1 components).
- File/folder layout inside each package.
