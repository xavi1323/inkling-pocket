# Inkling Pocket Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An iPad app that runs Inkling 2 Mini (abliterated, MLX 12GB build) on-device as a tool-using agent with web crawling, memory and OpenUI rendering, delivered as an unsigned IPA for LiveContainer.

**Architecture:** One XcodeGen app target (`InklingPocket`) on top of one local Swift package (`InklingKit`). Pure-logic targets build and test on Linux; Apple-only targets (MLX engine, WebKit, SwiftData, SwiftUI) build and test in GitHub Actions on `macos-26`. The engine has a slow **reference path** (plain MLX Swift port of mlx-vlm `inkling2`) and a **fast path** (mmapped weights + custom Metal kernels); the on-iPad Self-test compares them.

**Tech Stack:** Swift 6, SwiftUI, SwiftData, Swift Charts, WebKit, NaturalLanguage, mlx-swift 0.33.0 (`MLX`, `MLXNN`, `MLXFast`), SwiftSoup 2.14.4, XcodeGen 2.49.0, GitHub Actions (`actions/checkout@v7`, `actions/upload-artifact@v7`), Xcode 26.6.

**Spec:** `docs/Spec.md` (requirements with IDs) and `docs/Design.md` (approved design). Read both.

## Global Constraints

- iPadOS deployment target **26.0**; `TARGETED_DEVICE_FAMILY = 2`; Swift 6 language mode, strict concurrency complete.
- CI runner `macos-26`, `sudo xcode-select -s /Applications/Xcode_26.6.app`.
- App runs inside **LiveContainer**: never rely on our own entitlements, bundle ID, or app groups.
- **No real-model runs on the Linux PC.** Only the 4 small files (`config.json`, `model.safetensors.index.json`, `chat_template.jinja`, `tokenizer_config.json`) plus `tokenizer.json` / `tiktoken/tokenizer.model` may be read there, from `~/Downloads/orcarouter_Inkling-2-Mini-abliterated-MLX/Inkling-2-Mini-abliterated-MLX-12GB/`. Tiny random-weight models and Python tokenizers/jinja2 are allowed on the PC.
- **Never commit** model weights, `tokenizer.json`, `tiktoken/`, `chat_template.jinja` or any file copied from the model folder. Test fixtures that need tokenizer data are generated outputs (ids, strings), not the tokenizer itself.
- Reference sources (pin exactly): mlx-vlm commit `d998a6207b7dc613b52618646dbea13a4799f7ca` files `mlx_vlm/models/inkling2/{language,config,kernels,inkling2}.py`, `mlx_vlm/models/inkling/language.py`, `mlx_vlm/models/switch_layers.py`, `mlx_vlm/models/cache.py`; crawl4ai tag `v0.9.6` (`crawl4ai/content_filter_strategy.py` `PruningContentFilter`, `crawl4ai/markdown_generation_strategy.py`, `crawl4ai/html2text/`); OpenUI commit `7ce17a1181ebf3a8ba53191ec570e1949e20ae7c` (`spec/language.md`, `spec/prompt.md`). Ported files keep the upstream license header.
- Special tokens (from `tokenizer_config.json`): `<|message_user|>`, `<|message_model|>`, `<|message_system|>`, `<|message_tool|>`, `<|content_text|>`, `<|content_thinking|>`, `<|content_invoke_tool_json|>`, `<|content_tool_error|>`, `<|content_xml|>`, `<|end_message|>`, `<|content_model_end_sampling|>` (EOS, id 200006), `<|endoftext|>` (pad).
- Model text config facts: hidden 2304, 24 layers, loop `core_start 6, core_end 18, num_passes 3` (48 layer visits), 128 experts / 8 active, intermediate 1152, sliding window 1024 on `local_layer_ids`, `final_logit_softcapping 30.0`, `unpadded_vocab_size 200058`, vocab 201024.
- Defaults: `reasoning_effort = low`, max tool steps 8, tool result budget 2,000 tokens, context cap 16,384 tokens, prefill chunk 256, crawl top-N 3, crawl timeout 15 s, memory injection ≤300 tokens.
- User rules: commit after every task step marked Commit; **no destructive git or file actions** (no force-push, no `reset --hard`, no deleting files outside the task's list); end commit messages with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Network: generic User-Agent string `Mozilla/5.0 (iPad; CPU OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1`; never send personal data.

## Review Focus

1. **iPad locks / LiveContainer goes to background mid-generation** → generation is cancelled cleanly, the partial answer is saved with a "stopped" mark, no crash on return. (Task 17 test `cancelMidStreamSavesPartial`.)
2. **Chat longer than the context cap** → oldest turns are dropped (system prompt, memory block and last user turn kept), never an overflow or crash. (Task 6 test `contextWindowDropsOldestTurns`.)
3. **`open_url` on non-HTML or huge pages** (PDF, image, 20 MB HTML, non-UTF-8) → short "unsupported content type" or trimmed text, no hang. (Task 8 tests `openURLRejectsPDF`, `openURLTrimsHugePage`.)
4. **Model folder copied partially** (download/copy interrupted, truncated shard) → validator reports which shard is truncated by checking the safetensors header against file size. (Task 5 test `truncatedShardDetected`.)
5. **No network (airplane mode)** → `web_search` returns "search unavailable: offline" within 3 s and the agent still answers. (Task 8 test `searchFailsSoftWhenOffline`.)

---

## File Structure

```
inkling-pocket/
  project.yml                      XcodeGen app definition
  .github/workflows/ci.yml         linux-tests + ios-build jobs
  THIRD_PARTY_NOTICES.md
  App/                             InklingPocket app target (SwiftUI)
    InklingPocketApp.swift
    Smoke/SmokeView.swift          M0 LiveContainer checks
    Chat/  Sidebar/  Import/  Memory/  Bench/  Settings/  Components/
  InklingKit/Package.swift
  InklingKit/Sources/
    InklingCore/      (Linux) ChatTypes, PromptFormatter, Tokenizer, StreamParser, ModelFolder
    AgentCore/        (Linux) Agent, Tool, ToolRegistry, ContextWindow, built-in tools, memory protocols
    CrawlCore/        (Linux) PruningContentFilter, MarkdownGenerator, DuckDuckGoParser, crawl tools
    OpenUICore/       (Linux) AnswerSegmenter, OpenUIParser, ComponentRegistry
    InklingEngine/    (Apple) config, MappedSafetensors, Inkling2 model, caches, generator, residency, kernels, self-test
    PocketCrawler/    (Apple) WebPageFetcher (WKWebView)
    PocketMemory/     (Apple) SwiftData models, FactStore, ChatSearch, NLEmbedder
    OpenUIViews/      (Apple) SwiftUI renderers
  InklingKit/Tests/<Target>Tests/  Swift Testing suites + Fixtures/
  scripts/                         Python fixture generators (run on PC, outputs committed)
```

---

### Task 0: Toolchain, repo, CI, smoke app (M0)

**Files:**
- Create: `project.yml`, `App/InklingPocketApp.swift`, `App/Smoke/SmokeView.swift`, `.github/workflows/ci.yml`, `.gitignore`, `README.md`

**Interfaces:**
- Produces: CI workflow with job `ios-build` (artifact name `InklingPocket-ipa`, file `InklingPocket.ipa`); app target name `InklingPocket`, bundle id `com.inklingpocket.app`.

- [ ] **Step 1:** Install Swift on the PC via `swiftly` (user-local, no root): `curl -O https://download.swift.org/swiftly/linux/swiftly-$(uname -m).tar.gz`, extract, `./swiftly init --quiet-shell-followup`, then `swiftly install latest`. Verify `swift --version` prints 6.x.
- [ ] **Step 2:** `git init` in `/home/xavi/Projects/inkling-pocket`; `.gitignore` covers `*.safetensors`, `tokenizer.json`, `tiktoken/`, `chat_template.jinja`, `.build/`, `out/`, `*.ipa`, `Payload/`, `*.xcodeproj`, `.venv/`.
- [ ] **Step 3:** `project.yml`: iOS app `InklingPocket`, deployment 26.0, device family 2, Swift 6, `INFOPLIST_KEY_UISupportsDocumentBrowser = NO`, `UIFileSharingEnabled = YES`, `LSSupportsOpeningDocumentsInPlace = YES`, display name "Inkling Pocket"; local package `InklingKit` (added in Task 1; leave the hook commented now).
- [ ] **Step 4:** `SmokeView` shows 4 rows with pass/fail: Metal (`MTLCreateSystemDefaultDevice()?.name`), WebKit (hidden `WKWebView` loads `https://example.com`, reports page title), Files (folder picker → security-scoped bookmark resolves), Storage (write+read a file in Application Support), plus `os_proc_available_memory()` in GB.
- [ ] **Step 5:** `ci.yml` job `ios-build` on `macos-26`: checkout@v7 → `sudo xcode-select -s /Applications/Xcode_26.6.app` → `brew install xcodegen` → `xcodegen generate` → `xcodebuild -scheme InklingPocket -sdk iphoneos -configuration Release -derivedDataPath out CODE_SIGNING_ALLOWED=NO build` → package `Payload/InklingPocket.app` into `InklingPocket.ipa` → upload-artifact@v7.
- [ ] **Step 6:** Commit. `gh repo create inkling-pocket --public --source . --push`.
- [ ] **Step 7:** `gh run watch` the CI run; read logs with `gh run view --log-failed` on failure. Expected: green, artifact `InklingPocket-ipa` present. Hand the IPA to the owner for the LiveContainer smoke check (all 4 rows pass).

### Task 1: InklingKit package skeleton + Linux test job

**Files:**
- Create: `InklingKit/Package.swift`, one placeholder source file per target, `InklingKit/Tests/InklingCoreTests/SmokeTests.swift`
- Modify: `project.yml` (add local package, link Apple products), `.github/workflows/ci.yml` (add `linux-tests`, add package tests to `ios-build`)

**Interfaces:**
- Produces: library products `InklingCore`, `AgentCore`, `CrawlCore`, `OpenUICore` (all platforms) and `InklingEngine`, `PocketCrawler`, `PocketMemory`, `OpenUIViews` (Apple only, declared inside `#if !os(Linux)` in `Package.swift`). Dependencies: `mlx-swift` `exact: "0.33.0"` (InklingEngine), `SwiftSoup` `exact: "2.14.4"` (CrawlCore). Platforms `.iOS("26.0"), .macOS("26.0")`.

- [ ] **Step 1:** Test `@Test func packageBuilds() { #expect(InklingCore.version == "0.1.0") }`.
- [ ] **Step 2:** Run `swift test --package-path InklingKit` → FAIL (no symbol).
- [ ] **Step 3:** Add `public enum InklingCore { public static let version = "0.1.0" }` and the skeleton.
- [ ] **Step 4:** Run `swift test --package-path InklingKit` → PASS on Linux.
- [ ] **Step 5:** CI: `linux-tests` job on `ubuntu-latest` using the official `swift` Docker image (take the newest 6.x tag from Docker Hub at execution time) running `swift test --package-path InklingKit`. `ios-build` adds `xcodebuild test -scheme InklingKit-Package -destination 'platform=macOS'` (engine tests use MLX CPU device).
- [ ] **Step 6:** Commit, push, confirm both jobs green.

### Task 2: Chat types + prompt formatter (T2)

**Files:**
- Create: `InklingKit/Sources/InklingCore/ChatTypes.swift`, `PromptFormatter.swift`, `scripts/render_template_fixtures.py`, `InklingKit/Tests/InklingCoreTests/Fixtures/prompts/*.json`, `PromptFormatterTests.swift`

**Interfaces:**
- Produces:
  - `public enum Role: String, Codable, Sendable { case system, user, assistant, tool }`
  - `public struct ToolCall: Codable, Sendable, Equatable { public var name: String; public var arguments: [String: JSONValue] }`
  - `public enum JSONValue: Codable, Sendable, Equatable` (null/bool/number/string/array/object) with `func canonicalJSON(sortKeys: Bool) -> String` (separators `,` and `:`).
  - `public struct ChatMessage: Codable, Sendable, Equatable { role; content: String; reasoning: String?; toolCalls: [ToolCall]; toolName: String?; isToolError: Bool }`
  - `public struct ToolSpec: Codable, Sendable, Equatable { name; description; parameters: JSONValue }`
  - `public enum ReasoningEffort: String, CaseIterable, Codable, Sendable { case none, minimal, low, medium, high, xhigh, max }` with `var value: Double` (0, 0.1, 0.2, 0.7, 0.9, 0.99, 0.99).
  - `public enum PromptFormatter { static func render(messages: [ChatMessage], tools: [ToolSpec], effort: ReasoningEffort) -> String }`

- [ ] **Step 1:** `scripts/render_template_fixtures.py` (Python + `jinja2` only, in a `uv` venv) renders the model's `chat_template.jinja` for 6 cases and writes `{input, expected}` JSON: (a) user only; (b) system + user; (c) tools + user; (d) assistant with reasoning + text; (e) assistant tool call + tool result + final answer; (f) tool error. Replace `{% generation %}…{% endgeneration %}` with its body before rendering; add `raise_exception`. Effort `low`.
- [ ] **Step 2:** Test `@Test(arguments: fixtureNames) func matchesJinja(name:)` — `#expect(PromptFormatter.render(...) == expected)` byte-for-byte.
- [ ] **Step 3:** Run → FAIL.
- [ ] **Step 4:** Implement `render` natively (no Jinja engine): effort line `<|message_system|><|content_text|>Thinking effort level: <num><|end_message|>` emitted before the first non-system message (`0` for none, else Python `str(float)` form e.g. `0.2`); tools block `<|message_system|>tool_declare<|content_xml|>` + sorted-key compact JSON of `[{description,name,parameters,type:"function"}]` + `<|end_message|>`; tool calls `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{…sorted…}}<|end_message|>`; assistant turns end with `<|content_model_end_sampling|>`; tool messages `<|message_tool|>NAME<|content_text|>` (or `<|content_tool_error|>` when `isToolError`).
- [ ] **Step 5:** Run → PASS. Commit (fixture JSON + script, no template file).

### Task 3: Tokenizer (T1)

**Files:**
- Create: `InklingKit/Sources/InklingCore/Tokenizer.swift`, `scripts/tokenizer_fixtures.py`, `Tests/InklingCoreTests/Fixtures/tokens.json`, `TokenizerTests.swift`

**Interfaces:**
- Produces: `public final class InklingTokenizer: Sendable { public init(modelFolder: URL) throws; func encode(_ text: String) -> [Int32]; func decode(_ ids: [Int32]) -> String; func specialTokenID(_ token: String) -> Int32?; var eosID: Int32 }` and `public struct StreamingDecoder { mutating func append(_ id: Int32) -> String }` (emits only complete UTF-8).

- [ ] **Step 1:** Fixture script (Python `tokenizers` lib) encodes 30 strings (English, emoji, CJK, code, numbers `1234567`, whitespace runs, `\r\n`, every special token inline, a rendered prompt from Task 2) → `{text, ids}`.
- [ ] **Step 2:** Tests: `encodeMatchesReference` (all 30), `decodeRoundTrip`, `specialTokensAreSingleIDs` (`<|content_model_end_sampling|>` → 200006), `streamingDecoderNeverSplitsUTF8` (feed emoji ids one by one; concatenation == decode(all)). Tests skip with a clear message when env `INKLING_MODEL_DIR` is unset; CI passes nothing → tests skip on CI, run on PC.
- [ ] **Step 3:** Run → FAIL.
- [ ] **Step 4:** Implement tiktoken BPE from `tiktoken/tokenizer.model` (base64 token + rank per line) + added tokens from `tokenizer.json`; split regex copied verbatim from `tokenizer.json` `pre_tokenizer` (via `NSRegularExpression`); byte-level BPE merge by lowest rank; specials matched before regex split.
- [ ] **Step 5:** `INKLING_MODEL_DIR=… swift test --filter Tokenizer` → PASS. Commit.

### Task 4: Stream parser (T3)

**Files:** Create `InklingKit/Sources/InklingCore/StreamParser.swift`, `Tests/InklingCoreTests/StreamParserTests.swift`

**Interfaces:**
- Produces: `public enum ModelEvent: Sendable, Equatable { case thinking(String), answer(String), toolCall(ToolCall), toolCallError(raw: String), end }`; `public struct InklingStreamParser { public init(); mutating func feed(_ text: String) -> [ModelEvent]; mutating func finish() -> [ModelEvent] }`.

- [ ] **Step 1:** Tests: `thinkingThenAnswer` (`<|message_model|><|content_thinking|>hm<|end_message|><|message_model|><|content_text|>Hi<|end_message|><|content_model_end_sampling|>` → `[.thinking("hm"), .answer("Hi"), .end]`, adjacent same-kind events may be merged); `markerSplitAcrossChunks` (feed one character at a time → same result); `toolCallParsed` (`<|message_model|>web_search<|content_invoke_tool_json|>{"name":"web_search","args":{"query":"x"}}<|end_message|>` → `.toolCall(name "web_search", args query "x")`); `badToolJSONReported` → `.toolCallError(raw:)`; `textNeverLeaksMarkers` (no `<|` in any emitted text).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement as a small state machine holding back up to the longest marker length. **Step 4:** Run → PASS. **Step 5:** Commit.

### Task 5: Model folder validator (M2, M4)

**Files:** Create `InklingKit/Sources/InklingCore/ModelFolder.swift`, `Tests/InklingCoreTests/ModelFolderTests.swift` (fixtures built in a temp dir by the test itself)

**Interfaces:**
- Produces: `public struct ModelManifest: Sendable { folder: URL; shards: [URL]; totalBytes: Int64; modelType: String }`; `public enum ModelFolderError: Error, Equatable { case missingFile(String), wrongArchitecture(String), unreadableIndex, missingShard(String), truncatedShard(String) }` with `var userMessage: String`; `public enum ModelFolder { static func validate(_ url: URL) throws -> ModelManifest }`.

- [ ] **Step 1:** Tests: `validFolderPasses`, `missingConfigReported`, `wrongModelTypeReported` (`model_type` not prefixed `inkling2`), `missingShardReported`, `truncatedShardDetected` (shard whose size < 8 + headerLen + max `data_offsets` end).
- [ ] **Step 2:** FAIL. **Step 3:** Implement: required files `config.json`, `model.safetensors.index.json`, `tokenizer.json`, `tiktoken/tokenizer.model`, `chat_template.jinja`, `tokenizer_config.json`; read each shard's 8-byte little-endian header length + header JSON only. **Step 4:** PASS. **Step 5:** Commit.

### Task 6: Agent loop, context window, simple tools (A1, A3, A5, part A2)

**Files:** Create `AgentCore/LanguageModel.swift`, `Tool.swift`, `Agent.swift`, `ContextWindow.swift`, `Tools/CurrentTimeTool.swift`, `Tools/CalculatorTool.swift`, `Tests/AgentCoreTests/{AgentTests,ContextWindowTests,SimpleToolTests}.swift`, `Tests/AgentCoreTests/ScriptedModel.swift`

**Interfaces:**
- Consumes: `ChatMessage`, `ToolSpec`, `ToolCall`, `ModelEvent`, `ReasoningEffort` (Task 2/4).
- Produces:
  - `public struct GenerateParameters: Sendable { temperature: Float = 0.7; topP: Float = 0.95; topK: Int = 0; repetitionPenalty: Float = 1.0; maxTokens: Int = 4096 }`
  - `public protocol LanguageModel: Sendable { func generate(_ messages: [ChatMessage], tools: [ToolSpec], effort: ReasoningEffort, parameters: GenerateParameters) -> AsyncThrowingStream<ModelEvent, Error>; func countTokens(_ text: String) -> Int }`
  - `public protocol Tool: Sendable { var spec: ToolSpec { get }; func run(_ arguments: [String: JSONValue]) async throws -> String }`
  - `public struct ToolRegistry: Sendable { init(_ tools: [any Tool]); var specs: [ToolSpec]; func tool(named: String) -> (any Tool)? }`
  - `public struct AgentSettings: Sendable { maxSteps = 8; toolResultTokenBudget = 2000; contextCap = 16384; effort: ReasoningEffort = .low; parameters = GenerateParameters() }`
  - `public enum AgentEvent: Sendable, Equatable { case thinking(String), answer(String), toolStarted(ToolCall), toolFinished(name: String, preview: String, isError: Bool), stepLimitReached, finished([ChatMessage]) }`
  - `public actor Agent { init(model: any LanguageModel, tools: ToolRegistry, settings: AgentSettings); func run(history: [ChatMessage], systemPrompt: String) -> AsyncThrowingStream<AgentEvent, Error> }`
  - `public enum ContextWindow { static func fit(system: String, history: [ChatMessage], cap: Int, count: (String) -> Int) -> [ChatMessage] }`
  - `public func trimToBudget(_ text: String, budget: Int, count: (String) -> Int) -> String` (appends `\n[trimmed]`).

- [ ] **Step 1:** Tests with `ScriptedModel` (replays queued event lists): `answerWithoutTools`, `toolCallThenAnswer` (history gains assistant tool call + tool message + answer), `unknownToolReturnsError` (tool message has `isToolError`), `badToolJSONFedBack`, `stopsAtMaxSteps` (emits `.stepLimitReached`), `toolResultTrimmed` (5,000-token result → ≤2,000), `contextWindowDropsOldestTurns` (keeps system + last user turn), `calculatorEvaluates` (`"2*(3+4)"` → `"14"`, `"1/0"` → error text), `currentTimeISO8601`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. Calculator: own recursive-descent parser (+ − × ÷ ^ parentheses, decimals), no `NSExpression`. **Step 4:** PASS. **Step 5:** Commit.

### Task 7: crawl4ai cleaning port (C2)

**Files:** Create `CrawlCore/PruningContentFilter.swift`, `CrawlCore/MarkdownGenerator.swift`, `scripts/crawl_fixtures.py`, `Tests/CrawlCoreTests/Fixtures/pages/*.html` + `*.md`, `Tests/CrawlCoreTests/CleaningTests.swift`

**Interfaces:**
- Produces: `public struct PruningContentFilter { init(threshold: Double = 0.48, thresholdType: ThresholdType = .fixed, minWordThreshold: Int? = nil); func filter(html: String) throws -> [String] }`; `public struct MarkdownResult: Sendable { markdown: String; fitMarkdown: String; references: [URL] }`; `public enum MarkdownGenerator { static func generate(html: String, baseURL: URL, filter: PruningContentFilter?) throws -> MarkdownResult }` (citations as `⟨n⟩` + reference list, like crawl4ai `markdown_with_citations`).

- [ ] **Step 1:** Fixture script uses the owner's existing crawl4ai Python env (v0.9.6, run `-I`, no browser): for 8 saved HTML pages (Wikipedia article, news article, docs page, GitHub README, forum thread, recipe, product page, HTML with tables + code) write `DefaultMarkdownGenerator(content_filter=PruningContentFilter())` `raw_markdown` and `fit_markdown`.
- [ ] **Step 2:** Tests: `fitMarkdownSimilar` per page — line-level similarity (LCS lines / max lines) ≥ 0.90 after whitespace normalisation; `tablesBecomeMarkdownTables`; `codeBlocksFenced`; `linksBecomeCitations`.
- [ ] **Step 3:** FAIL. **Step 4:** Port with SwiftSoup, following v0.9.6 scoring (text density, link density, tag weights, class/id negative patterns). **Step 5:** PASS. **Step 6:** Commit.

### Task 8: Search + open_url tools (C3, C4)

**Files:** Create `CrawlCore/PageFetcher.swift`, `CrawlCore/DuckDuckGoParser.swift`, `CrawlCore/WebSearchTool.swift`, `CrawlCore/OpenURLTool.swift`, `Tests/CrawlCoreTests/Fixtures/ddg.html`, `Tests/CrawlCoreTests/CrawlToolTests.swift`

**Interfaces:**
- Consumes: `Tool` (Task 6), `MarkdownGenerator` (Task 7).
- Produces: `public struct FetchedPage: Sendable { html: String?; finalURL: URL; contentType: String; byteCount: Int }`; `public protocol PageFetcher: Sendable { func fetch(_ url: URL, timeout: TimeInterval) async throws -> FetchedPage }`; `public enum FetchError: Error { case offline, timeout, http(Int), unsupported(String) }`; `public struct SearchResult: Sendable, Equatable { title; url: URL; snippet }`; `DuckDuckGoParser.parse(html:) -> [SearchResult]` (decodes `uddg=` redirect links, skips ads); `WebSearchTool(fetcher:, crawlTopN: 3)` tool name `web_search` args `{query}`; `OpenURLTool(fetcher:)` tool name `open_url` args `{url}`. Search URL `https://html.duckduckgo.com/html/?q=<encoded>`.

- [ ] **Step 1:** Save one real DDG HTML results page as fixture (fetched once with curl, generic UA).
- [ ] **Step 2:** Tests (with `FakeFetcher`): `parsesDDGResults` (≥5 results, absolute URLs, no `duckduckgo.com/l/`), `searchCrawlsTopN` (3 fetches after the search page), `searchFailsSoftWhenOffline` (fetcher throws `.offline` → tool returns text starting `search unavailable: offline`, completes < 3 s), `openURLRejectsPDF` (`application/pdf` → `unsupported content type: application/pdf`), `openURLTrimsHugePage` (20 MB HTML → output ≤ budget, completes < 5 s).
- [ ] **Step 3:** FAIL. **Step 4:** Implement; also `URLSessionFetcher: PageFetcher` (Linux-capable fallback). **Step 5:** PASS. **Step 6:** Commit.

### Task 9: WKWebView fetcher (C1)

**Files:** Create `PocketCrawler/WebPageFetcher.swift`, `Tests/PocketCrawlerTests/WebPageFetcherTests.swift`

**Interfaces:**
- Consumes: `PageFetcher`, `FetchedPage`, `FetchError`, `URLSessionFetcher` (Task 8).
- Produces: `@MainActor public final class WebPageFetcher: PageFetcher` — one hidden reused `WKWebView` (`.nonPersistent()` store, UA from Global Constraints); navigates, waits for `didFinish` + 600 ms JS idle, returns `document.documentElement.outerHTML`; non-HTML content types → falls back to `URLSessionFetcher`; serialises requests.

- [ ] **Step 1:** Tests (macOS CI): `loadsLocalHTMLFile` (file URL fixture with JS that inserts text after 200 ms → text present), `timeoutThrows` (1 s timeout on a never-finishing page → `.timeout`).
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS in CI. **Step 5:** Commit.

### Task 10: Memory (R1–R3, memory tools, chat_search)

**Files:** Create `AgentCore/MemoryProtocols.swift`, `AgentCore/Tools/MemoryTools.swift`, `AgentCore/Tools/ChatSearchTool.swift`, `PocketMemory/Models.swift`, `PocketMemory/FactStore.swift`, `PocketMemory/ChatSearch.swift`, `PocketMemory/NLEmbedder.swift`, tests in `AgentCoreTests/MemoryToolTests.swift`, `PocketMemoryTests/FactStoreTests.swift`

**Interfaces:**
- Produces (AgentCore): `public struct StoredFact: Sendable, Equatable { id: UUID; text: String; created: Date }`; `public protocol FactStoring: Sendable { func save(_ text: String, sourceChat: UUID?) async throws -> UUID; func search(_ query: String, limit: Int) async throws -> [StoredFact]; func forget(_ id: UUID) async throws; func all() async throws -> [StoredFact] }`; `public protocol ChatSearching: Sendable { func search(_ query: String, limit: Int) async throws -> [(chatTitle: String, snippet: String)] }`; tools `memory_save {fact}`, `memory_search {query}`, `memory_forget {id}`, `chat_search {query}`; `public func memoryBlock(facts: [StoredFact], budget: Int = 300, count: (String) -> Int) -> String`.
- Produces (PocketMemory): `@Model ChatRecord { id: UUID; title: String; created; updated; effort: String; messages: [MessageRecord] }`, `@Model MessageRecord { id; role: String; content: String; reasoning: String?; toolName: String?; toolCallsJSON: String?; isToolError: Bool; stopped: Bool; created }`, `@Model FactRecord { id; text; created; sourceChat: UUID?; pinned: Bool; embedding: Data? }`; `actor SwiftDataFactStore: FactStoring` (score = 0.7 × cosine(NLContextualEmbedding mean-pooled) + 0.3 × keyword overlap); `actor SwiftDataChatSearch: ChatSearching`.

- [ ] **Step 1:** Tests: `saveThenSearchFindsFact` (in-memory store), `forgetRemoves`, `memoryBlockRespectsBudget`, `keywordFallbackWhenNoEmbedding` (embedder returns nil → keyword match still finds "favorite color is green" for "color"), `chatSearchFindsOldMessage`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement (`ModelConfiguration(isStoredInMemoryOnly: true)` in tests). **Step 4:** PASS. **Step 5:** Commit.

### Task 11: OpenUI parser (O1, O3)

**Files:** Create `OpenUICore/AnswerSegmenter.swift`, `OpenUICore/OpenUIParser.swift`, `OpenUICore/ComponentRegistry.swift`, `Tests/OpenUICoreTests/*.swift`

**Interfaces:**
- Produces: `public enum AnswerSegment: Equatable, Sendable { case markdown(String), openUI(source: String, complete: Bool) }`; `AnswerSegmenter.segments(_ text: String) -> [AnswerSegment]` (OpenUI source lives in fenced blocks tagged `openui`); `public indirect enum OUIValue: Equatable, Sendable { string, number, bool, null, array([OUIValue]), object([String: OUIValue]), node(OUINode), ref(String) }`; `public struct OUINode: Equatable, Sendable { component: String; args: [OUIValue] ; named: [String: OUIValue] }`; `public struct OpenUIDocument: Equatable, Sendable { root: OUINode?; bindings: [String: OUIValue] }`; `public struct OpenUIParser { mutating func feed(_ chunk: String) -> OpenUIDocument }`; `public struct ComponentSpec { name; summary; params: [(name: String, type: String, required: Bool)] }`; `ComponentRegistry.v1: [ComponentSpec]` (Card, Table, List, Chart, Callout, CodeBlock, Steps, Buttons, Form, Image); `ComponentRegistry.systemPrompt() -> String`.

- [ ] **Step 1:** Read OpenUI `spec/language.md` and `spec/prompt.md` at the pinned commit. Copy 6 example programs that use only v1 components into fixtures, with expected trees written by hand.
- [ ] **Step 2:** Tests: `segmentsMarkdownAndOpenUI`, `parsesSpecExamples` (6), `partialStreamRendersValidPrefix` (feed char by char; every intermediate document is valid and node count never decreases), `malformedStatementDropped`, `unknownComponentDropped`, `systemPromptListsAllV1Components`.
- [ ] **Step 3:** FAIL. **Step 4:** Implement per the spec grammar (statements `name = Component(...)`, references, `root` binding). **Step 5:** PASS. **Step 6:** Commit.

### Task 12: OpenUI SwiftUI components (O2, O4)

**Files:** Create `OpenUIViews/OpenUIView.swift`, one file per component in `OpenUIViews/Components/`, `OpenUIViews/OpenUIAction.swift`, `Tests/OpenUIViewsTests/ActionTests.swift`

**Interfaces:**
- Consumes: `OpenUIDocument`, `OUINode` (Task 11).
- Produces: `public struct OpenUIView: View { init(document: OpenUIDocument, onAction: @escaping (OpenUIAction) -> Void) }`; `public enum OpenUIAction: Equatable, Sendable { case sendMessage(String), submitForm(id: String, values: [String: String]) }`; `public func actionMessage(_ action: OpenUIAction) -> String` (form → `Form <id> submitted: key=value, …`).

- [ ] **Step 1:** Tests: `buttonActionSendsLabel`, `formSubmitMessageFormat`, `unknownChartTypeFallsBackToTable` (pure function `chartKind(_:)`).
- [ ] **Step 2:** FAIL. **Step 3:** Implement views: Chart via Swift Charts (bar/line/pie→`SectorMark`); all text Dynamic Type; VoiceOver labels; no glass on content. **Step 4:** PASS. **Step 5:** Commit.

### Task 13: App shell UI with fake model (U1–U4, P6, A4)

**Files:** Create `App/Chat/ChatView.swift`, `App/Chat/MessageRow.swift`, `App/Chat/Composer.swift`, `App/Chat/ThinkingSection.swift`, `App/Chat/ToolChip.swift`, `App/Chat/SpeedHUD.swift`, `App/Chat/ChatSession.swift`, `App/Sidebar/SidebarView.swift`, `App/Memory/MemoryView.swift`, `App/Settings/SettingsView.swift`, `App/Settings/AppSettings.swift`, `App/Preview/EchoModel.swift`; Modify `App/InklingPocketApp.swift`
- Test: `InklingKit/Tests/AgentCoreTests/ChatSessionLogicTests.swift` for logic moved into `AgentCore/ChatSessionLogic.swift`

**Interfaces:**
- Consumes: `Agent`, `AgentEvent`, `ToolRegistry`, tools (Tasks 6–10), `OpenUIView` (Task 12), `AnswerSegmenter` (Task 11), SwiftData models (Task 10).
- Produces: `@Observable @MainActor final class ChatSession { init(chat: ChatRecord, agentFactory: () -> Agent); func send(_ text: String); func stop(); var streamingAnswer: String; var streamingThinking: String; var toolChips: [ToolChipState]; var tokensPerSecond: Double? }`; `AgentCore.ChatSessionLogic.apply(_ event: AgentEvent, to state: inout ChatTurnState)` (pure, tested); `@Observable final class AppSettings` (UserDefaults-backed: effort, temperature, topP, maxSteps, contextCap, crawlTopN, memoryBudgetOverrideGB, enginePath, keepModelInPlace).

- [ ] **Step 1:** Tests: `thinkingAccumulates`, `toolChipLifecycle` (started → finished with preview), `stopMarksTurnStopped`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement logic, then views: `NavigationSplitView` (sidebar: chats list, Memory, Benchmark, Settings); detail `ChatView`; `Composer` + `SpeedHUD` as `.glassEffect()` inside one `GlassEffectContainer`, plain material fallback when `accessibilityReduceTransparency`; ⌘↩ send, ⌘N new chat; per-chat effort picker; Stop button. `EchoModel: LanguageModel` used until Task 17 (streams the user text back, emits one fake tool call when text starts with `/tool`).
- [ ] **Step 4:** PASS + CI green. **Step 5:** Commit; owner can install this IPA to try the UI with the echo model.

### Task 14: Model import flow (M1, M3)

**Files:** Create `App/Import/ImportView.swift`, `App/Import/ModelLocation.swift`, `InklingKit/Sources/InklingCore/ModelCopier.swift`, `Tests/InklingCoreTests/ModelCopierTests.swift`

**Interfaces:**
- Consumes: `ModelFolder.validate` (Task 5).
- Produces: `public struct ModelCopier { static func copy(from: URL, to: URL, progress: @Sendable (Double) -> Void) async throws }` (cancellable; copies to `<dest>.partial` then renames); `ModelLocation` (`@MainActor`): `func currentModelFolder() -> URL?` — reads either the copied folder in Application Support/`Model/` or the security-scoped bookmark (setting `keepModelInPlace`).

- [ ] **Step 1:** Tests: `copyReportsProgressToOne`, `cancelLeavesNoFinalFolder` (only `.partial` remains, then cleaned on next start).
- [ ] **Step 2:** FAIL. **Step 3:** Implement + `ImportView` (folder picker, validation errors via `userMessage`, progress bar, cancel). App shows `ImportView` when `currentModelFolder()` is nil. **Step 4:** PASS. **Step 5:** Commit.

### Task 15: Engine config + memory-mapped weights (E2, E3)

**Files:** Create `InklingCore/InklingConfig.swift` (pure, Linux-testable), `InklingEngine/MappedSafetensors.swift`, `docs/Engine/Notes.md`, `Tests/InklingCoreTests/ConfigTests.swift` (reads the real `config.json` from `INKLING_MODEL_DIR`; skips when unset — nothing from the model folder is committed), `Tests/InklingEngineTests/MappedSafetensorsTests.swift`

**Interfaces:**
- Produces: `public struct InklingTextConfig: Decodable, Sendable` (all `text_config` keys from Global Constraints + `local_layer_ids`, `sconv_kernel_size`, `n_layers_per_attn_res_block`, `mlp_gate_softcap`, `mlp_up_softcap`, `route_scale`, `use_gate_bias`, `qk_l2_clip_threshold`, `rel_l2_clip_threshold`, `d_rel`, `rel_extent`, `log_scaling_*`, `loop_config`) with `var layerVisits: [Int]` (0…5, then 6…17 ×3, then 18…23 → 48 entries); `public struct QuantSpec: Sendable, Equatable { bits: Int; groupSize: Int; mode: String }`; `public struct QuantizationMap { func spec(for modulePath: String) -> QuantSpec? }`; `public final class MappedSafetensors: @unchecked Sendable { init(manifest: ModelManifest) throws; func array(_ name: String) throws -> MLXArray; var names: [String]; var mappedBytes: Int64 }`.

- [ ] **Step 1:** Spike: inspect mlx-swift 0.33.0 sources (`Source/MLX/MLXArray+Init.swift`, `Source/Cmlx` headers) for an array initializer that wraps external memory without copying (`mlx_array_new_data_managed` or similar). Record the finding and the chosen approach in `docs/Engine/Notes.md`. If none exists, `array(_:)` copies (reference path only) and the fast path (Task 20) reads mmapped `MTLBuffer`s directly.
- [ ] **Step 2:** Tests: `layerVisitsHas48Entries` (`[0,1,2,3,4,5,6…17,6…17,6…17,18…23]`), `expertQuantIs2or3bit` (`layers.6.mlp.switch_mlp.gate_proj` → bits 2, group 128), `attentionQuantIs8bit`, `mapsTinySafetensors` (write a 2-tensor safetensors file in the test, map it, values equal), `mappedBytesEqualsFileSize`.
- [ ] **Step 3:** FAIL. **Step 4:** Implement (`mmap(PROT_READ, MAP_PRIVATE)`, page-aligned). **Step 5:** PASS (`ConfigTests` on the PC with `INKLING_MODEL_DIR`; `MappedSafetensorsTests` in macOS CI). **Step 6:** Commit.

### Task 16: Reference model port + tiny golden test (E1)

**Files:** Create `scripts/make_tiny_inkling.py`, `Tests/InklingEngineTests/Fixtures/tiny-inkling/{config.json,model.safetensors,golden.json}`, `InklingEngine/Model/{ShortConvolution,Attention,SparseMoE,AttentionResidual,MessageController,DecoderLayer,TextModel,Inkling2LanguageModel}.swift`, `InklingEngine/Kernels/ShortConvKernel.swift`, `Tests/InklingEngineTests/GoldenTests.swift`

**Interfaces:**
- Consumes: `InklingTextConfig`, `QuantizationMap`, `MappedSafetensors` (Task 15).
- Produces: `public final class Inkling2LanguageModel: Module { init(_ config: InklingTextConfig); func callAsFunction(_ tokens: MLXArray, cache: [LayerVisitCache]?) -> MLXArray /* logits [B,L,200058 or unpadded] */; func makeCache() -> [LayerVisitCache] /* 48 */; static func load(manifest:, weights: MappedSafetensors) throws -> Inkling2LanguageModel }`; `public protocol LayerVisitCache` placeholder implemented in Task 17 (Task 16 uses a minimal in-file version and Task 17 replaces it).

- [ ] **Step 1:** `make_tiny_inkling.py` (uv venv, Python 3.12, `mlx` CPU wheel, mlx-vlm at the pinned commit): build `inkling2_loop` text model with hidden 64, heads 4/kv 2, head_dim 16, 8 layers, loop core 2…5 ×3, 8 experts/2 active, intermediate 32, sliding window 8, `local_layer_ids` all except 1 and 7, vocab 512, random seed 0; quantize with the same mixed scheme shape (experts 2/3-bit group 32, attention 8-bit group 32); save weights + config; run 12 fixed token ids teacher-forced and save logits (float32) and greedy next-8 tokens to `golden.json`. Size < 5 MB.
- [ ] **Step 2:** Tests: `tinyLogitsMatchPython` (max abs diff ≤ 1e-2, top-1 equal at every position; run on `Device.cpu`), `tinyGreedyMatchesPython` (8 tokens equal), `loopPassesUseSeparateCaches` (48-visit analogue: cache count == `layerVisits.count`).
- [ ] **Step 3:** FAIL. **Step 4:** Port module by module from the pinned Python sources, keeping FP32 intermediates exactly where the Python does (`kernels.py` short convolution → `MLXFast.metalKernel` with the same source; CPU fallback = pure MLX ops for tests). **Step 5:** PASS. **Step 6:** Commit.

### Task 17: Caches, sampler, generator; wire real model into the app (E7, E8, E9, E11)

**Files:** Create `InklingEngine/Cache/{RotatingKVCache,KVCache,ConvStateCache,LayerVisitCache}.swift`, `InklingEngine/Sampler.swift`, `InklingEngine/InklingGenerator.swift`, `Tests/InklingEngineTests/{CacheTests,SamplerTests,GeneratorTests}.swift`; Modify `App/InklingPocketApp.swift` (use `InklingGenerator` when a model is present, `EchoModel` otherwise)

**Interfaces:**
- Consumes: `Inkling2LanguageModel` (16), `InklingTokenizer` (3), `PromptFormatter` (2), `InklingStreamParser` (4), `LanguageModel` (6).
- Produces: `public actor InklingGenerator: LanguageModel { init(modelFolder: URL, path: EnginePath) async throws; var stats: GenerationStats }`; `public enum EnginePath: String, Sendable { case reference, fast }`; `public struct GenerationStats: Sendable { promptTokens; generatedTokens; prefillTokensPerSecond; decodeTokensPerSecond; timeToFirstToken: Duration }`; `public struct Sampler { init(_ p: GenerateParameters); func sample(_ logits: MLXArray, history: [Int32]) -> MLXArray }`; prefix reuse: generator keeps the last rendered token prefix + caches and re-prefills only the new suffix.

- [ ] **Step 1:** Tests (tiny model): `rotatingCacheKeepsLast1024` (window from config: 8 for tiny), `prefixReuseSkipsSharedTokens` (second turn prefills only new tokens; logits equal to full recompute within 1e-3), `chunkedPrefillEqualsSinglePass` (chunk 4 vs 12), `greedyWhenTemperatureZero`, `topKOneIsGreedy`, `cancelStopsWithinOneToken`, `cancelMidStreamSavesPartial` (ChatSession logic marks `stopped` and keeps text), `stopsAtEOS200006`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. Generation loop runs on a dedicated task; cancellation checked between tokens; app scene phase `.background` → cancel. **Step 4:** PASS. **Step 5:** Commit; owner installs IPA, imports the model, sends "hi" (reference path), reports whether text appears.

### Task 18: Self-test + Benchmark screens (B1–B4)

**Files:** Create `InklingEngine/SelfTest.swift`, `InklingEngine/RoutingStats.swift`, `App/Bench/BenchmarkView.swift`, `App/Bench/SelfTestView.swift`, `Tests/InklingEngineTests/SelfTestTests.swift`

**Interfaces:**
- Consumes: `InklingGenerator`, `EnginePath`, `GenerationStats` (17).
- Produces: `public struct SelfTestReport: Sendable { top1Agreement: Double; meanKLD: Double; maxAbsLogitDiff: Float; knownAnswers: [(prompt: String, output: String, passed: Bool)]; passed: Bool; asText: String }`; `public enum SelfTest { static func run(modelFolder: URL) async throws -> SelfTestReport }` (pass = top-1 ≥ 0.97 and mean KLD ≤ 0.01 over 3 prompts × 64 teacher-forced tokens, and all known answers pass); known answers (effort none, temperature 0): "What is the capital of France? Answer in one word." → contains "Paris"; "What is 12 times 12? Reply with the number only." → contains "144"; "Spell the word cat backwards. Reply with the result only." → contains "tac" (case-insensitive). `public final class RoutingStats: @unchecked Sendable { func record(visit: Int, experts: MLXArray); func hitHistogram(visit: Int) -> [Int]; func passOverlap(layer: Int) -> Double /* share of experts reused across the 3 passes */ }`; `BenchmarkReport` with prefill tok/s, decode tok/s (256 tokens), TTFT, `os_proc_available_memory()`, `phys_footprint` peak (via `task_info` `TASK_VM_INFO`), expert hit rate, per-visit routing histograms, `ProcessInfo.thermalState`, `asText`.
- [ ] **Step 1:** Tests (tiny model, both paths = reference until Task 20): `selfTestPassesWhenPathsEqual`, `reportTextContainsAllFields`, `routingStatsCountsEveryVisit`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement + views (Copy button → `UIPasteboard`). **Step 4:** PASS. **Step 5:** Commit; owner runs Benchmark (reference path) and pastes the text back. This is the speed baseline.

### Task 19: Memory budget, expert residency, pressure handling (E4, E5, E10)

**Files:** Create `InklingEngine/Residency/{MemoryBudget,ExpertResidency}.swift`, `Tests/InklingEngineTests/ResidencyTests.swift`

**Interfaces:**
- Consumes: `MappedSafetensors` (15), `RoutingStats` (18).
- Produces: `public struct MemoryBudget { static func current(overrideGB: Double?) -> Int64 /* os_proc_available_memory - 1.5 GB safety, or override */ }`; `public struct ExpertKey: Hashable, Sendable { layer: Int; expert: Int }`; `public final class ExpertResidency: @unchecked Sendable { init(weights: MappedSafetensors, budget: Int64, expertBytes: (ExpertKey) -> Int64); func record(layer: Int, experts: [Int]); func rebalance() /* pin hottest within budget: madvise WILLNEED + touch; unpin: MADV_DONTNEED not used on file maps — just stop touching */; func prefetch(layer: Int, experts: [Int]); func handleMemoryWarning() /* drop pinned set to 50% */; var hitRate: Double }`.
- [ ] **Step 1:** Tests (pure accounting with fake byte sizes): `pinsHottestWithinBudget`, `memoryWarningHalvesPinned`, `hitRateCountsPinnedHits`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement; subscribe to `UIApplication.didReceiveMemoryWarningNotification` in the app and call `handleMemoryWarning()`. **Step 4:** PASS. **Step 5:** Commit; owner re-runs Benchmark.

### Task 20: Fast path (E6)

**Files:** Create `InklingEngine/Kernels/{FusedMoEDecode.metal-source.swift,FusedMoEDecode.swift,GPUSampler.swift}`, `InklingEngine/FastPath.swift`, `Tests/InklingEngineTests/FastPathTests.swift`; Modify `InklingGenerator.swift` (select path)

**Interfaces:**
- Consumes: everything in 15–19.
- Produces: `FusedMoEDecode.callAsFunction(x: MLXArray /*[1,2304]*/, expertIDs: MLXArray /*[8]*/, weights: MLXArray /*[8]*/, layer: Int) -> MLXArray` — one `MLXFast.metalKernel` doing gather + affine dequant (2/3/4-bit, group 128) + gate/up matmul + softcapped SwiGLU (`mlp_gate_softcap 4.0`, `mlp_up_softcap 25.0`) + down matmul + weighted sum; `GPUSampler` (temperature/top-p/top-k on GPU, returns one token id); `FastPath` compiles the per-token decode step with `compile(...)`, reuses buffers across the 3 loop passes.
- [ ] **Step 1:** Tests (tiny model): `fusedMoEMatchesReference` (max abs diff ≤ 2e-2), `fastPathLogitsMatchReference` (top-1 equal on 12 positions), `gpuSamplerGreedyEqualsArgmax`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement one kernel at a time; keep each behind `EnginePath.fast`. **Step 4:** PASS. **Step 5:** Commit; owner runs Self-test (must pass) and Benchmark on `fast`; speed work continues from those numbers (next levers in `docs/Engine/Notes.md`: lm_head re-quant option, residency tuning, prefill tile sizes).

### Task 21: Notices, settings wiring, final verification (G1, G2)

**Files:** Create `THIRD_PARTY_NOTICES.md`; Modify `App/Settings/SettingsView.swift`, `README.md`

- [ ] **Step 1:** `THIRD_PARTY_NOTICES.md`: mlx-vlm (MIT), crawl4ai (Apache-2.0), OpenUI (MIT), mlx-swift (MIT), SwiftSoup (MIT) with copyright lines from each upstream LICENSE.
- [ ] **Step 2:** Settings exposes every `AppSettings` field; README explains LiveContainer install + model import.
- [ ] **Step 3:** `git ls-files | grep -E 'safetensors|tokenizer.json|tiktoken|chat_template'` → no output.
- [ ] **Step 4:** CI green on `main`; IPA artifact present. Commit.

---

## Spec coverage

P1–P4 T0/T1 · P5 T0 · P6 T13 · M1/M3 T14 · M2/M4 T5 · E1 T16 · E2/E3 T15 · E4/E5/E10 T19 · E6 T20 · E7/E8/E9/E11 T17 · T1 T3 · T2 T2 · T3 T4 · A1/A3/A5 T6 · A2 T6/T8/T10 · A4 T13 · C1 T9 · C2 T7 · C3/C4 T8 · R1–R3 T10 · O1/O3 T11 · O2/O4 T12 · U1–U4 T13 · B1–B4 T18 · G1/G2 T21.

## Engine speed note (for the owner)

From the real `config.json`, each token reads roughly: experts ~1.0 GB (2–3-bit, 48 visits × 8), attention ~0.8 GB (8-bit, 48 visits), output head ~0.46 GB (8-bit, 201K vocab) ≈ **2.3 GB/token** → bandwidth ceiling ≈ 52 tok/s; realistic with everything in RAM ≈ **26–31 tok/s**, lower when experts are read from flash. Numbers are confirmed only by the iPad Benchmark (Task 18 onward).
