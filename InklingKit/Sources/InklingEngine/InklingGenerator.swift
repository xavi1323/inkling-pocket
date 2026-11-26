import Foundation
import MLX
import InklingCore
import AgentCore

public enum EnginePath: String, Sendable, CaseIterable {
    /// Turbo (`TurboPath`: routing on the GPU, no waits inside a token) guesses tokens ahead and the exact path
    /// checks them, so the output is exact.
    case turbo
    /// Custom Metal kernels; exact.
    case fast
    /// Plain MLX ops.
    case reference
}

/// The on-device Inkling 2 Mini as an agent `LanguageModel`: chat → prompt tokens → sampled tokens → parsed events.
public actor InklingGenerator: LanguageModel {
    public nonisolated let tokenizer: InklingTokenizer
    public let path: EnginePath
    let generator: TokenGenerator
    /// Bytes of memory-mapped weight shards (0 for test models).
    public let mappedWeightBytes: Int64
    /// Expert slots for each MoE layer (nil when experts are not streamed, e.g. test models).
    public let expertSlots: [Int]?
    /// Mapped bytes that are not expert weights.
    let nonExpertBytes: Int64
    /// Working memory kept free for the KV cache, activations and Metal scratch on top of `MemoryBudget`'s margin.
    /// Peak use stayed ~1 GB under the budget with 0.8 GB here; 0.5 GB still covers a long chat's K/V.
    static let workingReserve: Int64 = 500_000_000
    static let mlxCacheLimit = 256 << 20

    /// Gives memory freed by a dropped engine back to iOS (call after releasing the old engine, before loading).
    public static func releaseFreedMemory() { Memory.clearCache() }
    public private(set) var stats = GenerationStats()
    /// Where the instructions' cache state is kept between launches (nil for test models).
    var promptCacheURL: URL?
    /// Where expert use counts are kept between launches (nil for test models).
    var heatURL: URL?
    private var savedPrefix: [Int32]?
    private var triedSavedPrefix = false
    private var promptCacheIdentity: String { "inkling-\(mappedWeightBytes)" }

    /// Loads config, tokenizer and memory-mapped weights from an imported model folder.
    public init(modelFolder: URL, path: EnginePath, memoryBudgetGB: Double? = nil) async throws {
        let manifest = try ModelFolder.validate(modelFolder)
        // The tokenizer (200k ranks to parse) loads on another core while the weights are mapped and slots made.
        let tokenizerTask = Task.detached(priority: .userInitiated) { try InklingTokenizer(modelFolder: modelFolder) }
        // Freed MLX buffers are kept for reuse up to this size; the default (most of RAM) is far too much here.
        Memory.cacheLimit = Self.mlxCacheLimit
        let weights = try MappedSafetensors(manifest: manifest)
        let data = try Data(contentsOf: manifest.folder.appending(path: "config.json"))
        let config = try InklingModelConfig.decode(data)
        let text = config.text
        let perLayer = (0..<text.numHiddenLayers).map {
            ExpertCache.bytesPerExpert(path: "model.layers.\($0).mlp", weights: weights, prefix: "language_model.", experts: text.numExperts)
        }
        let nonExpert = weights.mappedBytes - perLayer.reduce(0, +) * Int64(text.numExperts)
        let visits = (0..<text.numHiddenLayers).map { layer in text.layerVisits.filter { $0 == layer }.count }
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
        let heatURL = caches?.appending(path: "inkling-expert-heat.json")
        let identity = "inkling-\(weights.mappedBytes)"
        let heat = heatURL.flatMap { try? Data(contentsOf: $0) }
            .flatMap { try? JSONDecoder().decode(Inkling2LanguageModel.HeatFile.self, from: $0) }
            .flatMap { $0.identity == identity ? $0 : nil }
        let budget = MemoryBudget.current(overrideGB: memoryBudgetGB) - nonExpert - Self.workingReserve
        let slots = heat.flatMap { Self.heatPlan(budget: budget, perLayer: perLayer, heat: $0.layers, experts: text.numExperts,
                                                 topK: text.expertsPerToken) }
            ?? Self.slotPlan(budget: budget, perLayer: perLayer, visits: visits, experts: text.numExperts, topK: text.expertsPerToken)
        let model = try Inkling2LanguageModel.load(config: config, weights: weights, slotPlan: slots)
        let tokenizer = try await tokenizerTask.value
        self.init(model: model, tokenizer: tokenizer, path: path, mappedBytes: weights.mappedBytes, expertSlots: slots,
                  nonExpertBytes: nonExpert)
        promptCacheURL = caches?.appending(path: "inkling-prompt-cache.safetensors")
        self.heatURL = heatURL
        // Start with last session's most-used experts in their slots instead of empty ones.
        if let heat { model.warm(from: heat, identity: identity) }
        // Build the Metal kernels and compiled graphs now, while the user types, not on the first reply.
        Task(priority: .utility) { [weak self] in await self?.warmUp() }
    }

    /// Whether `warmUp` ran.
    public private(set) var warmedUp = false

    /// Runs a few tokens through every path chat uses (prompt chunk, guesses, checks, plain steps, sampling) on a
    /// scratch cache, so their kernels are compiled before the first reply. Chat state is not touched.
    public func warmUp() {
        guard !warmedUp else { return }
        warmedUp = true
        let scratch = TokenGenerator(model: generator.model, eos: [])
        scratch.draft = generator.draft
        scratch.exactStep = generator.exactStep
        scratch.pipelined = generator.pipelined
        let lm = paths().reference as? Inkling2LanguageModel
        lm?.smartRouting = smartRouting
        // Pick the fastest kernel launch settings for this GPU first, so the warm-up builds the chosen ones.
        if let lm { KernelTuning.shared.tune(lm) }
        let prompt = tokenizer.encode("Hello! Say hi back in one short sentence.")
        guard !prompt.isEmpty else { return }
        _ = try? withError {
            scratch.generate(prompt: prompt, sampler: Sampler(GenerateParameters()), maxTokens: 6, shouldStop: { false },
                             onToken: { _ in })
        }
    }

    /// Slots per layer from last session's expert use counts: each slot goes where it saves the most reads (the
    /// layer whose next most-used expert is used most, per byte), so layers that spread their routing get more
    /// slots and layers that keep using a few experts fewer. Every layer keeps at least `topK`. Nil when the counts
    /// do not fit the model.
    static func heatPlan(budget: Int64, perLayer: [Int64], heat: [[Int]], experts: Int, topK: Int) -> [Int]? {
        guard heat.count == perLayer.count, heat.allSatisfy({ $0.count == experts }),
              heat.contains(where: { $0.contains { $0 > 0 } }) else { return nil }
        let sorted = heat.map { $0.sorted(by: >) }
        var plan = Array(repeating: topK, count: perLayer.count)
        var used = zip(plan, perLayer).map { Int64($0) * $1 }.reduce(0, +)
        guard used <= budget else { return nil }
        // Gain of one more slot in layer l: uses of its next expert per byte (+1 so unseen experts still count).
        func gain(_ l: Int) -> Double { plan[l] < experts ? Double(sorted[l][plan[l]] + 1) / Double(max(perLayer[l], 1)) : -1 }
        var queue = (0..<perLayer.count).map { (gain($0), $0) }
        while true {
            guard let best = queue.indices.max(by: { queue[$0].0 < queue[$1].0 }), queue[best].0 >= 0 else { break }
            let l = queue[best].1
            guard used + perLayer[l] <= budget else { queue[best].0 = -1; continue }
            plan[l] += 1
            used += perLayer[l]
            queue[best].0 = gain(l)
        }
        return plan
    }

    /// Slots per layer within `budget` bytes, in proportion to how often each layer is visited per token (the loop
    /// core runs three times), each between one token's experts (`topK`) and all of them.
    static func slotPlan(budget: Int64, perLayer: [Int64], visits: [Int], experts: Int, topK: Int) -> [Int] {
        func plan(_ c: Double) -> [Int] {
            zip(perLayer, visits).map { b, v in b == 0 ? experts : min(experts, max(topK, Int(c * Double(max(v, 1))))) }
        }
        func cost(_ p: [Int]) -> Int64 { zip(p, perLayer).map { Int64($0) * $1 }.reduce(0, +) }
        if cost(plan(Double(experts))) <= budget { return plan(Double(experts)) }
        var lo = 0.0, hi = Double(experts)
        for _ in 0..<60 {
            let mid = (lo + hi) / 2
            if cost(plan(mid)) <= budget { lo = mid } else { hi = mid }
        }
        return plan(lo)
    }

    init(model: CausalLM, tokenizer: InklingTokenizer, path: EnginePath = .reference, mappedBytes: Int64 = 0,
         expertSlots: [Int]? = nil, nonExpertBytes: Int64 = 0) {
        self.nonExpertBytes = nonExpertBytes
        self.tokenizer = tokenizer
        self.path = path
        mappedWeightBytes = mappedBytes
        self.expertSlots = expertSlots
        if path == .turbo, let lm = model as? Inkling2LanguageModel {
            generator = TokenGenerator(model: TurboPath(lm), eos: [tokenizer.eosID])
            generator.pipelined = lm.expertCaches.isEmpty  // the table changes between tokens
            // Streaming: turbo only guesses ahead; every token is checked exact (multi-token passes run exact).
            if !lm.expertCaches.isEmpty {
                generator.draft = TurboPath(lm, redo: false)
                generator.exactStep = FastPath(lm)
            }
        } else if path == .fast, let lm = model as? Inkling2LanguageModel {
            generator = TokenGenerator(model: FastPath(lm), eos: [tokenizer.eosID])
            // Streaming reads routing back every layer, so there is nothing to overlap.
            generator.pipelined = lm.expertCaches.isEmpty
        } else {
            generator = TokenGenerator(model: model, eos: [tokenizer.eosID])
        }
        // A prompt chunk needs nearly every expert, so streaming reads the experts once per chunk: fewer, bigger chunks.
        if expertSlots != nil { generator.prefillChunk = Self.streamingPrefillChunk }
        // Keep the cache right after each end of message (a thought's end): the next agent step resends it.
        if let end = tokenizer.specialTokenID(PromptFormatter.Token.endMessage) { generator.boundaryTokens = [end] }
    }

    /// Every chunk reads nearly all experts once per layer visit, so bigger chunks read less per token (4096: half the
    /// reads of 2048 on long prompts); attention runs in 512-query blocks (`Attention.queryBlock`), so the mask stays
    /// ~0.5 GB.
    static let streamingPrefillChunk = 4096

    /// Share of expert lookups already in a slot (nil when not streaming).
    var expertHitRate: Double? {
        guard let lm = paths().reference as? Inkling2LanguageModel else { return nil }
        let caches = lm.expertCaches
        let hits = caches.map(\.hits).reduce(0, +), total = hits + caches.map(\.misses).reduce(0, +)
        return total == 0 ? nil : Double(hits) / Double(total)
    }

    /// (reference, fast) models over the same weights; test models without the Inkling layers use themselves for both.
    func paths() -> (reference: CausalLM, fast: CausalLM) {
        if let fast = generator.model as? FastPath { return (fast.base, fast) }
        if let turbo = generator.model as? TurboPath { return (turbo.base, turbo) }
        if let lm = generator.model as? Inkling2LanguageModel { return (lm, FastPath(lm)) }
        return (generator.model, generator.model)
    }

    /// iOS memory warning. Expert slots are a fixed size chosen at load, so there is nothing to shrink here.
    public nonisolated func handleMemoryWarning() {}

    /// Settings → Memory budget changed (GB, nil = automatic). Slots are sized at load: applies on the next load.
    public nonisolated func setMemoryBudget(gb: Double?) {}

    public nonisolated func countTokens(_ text: String) -> Int { tokenizer.encode(text).count }

    public func setContextLimit(_ tokens: Int) { generator.contextLimit = tokens }

    /// Smart expert picks (faster, near-exact; tool calls stay exact). On by default.
    public private(set) var smartRouting = true
    public func setSmartRouting(_ on: Bool) { smartRouting = on }

    public nonisolated func generate(_ messages: [ChatMessage], tools: [ToolSpec], effort: ReasoningEffort,
                                     parameters: GenerateParameters) -> AsyncThrowingStream<ModelEvent, Error> {
        AsyncThrowingStream { continuation in
            // High priority: the thread waits on the GPU ~48 times per token and wakes faster at a higher QoS.
            let task = Task(priority: .high) {
                do {
                    try await self.run(messages, tools, effort, parameters, continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func run(_ messages: [ChatMessage], _ tools: [ToolSpec], _ effort: ReasoningEffort,
                     _ parameters: GenerateParameters, _ out: AsyncThrowingStream<ModelEvent, Error>.Continuation) throws {
        let prompt = tokenizer.encode(PromptFormatter.render(messages: messages, tools: tools, effort: effort))
        let stablePrefix = Self.stablePrefixLength(prompt, messages, tools, effort, tokenizer)
        guard prompt.count < generator.contextLimit else {
            throw GenerationError.contextFull(prompt: prompt.count, limit: generator.contextLimit)
        }
        restoreSavedPrefix(for: prompt)
        // MLX errors become a thrown error (shown in the chat) instead of crashing the app.
        let generator = self.generator, tokenizer = self.tokenizer
        let ended = try withError { () -> Bool in
            var decoder = StreamingDecoder(tokenizer: tokenizer)
            var parser = InklingStreamParser()
            var ended = false
            let budget = parameters.maxThinkingTokens.flatMap { cap in
                tokenizer.specialTokenID(PromptFormatter.Token.thinking).flatMap { open in
                    tokenizer.specialTokenID(PromptFormatter.Token.endMessage).map { ThinkingBudget(cap: cap, open: open, close: $0) }
                }
            }
            generator.forcedToken = budget.map { b in { b.forced } }
            defer { generator.forcedToken = nil }
            // Smart expert picks, except inside tool calls (their JSON must be exact).
            let lm = (generator.model as? TurboPath)?.base ?? (generator.model as? FastPath)?.base ?? (generator.model as? Inkling2LanguageModel)
            let smart = smartRouting
            lm?.smartRouting = smart
            let toolSpan = tokenizer.specialTokenID(PromptFormatter.Token.invokeToolJSON).flatMap { open in
                tokenizer.specialTokenID(PromptFormatter.Token.endMessage).map { ExactSpan(open: open, close: $0) }
            }
            generator.generate(prompt: prompt, stablePrefix: stablePrefix, sampler: Sampler(parameters), maxTokens: parameters.maxTokens,
                               shouldStop: { Task.isCancelled || ended }) { token in
                budget?.observe(token)
                if let toolSpan {
                    toolSpan.observe(token)
                    lm?.smartRouting = smart && !toolSpan.inside
                }
                for event in parser.feed(decoder.append(token)) {
                    out.yield(event)
                    if event == .end { ended = true }
                }
            }
            if !ended {
                for event in parser.feed(decoder.flush()) + parser.finish() { out.yield(event) }
            }
            return ended
        }
        stats = generator.stats
        if !Task.isCancelled, !ended { out.yield(.end) }
        keepPrefixForNextLaunch()
        keepHeatForNextLaunch()
    }

    private func keepHeatForNextLaunch() {
        guard let heatURL, let lm = paths().reference as? Inkling2LanguageModel, !lm.expertCaches.isEmpty,
              let data = try? JSONEncoder().encode(lm.heat(identity: promptCacheIdentity)) else { return }
        try? data.write(to: heatURL, options: .atomic)
    }

    /// Tokens of tools + leading system messages + effort line: the same for every chat with these settings.
    static func stablePrefixLength(_ prompt: [Int32], _ messages: [ChatMessage], _ tools: [ToolSpec], _ effort: ReasoningEffort,
                                   _ tokenizer: InklingTokenizer) -> Int {
        let leading = Array(messages.prefix { $0.role == .system })
        guard !leading.isEmpty else { return 0 }
        let head = tokenizer.encode(PromptFormatter.render(messages: leading, tools: tools, effort: effort))
        return head.count < prompt.count && prompt.starts(with: head) ? head.count : 0
    }

    /// First turn after launch: start from the instructions' saved cache state when it matches.
    private func restoreSavedPrefix(for prompt: [Int32]) {
        guard let url = promptCacheURL, !triedSavedPrefix, generator.prefixTokens == nil else { return }
        triedSavedPrefix = true
        let identity = promptCacheIdentity
        if (try? generator.loadPrefix(from: url, identity: identity, ifPrefixOf: prompt)) == true {
            savedPrefix = generator.prefixTokens
        }
    }

    /// New instructions state (first launch, or the date or memory changed): keep it for the next launch.
    private func keepPrefixForNextLaunch() {
        guard let url = promptCacheURL, let tokens = generator.prefixTokens, tokens != savedPrefix else { return }
        let identity = promptCacheIdentity
        if (try? generator.savePrefix(to: url, identity: identity)) != nil { savedPrefix = tokens }
    }
}
