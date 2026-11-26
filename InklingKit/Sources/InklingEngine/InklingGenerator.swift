import Foundation
import MLX
import InklingCore
import AgentCore

public enum EnginePath: String, Sendable, CaseIterable {
    /// Custom Metal kernels (Task 20); until then the same as `.reference`.
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
    static let workingReserve: Int64 = 800_000_000
    static let mlxCacheLimit = 256 << 20

    /// Gives memory freed by a dropped engine back to iOS (call after releasing the old engine, before loading).
    public static func releaseFreedMemory() { Memory.clearCache() }
    public private(set) var stats = GenerationStats()
    /// Where the instructions' cache state is kept between launches (nil for test models).
    var promptCacheURL: URL?
    private var savedPrefix: [Int32]?
    private var triedSavedPrefix = false
    private var promptCacheIdentity: String { "inkling-\(mappedWeightBytes)" }

    /// Loads config, tokenizer and memory-mapped weights from an imported model folder.
    public init(modelFolder: URL, path: EnginePath, memoryBudgetGB: Double? = nil) async throws {
        let manifest = try ModelFolder.validate(modelFolder)
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
        let slots = Self.slotPlan(budget: MemoryBudget.current(overrideGB: memoryBudgetGB) - nonExpert - Self.workingReserve,
                                  perLayer: perLayer, visits: visits, experts: text.numExperts, topK: text.expertsPerToken)
        let model = try Inkling2LanguageModel.load(config: config, weights: weights, slotPlan: slots)
        let tokenizer = try InklingTokenizer(modelFolder: modelFolder)
        self.init(model: model, tokenizer: tokenizer, path: path, mappedBytes: weights.mappedBytes, expertSlots: slots,
                  nonExpertBytes: nonExpert)
        promptCacheURL = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first?
            .appending(path: "inkling-prompt-cache.safetensors")
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
        if path == .fast, let lm = model as? Inkling2LanguageModel {
            generator = TokenGenerator(model: FastPath(lm), eos: [tokenizer.eosID])
            // Streaming reads routing back every layer, so there is nothing to overlap.
            generator.pipelined = lm.expertCaches.isEmpty
        } else {
            generator = TokenGenerator(model: model, eos: [tokenizer.eosID])
        }
        // A prompt chunk needs nearly every expert, so streaming reads the experts once per chunk: fewer, bigger chunks.
        if expertSlots != nil { generator.prefillChunk = Self.streamingPrefillChunk }
    }

    static let streamingPrefillChunk = 512  // the attention mask for a chunk grows with chunk × context; 512 keeps it ~0.5 GB

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
        if let lm = generator.model as? Inkling2LanguageModel { return (lm, FastPath(lm)) }
        return (generator.model, generator.model)
    }

    /// iOS memory warning. Expert slots are a fixed size chosen at load, so there is nothing to shrink here.
    public nonisolated func handleMemoryWarning() {}

    /// Settings → Memory budget changed (GB, nil = automatic). Slots are sized at load: applies on the next load.
    public nonisolated func setMemoryBudget(gb: Double?) {}

    public nonisolated func countTokens(_ text: String) -> Int { tokenizer.encode(text).count }

    public func setContextLimit(_ tokens: Int) { generator.contextLimit = tokens }

    public nonisolated func generate(_ messages: [ChatMessage], tools: [ToolSpec], effort: ReasoningEffort,
                                     parameters: GenerateParameters) -> AsyncThrowingStream<ModelEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
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
            generator.generate(prompt: prompt, stablePrefix: stablePrefix, sampler: Sampler(parameters), maxTokens: parameters.maxTokens,
                               shouldStop: { Task.isCancelled || ended }) { token in
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
