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
    /// Expert slots per MoE layer (nil when experts are not streamed, e.g. test models).
    public let expertSlots: Int?
    /// Mapped bytes that are not expert weights.
    let nonExpertBytes: Int64
    /// Working memory kept free for the KV cache, activations and Metal scratch on top of `MemoryBudget`'s margin.
    static let workingReserve: Int64 = 600_000_000
    static let mlxCacheLimit = 256 << 20

    /// Gives memory freed by a dropped engine back to iOS (call after releasing the old engine, before loading).
    public static func releaseFreedMemory() { Memory.clearCache() }
    public private(set) var stats = GenerationStats()

    /// Loads config, tokenizer and memory-mapped weights from an imported model folder.
    public init(modelFolder: URL, path: EnginePath, memoryBudgetGB: Double? = nil) async throws {
        let manifest = try ModelFolder.validate(modelFolder)
        // Freed MLX buffers are kept for reuse up to this size; the default (most of RAM) is far too much here.
        Memory.cacheLimit = Self.mlxCacheLimit
        let weights = try MappedSafetensors(manifest: manifest)
        let data = try Data(contentsOf: manifest.folder.appending(path: "config.json"))
        let config = try InklingModelConfig.decode(data)
        let perSlot = Self.bytesPerSlot(weights: weights, config: config.text)
        let nonExpert = weights.mappedBytes - perSlot * Int64(config.text.numExperts)
        let slots = Self.slots(budget: MemoryBudget.current(overrideGB: memoryBudgetGB) - nonExpert - Self.workingReserve,
                               perSlot: perSlot, config: config.text)
        let model = try Inkling2LanguageModel.load(config: config, weights: weights, expertSlots: slots)
        let tokenizer = try InklingTokenizer(modelFolder: modelFolder)
        self.init(model: model, tokenizer: tokenizer, path: path, mappedBytes: weights.mappedBytes, expertSlots: slots,
                  nonExpertBytes: nonExpert)
    }

    /// Bytes of one expert in every MoE layer (one slot everywhere).
    static func bytesPerSlot(weights: MappedSafetensors, config: InklingTextConfig) -> Int64 {
        (0..<config.numHiddenLayers).map {
            ExpertCache.bytesPerExpert(path: "model.layers.\($0).mlp", weights: weights, prefix: "language_model.", experts: config.numExperts)
        }.reduce(0, +)
    }

    /// Slots per layer that fit `budget`: at least one token's experts, at most all of them.
    static func slots(budget: Int64, perSlot: Int64, config: InklingTextConfig) -> Int {
        guard perSlot > 0 else { return config.numExperts }
        return Int(min(max(budget / perSlot, Int64(config.expertsPerToken)), Int64(config.numExperts)))
    }

    init(model: CausalLM, tokenizer: InklingTokenizer, path: EnginePath = .reference, mappedBytes: Int64 = 0,
         expertSlots: Int? = nil, nonExpertBytes: Int64 = 0) {
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
    }

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
        guard prompt.count < generator.contextLimit else {
            throw GenerationError.contextFull(prompt: prompt.count, limit: generator.contextLimit)
        }
        // MLX errors become a thrown error (shown in the chat) instead of crashing the app.
        let generator = self.generator, tokenizer = self.tokenizer
        let ended = try withError { () -> Bool in
            var decoder = StreamingDecoder(tokenizer: tokenizer)
            var parser = InklingStreamParser()
            var ended = false
            generator.generate(prompt: prompt, sampler: Sampler(parameters), maxTokens: parameters.maxTokens,
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
    }
}
