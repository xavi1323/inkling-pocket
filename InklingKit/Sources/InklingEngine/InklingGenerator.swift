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
    public private(set) var stats = GenerationStats()

    /// Loads config, tokenizer and memory-mapped weights from an imported model folder.
    public init(modelFolder: URL, path: EnginePath) async throws {
        let manifest = try ModelFolder.validate(modelFolder)
        let weights = try MappedSafetensors(manifest: manifest)
        let model = try Inkling2LanguageModel.load(manifest: manifest, weights: weights)
        let tokenizer = try InklingTokenizer(modelFolder: modelFolder)
        self.init(model: model, tokenizer: tokenizer, path: path, mappedBytes: weights.mappedBytes)
    }

    init(model: CausalLM, tokenizer: InklingTokenizer, path: EnginePath = .reference, mappedBytes: Int64 = 0) {
        self.tokenizer = tokenizer
        self.path = path
        mappedWeightBytes = mappedBytes
        generator = TokenGenerator(model: model, eos: [tokenizer.eosID])
    }

    /// (reference, fast) models over the same weights. The fast path arrives in Task 20; until then both are the reference.
    func paths() -> (reference: CausalLM, fast: CausalLM) { (generator.model, generator.model) }

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
        stats = generator.stats
        if !Task.isCancelled, !ended { out.yield(.end) }
    }
}
