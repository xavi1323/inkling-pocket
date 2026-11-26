import MLX

/// The fast engine over the same weights as the reference: fused single-token MoE kernels; `TokenGenerator.pipelined`
/// overlaps building the next token's graph with the GPU running the current one (E6).
public final class FastPath: CausalLM {
    public let base: Inkling2LanguageModel
    public init(_ base: Inkling2LanguageModel) { self.base = base }

    public func callAsFunction(_ tokens: MLXArray, cache: [LayerVisitCache]?) -> MLXArray {
        base(tokens, cache: cache, fast: true)
    }

    public func makeCache() -> [LayerVisitCache] { base.makeCache() }
}
