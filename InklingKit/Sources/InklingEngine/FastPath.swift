import MLX

/// The fast engine over the same weights as the reference: fused single-token MoE kernels; `TokenGenerator.pipelined`
/// overlaps building the next token's graph with the GPU running the current one (E6).
public final class FastPath: CausalLM {
    public let base: Inkling2LanguageModel
    public init(_ base: Inkling2LanguageModel) { self.base = base }

    public func callAsFunction(_ tokens: MLXArray, cache: [LayerVisitCache]?) -> MLXArray {
        base.turboReset()  // exact routing even right after turbo drafts
        return base(tokens, cache: cache, fast: true, compiled: true)
    }

    public func lastLogits(_ tokens: MLXArray, cache: [LayerVisitCache]?) -> MLXArray {
        base.turboReset()
        return base(tokens, cache: cache, fast: true, compiled: true, lastOnly: true)[0, -1]
    }

    public func makeCache() -> [LayerVisitCache] { base.makeCache() }
}
