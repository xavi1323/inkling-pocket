import MLX

/// The turbo engine: the fast path with routing kept on the GPU. A token runs start to end without waiting for the
/// CPU or storage: experts not in a slot are swapped for the best ones that are, and read in after the token for the
/// next ones. Not bit-exact with `.fast`; the self-test measures how close it is. Prompt chunks run exact.
public final class TurboPath: CausalLM {
    public let base: Inkling2LanguageModel
    public init(_ base: Inkling2LanguageModel) { self.base = base }

    public func callAsFunction(_ tokens: MLXArray, cache: [LayerVisitCache]?) -> MLXArray {
        if tokens.dim(1) == 1, !base.expertCaches.isEmpty { base.turboBoundary() } else { base.turboReset() }
        return base(tokens, cache: cache, fast: true, compiled: true)
    }

    public func makeCache() -> [LayerVisitCache] { base.makeCache() }
}
