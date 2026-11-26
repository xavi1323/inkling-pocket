import MLX

/// The turbo engine: the fast path with routing kept on the GPU. A token runs start to end without waiting for the
/// CPU or storage: experts not in a slot are swapped for the best ones that are, and read in after the token for the
/// next ones. Not bit-exact with `.fast`; the self-test measures how close it is. Prompt chunks run exact.
public final class TurboPath: CausalLM {
    public let base: Inkling2LanguageModel
    public init(_ base: Inkling2LanguageModel) { self.base = base }

    public func callAsFunction(_ tokens: MLXArray, cache: [LayerVisitCache]?) -> MLXArray {
        run(tokens, cache: cache, lastOnly: false)
    }

    public func lastLogits(_ tokens: MLXArray, cache: [LayerVisitCache]?) -> MLXArray {
        run(tokens, cache: cache, lastOnly: true)[0, -1]
    }

    /// A token whose swapped experts carried more than this share of one visit's routing weight, or more than
    /// `maxTokenMiss` summed over its visits, is rewound and run exact.
    nonisolated(unsafe) static var maxVisitMiss: Float = 0.35
    nonisolated(unsafe) static var maxTokenMiss: Float = 2.0

    private func run(_ tokens: MLXArray, cache: [LayerVisitCache]?, lastOnly: Bool) -> MLXArray {
        guard tokens.dim(1) == 1, !base.expertCaches.isEmpty else {
            base.turboReset()
            return base(tokens, cache: cache, fast: true, compiled: true, lastOnly: lastOnly)
        }
        base.turboBoundary()
        let visits = cache?.compactMap { $0 as? InklingVisitCache } ?? []
        let points = visits.map { $0.checkpoint() }
        let out = base(tokens, cache: cache, fast: true, compiled: true, lastOnly: lastOnly)
        guard let missed = base.turboMissed() else { return out }
        eval(out, missed)
        let m = missed.asArray(Float.self)
        let rewindable = visits.count == cache?.count && zip(visits, points).allSatisfy { $0.canRewind(to: $1) }
        guard m[0] > Self.maxVisitMiss || m[1] > Self.maxTokenMiss, rewindable else { return out }
        for (v, p) in zip(visits, points) { v.rewind(to: p) }
        base.turboWillRedo()
        return base(tokens, cache: cache, fast: true, compiled: true, lastOnly: lastOnly)
    }

    public func makeCache() -> [LayerVisitCache] { base.makeCache() }
}
