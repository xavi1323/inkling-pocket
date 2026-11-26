import MLX

/// `Sampler` with the history kept as an array on the device, so the next token can be sampled before the previous
/// one is read back (pipelined decode). Returns a lazy scalar int32.
public struct GPUSampler {
    public let base: Sampler
    public init(_ base: Sampler) { self.base = base }

    public var usesHistory: Bool { base.parameters.repetitionPenalty != 1 }

    public func sample(_ logits: MLXArray, history: MLXArray?) -> MLXArray {
        let p = base.parameters
        var l = logits.reshaped([-1, logits.dim(-1)])[-1].asType(.float32)
        if p.repetitionPenalty != 1, let history, history.size > 0 {
            // Duplicate ids write the same penalised value, so no de-duplication is needed.
            let seen = history.reshaped([-1]).asType(.int32)
            let old = take(l, seen, axis: 0)
            l = putAlong(l, seen, values: which(old .> 0, old / p.repetitionPenalty, old * p.repetitionPenalty), axis: 0)
        }
        return base.sample(l, history: [])
    }

    /// Last `window` ids of `history` followed by `token`.
    public func append(_ history: MLXArray?, _ token: MLXArray) -> MLXArray? {
        guard usesHistory else { return nil }
        let joined = history.map { concatenated([$0.reshaped([-1]).asType(.int32), token.reshaped([1]).asType(.int32)]) }
            ?? token.reshaped([1]).asType(.int32)
        let n = joined.dim(0)
        return n > base.penaltyWindow ? joined[(n - base.penaltyWindow)...] : joined
    }
}
