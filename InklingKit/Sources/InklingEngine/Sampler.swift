import MLX
import AgentCore

/// Picks the next token: repetition penalty, then temperature, top-k, top-p; argmax when temperature is 0.
public struct Sampler {
    public let parameters: GenerateParameters
    /// How many recent tokens the repetition penalty looks at.
    public var penaltyWindow = 64
    /// Top-p without top-k samples among this many best tokens (their mass is nearly always far past top-p).
    static let topPCandidates = 256

    public init(_ parameters: GenerateParameters) { self.parameters = parameters }

    /// `logits` [..., V] (last position is used); returns a scalar int32 token id.
    public func sample(_ logits: MLXArray, history: [Int32]) -> MLXArray {
        let p = parameters
        var l = logits.reshaped([-1, logits.dim(-1)])[-1].asType(.float32)
        if p.repetitionPenalty != 1, !history.isEmpty {
            let seen = MLXArray(Array(Set(history.suffix(penaltyWindow))))
            let old = take(l, seen, axis: 0)
            let penalised = which(old .> 0, old / p.repetitionPenalty, old * p.repetitionPenalty)
            l = putAlong(l, seen, values: penalised, axis: 0)
        }
        if p.temperature <= 0 { return argMax(l, axis: -1) }
        l = l / p.temperature
        let vocab = l.dim(0)
        let useTopK = p.topK > 0 && p.topK < vocab
        let useTopP = p.topP > 0 && p.topP < 1
        guard useTopK || useTopP else { return categorical(l) }
        // Work on the best `count` candidates sorted best-first: a partial selection over the vocabulary, then a sort
        // of only those (not a sort of all ~200k logits every token). Top-p alone keeps at most `topPCandidates`.
        let count = useTopK ? p.topK : min(Self.topPCandidates, vocab)
        let best = count < vocab ? argPartition(-l, kth: count - 1, axis: -1)[..<count] : MLXArray(Int32(0)..<Int32(vocab))
        let bestLogits = take(l, best, axis: 0)
        let byRank = argSort(-bestLogits, axis: -1)
        let order = take(best, byRank, axis: 0)
        var sorted = take(bestLogits, byRank, axis: 0)
        let rank = MLXArray(Int32(0)..<Int32(count))
        let negInf = MLXArray(-Float.infinity)
        if useTopK { sorted = which(rank .< Int32(p.topK), sorted, negInf) }
        if useTopP {
            let probs = softmax(sorted, axis: -1)
            let before = cumsum(probs, axis: -1) - probs
            sorted = which((before .< p.topP) .|| (rank .== 0), sorted, negInf)
        }
        return take(order, categorical(sorted), axis: 0)
    }
}
