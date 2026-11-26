// Ported from mlx-vlm inkling2/language.py AttentionResidual (MIT) at d998a62.
import Foundation
import MLX

/// Mixes the completed block outputs and the current partial with learned query attention (FP32).
/// Output [..., queries, D].
struct AttentionResidual {
    let query: MLXArray  // [D] or [n, D]
    let normalizeValues: Bool

    func callAsFunction(_ blocks: [MLXArray], _ partial: MLXArray) -> MLXArray {
        let states = stacked((blocks + [partial]).map { $0.asType(.float32) }, axis: -2)
        let d = states.dim(-1)
        let keys = Ops.rmsNorm(states, weight: nil, eps: 1e-6)
        let scores = matmul(keys, query.reshaped([-1, d]).T) * Float(1 / sqrt(Double(d)))
        let weights = softmax(scores, axis: -2).swappedAxes(-1, -2)
        return matmul(weights, normalizeValues ? keys : states)
    }
}
