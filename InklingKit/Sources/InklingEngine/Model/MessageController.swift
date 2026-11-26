// Ported from mlx-vlm inkling2/language.py MessageController (MIT) at d998a62.
import MLX

/// Blends a previous loop pass's message into one sublayer input through a low-rank forget/gain gate.
struct MessageController {
    let down: MLXArray    // [rank, 2D]
    let forget: MLXArray  // [D, rank]
    let gain: MLXArray    // [D, rank]
    let eps: Float

    func callAsFunction(_ x: MLXArray, _ message: MLXArray) -> MLXArray {
        let pair = concatenated([Ops.rmsNorm(x, weight: nil, eps: eps), Ops.rmsNorm(message, weight: nil, eps: eps)], axis: -1)
        let hidden = matmul(pair.asType(down.dtype), down.T).asType(.float32)
        let f = tanh(matmul(hidden, forget.T))
        let g = tanh(matmul(hidden, gain.T))
        return (1 - f) * x + g * message
    }
}
