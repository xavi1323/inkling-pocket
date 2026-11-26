// Ported from mlx-vlm inkling2/language.py ShortConvolution / DynamicShortConvolution (MIT) at d998a62.
import Foundation
import MLX

/// Depthwise causal conv over the previous `kernelSize - 1` inputs plus a residual, all in FP32.
struct ShortConvolution {
    let weight: MLXArray  // [C, K, 1]
    let kernelSize: Int
    let slot: Int

    func callAsFunction(_ x: MLXArray, cache: LayerVisitCache?) -> MLXArray {
        let k = kernelSize
        let xf = x.asType(.float32)
        let context = convContext(xf, kernel: k, slot: slot, cache: cache)
        let out = conv1d(context, weight.asType(.float32), groups: x.dim(-1))
        return ((out + xf) / Float(sqrt(Double(k + 1)))).asType(x.dtype)
    }
}

/// Short conv with per-token taps: static taps plus a projection of `dynamicInput`, per KV head.
struct DynamicShortConvolution {
    let staticWeight: MLXArray  // [K, C]
    let weightProj: QLinear     // hidden → heads * K
    let numHeads: Int
    let kernelSize: Int
    let slot: Int

    func callAsFunction(_ x: MLXArray, dynamicInput: MLXArray, cache: LayerVisitCache?) -> MLXArray {
        let (b, length, channels) = (x.dim(0), x.dim(1), x.dim(2))
        let k = kernelSize
        let xf = x.asType(.float32)
        let context = convContext(xf, kernel: k, slot: slot, cache: cache)
        let dynamic = weightProj(dynamicInput).asType(.float32).reshaped([b, length, numHeads, 1, k])
        let weights = staticWeight.T.reshaped([numHeads, -1, k]) + dynamic
        var out = MLXArray.zeros([b, length, numHeads, channels / numHeads], dtype: .float32)
        for lag in 0..<k {
            let values = context[0..., (k - 1 - lag)..<(k - 1 - lag + length), 0...].reshaped(out.shape)
            out = out + values * weights[.ellipsis, lag]
        }
        return ((out.reshaped(x.shape) + xf) / Float(sqrt(Double(2 * k + 1)))).asType(x.dtype)
    }
}

/// `[state; inputs]` along time; stores the last `kernel - 1` rows back into the cache slot.
func convContext(_ inputs: MLXArray, kernel k: Int, slot: Int, cache: LayerVisitCache?) -> MLXArray {
    let state = cache?.conv[slot] ?? MLXArray.zeros([inputs.dim(0), k - 1, inputs.dim(-1)], dtype: .float32)
    let context = concatenated([state, inputs], axis: 1)
    cache?.conv[slot] = context[0..., (context.dim(1) - (k - 1))..., 0...]
    return context
}
