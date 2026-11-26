// Numerics shared by the reference model. Ported from mlx-vlm inkling2/language.py (MIT) at d998a62.
import MLX

enum Ops {
    /// `mx.fast.rms_norm`: normalise in FP32, cast to result_type(x, weight), then scale by weight.
    static func rmsNorm(_ x: MLXArray, weight: MLXArray?, eps: Float) -> MLXArray {
        #if canImport(Metal)
        if useFused, let weight { return MLXFast.rmsNorm(x, weight: weight, eps: eps) }
        #endif
        return rmsNormPortable(x, weight: weight, eps: eps)
    }

    /// Use `MLXFast` kernels where they exist (off on the CPU device: they are only checked against Python on Metal).
    static var useFused: Bool { Device.defaultDevice().deviceType == .gpu }

    static func rmsNormPortable(_ x: MLXArray, weight: MLXArray?, eps: Float) -> MLXArray {
        let outType = weight.map { promote(x.dtype, $0.dtype) } ?? x.dtype
        let xf = x.asType(.float32)
        var y = (xf * rsqrt(mean(square(xf), axis: -1, keepDims: true) + eps)).asType(outType)
        if let weight { y = y * weight }
        return y
    }

    static func promote(_ a: DType, _ b: DType) -> DType {
        if a == b { return a }
        if a == .float32 || b == .float32 { return .float32 }
        return (MLXArray.zeros([1], dtype: a) + MLXArray.zeros([1], dtype: b)).dtype
    }

    /// `mx.fast.scaled_dot_product_attention` with an additive mask; q [B,Hq,L,D], k/v [B,Hkv,S,D], mask [B,Hq,L,S].
    static func attention(q: MLXArray, k: MLXArray, v: MLXArray, scale: Float, mask: MLXArray) -> MLXArray {
        #if canImport(Metal)
        if useFused { return MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask) }
        #endif
        return attentionPortable(q: q, k: k, v: v, scale: scale, mask: mask)
    }

    static func attentionPortable(q: MLXArray, k: MLXArray, v: MLXArray, scale: Float, mask: MLXArray) -> MLXArray {
        let (b, hq, l, d) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
        let hkv = k.dim(1), rep = hq / hkv
        var q = MLXArray(scale).asType(q.dtype) * q
        var k = k, v = v, mask = mask
        if rep > 1 {
            q = q.reshaped([b, hkv, rep, l, d])
            k = k.expandedDimensions(axis: 2)
            v = v.expandedDimensions(axis: 2)
            mask = mask.dim(-3) == 1 ? mask.expandedDimensions(axis: -3)
                : mask.reshaped(Array(mask.shape.dropLast(3)) + [hkv, rep] + Array(mask.shape.suffix(2)))
        }
        var scores = matmul(q, k.swappedAxes(-1, -2)) + mask
        scores = softmax(scores, axis: -1, precise: true)
        let out = matmul(scores, v)
        return rep > 1 ? out.reshaped([b, hq, l, d]) : out
    }

    /// `l2_clip`: scale rows so their L2 norm is at most `threshold` (FP32 inside).
    static func l2Clip(_ x: MLXArray, _ threshold: Float, scale: MLXArray? = nil) -> MLXArray {
        var xf = x.asType(.float32)
        if let scale { xf = xf * scale.asType(.float32) }
        let norm = sqrt(sum(square(xf), axis: -1, keepDims: true))
        return (xf * minimum(threshold / (norm + 1e-6), 1)).asType(x.dtype)
    }

    /// `banded_additive_mask` (portable path): relative-position logits within `relExtent`, -1e30 outside the band.
    /// rel [B,LQ,H,dRel], proj [dRel,relExtent] → [B,H,LQ,S].
    static func bandedMask(rel: MLXArray, proj: MLXArray, qOffset: Int, keys s: Int, sliding: Int, relExtent: Int) -> MLXArray {
        let (b, lq, h) = (rel.dim(0), rel.dim(1), rel.dim(2))
        let dtype = rel.dtype
        let rl = matmul(rel, proj).transposed(0, 2, 1, 3)
        let qp = MLXArray(Int32(qOffset)..<Int32(qOffset + lq)).expandedDimensions(axis: 1)
        let kp = MLXArray(Int32(0)..<Int32(s)).expandedDimensions(axis: 0)
        let dist = qp - kp
        let gidx = broadcast(clip(dist, min: 0, max: relExtent - 1).reshaped([1, 1, lq, s]), to: [b, h, lq, s])
        var pb = takeAlong(rl, gidx, axis: -1)
        pb = which((dist .>= relExtent).reshaped([1, 1, lq, s]), MLXArray(Float(0)).asType(dtype), pb)
        var masked = dist .< 0
        if sliding > 0 { masked = masked .|| (dist .>= sliding) }
        return which(masked.reshaped([1, 1, lq, s]), MLXArray(Float(-1e30)).asType(dtype), pb).asType(dtype)
    }

    /// `x[..., i, :]`
    static func pick(_ x: MLXArray, _ i: Int) -> MLXArray { x[.ellipsis, i, 0...] }
}
