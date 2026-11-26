// Ported from mlx-vlm inkling2/language.py Attention (MIT) at d998a62 (single device, no speculation).
import Foundation
import MLX
import InklingCore

/// Attention with relative-position logits, a query-dependent sink, short convs on K/V and a sigmoid output gate.
final class Attention {
    enum KVConv {
        case fixed(ShortConvolution)
        case dynamic(DynamicShortConvolution)
    }

    let config: InklingTextConfig
    let isSliding: Bool
    let headDim, nHeads, nKV, sliding, relExtent: Int
    let qProj, kProj, vProj, rProj, oProj, gateProj, sinkProj: QLinear
    let qScale, relScale, relProj: MLXArray
    let kConv, vConv: KVConv

    init(_ config: InklingTextConfig, layer: Int, path: String, loader: WeightLoader) throws {
        self.config = config
        isSliding = config.isSliding(layer)
        headDim = isSliding ? config.swaHeadDim : config.headDim
        nHeads = isSliding ? config.swaNumAttentionHeads : config.numAttentionHeads
        nKV = isSliding ? config.swaNumKeyValueHeads : config.numKeyValueHeads
        sliding = isSliding ? config.slidingWindowSize : 0
        relExtent = isSliding ? config.slidingWindowSize : config.relExtent
        qProj = try loader.linear(path + ".q_proj")
        kProj = try loader.linear(path + ".k_proj")
        vProj = try loader.linear(path + ".v_proj")
        rProj = try loader.linear(path + ".r_proj")
        oProj = try loader.linear(path + ".o_proj")
        gateProj = try loader.linear(path + ".gate_proj")
        sinkProj = try loader.linear(path + ".sink_proj")
        qScale = try loader.tensor(path + ".q_scale")
        relScale = try loader.tensor(path + ".rel_scale")
        relProj = try loader.tensor(path + ".rel_proj")
        let k = config.sconvKernelSize, sliding = isSliding, heads = nKV
        func conv(_ name: String, _ slot: Int) throws -> KVConv {
            if sliding { return .fixed(ShortConvolution(weight: try loader.tensor("\(path).\(name).conv.weight"), kernelSize: k, slot: slot)) }
            return .dynamic(DynamicShortConvolution(staticWeight: try loader.tensor("\(path).\(name).static_weight"),
                                                    weightProj: try loader.linear("\(path).\(name).weight_proj"),
                                                    numHeads: heads, kernelSize: k, slot: slot))
        }
        kConv = try conv("k_sconv", 0)
        vConv = try conv("v_sconv", 1)
    }

    private func apply(_ conv: KVConv, _ x: MLXArray, _ input: MLXArray, _ cache: LayerVisitCache?) -> MLXArray {
        switch conv {
        case .fixed(let c): c(x, cache: cache)
        case .dynamic(let c): c(x, dynamicInput: input, cache: cache)
        }
    }

    func callAsFunction(_ x: MLXArray, keyInput: MLXArray, valueInput: MLXArray, cache: LayerVisitCache?) -> MLXArray {
        let (b, length) = (x.dim(0), x.dim(1))
        let cfg = config
        var k = apply(kConv, kProj(keyInput), keyInput, cache)
        var v = apply(vConv, vProj(valueInput), valueInput, cache)
        var q = Ops.l2Clip(qProj(x).reshaped([b, length, nHeads, headDim]), cfg.qkL2ClipThreshold, scale: qScale)
        k = Ops.l2Clip(k.reshaped([b, length, nKV, headDim]), cfg.qkL2ClipThreshold)
        v = v.reshaped([b, length, nKV, headDim])
        if cfg.useValueNormNoGamma {
            v = Ops.rmsNorm(v.asType(.float32), weight: nil, eps: cfg.rmsNormEps).asType(x.dtype)
        }
        var r = Ops.l2Clip(rProj(x).reshaped([b, length, nHeads, cfg.dRel]), cfg.relL2ClipThreshold, scale: relScale)
        if !isSliding, let floor = cfg.logScalingNFloor {
            let offset = cache?.offset ?? 0
            let positions = MLXArray(Int32(offset + 1)..<Int32(offset + 1 + length)).asType(.float32)
            let tau = (1 + cfg.logScalingAlpha * log(maximum(positions / Float(floor), 1))).reshaped([length, 1, 1])
            q = (q.asType(.float32) * tau).asType(x.dtype)
            r = (r.asType(.float32) * tau).asType(x.dtype)
        }
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)
        if let cache { (k, v) = cache.updateAndFetch(keys: k, values: v) }
        let size = k.dim(2)
        var mask = Ops.bandedMask(rel: r, proj: relProj.asType(x.dtype), qOffset: size - length, keys: size,
                                  sliding: sliding, relExtent: relExtent)
        // The sink is a zero key/value with its own query-dependent logit.
        let sinks = sinkProj(x).transposed(0, 2, 1).expandedDimensions(axis: -1)
        mask = concatenated([mask, sinks], axis: -1)
        k = concatenated([k, MLXArray.zeros([b, nKV, 1, headDim], dtype: k.dtype)], axis: 2)
        v = concatenated([v, MLXArray.zeros([b, nKV, 1, headDim], dtype: v.dtype)], axis: 2)
        var out = Ops.attention(q: q, k: k, v: v, scale: Float(1 / sqrt(Double(headDim))), mask: mask)
        out = out.transposed(0, 2, 1, 3).reshaped([b, length, -1])
        out = (out.asType(.float32) * sigmoid(gateProj(x).asType(.float32))).asType(x.dtype)
        return oProj(out)
    }
}
