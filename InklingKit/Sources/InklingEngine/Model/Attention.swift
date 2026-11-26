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

    private func apply(_ conv: KVConv, _ x: MLXArray, _ input: MLXArray, _ state: MLXArray?) -> (MLXArray, MLXArray) {
        switch conv {
        case .fixed(let c): c(x, state: state)
        case .dynamic(let c): c(x, dynamicInput: input, state: state)
        }
    }

    /// Everything before the K/V cache, pure: q, k, v [B,H,L,D], r [B,L,H,dRel], sinks [B,H,L,1], the FP32 output
    /// gate, and the new k/v conv states. `positions` [L] FP32 are the 1-based token positions.
    struct Projected {
        var q, k, v, r, sinks, gate, stateK, stateV: MLXArray
        var arrays: [MLXArray] { [q, k, v, r, sinks, gate, stateK, stateV] }
        init(q: MLXArray, k: MLXArray, v: MLXArray, r: MLXArray, sinks: MLXArray, gate: MLXArray, stateK: MLXArray, stateV: MLXArray) {
            (self.q, self.k, self.v, self.r, self.sinks, self.gate, self.stateK, self.stateV) = (q, k, v, r, sinks, gate, stateK, stateV)
        }
        init(_ a: [MLXArray]) { self.init(q: a[0], k: a[1], v: a[2], r: a[3], sinks: a[4], gate: a[5], stateK: a[6], stateV: a[7]) }
    }

    func project(_ x: MLXArray, keyInput: MLXArray, valueInput: MLXArray, stateK: MLXArray?, stateV: MLXArray?,
                 positions: MLXArray) -> Projected {
        let (b, length) = (x.dim(0), x.dim(1))
        let cfg = config
        var (k, nextK) = apply(kConv, kProj(keyInput), keyInput, stateK)
        var (v, nextV) = apply(vConv, vProj(valueInput), valueInput, stateV)
        var q = Ops.l2Clip(qProj(x).reshaped([b, length, nHeads, headDim]), cfg.qkL2ClipThreshold, scale: qScale)
        k = Ops.l2Clip(k.reshaped([b, length, nKV, headDim]), cfg.qkL2ClipThreshold)
        v = v.reshaped([b, length, nKV, headDim])
        if cfg.useValueNormNoGamma {
            v = Ops.rmsNorm(v.asType(.float32), weight: nil, eps: cfg.rmsNormEps).asType(x.dtype)
        }
        var r = Ops.l2Clip(rProj(x).reshaped([b, length, nHeads, cfg.dRel]), cfg.relL2ClipThreshold, scale: relScale)
        if !isSliding, let floor = cfg.logScalingNFloor {
            let tau = (1 + cfg.logScalingAlpha * log(maximum(positions / Float(floor), 1))).reshaped([length, 1, 1])
            q = (q.asType(.float32) * tau).asType(x.dtype)
            r = (r.asType(.float32) * tau).asType(x.dtype)
        }
        let sinks = sinkProj(x).transposed(0, 2, 1).expandedDimensions(axis: -1)
        let gate = sigmoid(gateProj(x).asType(.float32))
        return Projected(q: q.transposed(0, 2, 1, 3), k: k.transposed(0, 2, 1, 3), v: v.transposed(0, 2, 1, 3), r: r,
                         sinks: sinks, gate: gate, stateK: nextK, stateV: nextV)
    }

    /// 1-based positions of `length` tokens after `offset`, FP32.
    static func positions(offset: Int, length: Int) -> MLXArray {
        MLXArray(Int32(offset + 1)..<Int32(offset + 1 + length)).asType(.float32)
    }

    /// Attention over the cache (updated with this step's k/v): [B,H,L,D].
    func attend(_ p: Projected, cache: LayerVisitCache?) -> MLXArray {
        let (b, length) = (p.q.dim(0), p.q.dim(2))
        let dtype = p.q.dtype
        var (k, v) = (p.k, p.v)
        if let cache { (k, v) = cache.updateAndFetch(keys: k, values: v) }
        let size = k.dim(2)
        var mask = Ops.bandedMask(rel: p.r, proj: relProj.asType(dtype), qOffset: size - length, keys: size,
                                  sliding: sliding, relExtent: relExtent)
        // The sink is a zero key/value with its own query-dependent logit.
        mask = concatenated([mask, p.sinks], axis: -1)
        k = concatenated([k, MLXArray.zeros([b, nKV, 1, headDim], dtype: k.dtype)], axis: 2)
        v = concatenated([v, MLXArray.zeros([b, nKV, 1, headDim], dtype: v.dtype)], axis: 2)
        return Ops.attention(q: p.q, k: k, v: v, scale: Float(1 / sqrt(Double(headDim))), mask: mask)
    }

    /// Output gate and projection of `attend`'s result, pure.
    func finish(_ out: MLXArray, gate: MLXArray) -> MLXArray {
        let (b, length) = (out.dim(0), out.dim(2))
        let o = out.transposed(0, 2, 1, 3).reshaped([b, length, -1])
        return oProj((o.asType(.float32) * gate).asType(out.dtype))
    }

    func callAsFunction(_ x: MLXArray, keyInput: MLXArray, valueInput: MLXArray, cache: LayerVisitCache?) -> MLXArray {
        let p = project(x, keyInput: keyInput, valueInput: valueInput, stateK: cache?.conv[0], stateV: cache?.conv[1],
                        positions: Self.positions(offset: cache?.offset ?? 0, length: x.dim(1)))
        cache?.conv[0] = p.stateK
        cache?.conv[1] = p.stateV
        return finish(attend(p, cache: cache), gate: p.gate)
    }
}
