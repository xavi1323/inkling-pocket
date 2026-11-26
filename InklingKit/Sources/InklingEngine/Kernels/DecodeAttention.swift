import MLX

/// One token's attention straight over the K/V cache buffers: relative-position logits within `relExtent`, the
/// sliding window, grouped KV heads and the query-dependent zero "sink", in one pass. Replaces building the
/// [heads × keys] mask, appending the sink column and copying K/V with a zero row every step.
/// On an Apple GPU one Metal kernel (one threadgroup of 8–32 simdgroups per head, online softmax); elsewhere MLX ops
/// in the reference's precision.
enum DecodeAttention {
    /// q [1,H,1,D], rel logits [H, relExtent], sinks [H] → [1,H,1,D].
    static func callAsFunction(q: MLXArray, kv: DecodeKV, rel: MLXArray, sinks: MLXArray, kvHeads: Int,
                               sliding: Int, relExtent: Int) -> MLXArray {
        let d = q.dim(-1)
        #if canImport(Metal)
        if Ops.useFused, d % 32 == 0, d <= 128 {
            return kernel(q: q, kv: kv, rel: rel, sinks: sinks, kvHeads: kvHeads, sliding: sliding, relExtent: relExtent)
        }
        #endif
        return portable(q: q, kv: kv, rel: rel, sinks: sinks, kvHeads: kvHeads, sliding: sliding, relExtent: relExtent)
    }

    /// Same math as `Ops.attentionPortable` with the banded mask (dtype scores, precise softmax), keys in slot order.
    static func portable(q: MLXArray, kv: DecodeKV, rel: MLXArray, sinks: MLXArray, kvHeads: Int,
                         sliding: Int, relExtent: Int) -> MLXArray {
        let (h, d) = (q.dim(1), q.dim(3))
        let cap = kv.keys.dim(2), rep = h / kvHeads
        let dtype = q.dtype
        let dist = kv.distances()
        let slot = MLXArray(Int32(0)..<Int32(cap))
        var masked = slot .>= Int32(kv.valid)
        if sliding > 0 { masked = masked .|| (dist .>= Int32(sliding)) }
        let bias = which(dist .>= Int32(relExtent), MLXArray(Float(0)).asType(dtype),
                         take(rel, clip(dist, min: 0, max: relExtent - 1), axis: 1))  // [H, cap]
        let scale = MLXArray(Float(1 / Double(d).squareRoot())).asType(dtype)
        let qs = (scale * q).reshaped([kvHeads, rep, 1, d])
        let k = kv.keys.reshaped([kvHeads, 1, cap, d]), v = kv.values.reshaped([kvHeads, 1, cap, d])
        var scores = matmul(qs, k.swappedAxes(-1, -2)).reshaped([h, cap]) + bias
        scores = which(masked, MLXArray(Float(-1e30)).asType(dtype), scores)
        let all = softmax(concatenated([scores, sinks.reshaped([h, 1]).asType(dtype)], axis: -1), axis: -1, precise: true)
        let p = all[0..., ..<cap].reshaped([kvHeads, rep, 1, cap])
        return matmul(p, v).reshaped([1, h, 1, d])
    }

    #if canImport(Metal)
    /// `groups`: simdgroups per head (each takes every `groups`-th key); picked on the device by `KernelTuning`.
    /// 32 × groups threads must cover D for the final step.
    static func kernel(q: MLXArray, kv: DecodeKV, rel: MLXArray, sinks: MLXArray, kvHeads: Int,
                       sliding: Int, relExtent: Int, groups requested: Int? = nil) -> MLXArray {
        let (h, d) = (q.dim(1), q.dim(3))
        let wanted = requested ?? KernelTuning.shared.attentionGroups
        let groups = 32 * wanted >= d ? wanted : 32
        let meta = MLXArray([Int32(kv.valid), Int32(kv.index), Int32(kv.keys.dim(2))])
        let out = decodeKernel(
            [q.reshaped([-1]), kv.keys, kv.values, rel.reshaped([-1]), sinks.reshaped([-1]).asType(q.dtype), meta],
            template: [("T", q.dtype), ("D", d), ("REP", h / kvHeads), ("REL", relExtent), ("SLIDING", sliding), ("NSG", groups)],
            grid: (32 * groups * h, 1, 1), threadGroup: (32 * groups, 1, 1),
            outputShapes: [[h * d]], outputDTypes: [q.dtype])[0]
        return out.reshaped([1, h, 1, d])
    }

    static let decodeKernel = MLXFast.metalKernel(
        name: "inkling_decode_attention", inputNames: ["q", "keys", "values", "rel", "sinks", "meta"],
        outputNames: ["out"], source: source)

    static let source = """
        constexpr int E = D / 32;
        const uint head = threadgroup_position_in_grid.x;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const int valid = meta[0], index = meta[1], cap = meta[2];
        const size_t base = size_t(head / REP) * size_t(cap) * D;
        const float scale = rsqrt(float(D));
        float qv[E], o[E];
        for (int i = 0; i < E; i++) {
            qv[i] = float(q[head * D + lane * E + i]) * scale;
            o[i] = 0.0f;
        }
        float m = -INFINITY, l = 0.0f;
        for (int j = int(sg); j < valid; j += NSG) {
            const int dist = j < index ? index - 1 - j : index - 1 - j + valid;
            if (SLIDING > 0 && dist >= SLIDING) continue;
            const size_t row = base + size_t(j) * D + lane * E;
            float s = 0.0f;
            for (int i = 0; i < E; i++) s += qv[i] * float(keys[row + i]);
            s = simd_sum(s);
            if (dist < REL) s += float(rel[head * REL + dist]);
            const float mn = max(m, s);
            const float a = metal::exp(m - mn), p = metal::exp(s - mn);
            for (int i = 0; i < E; i++) o[i] = o[i] * a + p * float(values[row + i]);
            l = l * a + p;
            m = mn;
        }
        threadgroup float tm[NSG];
        threadgroup float tl[NSG];
        threadgroup float to[NSG * D];
        if (lane == 0) { tm[sg] = m; tl[sg] = l; }
        for (int i = 0; i < E; i++) to[sg * D + lane * E + i] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint t = thread_position_in_threadgroup.x;
        if (t < uint(D)) {
            const float sink = float(sinks[head]);
            float mx = sink;
            for (int g = 0; g < NSG; g++) mx = max(mx, tm[g]);
            float den = metal::exp(sink - mx), acc = 0.0f;
            for (int g = 0; g < NSG; g++) {
                const float w = metal::exp(tm[g] - mx);
                den += tl[g] * w;
                acc += to[g * D + t] * w;
            }
            out[head * D + t] = T(acc / den);
        }
        """
    #endif
}
