import Foundation
import MLX
#if canImport(Metal)
import Metal
#endif

/// Launch settings for the decode kernels, picked on the device: each candidate runs on this model's real expert
/// slots (and attention shapes), must match the default kernel's output, and the fastest wins if it beats the
/// default by `minGain`. Apple's Metal guidance: tile and threadgroup sizes are workload and GPU specific; measure.
final class KernelTuning: @unchecked Sendable {
    static let shared = KernelTuning()

    struct Launch: Equatable, Codable, CustomStringConvertible {
        var values: Int   // values per lane step (8 = one 8-value block; 16/32 = one wide load)
        var threads: Int  // threadgroup size
        var description: String { "\(values)×\(threads)" }
        static let standard = Launch(values: 8, threads: 256)
    }

    struct MoE: Equatable, Codable { var gateUp = Launch.standard, down = Launch.standard }

    /// A tuning result kept between launches; `key` names the GPU, OS and kernel shapes it is valid for.
    struct Saved: Codable { var key: String, moe: MoE, attentionGroups: Int, report: String }

    private let lock = NSLock()
    private var _moe = MoE()
    private var _attentionGroups = 32
    private(set) var report = "not tuned"

    var moe: MoE { lock.withLock { _moe } }
    var attentionGroups: Int { lock.withLock { _attentionGroups } }

    static let launches = [8, 16, 32].flatMap { v in [128, 256, 512].map { Launch(values: v, threads: $0) } }
    static let groupChoices = [8, 16, 32]
    /// A candidate must be this much faster than the default to replace it (timing noise).
    static let minGain = 0.03

    /// Close enough to the default kernel: only the order of float additions differs.
    static func matches(_ a: MLXArray, _ b: MLXArray) -> Bool {
        let a = a.asType(.float32), b = b.asType(.float32)
        guard !isNaN(a).any().item(Bool.self), !isInf(a).any().item(Bool.self) else { return false }
        let scale = max(1, abs(b).max().item(Float.self))
        return abs(a - b).max().item(Float.self) <= 0.02 * scale
    }

    /// Seconds per call: `calls` kernels queued together, best of 3 rounds (first run builds the pipeline).
    static func time(calls: Int, _ make: (Int) -> MLXArray) -> Double {
        eval((0..<calls).map(make))
        var best = Double.infinity
        for _ in 0..<3 {
            let start = ContinuousClock.now
            eval((0..<calls).map(make))
            best = min(best, (ContinuousClock.now - start) / .seconds(1) / Double(calls))
        }
        return best
    }

    /// Fastest candidate that matches `reference`; the default unless one beats it by `minGain`.
    static func pick<C>(_ candidates: [C], standard: C, run: (C, Int) -> MLXArray, reference: MLXArray,
                        calls: Int) -> (C, [String]) {
        let base = time(calls: calls) { run(standard, $0) }
        var best = (standard, base), notes: [String] = []
        for c in candidates {
            guard matches(run(c, 0), reference) else { notes.append("\(c) wrong"); continue }
            let t = time(calls: calls) { run(c, $0) }
            if t < best.1 { best = (c, t) }
        }
        if best.1 > base * (1 - minGain) { best = (standard, base) }
        notes.append(String(format: "%@ %.0f%% faster", "\(best.0)", (base / best.1 - 1) * 100))
        return (best.0, notes)
    }

    /// Tunes on `model`'s first streamed MoE layer and first global attention layer. GPU (fused path) only.
    /// With `saved`: reuses the result kept there for this GPU, OS and model shapes (tuning costs ~1 s of GPU time
    /// and hundreds of kernel builds at every launch otherwise), and keeps a new result there.
    func tune(_ model: Inkling2LanguageModel, saved url: URL? = nil) {
        #if canImport(Metal)
        guard Ops.useFused, Device.defaultDevice().deviceType == .gpu,
              let layer = model.model.layers.first(where: { $0.moe.fused.kernelSupported }) else { return }
        let fused = layer.moe.fused
        let attention0 = model.model.layers.map(\.attention).first { $0.sliding == 0 } ?? layer.attention
        let key = [MTLCreateSystemDefaultDevice()?.name ?? "gpu", ProcessInfo.processInfo.operatingSystemVersionString,
                   "\(fused.down.weight.dim(1))x\(fused.gate.weight.dim(1))", "\(fused.gate.bits)/\(fused.up.bits)/\(fused.down.bits)",
                   "k\(layer.moe.topK)", "h\(attention0.nHeads)/\(attention0.nKV)x\(attention0.headDim)"].joined(separator: "|")
        if let url, let data = try? Data(contentsOf: url), let s = try? JSONDecoder().decode(Saved.self, from: data), s.key == key {
            lock.withLock {
                _moe = s.moe
                _attentionGroups = s.attentionGroups
                report = s.report + " (saved)"
            }
            return
        }
        defer {
            if let url, let data = try? JSONEncoder().encode(Saved(key: key, moe: moe, attentionGroups: attentionGroups, report: report)) {
                try? data.write(to: url, options: .atomic)
            }
        }
        let slots = fused.gate.weight.dim(0), k = layer.moe.topK, d = fused.down.weight.dim(1)
        let dtype = fused.gate.scales!.dtype
        let x = (MLXRandom.normal([d], key: MLXRandom.key(7)) * 0.5).asType(dtype)
        let w = softmax(MLXRandom.normal([k], key: MLXRandom.key(8)), axis: -1).asType(.float32)
        // Different slots each call, so weights come from memory, not the GPU cache.
        func ids(_ call: Int) -> MLXArray { MLXArray((0..<k).map { Int32((call * k + $0) % slots) }) }
        let calls = max(1, min(24, slots / k))
        let h0 = fused.gateUpStep(x, ids(0), .standard)
        let (gateUp, n1) = Self.pick(Self.launches, standard: .standard, run: { fused.gateUpStep(x, ids($1), $0) },
                                     reference: h0, calls: calls)
        let hs = (0..<calls).map { fused.gateUpStep(x, ids($0), .standard) }
        eval(hs)
        let out0 = fused.downStep(hs[0], ids(0), w, .standard)
        let (down, n2) = Self.pick(Self.launches, standard: .standard, run: { fused.downStep(hs[$1], ids($1), w, $0) },
                                   reference: out0, calls: calls)
        var groups = 32, n3: [String] = []
        let attention = model.model.layers.map(\.attention).first { $0.sliding == 0 } ?? layer.attention
        if attention.headDim % 32 == 0, attention.headDim <= 128 {
            let (hd, heads, kvHeads) = (attention.headDim, attention.nHeads, attention.nKV)
            let cap = 1024, valid = 640
            let q = MLXRandom.normal([1, heads, 1, hd], key: MLXRandom.key(9)).asType(dtype)
            let keys = MLXRandom.normal([1, kvHeads, cap, hd], key: MLXRandom.key(10)).asType(dtype)
            let values = MLXRandom.normal([1, kvHeads, cap, hd], key: MLXRandom.key(11)).asType(dtype)
            let rel = (MLXRandom.normal([heads, attention.relExtent], key: MLXRandom.key(12)) * 0.1).asType(dtype)
            let sinks = MLXArray.zeros([heads], dtype: dtype)
            let kv = DecodeKV(keys: keys, values: values, valid: valid, index: valid)
            func run(_ g: Int, _ call: Int) -> MLXArray {
                DecodeAttention.kernel(q: q, kv: kv, rel: rel, sinks: sinks, kvHeads: kvHeads,
                                       sliding: attention.sliding, relExtent: attention.relExtent, groups: g)
            }
            let ref = run(32, 0)
            (groups, n3) = Self.pick(Self.groupChoices.filter { 32 * $0 >= hd }, standard: 32, run: run,
                                     reference: ref, calls: 24)
        }
        lock.withLock {
            _moe = MoE(gateUp: gateUp, down: down)
            _attentionGroups = groups
            report = "gate/up \(n1.last ?? ""), down \(n2.last ?? ""), attention \(groups) groups: \(n3.last ?? "")"
                + ((n1 + n2 + n3).contains { $0.hasSuffix("wrong") } ? " (some variants rejected)" : "")
        }
        #endif
    }

    /// Back to the defaults (tests).
    func reset() {
        lock.withLock {
            _moe = MoE()
            _attentionGroups = 32
            report = "not tuned"
        }
    }
}
