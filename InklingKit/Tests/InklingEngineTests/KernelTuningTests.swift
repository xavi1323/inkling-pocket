import Foundation
import Testing
import MLX
@testable import InklingEngine
#if canImport(Metal)
import Metal
#endif

/// Every launch setting `KernelTuning` may pick gives the stock-ops result (Apple GPU only; macOS CI runs these).
@Suite(.serialized) struct KernelTuningTests {
    #if canImport(Metal)
    var hasGPU: Bool { MTLCreateSystemDefaultDevice()?.supportsFamily(.apple7) ?? false }

    func experts(bits: Int, e: Int = 6, d: Int = 256, i: Int = 128) -> FusedMoEDecode {
        func q(_ shape: [Int], _ seed: UInt64) -> QLinear {
            let w = (MLXRandom.normal(shape, key: MLXRandom.key(seed)) * 0.05).asType(.bfloat16)
            let (wq, s, b) = quantized(w, groupSize: 64, bits: bits)
            return QLinear(weight: wq, scales: s, biases: b, groupSize: 64, bits: bits)
        }
        return FusedMoEDecode(gate: q([e, i, d], 1), up: q([e, i, d], 2), down: q([e, d, i], 3), gateCap: 10, upCap: 10)
    }

    @Test(arguments: [4, 8])
    func everyMoELaunchMatchesStockOps(bits: Int) throws {
        guard hasGPU else { return }
        Device.withDefaultDevice(.gpu) {
            let fused = experts(bits: bits)
            let x = (MLXRandom.normal([256], key: MLXRandom.key(4)) * 0.5).asType(.bfloat16)
            let ids = MLXArray([Int32(5), 0, 3, 1])
            let w = softmax(MLXRandom.normal([4], key: MLXRandom.key(5)), axis: -1)
            let want = fused.fallback(x, ids, w).asType(.float32)
            for launch in KernelTuning.launches {
                let got = fused.downStep(fused.gateUpStep(x, ids, launch), ids, w, launch).asType(.float32)
                let diff = abs(got - want).max().item(Float.self), scale = abs(want).max().item(Float.self)
                #expect(diff <= 0.02 * max(scale, 1e-3), "\(bits)-bit \(launch): diff \(diff) of \(scale)")
            }
        }
    }

    @Test func everyAttentionGroupCountMatchesPortable() {
        guard hasGPU else { return }
        Device.withDefaultDevice(.gpu) {
            let (h, kvh, d, cap, valid, rel) = (8, 2, 64, 96, 70, 16)
            let q = MLXRandom.normal([1, h, 1, d], key: MLXRandom.key(1)).asType(.bfloat16)
            let kv = DecodeKV(keys: MLXRandom.normal([1, kvh, cap, d], key: MLXRandom.key(2)).asType(.bfloat16),
                              values: MLXRandom.normal([1, kvh, cap, d], key: MLXRandom.key(3)).asType(.bfloat16),
                              valid: valid, index: 40)
            let r = (MLXRandom.normal([h, rel], key: MLXRandom.key(4)) * 0.1).asType(.bfloat16)
            let sinks = MLXRandom.normal([h], key: MLXRandom.key(5)).asType(.bfloat16)
            for sliding in [0, 32] {
                let want = DecodeAttention.portable(q: q, kv: kv, rel: r, sinks: sinks, kvHeads: kvh, sliding: sliding,
                                                    relExtent: rel).asType(.float32)
                for g in KernelTuning.groupChoices {
                    let got = DecodeAttention.kernel(q: q, kv: kv, rel: r, sinks: sinks, kvHeads: kvh, sliding: sliding,
                                                     relExtent: rel, groups: g).asType(.float32)
                    let diff = abs(got - want).max().item(Float.self)
                    #expect(diff < 0.05, "groups \(g) sliding \(sliding): \(diff)")
                }
            }
        }
    }
    #endif

    @Test func wideLoadsNeedAlignedRows() throws {
        let fused = FusedMoEDecode(gate: QLinear(weight: MLXArray(0), scales: nil, biases: nil, groupSize: 64, bits: 4),
                                   up: QLinear(weight: MLXArray(0), scales: nil, biases: nil, groupSize: 64, bits: 4),
                                   down: QLinear(weight: MLXArray(0), scales: nil, biases: nil, groupSize: 64, bits: 4),
                                   gateCap: 1, upCap: 1)
        #expect(fused.supports(8, inputs: 24, bits: [3]))
        #expect(fused.supports(32, inputs: 256, bits: [4, 4]))
        #expect(!fused.supports(32, inputs: 256, bits: [8]))   // 32 bytes: wider than one load
        #expect(fused.supports(16, inputs: 256, bits: [8]))
        #expect(!fused.supports(16, inputs: 24, bits: [4]))    // not whole steps
        #expect(!fused.supports(16, inputs: 256, bits: [4, 8])) // gate and up differ
        #expect(!fused.supports(16, inputs: 256, bits: [3]))
    }
}
