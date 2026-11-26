import Foundation
import Testing
import MLX
@testable import InklingEngine
#if canImport(Metal)
import Metal
#endif

/// MLXFast vs the portable ops, with the shapes and dtypes the model uses (Apple only).
@Suite(.serialized) struct FusedOpsTests {
    func devices() -> [Device] {
        #if canImport(Metal)
        var list: [Device] = [.cpu]
        if let d = MTLCreateSystemDefaultDevice(), d.supportsFamily(.apple7) { list.append(.gpu) }
        return list
        #else
        return []
        #endif
    }

    @Test(arguments: zip([1, 12, 4], [9, 13, 1025]))
    func sdpaMatchesPortable(l: Int, s: Int) {
        #if canImport(Metal)
        for device in devices() {
            Device.withDefaultDevice(device) {
                let key = MLXRandom.key(UInt64(l * 1000 + s))
                let ks = MLXRandom.split(key: key, into: 4)
                let q = MLXRandom.normal([1, 4, l, 16], key: ks[0]).asType(.bfloat16)
                let k = MLXRandom.normal([1, 2, s, 16], key: ks[1]).asType(.bfloat16)
                let v = MLXRandom.normal([1, 2, s, 16], key: ks[2]).asType(.bfloat16)
                var mask = MLXRandom.normal([1, 4, l, s], key: ks[3]).asType(.bfloat16)
                mask = which(MLXArray(Int32(0)..<Int32(s)) .> MLXArray(Int32(s - l)) + MLXArray(Int32(0)..<Int32(l)).expandedDimensions(axis: 1),
                             MLXArray(Float(-1e30)).asType(.bfloat16), mask)
                let fused = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.25, mask: mask).asType(.float32)
                let portable = Ops.attentionPortable(q: q, k: k, v: v, scale: 0.25, mask: mask).asType(.float32)
                let diff = abs(fused - portable).max().item(Float.self)
                #expect(diff < 0.05, "\(device) L=\(l) S=\(s): \(diff), fused nan: \(isNaN(fused).any().item(Bool.self))")
            }
        }
        #endif
    }

    @Test func rmsNormMatchesPortable() {
        #if canImport(Metal)
        for device in devices() {
            Device.withDefaultDevice(device) {
                for (xt, wt) in [(DType.float32, DType.bfloat16), (.bfloat16, .bfloat16), (.bfloat16, .float32)] {
                    let x = (MLXRandom.normal([1, 3, 64], key: MLXRandom.key(1)) * 3).asType(xt)
                    let w = (1 + 0.1 * MLXRandom.normal([64], key: MLXRandom.key(2))).asType(wt)
                    let fused = MLXFast.rmsNorm(x, weight: w, eps: 1e-6)
                    let portable = Ops.rmsNormPortable(x, weight: w, eps: 1e-6)
                    #expect(fused.dtype == portable.dtype, "\(device) \(xt)/\(wt): \(fused.dtype) vs \(portable.dtype)")
                    let diff = abs(fused.asType(.float32) - portable.asType(.float32)).max().item(Float.self)
                    #expect(diff < 0.05, "\(device) \(xt)/\(wt): \(diff)")
                }
            }
        }
        #endif
    }

    /// The whole tiny model on the GPU with the fused ops must still match Python.
    @Test func tinyModelOnGPUMatchesPython() throws {
        #if canImport(Metal)
        guard devices().contains(.gpu) else { return }
        try Device.withDefaultDevice(.gpu) {
            let g = try Tiny.golden()
            let model = try Tiny.model()
            let logits = model(MLXArray(g.tokens.map(Int32.init)).reshaped([1, -1]), cache: model.makeCache())[0].asType(.float32)
            let top = argMax(logits, axis: -1).asArray(Int32.self).map(Int.init)
            let expected = g.logits.map { argmax($0) }
            #expect(top == expected)
        }
        #endif
    }
}
