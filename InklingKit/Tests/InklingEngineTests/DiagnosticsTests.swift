import Foundation
import Testing
import MLX
@testable import InklingEngine

/// Pinpoints where NaNs appear (macOS CI showed NaN logits while Linux matches Python).
@Suite(.serialized) struct DiagnosticsTests {
    init() { Device.setDefault(device: .cpu) }

    static func bf16(_ bits: UInt16) -> Float { Float(bitPattern: UInt32(bits) << 16) }

    @Test func loadedWeightsMatchFileBytes() throws {
        let weights = try MappedSafetensors(files: [Tiny.dir.appending(path: "model.safetensors")])
        var bad: [String] = []
        for name in weights.names {
            guard let e = weights.entry(name), e.dtype == "BF16" || e.dtype == "F32" else { continue }
            let p = try weights.pointer(name)
            let count = e.shape.reduce(1, *)
            var fileSum = 0.0
            for i in 0..<count {
                fileSum += Double(e.dtype == "BF16" ? Self.bf16(p.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self))
                                                    : p.loadUnaligned(fromByteOffset: i * 4, as: Float.self))
            }
            let arraySum = Double(try weights.array(name).asType(.float32).sum().item(Float.self))
            if !(abs(arraySum - fileSum) <= 1e-3 * max(1, abs(fileSum))) { bad.append("\(name) \(e.dtype) file \(fileSum) array \(arraySum)") }
        }
        let summary = bad.prefix(8).joined(separator: "; ")
        #expect(bad.isEmpty, "\(bad.count) tensors differ: \(summary)")
    }

    @Test func moeStagesAreFinite() throws {
        let model = try Tiny.model()
        let moe = model.model.layers[0].moe
        let x = (MLXRandom.normal([1, 1, model.config.hiddenSize], key: MLXRandom.key(1)) * 0.5).asType(model.model.embedNorm.dtype)
        func finite(_ a: MLXArray) -> Bool { !isNaN(a.asType(.float32)).any().item(Bool.self) && isInf(a.asType(.float32)).any().item(Bool.self) == false }
        let scores = sigmoid(matmul(x.asType(.float32), moe.gateWeight.asType(.float32).T))
        let (ids, w) = moe.route(x)
        let rows = x.expandedDimensions(axes: [-2, -3])
        let gate = moe.gateProj(rows, ids, sorted: false)
        let l = moe.gateProj.linear
        let deq = dequantized(l.weight, scales: l.scales!, biases: l.biases, groupSize: l.groupSize, bits: l.bits)
        let qmm = quantizedMM(x.reshaped([1, -1]), l.weight[0], scales: l.scales![0], biases: l.biases?[0], transpose: true,
                              groupSize: l.groupSize, bits: l.bits)
        let report = [("x", finite(x)), ("gateWeight", finite(moe.gateWeight)), ("scores", finite(scores)), ("weights", finite(w)),
                      ("scales", finite(l.scales!)), ("biases", finite(l.biases!)), ("dequantized", finite(deq)),
                      ("quantizedMM", finite(qmm)), ("gatherQMM", finite(gate)), ("moe", finite(moe(x)))]
        let text = report.map { "\($0.0)=\($0.1)" }.joined(separator: " ") + " ids=\(ids.asArray(Int32.self)) bits=\(l.bits) group=\(l.groupSize) dtype=\(x.dtype)"
        let ok = report.allSatisfy(\.1)
        #expect(ok, "\(text)")
    }
}
