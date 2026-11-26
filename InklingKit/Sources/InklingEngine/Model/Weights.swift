// Weight loading and (quantized) projections for the reference model.
import MLX
import InklingCore

/// Anything that hands out tensors by safetensors name.
public protocol WeightSource: AnyObject {
    func array(_ name: String) throws -> MLXArray
    func contains(_ name: String) -> Bool
}

extension MappedSafetensors: WeightSource {
    public func contains(_ name: String) -> Bool { entry(name) != nil }
}

public enum ModelLoadError: Error, Equatable {
    case missingWeight(String)
    case badQuantization(String)
    case unsupported(String)
}

/// Dense or affine-quantized `x @ W.T`.
struct QLinear {
    let weight: MLXArray
    let scales: MLXArray?
    let biases: MLXArray?
    let groupSize: Int
    let bits: Int

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let scales else { return matmul(x, weight.T) }
        return quantizedMM(x, weight, scales: scales, biases: biases, transpose: true, groupSize: groupSize, bits: bits)
    }
}

/// Per-expert projection, `gather_qmm` over routed experts.
struct QSwitchLinear {
    let linear: QLinear

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sorted: Bool) -> MLXArray {
        let l = linear
        guard let scales = l.scales else {
            return gatherMM(x, l.weight.swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: sorted)
        }
        return gatherQuantizedMM(x, l.weight, scales: scales, biases: l.biases, rhsIndices: indices, transpose: true,
                                 groupSize: l.groupSize, bits: l.bits, sortedIndices: sorted)
    }
}

/// Token embedding; quantized rows are dequantized on lookup (output dtype = scales dtype).
struct QEmbedding {
    let table: QLinear

    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        let flat = ids.flattened()
        let rows = take(table.weight, flat, axis: 0)
        let out: MLXArray
        if let scales = table.scales {
            out = dequantized(rows, scales: take(scales, flat, axis: 0), biases: table.biases.map { take($0, flat, axis: 0) },
                              groupSize: table.groupSize, bits: table.bits)
        } else {
            out = rows
        }
        return out.reshaped(ids.shape + [-1])
    }
}

/// Reads `language_model.<path>` tensors and builds projections using the config's quantization map.
struct WeightLoader {
    let source: WeightSource
    let quant: QuantizationMap
    var prefix = "language_model."
    /// When set (and the source is memory-mapped), MoE experts are streamed through this many slots per layer.
    var expertSlots: Int? = nil

    func has(_ path: String) -> Bool { source.contains(prefix + path) }

    func tensor(_ path: String) throws -> MLXArray {
        guard has(path) else { throw ModelLoadError.missingWeight(prefix + path) }
        return try source.array(prefix + path)
    }

    /// `path` is the module (e.g. `model.layers.0.self_attn.q_proj`); reads `.weight` and, if present, `.scales`/`.biases`.
    func linear(_ path: String) throws -> QLinear {
        let weight = try tensor(path + ".weight")
        guard has(path + ".scales") else { return QLinear(weight: weight, scales: nil, biases: nil, groupSize: 0, bits: 0) }
        let scales = try tensor(path + ".scales")
        let biases = has(path + ".biases") ? try tensor(path + ".biases") : nil
        guard let spec = quant.spec(for: path) else { throw ModelLoadError.badQuantization("\(path): quantized but not in config") }
        guard spec.mode == "affine" else { throw ModelLoadError.unsupported("\(path): quantization mode \(spec.mode)") }
        let inDims = scales.dim(-1) * spec.groupSize
        guard weight.dim(-1) * 32 == inDims * spec.bits else {
            throw ModelLoadError.badQuantization("\(path): \(spec.bits)-bit/\(spec.groupSize) does not fit \(weight.shape) with scales \(scales.shape)")
        }
        return QLinear(weight: weight, scales: scales, biases: biases, groupSize: spec.groupSize, bits: spec.bits)
    }
}
