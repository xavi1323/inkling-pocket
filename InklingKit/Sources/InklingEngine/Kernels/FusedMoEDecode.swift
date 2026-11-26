import MLX

/// Single-token MoE: gather + affine dequant + gate/up + softcapped SwiGLU + down + weighted sum (E6).
/// On an Apple GPU this is two Metal kernels (the down projection needs every intermediate row first, and Metal has
/// no grid-wide barrier); elsewhere it is a dequantize + matmul composition that never calls `gatherQuantizedMM`.
final class FusedMoEDecode {
    let gate, up, down: QLinear
    let gateCap, upCap: Float

    /// Softcaps as one array, made once (not copied to the GPU every step).
    private lazy var caps = MLXArray([gateCap, upCap])

    init(gate: QLinear, up: QLinear, down: QLinear, gateCap: Float, upCap: Float) {
        self.gate = gate; self.up = up; self.down = down
        self.gateCap = gateCap; self.upCap = upCap
    }

    /// x [D] or [1,1,D], expert ids [k], route weights [k] (FP32) → [D] in x's dtype.
    func callAsFunction(x: MLXArray, expertIDs: MLXArray, weights: MLXArray) -> MLXArray {
        let x = x.reshaped([-1]), ids = expertIDs.reshaped([-1]).asType(.int32), w = weights.reshaped([-1]).asType(.float32)
        if kernelSupported, Device.defaultDevice().deviceType == .gpu { return kernel(x, ids, w) }
        return fallback(x, ids, w)
    }

    /// Shapes the kernels handle: quantized with biases, 8-aligned dims and groups, ≤ 8 bits.
    var kernelSupported: Bool {
        #if canImport(Metal)
        let all = [gate, up, down]
        guard all.allSatisfy({ $0.scales != nil && $0.biases != nil && $0.bits <= 8 && $0.groupSize % 8 == 0 }) else { return false }
        let d = down.weight.dim(1), i = gate.weight.dim(1)
        return d % 8 == 0 && i % 8 == 0
        #else
        return false
        #endif
    }

    func kernel(_ x: MLXArray, _ ids: MLXArray, _ w: MLXArray) -> MLXArray {
        let config = KernelTuning.shared.moe
        return downStep(gateUpStep(x, ids, config.gateUp), ids, w, config.down)
    }

    /// The gate/up kernel with `launch` (values per lane step, threadgroup size): [k * I] activations.
    func gateUpStep(_ x: MLXArray, _ ids: MLXArray, _ launch: KernelTuning.Launch) -> MLXArray {
        #if canImport(Metal)
        let d = x.dim(0), i = gate.weight.dim(1), k = ids.dim(0)
        let v = supports(launch.values, inputs: d, bits: [gate.bits, up.bits]) ? launch.values : 8
        return Self.gateUpKernel(
            [x, ids, gate.weight, gate.scales!, gate.biases!, up.weight, up.scales!, up.biases!, caps],
            template: [("T", x.dtype), ("D", d), ("I", i), ("GB", gate.bits), ("GG", gate.groupSize),
                       ("UB", up.bits), ("UG", up.groupSize), ("V", v)],
            grid: (32 * i, k, 1), threadGroup: (launch.threads, 1, 1),
            outputShapes: [[k * i]], outputDTypes: [x.dtype])[0]
        #else
        fatalError("Metal kernels need an Apple GPU")
        #endif
    }

    /// The down kernel with `launch`: [D] weighted sum over the k experts.
    func downStep(_ h: MLXArray, _ ids: MLXArray, _ w: MLXArray, _ launch: KernelTuning.Launch) -> MLXArray {
        #if canImport(Metal)
        let d = down.weight.dim(1), i = gate.weight.dim(1), k = ids.dim(0)
        let v = supports(launch.values, inputs: i, bits: [down.bits]) ? launch.values : 8
        return Self.downKernel(
            [h, ids, w, down.weight, down.scales!, down.biases!],
            template: [("T", h.dtype), ("D", d), ("I", i), ("K", k), ("DB", down.bits), ("DG", down.groupSize), ("V", v)],
            grid: (32 * d, 1, 1), threadGroup: (launch.threads, 1, 1),
            outputShapes: [[d]], outputDTypes: [h.dtype])[0]
        #else
        fatalError("Metal kernels need an Apple GPU")
        #endif
    }

    /// Wide loads need 4- or 8-bit rows (same width for gate and up), whole lane steps per row, and rows that
    /// start on the load's alignment (8 or 16 bytes).
    func supports(_ values: Int, inputs: Int, bits: [Int]) -> Bool {
        guard values > 8 else { return true }
        guard let b = bits.first, bits.allSatisfy({ $0 == b }), b == 4 || b == 8, values * b <= 128 else { return false }
        let bytes = values * b / 8
        return inputs % values == 0 && (inputs * b / 8) % bytes == 0
    }

    #if canImport(Metal)
    static let gateUpKernel = MLXFast.metalKernel(
        name: "inkling_moe_gate_up", inputNames: ["x", "ids", "gw", "gs", "gbias", "uw", "us", "ubias", "caps"],
        outputNames: ["h"], source: FusedMoESource.gateUp)
    static let downKernel = MLXFast.metalKernel(
        name: "inkling_moe_down", inputNames: ["h", "ids", "rw", "dw", "ds", "dbias"],
        outputNames: ["out"], source: FusedMoESource.down)
    #endif

    /// Same maths with stock ops: dequantize only the routed experts' rows in FP32, plain matmuls, and round to x's
    /// dtype at the same points as `gather_qmm` (each projection's output).
    func fallback(_ x: MLXArray, _ ids: MLXArray, _ w: MLXArray) -> MLXArray {
        func rows(_ l: QLinear) -> MLXArray {
            let q = take(l.weight, ids, axis: 0)
            guard let scales = l.scales else { return q.asType(.float32) }
            return dequantized(q, scales: take(scales, ids, axis: 0).asType(.float32),
                               biases: l.biases.map { take($0, ids, axis: 0).asType(.float32) },
                               groupSize: l.groupSize, bits: l.bits)
        }
        func project(_ l: QLinear, _ v: MLXArray) -> MLXArray {  // v [k, in] or [in] → [k, out] in x's dtype
            let input = (v.ndim == 1 ? broadcast(v, to: [ids.dim(0), v.dim(0)]) : v).asType(.float32)
            return matmul(rows(l), input.expandedDimensions(axis: -1)).squeezed(axis: -1).asType(x.dtype)
        }
        let g = project(gate, x).asType(.float32), u = project(up, x).asType(.float32)
        let h = (tanh(g / gateCap) * gateCap * sigmoid(g) * tanh(u / upCap) * upCap).asType(x.dtype)
        let out = project(down, h)
        return (out.asType(.float32) * w.expandedDimensions(axis: -1)).sum(axis: 0).asType(x.dtype)
    }
}
