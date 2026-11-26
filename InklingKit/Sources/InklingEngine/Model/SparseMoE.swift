// Ported from mlx-vlm inkling2/language.py SparseMoE + switch_layers.SwitchGLU (MIT) at d998a62.
import MLX
import InklingCore

/// Sigmoid-routed top-k experts with a softcapped SwiGLU; route weights are renormalised and scaled.
final class SparseMoE {
    let gateWeight: MLXArray  // [E, D]
    let scoreBias: MLXArray   // [E]
    let gateProj, upProj, downProj: QSwitchLinear
    let topK: Int
    let routeScale, gateCap, upCap: Float
    /// Single-token kernels over the same weights (fast path).
    lazy var fused = FusedMoEDecode(gate: gateProj.linear, up: upProj.linear, down: downProj.linear, gateCap: gateCap, upCap: upCap)

    init(_ config: InklingTextConfig, path: String, loader: WeightLoader) throws {
        gateWeight = try loader.tensor(path + ".gate_weight")
        scoreBias = try loader.tensor(path + ".e_score_correction_bias")
        gateProj = QSwitchLinear(linear: try loader.linear(path + ".switch_mlp.gate_proj"))
        upProj = QSwitchLinear(linear: try loader.linear(path + ".switch_mlp.up_proj"))
        downProj = QSwitchLinear(linear: try loader.linear(path + ".switch_mlp.down_proj"))
        topK = config.expertsPerToken
        routeScale = config.routeScale
        gateCap = config.mlpGateSoftcap
        upCap = config.mlpUpSoftcap
    }

    /// Expert ids [B,L,k] and their weights [B,L,k] (FP32).
    func route(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let scores = sigmoid(matmul(x.asType(.float32), gateWeight.asType(.float32).T))
        let indices = argPartition(-(scores + scoreBias), kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, indices, axis: -1)
        weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20) * routeScale
        return (indices, weights)
    }

    func activation(up: MLXArray, gate: MLXArray) -> MLXArray {
        let g = gate.asType(.float32), u = up.asType(.float32)
        return (tanh(g / gateCap) * gateCap * sigmoid(g) * tanh(u / upCap) * upCap).asType(gate.dtype)
    }

    func callAsFunction(_ x: MLXArray, onRoute: ((MLXArray) -> Void)? = nil, fast: Bool = false) -> MLXArray {
        let (indices, weights) = route(x)
        onRoute?(indices)
        let (b, length, d) = (x.dim(0), x.dim(1), x.dim(2))
        if fast, b * length == 1 {
            return fused(x: x, expertIDs: indices, weights: weights).reshaped([b, length, d])
        }
        let routes = indices.dim(-1)
        let out: MLXArray
        if indices.size >= 64 {
            // Sort routes by expert so each expert's rows are contiguous (matches `_gather_sort`).
            let flat = indices.flattened()
            let order = argSort(flat, axis: -1)
            let inverse = argSort(order, axis: -1)
            let rows = take(x.reshaped([b * length, 1, d]), order / Int32(routes), axis: 0)
            let idx = take(flat, order, axis: 0)
            let up = upProj(rows, idx, sorted: true)
            let gate = gateProj(rows, idx, sorted: true)
            let down = downProj(activation(up: up, gate: gate), idx, sorted: true)
            out = take(down, inverse, axis: 0).reshaped([b, length, routes, d])
        } else {
            let rows = x.expandedDimensions(axes: [-2, -3])
            let up = upProj(rows, indices, sorted: false)
            let gate = gateProj(rows, indices, sorted: false)
            out = downProj(activation(up: up, gate: gate), indices, sorted: false).squeezed(axis: -2)
        }
        return (out.asType(.float32) * weights.expandedDimensions(axis: -1)).sum(axis: -2).asType(out.dtype)
    }
}
