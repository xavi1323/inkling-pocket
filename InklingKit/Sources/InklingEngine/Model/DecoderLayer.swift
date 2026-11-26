// Ported from mlx-vlm inkling2/language.py DecoderLayer (MIT) at d998a62.
import MLX
import InklingCore

final class DecoderLayer {
    let isSliding: Bool
    let isBlockBoundary: Bool
    let eps: Float
    let attentionInput: AttentionResidual?
    let mlpInput: AttentionResidual
    let attention: Attention
    let moe: SparseMoE
    /// Sliding: [input]; global: [q, k, v].
    let inputNorms: [MLXArray]
    let postAttentionNorm: MLXArray
    let attnConv, mlpConv: ShortConvolution
    /// attn, k, v, mlp — only on looped core layers.
    let controllers: [MessageController]?

    init(_ config: InklingTextConfig, layer: Int, loader: WeightLoader) throws {
        let path = "model.layers.\(layer)"
        isSliding = config.isSliding(layer)
        isBlockBoundary = layer % config.layersPerAttnResBlock == 0
        eps = config.rmsNormEps
        let normalize = config.attnResNormalizeValues
        attentionInput = layer == 0 ? nil
            : AttentionResidual(query: try loader.tensor(path + ".attention_input.query"), normalizeValues: normalize)
        mlpInput = AttentionResidual(query: try loader.tensor(path + ".mlp_input.query"), normalizeValues: normalize)
        attention = try Attention(config, layer: layer, path: path + ".self_attn", loader: loader)
        moe = try SparseMoE(config, path: path + ".mlp", loader: loader)
        let norms = isSliding ? ["input_layernorm"] : ["q_input_layernorm", "k_input_layernorm", "v_input_layernorm"]
        inputNorms = try norms.map { try loader.tensor("\(path).\($0).weight") }
        postAttentionNorm = try loader.tensor(path + ".post_attention_layernorm.weight")
        let k = config.sconvKernelSize
        attnConv = ShortConvolution(weight: try loader.tensor(path + ".attn_sconv.conv.weight"), kernelSize: k, slot: 2)
        mlpConv = ShortConvolution(weight: try loader.tensor(path + ".mlp_sconv.conv.weight"), kernelSize: k, slot: 3)
        if let loop = config.loopConfig, loop.coreStart <= layer, layer < loop.coreEnd {
            controllers = try ["attn", "k", "v", "mlp"].map { name in
                let p = "\(path).controllers.\(name)"
                return MessageController(down: try loader.tensor(p + ".down_weight"), forget: try loader.tensor(p + ".forget_weight"),
                                         gain: try loader.tensor(p + ".gain_weight"), eps: config.rmsNormEps)
            }
        } else {
            controllers = nil
        }
    }

    /// `blocks` collects block-boundary inputs for the attention residuals and is appended to in place.
    func callAsFunction(_ x: MLXArray, blocks: inout [MLXArray], cache: LayerVisitCache?, messages: MLXArray?,
                        onRoute: ((MLXArray) -> Void)? = nil) -> MLXArray {
        let mixed = attentionInput.map { $0(blocks, x) } ?? x.expandedDimensions(axis: -2)
        let attn: MLXArray
        if isSliding && messages == nil {
            let h = Ops.rmsNorm(Ops.pick(mixed, 0), weight: inputNorms[0], eps: eps).asType(x.dtype)
            attn = attention(h, keyInput: h, valueInput: h, cache: cache)
        } else {
            let count = mixed.dim(-2)
            var parts = (0..<3).map { Ops.pick(mixed, min($0, count - 1)) }
            if let messages, let controllers {
                parts = (0..<3).map { controllers[$0](parts[$0], Ops.pick(messages, $0)) }
            }
            parts = (0..<3).map { Ops.rmsNorm(parts[$0], weight: inputNorms[isSliding ? 0 : $0], eps: eps).asType(x.dtype) }
            attn = attention(parts[0], keyInput: parts[1], valueInput: parts[2], cache: cache)
        }
        var residual = attnConv(attn, cache: cache).asType(.float32)
        if isBlockBoundary {
            blocks.append(x)
        } else {
            residual = residual + x.asType(.float32)
        }
        var hidden = Ops.pick(mlpInput(blocks, residual), 0)
        if let messages, let controllers { hidden = controllers[3](hidden, Ops.pick(messages, 3)) }
        hidden = moe(Ops.rmsNorm(hidden, weight: postAttentionNorm, eps: eps).asType(x.dtype), onRoute: onRoute)
        return (residual + mlpConv(hidden, cache: cache).asType(.float32)).asType(x.dtype)
    }
}
