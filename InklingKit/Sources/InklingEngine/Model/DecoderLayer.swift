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
        moe = try SparseMoE(config, layer: layer, path: path + ".mlp", loader: loader)
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
    /// `compiled`: single-token steps run the layer's pure parts as compiled (fused) functions.
    func callAsFunction(_ x: MLXArray, blocks: inout [MLXArray], cache: LayerVisitCache?, messages: MLXArray?,
                        onRoute: ((MLXArray) -> Void)? = nil, fast: Bool = false, compiled: Bool = false) -> MLXArray {
        let states = cache?.conv ?? [nil, nil, nil, nil]
        if compiled, Ops.useFused || Self.splitOnCPU, x.dim(0) * x.dim(1) == 1, let cache, states.allSatisfy({ $0 != nil }) {
            return compiledStep(x, blocks: &blocks, cache: cache, messages: messages, onRoute: onRoute, fast: fast)
        }
        let positions = Attention.positions(offset: cache?.offset ?? 0, length: x.dim(1))
        let p = Attention.Projected(pre(x, blocks: blocks, messages: messages, stateK: states[0], stateV: states[1], positions: positions))
        cache?.conv[0] = p.stateK
        cache?.conv[1] = p.stateV
        let mlpBlocks = isBlockBoundary ? blocks + [x] : blocks
        let post = post(attention.attend(p, cache: cache), gate: p.gate, x: x, blocks: mlpBlocks, messages: messages, state: states[2])
        cache?.conv[2] = post[2]
        blocks = mlpBlocks
        let hidden = moe(post[1], onRoute: onRoute, fast: fast).asType(x.dtype)
        let out = tail(residual: post[0], hidden: hidden, state: states[3])
        cache?.conv[3] = out[1]
        return out[0]
    }

    /// Mixing, message controllers, norms and the attention projections (pure; `Attention.Projected.arrays`).
    func pre(_ x: MLXArray, blocks: [MLXArray], messages: MLXArray?, stateK: MLXArray?, stateV: MLXArray?,
             positions: MLXArray) -> [MLXArray] {
        let mixed = attentionInput.map { $0(blocks, x) } ?? x.expandedDimensions(axis: -2)
        if isSliding && messages == nil {
            let h = Ops.rmsNorm(Ops.pick(mixed, 0), weight: inputNorms[0], eps: eps).asType(x.dtype)
            return attention.project(h, keyInput: h, valueInput: h, stateK: stateK, stateV: stateV, positions: positions).arrays
        }
        let count = mixed.dim(-2)
        var parts = (0..<3).map { Ops.pick(mixed, min($0, count - 1)) }
        if let messages, let controllers {
            parts = (0..<3).map { controllers[$0](parts[$0], Ops.pick(messages, $0)) }
        }
        parts = (0..<3).map { Ops.rmsNorm(parts[$0], weight: inputNorms[isSliding ? 0 : $0], eps: eps).asType(x.dtype) }
        return attention.project(parts[0], keyInput: parts[1], valueInput: parts[2], stateK: stateK, stateV: stateV,
                                 positions: positions).arrays
    }

    /// Attention output → [residual (FP32), MoE input, new attention-conv state] (pure). `blocks` already holds `x`
    /// when this layer starts a block.
    func post(_ attended: MLXArray, gate: MLXArray, x: MLXArray, blocks: [MLXArray], messages: MLXArray?, state: MLXArray?) -> [MLXArray] {
        let attn = attention.finish(attended, gate: gate)
        let (conv, next) = attnConv(attn, state: state)
        var residual = conv.asType(.float32)
        if !isBlockBoundary { residual = residual + x.asType(.float32) }
        var hidden = Ops.pick(mlpInput(blocks, residual), 0)
        if let messages, let controllers { hidden = controllers[3](hidden, Ops.pick(messages, 3)) }
        return [residual, Ops.rmsNorm(hidden, weight: postAttentionNorm, eps: eps).asType(x.dtype), next]
    }

    /// MoE output → [layer output, new MLP-conv state] (pure).
    func tail(residual: MLXArray, hidden: MLXArray, state: MLXArray?) -> [MLXArray] {
        let (conv, next) = mlpConv(hidden, state: state)
        return [(residual + conv.asType(.float32)).asType(hidden.dtype), next]
    }

    /// Single-token steps run through compiled `pre`/`post`/`tail`.
    /// Tests: take the compiled-step path on the CPU too (MLX cannot JIT CPU kernels there: run with compile disabled).
    nonisolated(unsafe) static var splitOnCPU = false
    private(set) var compiledSteps = 0

    // Inputs: pre [x, stateK, stateV, positions, messages?, blocks…]; post [attended, gate, x, state, messages?, blocks…];
    // tail [residual, hidden, state]. One compiled function per messages/no-messages (block counts retrace by shape).
    private lazy var compiledPre: [Bool: ([MLXArray]) -> [MLXArray]] = [false: compilePre(false), true: compilePre(true)]
    private lazy var compiledPost: [Bool: ([MLXArray]) -> [MLXArray]] = [false: compilePost(false), true: compilePost(true)]
    private lazy var compiledTail: ([MLXArray]) -> [MLXArray] = compile(inputs: [], outputs: []) { [unowned self] a in
        tail(residual: a[0], hidden: a[1], state: a[2])
    }

    private func compilePre(_ hasMessages: Bool) -> ([MLXArray]) -> [MLXArray] {
        compile(inputs: [], outputs: []) { [unowned self] a in
            let blocks = Array(a[(hasMessages ? 5 : 4)...])
            return pre(a[0], blocks: blocks, messages: hasMessages ? a[4] : nil, stateK: a[1], stateV: a[2], positions: a[3])
        }
    }

    private func compilePost(_ hasMessages: Bool) -> ([MLXArray]) -> [MLXArray] {
        compile(inputs: [], outputs: []) { [unowned self] a in
            let blocks = Array(a[(hasMessages ? 5 : 4)...])
            return post(a[0], gate: a[1], x: a[2], blocks: blocks, messages: hasMessages ? a[4] : nil, state: a[3])
        }
    }

    private func compiledStep(_ x: MLXArray, blocks: inout [MLXArray], cache: LayerVisitCache, messages: MLXArray?,
                              onRoute: ((MLXArray) -> Void)?, fast: Bool) -> MLXArray {
        compiledSteps += 1
        let states = cache.conv.map { $0! }
        let hasMessages = messages != nil
        let message = hasMessages ? [messages!] : []
        let positions = Attention.positions(offset: cache.offset, length: 1)
        let p = Attention.Projected(compiledPre[hasMessages]!([x, states[0], states[1], positions] + message + blocks))
        cache.conv[0] = p.stateK
        cache.conv[1] = p.stateV
        let mlpBlocks = isBlockBoundary ? blocks + [x] : blocks
        let post = compiledPost[hasMessages]!([attention.attend(p, cache: cache, inPlace: true), p.gate, x, states[2]] + message + mlpBlocks)
        cache.conv[2] = post[2]
        blocks = mlpBlocks
        let hidden = moe(post[1], onRoute: onRoute, fast: fast).asType(x.dtype)
        let out = compiledTail([post[0], hidden, states[3]])
        cache.conv[3] = out[1]
        return out[0]
    }
}
