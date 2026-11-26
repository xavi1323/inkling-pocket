// Ported from mlx-vlm inkling2/language.py TextModel (MIT) at d998a62.
import Foundation
import MLX
import InklingCore

/// Embedding → 24 layers visited 48 times (core looped with message passing between passes) → final mix.
final class TextModel {
    let config: InklingTextConfig
    let embedTokens: QEmbedding
    let embedNorm: MLXArray
    let layers: [DecoderLayer]
    let visits: [Int]
    let messageQueries: MLXArray?  // [coreLayers, 4, D]
    let finalInput: AttentionResidual
    let norm: MLXArray
    var routing: RoutingStats?
    var collector: RouteCollector?

    init(_ config: InklingTextConfig, loader: WeightLoader) throws {
        self.config = config
        embedTokens = QEmbedding(table: try loader.linear("model.embed_tokens"))
        embedNorm = try loader.tensor("model.embed_norm.weight")
        layers = try (0..<config.numHiddenLayers).map { try DecoderLayer(config, layer: $0, loader: loader) }
        visits = config.layerVisits
        messageQueries = config.loopConfig == nil ? nil : try loader.tensor("model.message_queries")
        finalInput = AttentionResidual(query: try loader.tensor("model.final_input.query"), normalizeValues: config.attnResNormalizeValues)
        norm = try loader.tensor("model.norm.weight")
    }

    func embed(_ ids: MLXArray) -> MLXArray {
        Ops.rmsNorm(embedTokens(ids), weight: embedNorm, eps: config.rmsNormEps)
    }

    /// Hidden state before the final norm, [B, L, D] FP32.
    func callAsFunction(_ inputs: MLXArray, cache: [LayerVisitCache]?, fast: Bool = false) -> MLXArray {
        var x = embed(inputs)
        var blocks: [MLXArray] = []
        var bank: MLXArray?
        var prelude = x
        var preludeBlocks: [MLXArray] = []
        let d = Float(x.dim(-1))
        for (visit, index) in visits.enumerated() {
            if let loop = config.loopConfig, index == loop.coreStart {
                if visit == index {
                    prelude = x
                    preludeBlocks = blocks
                } else {
                    // The first core block is the prelude partial; later passes read completed core blocks + last partial.
                    let states = stacked(blocks[(preludeBlocks.count + 1)...].map { $0 } + [x], axis: -2).asType(.float32)
                    bank = Ops.rmsNorm(states, weight: nil, eps: 1e-6)
                    x = prelude
                    blocks = preludeBlocks
                }
            }
            var messages: MLXArray?
            if let bank, let loop = config.loopConfig, let queries = messageQueries, index < loop.coreEnd {
                let scores = matmul(queries[index - loop.coreStart], bank.swappedAxes(-1, -2)) * (1 / d.squareRoot())
                messages = matmul(softmax(scores, axis: -1), bank)
            }
            var onRoute: ((MLXArray) -> Void)?
            if routing != nil || collector != nil {
                onRoute = { [routing, collector] experts in
                    routing?.record(visit: visit, experts: experts)
                    collector?.add(visit: visit, experts: experts)
                }
            }
            layers[index].moe.next = visit + 1 < visits.count ? layers[visits[visit + 1]].moe : nil
            x = layers[index](x, blocks: &blocks, cache: cache?[visit], messages: messages, onRoute: onRoute, fast: fast)
        }
        collector?.endPass()
        return Ops.pick(finalInput(blocks, x), 0)
    }
}
