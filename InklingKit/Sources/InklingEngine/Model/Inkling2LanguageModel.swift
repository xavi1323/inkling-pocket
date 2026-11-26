// Ported from mlx-vlm inkling2/language.py LanguageModel (MIT) at d998a62: text-only, single device.
import Foundation
import MLX
import InklingCore

/// Reference (portable MLX ops) implementation of the inkling2_loop text model.
public final class Inkling2LanguageModel: CausalLM {
    public let config: InklingTextConfig
    let model: TextModel
    let lmHead: QLinear
    /// When set, every layer visit records its routed experts (slow: reads ids back to the CPU).
    public var routing: RoutingStats? {
        get { model.routing }
        set { model.routing = newValue }
    }
    /// When set, routed expert ids are kept (not read back) for residency bookkeeping after each step.
    public var collector: RouteCollector? {
        get { model.collector }
        set { model.collector = newValue }
    }

    init(config: InklingTextConfig, loader: WeightLoader) throws {
        guard config.numSharedExperts == 0 else { throw ModelLoadError.unsupported("shared experts") }
        if let loop = config.loopConfig, loop.isVarloop { throw ModelLoadError.unsupported("varloop") }
        self.config = config
        model = try TextModel(config, loader: loader)
        lmHead = try loader.linear("lm_head")
    }

    public static func load(config: InklingModelConfig, weights: WeightSource) throws -> Inkling2LanguageModel {
        try Inkling2LanguageModel(config: config.text, loader: WeightLoader(source: weights, quant: config.quantization))
    }

    public static func load(manifest: ModelManifest, weights: WeightSource) throws -> Inkling2LanguageModel {
        let data = try Data(contentsOf: manifest.folder.appending(path: "config.json"))
        return try load(config: InklingModelConfig.decode(data), weights: weights)
    }

    /// One cache per layer visit (48 for Inkling 2 Mini).
    public func makeCache() -> [LayerVisitCache] {
        config.layerVisits.map { InklingVisitCache(window: config.isSliding($0) ? config.slidingWindowSize : nil) }
    }

    /// Logits [B, L, outputVocabSize] (FP32, softcapped) for token ids [B, L]; advances `cache`.
    public func callAsFunction(_ tokens: MLXArray, cache: [LayerVisitCache]?) -> MLXArray {
        let h = model(tokens, cache: cache)
        // Round through the embedding dtype like the reference, then project in FP32.
        let normed = Ops.rmsNorm(h, weight: model.norm, eps: config.rmsNormEps)
            .asType(model.embedNorm.dtype).asType(.float32)
        var logits = lmHead(normed)
        let cap = config.finalLogitSoftcapping
        if cap > 0 { logits = tanh(logits / cap) * cap }
        return logits[.ellipsis, ..<config.outputVocabSize]
    }
}
