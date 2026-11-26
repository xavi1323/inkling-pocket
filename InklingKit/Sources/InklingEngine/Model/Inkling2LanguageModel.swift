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
        let caches = model.layers.compactMap { $0.moe.cache }
        profile = caches.isEmpty ? nil : StreamProfile()
        for c in caches { c.profile = profile }
    }

    /// Streaming time and reads, shared by every layer's cache (nil when experts are not streamed).
    let profile: StreamProfile?

    /// `expertSlots`: stream experts through that many slots per layer instead of handing MLX the mapped tensors.
    public static func load(config: InklingModelConfig, weights: WeightSource, expertSlots: Int) throws -> Inkling2LanguageModel {
        try load(config: config, weights: weights, slotPlan: Array(repeating: expertSlots, count: config.text.numHiddenLayers))
    }

    /// `slotPlan`: expert slots for each layer (nil = hand MLX the whole expert tensors).
    public static func load(config: InklingModelConfig, weights: WeightSource, slotPlan: [Int]? = nil) throws -> Inkling2LanguageModel {
        var loader = WeightLoader(source: weights, quant: config.quantization)
        loader.expertSlots = slotPlan
        return try Inkling2LanguageModel(config: config.text, loader: loader)
    }

    public static func load(manifest: ModelManifest, weights: WeightSource) throws -> Inkling2LanguageModel {
        let data = try Data(contentsOf: manifest.folder.appending(path: "config.json"))
        return try load(config: InklingModelConfig.decode(data), weights: weights)
    }

    /// Benchmark only: see `SparseMoE.ignoreRouting`.
    var ignoreRouting: Bool {
        get { model.layers.first?.moe.ignoreRouting ?? false }
        set { for l in model.layers { l.moe.ignoreRouting = newValue } }
    }

    /// One per MoE layer when experts are streamed.
    var expertCaches: [ExpertCache] { model.layers.compactMap { $0.moe.cache } }

    /// Turbo: expert picks replaced by a ready expert, and experts read in after a token (totals).
    private(set) var turboSkipped = 0, turboLoaded = 0, turboPicks = 0

    /// Turbo, between tokens (no GPU work in flight): reads the last token's routing, counts uses, starts reading the
    /// experts it missed, then gives every streamed layer its table for the next token.
    func turboBoundary() {
        let moes = model.layers.map(\.moe).filter { $0.cache != nil }
        let routes = moes.flatMap { $0.turboRoutes.flatMap { [$0.wanted, $0.used] } }
        if !routes.isEmpty { eval(routes) }
        for moe in moes {
            guard let cache = moe.cache else { continue }
            var wanted: [Int] = [], used: [Int] = []
            for r in moe.turboRoutes {
                wanted += r.wanted.asArray(UInt32.self).map(Int.init)
                used += r.used.asArray(UInt32.self).map(Int.init)
            }
            moe.turboRoutes = []
            if !wanted.isEmpty {
                cache.touch(wanted: wanted, used: used)
                turboPicks += wanted.count
                let ran = Set(used)
                let missed = wanted.filter { !ran.contains($0) }
                turboSkipped += missed.count
                if !missed.isEmpty {
                    let distinct = Array(Set(missed))
                    turboLoaded += distinct.count
                    cache.readIn(distinct, keepReady: moe.topK)
                }
            }
            let (slots, ready) = cache.turboTable()
            moe.turboTable = (MLXArray(slots), MLXArray(ready))
        }
    }

    /// Back to exact routing (prompt chunks): drops the tables and pending routes.
    func turboReset() {
        for l in model.layers {
            l.moe.turboTable = nil
            l.moe.turboRoutes = []
        }
    }

    /// Waits for turbo's background reads (tests).
    func finishTurboLoads() { for c in expertCaches { c.waitForLoads() } }

    /// One cache per layer visit (48 for Inkling 2 Mini).
    public func makeCache() -> [LayerVisitCache] {
        config.layerVisits.map { InklingVisitCache(window: config.isSliding($0) ? config.slidingWindowSize : nil) }
    }

    /// Logits [B, L, outputVocabSize] (FP32, softcapped) for token ids [B, L]; advances `cache`.
    public func callAsFunction(_ tokens: MLXArray, cache: [LayerVisitCache]?) -> MLXArray {
        callAsFunction(tokens, cache: cache, fast: false)
    }

    /// `fast`: single-token steps use the fused MoE kernels. `compiled`: and run each layer's pure parts compiled
    /// (fused elementwise kernels: far fewer GPU dispatches per token).
    public func callAsFunction(_ tokens: MLXArray, cache: [LayerVisitCache]?, fast: Bool, compiled: Bool = false) -> MLXArray {
        let h = model(tokens, cache: cache, fast: fast, compiled: compiled)
        // Round through the embedding dtype like the reference, then project in FP32.
        let normed = Ops.rmsNorm(h, weight: model.norm, eps: config.rmsNormEps)
            .asType(model.embedNorm.dtype).asType(.float32)
        var logits = lmHead(normed)
        let cap = config.finalLogitSoftcapping
        if cap > 0 { logits = tanh(logits / cap) * cap }
        return logits[.ellipsis, ..<config.outputVocabSize]
    }
}
