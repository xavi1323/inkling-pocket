import Foundation
import MLX

/// A model that maps token ids [B, L] to logits [B, L, V], advancing per-visit caches.
public protocol CausalLM: AnyObject {
    func callAsFunction(_ tokens: MLXArray, cache: [LayerVisitCache]?) -> MLXArray
    func makeCache() -> [LayerVisitCache]
}

public struct GenerationStats: Sendable, Equatable {
    /// Length of the last prompt.
    public var promptTokens = 0
    /// Prompt tokens actually computed (the rest came from the reused prefix).
    public var prefilledTokens = 0
    /// Tokens produced, not counting the end-of-sampling token.
    public var generatedTokens = 0
    public var prefillTokensPerSecond: Double = 0
    public var decodeTokensPerSecond: Double = 0
    public var timeToFirstToken: Duration = .zero
    public init() {}
}

public enum GenerationError: Error, Equatable, CustomStringConvertible {
    case contextFull(prompt: Int, limit: Int)
    public var description: String {
        switch self {
        case .contextFull(let prompt, let limit): "This chat is too long for the model (\(prompt) of \(limit) tokens). Start a new chat."
        }
    }
}

/// Token-level generation: chunked prefill, prompt-prefix reuse across turns, cancellation between tokens.
/// Not thread-safe — owned by one actor.
public final class TokenGenerator {
    public let model: CausalLM
    public var eos: Set<Int32>
    public var prefillChunk = 256
    public var contextLimit = 16_384
    public private(set) var stats = GenerationStats()
    /// Called after the prompt and after every generated token, once its logits are evaluated. The argument is how
    /// many newer forward passes are still running (pipelined decode keeps one in flight).
    public var afterStep: ((Int) -> Void)?
    /// Queue token n+1 on the GPU before reading token n back (fast path).
    public var pipelined = false

    private var cache: [LayerVisitCache]
    /// The last prompt, the cache state right after it, and its final logits.
    private var mark: (tokens: [Int32], snapshot: [VisitSnapshot], logits: MLXArray)?
    /// The cache state after the last prompt's stable prefix (tools + instructions), for new chats.
    private var prefixMark: (tokens: [Int32], snapshot: [VisitSnapshot])?

    public init(model: CausalLM, eos: Set<Int32>) {
        self.model = model
        self.eos = eos
        cache = model.makeCache()
    }

    /// Feeds the prompt (only the part after the previous prompt, when that is a prefix) and returns logits [V].
    /// `stablePrefix`: how many leading tokens are the same for every chat (tools + instructions); the cache state
    /// after them is kept so a new chat starts from there.
    @discardableResult
    public func prefill(_ prompt: [Int32], stablePrefix: Int = 0) -> MLXArray {
        prefill(prompt, stablePrefix: stablePrefix, shouldStop: { false })!
    }

    /// As `prefill(_:)`, but checks `shouldStop` between chunks; returns nil (and forgets the prompt) when stopped.
    func prefill(_ prompt: [Int32], stablePrefix: Int = 0, shouldStop: () -> Bool) -> MLXArray? {
        precondition(!prompt.isEmpty, "empty prompt")
        let start = Date()
        var from = 0
        var logits: MLXArray?
        if let mark, prompt.count >= mark.tokens.count, prompt.starts(with: mark.tokens) {
            for (c, s) in zip(cache, mark.snapshot) { (c as? InklingVisitCache)?.restore(s) }
            from = mark.tokens.count
            logits = mark.logits
        } else if let prefixMark, prompt.count > prefixMark.tokens.count, prompt.starts(with: prefixMark.tokens) {
            // Same caches the prefix was saved on (a fresh cache drops `prefixMark`), so its K/V are still there.
            for (c, s) in zip(cache, prefixMark.snapshot) { (c as? InklingVisitCache)?.restore(s) }
            from = prefixMark.tokens.count
            mark = nil
        } else {
            cache = model.makeCache()
            mark = nil
            prefixMark = nil
        }
        let keepPrefix = stablePrefix > from && stablePrefix < prompt.count
        var i = from
        while i < prompt.count {
            var end = min(i + prefillChunk, prompt.count)
            if keepPrefix, i < stablePrefix { end = min(end, stablePrefix) }
            let out = model(MLXArray(Array(prompt[i..<end])).reshaped([1, -1]), cache: cache)
            logits = out[0, -1]
            eval(logits!)
            i = end
            if keepPrefix, i == stablePrefix {
                let snapshots = cache.compactMap { ($0 as? InklingVisitCache)?.snapshot() }
                prefixMark = snapshots.count == cache.count ? (Array(prompt[..<stablePrefix]), snapshots) : nil
            }
            if i < prompt.count, shouldStop() {
                mark = nil
                prefixMark = nil
                cache = model.makeCache()
                return nil
            }
        }
        let snapshots = cache.compactMap { ($0 as? InklingVisitCache)?.snapshot() }
        mark = (prompt, snapshots.count == cache.count ? snapshots : [], logits!)
        if snapshots.count != cache.count { mark = nil }
        afterStep?(0)
        stats.promptTokens = prompt.count
        stats.prefilledTokens = prompt.count - from
        let seconds = Date().timeIntervalSince(start)
        stats.prefillTokensPerSecond = seconds > 0 ? Double(stats.prefilledTokens) / seconds : 0
        return logits!
    }

    /// Tokens of the kept stable prefix, if any.
    var prefixTokens: [Int32]? { prefixMark?.tokens }

    /// Saves the kept stable prefix's cache state (nothing to save → no file written).
    func savePrefix(to url: URL, identity: String) throws {
        guard let prefixMark else { return }
        var arrays: [String: MLXArray] = [:]
        var meta = ["identity": identity, "tokens": prefixMark.tokens.map(String.init).joined(separator: ","),
                    "visits": "\(cache.count)"]
        for (i, (c, s)) in zip(cache, prefixMark.snapshot).enumerated() {
            (c as? InklingVisitCache)?.export(s, key: "visit\(i)", arrays: &arrays, meta: &meta)
        }
        let partial = url.deletingPathExtension().appendingPathExtension("partial.safetensors")  // MLX needs the extension
        try save(arrays: arrays, metadata: meta, url: partial)
        _ = try? FileManager.default.removeItem(at: url)
        try FileManager.default.moveItem(at: partial, to: url)
    }

    /// Loads a saved stable prefix when it was saved for `identity` and starts `prompt`; returns whether it did.
    @discardableResult
    func loadPrefix(from url: URL, identity: String, ifPrefixOf prompt: [Int32]) throws -> Bool {
        guard FileManager.default.fileExists(atPath: url.path) else { return false }
        let (arrays, meta) = try loadArraysAndMetadata(url: url)
        guard meta["identity"] == identity, let list = meta["tokens"], meta["visits"] == "\(cache.count)" else { return false }
        let tokens = list.split(separator: ",").compactMap { Int32($0) }
        guard !tokens.isEmpty, prompt.count > tokens.count, prompt.starts(with: tokens) else { return false }
        let fresh = model.makeCache()
        for (i, c) in fresh.enumerated() {
            guard let c = c as? InklingVisitCache, c.load(key: "visit\(i)", arrays: arrays, meta: meta) else { return false }
        }
        eval(Array(arrays.values))
        cache = fresh
        mark = nil
        let snapshots = cache.compactMap { ($0 as? InklingVisitCache)?.snapshot() }
        prefixMark = (tokens, snapshots)
        return true
    }

    /// Generates up to `maxTokens` (stopping at EOS, the context limit or when `shouldStop()` is true),
    /// calling `onToken` for each token. Returns the tokens (without EOS).
    @discardableResult
    public func generate(prompt: [Int32], stablePrefix: Int = 0, sampler: Sampler, maxTokens: Int, shouldStop: () -> Bool,
                         onToken: (Int32) -> Void) -> [Int32] {
        if pipelined {
            return generatePipelined(prompt: prompt, stablePrefix: stablePrefix, sampler: sampler, maxTokens: maxTokens,
                                     shouldStop: shouldStop, onToken: onToken)
        }
        let started = ContinuousClock.now
        guard var logits = prefill(prompt, stablePrefix: stablePrefix, shouldStop: shouldStop) else { return [] }
        var produced: [Int32] = []
        var history = prompt.suffix(sampler.penaltyWindow).map { $0 }
        var decodeStart: ContinuousClock.Instant?
        stats.generatedTokens = 0
        stats.decodeTokensPerSecond = 0
        while produced.count < maxTokens, prompt.count + produced.count < contextLimit, !shouldStop() {
            let token = sampler.sample(logits, history: history).item(Int32.self)
            if !produced.isEmpty { afterStep?(0) }
            if produced.isEmpty {
                stats.timeToFirstToken = ContinuousClock.now - started
                decodeStart = .now
            }
            if eos.contains(token) { break }
            produced.append(token)
            history.append(token)
            stats.generatedTokens = produced.count
            onToken(token)
            if produced.count == maxTokens || shouldStop() { break }
            logits = model(MLXArray([token]).reshaped([1, 1]), cache: cache)[0, -1]
        }
        if let decodeStart, produced.count > 1 {
            let s = (ContinuousClock.now - decodeStart) / .seconds(1)
            stats.decodeTokensPerSecond = s > 0 ? Double(produced.count - 1) / s : 0
        }
        return produced
    }

    /// Same tokens as `generate`; the next step's graph is built and queued while the GPU still runs the current one.
    /// Stopping at EOS leaves one extra token in the cache, which the next prefill discards (it restores the prompt snapshot).
    func generatePipelined(prompt: [Int32], stablePrefix: Int = 0, sampler: Sampler, maxTokens: Int, shouldStop: () -> Bool,
                           onToken: (Int32) -> Void) -> [Int32] {
        let started = ContinuousClock.now
        let gpu = GPUSampler(sampler)
        guard let logits = prefill(prompt, stablePrefix: stablePrefix, shouldStop: shouldStop) else { return [] }
        var history: MLXArray? = gpu.usesHistory ? MLXArray(Array(prompt.suffix(sampler.penaltyWindow))) : nil
        var produced: [Int32] = []
        var decodeStart: ContinuousClock.Instant?
        stats.generatedTokens = 0
        stats.decodeTokensPerSecond = 0
        guard maxTokens > 0, prompt.count < contextLimit, !shouldStop() else { return [] }
        var y = gpu.sample(logits, history: history)
        asyncEval([y])
        while true {
            var next: MLXArray?
            if produced.count + 1 < maxTokens, prompt.count + produced.count + 1 < contextLimit {
                history = gpu.append(history, y)
                let l = model(y.reshaped([1, 1]), cache: cache)[0, -1]
                next = gpu.sample(l, history: history)
                asyncEval([next!])
            }
            let token = y.item(Int32.self)
            if produced.isEmpty {
                stats.timeToFirstToken = ContinuousClock.now - started
                decodeStart = .now
            } else {
                afterStep?(next == nil ? 0 : 1)
            }
            if eos.contains(token) { break }
            produced.append(token)
            stats.generatedTokens = produced.count
            onToken(token)
            guard let next, !shouldStop() else { break }
            y = next
        }
        if let decodeStart, produced.count > 1 {
            let s = (ContinuousClock.now - decodeStart) / .seconds(1)
            stats.decodeTokensPerSecond = s > 0 ? Double(produced.count - 1) / s : 0
        }
        return produced
    }
}
