import Foundation
import MLX

/// A model that maps token ids [B, L] to logits [B, L, V], advancing per-visit caches.
public protocol CausalLM: AnyObject {
    func callAsFunction(_ tokens: MLXArray, cache: [LayerVisitCache]?) -> MLXArray
    func makeCache() -> [LayerVisitCache]
    /// Logits [V] of the last position only (a prompt chunk needs no others).
    func lastLogits(_ tokens: MLXArray, cache: [LayerVisitCache]?) -> MLXArray
}

public extension CausalLM {
    func lastLogits(_ tokens: MLXArray, cache: [LayerVisitCache]?) -> MLXArray { self(tokens, cache: cache)[0, -1] }
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
    /// Speculative decoding: tokens guessed ahead, how many the exact check kept, and how many checks ran.
    public var draftedTokens = 0
    public var acceptedTokens = 0
    public var checks = 0
    /// Speculative decoding: tokens made by a plain exact step because guessing ahead cost more than it saved.
    public var plainSteps = 0
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

/// Whether generation is inside a span (a tool call's JSON: from `open` to `close`), to run it exact.
final class ExactSpan {
    let open, close: Int32
    private(set) var inside = false
    init(open: Int32, close: Int32) {
        self.open = open
        self.close = close
    }
    func observe(_ token: Int32) {
        if token == open { inside = true } else if token == close { inside = false }
    }
}

/// Closes a thought that ran past `cap` tokens: after `open` (the thinking marker) counts tokens until `close`
/// (end of message), and once `cap` is reached asks for `close` to be forced.
final class ThinkingBudget {
    let cap: Int
    let open, close: Int32
    private var thinking = false
    private var count = 0

    init(cap: Int, open: Int32, close: Int32) {
        self.cap = cap
        self.open = open
        self.close = close
    }

    func observe(_ token: Int32) {
        if token == open { thinking = true; count = 0 } else if token == close { thinking = false } else if thinking { count += 1 }
    }

    var forced: Int32? { thinking && count >= cap ? close : nil }
}

/// A draft that guesses `count` tokens after `token` in one go (caches are restored by the caller).
public protocol ChainedDraft {
    func propose(after token: Int32, count: Int, cache: [LayerVisitCache]) -> [Int32]
}

/// Token-level generation: chunked prefill, prompt-prefix reuse across turns, cancellation between tokens.
/// Not thread-safe — owned by one actor.
public final class TokenGenerator {
    public let model: CausalLM
    public var eos: Set<Int32>
    public var prefillChunk = 256
    /// Chunk sizes to choose between for long prompts, by measured speed (empty: always `prefillChunk`).
    public var prefillChunkChoices: [Int] = []
    /// Seconds per prefilled token for each tried chunk size (recency-weighted).
    private var prefillChunkCost: [Int: Double] = [:]
    /// How many prompts used each chunk size.
    var prefillChunkTries: [Int: Int] = [:]
    /// Long prompts seen so far (every 8th tries a size other than the fastest).
    private var longPrompts = 0
    /// The fastest measured chunk size, or `prefillChunk` before any is measured.
    public var chosenPrefillChunk: Int {
        prefillChunkChoices.filter { prefillChunkCost[$0] != nil }.min { prefillChunkCost[$0]! < prefillChunkCost[$1]! }
            ?? prefillChunk
    }
    public var contextLimit = 16_384
    public private(set) var stats = GenerationStats()
    /// Called after the prompt and after every generated token, once its logits are evaluated. The argument is how
    /// many newer forward passes are still running (pipelined decode keeps one in flight).
    public var afterStep: ((Int) -> Void)?
    /// Queue token n+1 on the GPU before reading token n back (fast path).
    public var pipelined = false
    /// Speculative decoding: a cheaper model over the same caches guesses up to `maxDraft` tokens ahead, then `model`
    /// checks them all in one multi-token pass and keeps only the tokens it would have sampled itself. The output is
    /// `model`'s (exact), whatever the draft guesses. Text repeated from the context is guessed by lookup instead.
    public var draft: CausalLM?
    /// Asked before each token is picked; a non-nil token is used instead of sampling (the thinking cap closing a
    /// thought). It sees the tokens handed out so far through `onToken`.
    public var forcedToken: (() -> Int32?)?
    public var maxDraft = 8
    /// How far back prompt lookup searches for the last tokens.
    static let lookupWindow = 4096

    private var cache: [LayerVisitCache]
    /// The last prompt, the cache state right after it, and its final logits.
    private var mark: (tokens: [Int32], snapshot: [VisitSnapshot], logits: MLXArray)?
    /// The cache state after the last prompt's stable prefix (tools + instructions), for new chats.
    private var prefixMark: (tokens: [Int32], snapshot: [VisitSnapshot])?
    /// The cache state after what the model wrote (the whole reply, and the last `boundaryTokens` in it), with the
    /// logits after it when known. An agent's next step resends the reply, so its thinking and tool call are not
    /// computed again. Valid only for the next prompt (it overwrites what comes after the restored point).
    private var resumeMarks: [(tokens: [Int32], snapshot: [VisitSnapshot], logits: MLXArray?)] = []
    /// Tokens after which the cache state is kept for the next prompt (end of message: the end of a thought).
    public var boundaryTokens: Set<Int32> = []
    /// An exact single-token model for plain steps between speculative checks (`model` may be turbo for one token).
    public var exactStep: CausalLM?

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
        let resume = resumeMarks.filter { m in
            prompt.starts(with: m.tokens) && (prompt.count > m.tokens.count || m.logits != nil)
        }.max { $0.tokens.count < $1.tokens.count }
        resumeMarks = []
        if let resume, resume.snapshot.count == cache.count {
            for (c, s) in zip(cache, resume.snapshot) { (c as? InklingVisitCache)?.restore(s) }
            from = resume.tokens.count
            logits = resume.logits
        } else if let mark, prompt.count >= mark.tokens.count, prompt.starts(with: mark.tokens) {
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
        let tuned = pickPrefillChunk(remaining: prompt.count - from)
        let chunk = tuned ?? prefillChunk
        let loopStart = Date()
        var i = from
        while i < prompt.count {
            var end = min(i + chunk, prompt.count)
            if keepPrefix, i < stablePrefix { end = min(end, stablePrefix) }
            logits = model.lastLogits(MLXArray(Array(prompt[i..<end])).reshaped([1, -1]), cache: cache)
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
        if let tuned { recordPrefillChunk(tuned, seconds: Date().timeIntervalSince(loopStart), tokens: prompt.count - from) }
        afterStep?(0)
        stats.promptTokens = prompt.count
        stats.prefilledTokens = prompt.count - from
        let seconds = Date().timeIntervalSince(start)
        stats.prefillTokensPerSecond = seconds > 0 ? Double(stats.prefilledTokens) / seconds : 0
        return logits!
    }

    /// The chunk size for a prompt with `remaining` tokens to feed, or nil for `prefillChunk` (no choices, or short).
    /// Each choice is tried twice, then the fastest is used, with a different one every 8th long prompt.
    private func pickPrefillChunk(remaining: Int) -> Int? {
        guard let smallest = prefillChunkChoices.min(), remaining > smallest else { return nil }
        longPrompts += 1
        if let untried = prefillChunkChoices.first(where: { prefillChunkTries[$0, default: 0] < 2 }) { return untried }
        let fastest = chosenPrefillChunk
        let others = prefillChunkChoices.filter { $0 != fastest }
        if longPrompts % 8 == 0, !others.isEmpty { return others[(longPrompts / 8) % others.count] }
        return fastest
    }

    /// Folds one finished prefill's speed into its chunk size's average (newer counts 30%).
    private func recordPrefillChunk(_ chunk: Int, seconds: Double, tokens: Int) {
        guard tokens > 0 else { return }
        let cost = seconds / Double(tokens)
        prefillChunkCost[chunk] = prefillChunkCost[chunk].map { $0 * 0.7 + cost * 0.3 } ?? cost
        prefillChunkTries[chunk, default: 0] += 1
    }

    /// Tokens of the kept stable prefix, if any.
    var prefixTokens: [Int32]? { prefixMark?.tokens }

    /// Saves the kept stable prefix's cache state (nothing to save → no file written). MLX errors are thrown.
    func savePrefix(to url: URL, identity: String) throws {
        try withError { try savePrefixUnchecked(to: url, identity: identity) }
    }

    private func savePrefixUnchecked(to url: URL, identity: String) throws {
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
    /// MLX errors are thrown.
    @discardableResult
    func loadPrefix(from url: URL, identity: String, ifPrefixOf prompt: [Int32]) throws -> Bool {
        try withError { try loadPrefixUnchecked(from: url, identity: identity, ifPrefixOf: prompt) }
    }

    private func loadPrefixUnchecked(from url: URL, identity: String, ifPrefixOf prompt: [Int32]) throws -> Bool {
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
        resumeMarks = []
        let snapshots = cache.compactMap { ($0 as? InklingVisitCache)?.snapshot() }
        prefixMark = (tokens, snapshots)
        return true
    }

    /// Generates up to `maxTokens` (stopping at EOS, the context limit or when `shouldStop()` is true),
    /// calling `onToken` for each token. Returns the tokens (without EOS).
    @discardableResult
    public func generate(prompt: [Int32], stablePrefix: Int = 0, sampler: Sampler, maxTokens: Int, shouldStop: () -> Bool,
                         onToken: (Int32) -> Void) -> [Int32] {
        if let draft, cache.allSatisfy({ $0 is InklingVisitCache }) {
            return generateSpeculative(prompt: prompt, stablePrefix: stablePrefix, sampler: sampler, maxTokens: maxTokens,
                                       shouldStop: shouldStop, onToken: onToken, draft: draft)
        }
        if pipelined, forcedToken == nil {
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
        var fed: [Int32] = []
        defer { keepResume(prompt + fed, logits: fed.isEmpty ? nil : logits) }
        while produced.count < maxTokens, prompt.count + produced.count < contextLimit, !shouldStop() {
            let token = forcedToken?() ?? sampler.sample(logits, history: history).item(Int32.self)
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
            fed.append(token)
            if boundaryTokens.contains(token) { keepBoundary(prompt + fed, logits: logits) }
        }
        if let decodeStart, produced.count > 1 {
            let s = (ContinuousClock.now - decodeStart) / .seconds(1)
            stats.decodeTokensPerSecond = s > 0 ? Double(produced.count - 1) / s : 0
        }
        return produced
    }

    /// The tokens after an earlier occurrence of the context's last tokens (up to `max`), or none. Tries the last
    /// 6, 5, 4, then 3 tokens (a longer match is more likely right); among matches of one length, the one with the
    /// longest continuation wins, ties going to the most recent.
    static func lookupDraft(_ context: [Int32], max: Int) -> [Int32] {
        guard max > 0 else { return [] }
        let count = context.count
        return context.withUnsafeBufferPointer { c in
            for n in stride(from: 6, through: 3, by: -1) where count > n {
                var best = -1, bestLength = 0
                var start = count - n - 1
                while start >= 0 {
                    var k = 0
                    while k < n, c[start + k] == c[count - n + k] { k += 1 }
                    if k == n {
                        let length = Swift.min(max, count - start - n)
                        if length > bestLength {
                            best = start
                            bestLength = length
                            if length == max { break }
                        }
                    }
                    start -= 1
                }
                if best >= 0 { return Array(c[(best + n)..<(best + n + bestLength)]) }
            }
            return []
        }
    }

    /// Memory warning: forget the saved cache states (they keep K/V buffers alive). The next prompt starts over
    /// (the instructions' prefix is still restored from its file or recomputed).
    public func dropSavedStates() {
        resumeMarks = []
        mark = nil
    }

    /// The state after `tokens` (prompt + what was fed since) is kept for the next prompt: the end of the reply.
    private func keepResume(_ tokens: [Int32], logits: MLXArray?) {
        guard tokens.count > (mark?.tokens.count ?? 0) else { return }
        let snapshots = cache.compactMap { ($0 as? InklingVisitCache)?.snapshot() }
        guard snapshots.count == cache.count else { return }
        resumeMarks = resumeMarks.filter { $0.tokens.count < tokens.count }.suffix(1) + [(tokens, snapshots, logits)]
    }

    /// A boundary token (end of a thought) is now in the cache: keep this point too (only the latest).
    private func keepBoundary(_ tokens: [Int32], logits: MLXArray?) {
        let snapshots = cache.compactMap { ($0 as? InklingVisitCache)?.snapshot() }
        guard snapshots.count == cache.count else { return }
        resumeMarks = [(tokens, snapshots, logits)]
    }

    /// Guessing ahead pays only when a check's tokens cost less than plain steps: plain steps are timed now and then
    /// (`probeEvery` checks, `probeSteps` steps) and run instead for `plainRun` tokens while they are cheaper.
    /// (Flash-MoE found MoE speculation break-even: every guessed token needs its own experts.)
    static let probeEvery = 16, probeSteps = 2, plainRun = 32

    /// `generate` with speculative decoding (see `draft`): same tokens as `model` alone.
    func generateSpeculative(prompt: [Int32], stablePrefix: Int, sampler: Sampler, maxTokens: Int, shouldStop: () -> Bool,
                             onToken: (Int32) -> Void, draft: CausalLM) -> [Int32] {
        let started = ContinuousClock.now
        guard let first = prefill(prompt, stablePrefix: stablePrefix, shouldStop: shouldStop) else { return [] }
        var produced: [Int32] = []
        var history = prompt.suffix(sampler.penaltyWindow).map { $0 }
        var context = prompt.suffix(Self.lookupWindow).map { $0 }
        var decodeStart: ContinuousClock.Instant?
        stats.generatedTokens = 0
        stats.decodeTokensPerSecond = 0
        stats.draftedTokens = 0
        stats.acceptedTokens = 0
        stats.checks = 0
        stats.plainSteps = 0
        var fed: [Int32] = []
        var lastLogits: MLXArray?
        defer {
            if let decodeStart, produced.count > 1 {
                let s = (ContinuousClock.now - decodeStart) / .seconds(1)
                stats.decodeTokensPerSecond = s > 0 ? Double(produced.count - 1) / s : 0
            }
            keepResume(prompt + fed, logits: fed.isEmpty ? nil : lastLogits)
        }
        // Room for one more token (as `generate`'s loop condition).
        func room() -> Bool { produced.count < maxTokens && prompt.count + produced.count < contextLimit && !shouldStop() }
        // Hands out a sampled token; false when generation ends with it (EOS, limits, stop).
        func emit(_ token: Int32) -> Bool {
            if produced.isEmpty {
                stats.timeToFirstToken = ContinuousClock.now - started
                decodeStart = .now
            } else {
                afterStep?(0)
            }
            if eos.contains(token) { return false }
            produced.append(token)
            history.append(token)
            context.append(token)
            stats.generatedTokens = produced.count
            onToken(token)
            return room()
        }
        func pick(_ logits: MLXArray) -> Int32 { forcedToken?() ?? sampler.sample(logits, history: history).item(Int32.self) }
        func seconds(since t: ContinuousClock.Instant) -> Double { (ContinuousClock.now - t) / .seconds(1) }
        func average(_ old: Double?, _ new: Double) -> Double { old.map { $0 * 0.7 + new * 0.3 } ?? new }
        guard room() else { return [] }
        var pending = pick(first)  // sampled, not yet in the cache
        var length = 4
        var draftCost: Double?, plainCost: Double?  // seconds per token: drafted checks, plain steps
        var plainLeft = 0, draftChecks = 0
        let stepModel = exactStep ?? model
        while emit(pending) {
            let begin = ContinuousClock.now
            let limit = min(length, maxTokens - produced.count, contextLimit - prompt.count - produced.count)
            var guesses = Self.lookupDraft(context, max: limit)
            let lookup = guesses.count >= 2
            // A plain exact step: cheaper than guessing right now, or a boundary token to keep the state after.
            if !lookup && (plainLeft > 0 || boundaryTokens.contains(pending)) {
                let logits = stepModel(MLXArray([pending]).reshaped([1, 1]), cache: cache)[0, -1]
                fed.append(pending)
                lastLogits = logits
                let token = pick(logits)
                if boundaryTokens.contains(pending) { keepBoundary(prompt + fed, logits: logits) }
                stats.plainSteps += 1
                if plainLeft > 0 {
                    plainCost = average(plainCost, seconds(since: begin))
                    plainLeft -= 1
                    // Stay plain while it is clearly cheaper than guessing.
                    if plainLeft == 0, let d = draftCost, let p = plainCost, p < d * 0.95 { plainLeft = Self.plainRun }
                }
                pending = token
                continue
            }
            if !lookup {
                let snapshots = cache.map { ($0 as! InklingVisitCache).snapshot() }
                guesses = (draft as? ChainedDraft)?.propose(after: pending, count: limit, cache: cache)
                    ?? Self.stepDraft(draft, after: pending, count: limit, cache: cache)
                if let end = guesses.firstIndex(where: { eos.contains($0) }) { guesses = Array(guesses[...end]) }
                for (c, s) in zip(cache, snapshots) { (c as! InklingVisitCache).restore(s) }
            }
            let input = [pending] + guesses
            let logits = ConvContext.$keepAll.withValue(true) { model(MLXArray(input).reshaped([1, -1]), cache: cache) }
            eval(logits)
            stats.checks += 1
            stats.draftedTokens += guesses.count
            var kept = 0, done = false
            pending = -1
            for (i, guess) in guesses.enumerated() {
                let token = pick(logits[0, i])
                guard token == guess else { pending = token; break }
                kept += 1
                if !emit(token) { done = true; break }
                // Stop at a boundary token so the cache can be kept right after it.
                if boundaryTokens.contains(token), i + 1 < guesses.count {
                    pending = pick(logits[0, i + 1])
                    break
                }
            }
            for c in cache { (c as! InklingVisitCache).keep(1 + kept, of: input.count) }
            fed += input.prefix(1 + kept)
            lastLogits = logits[0, kept]
            if let last = fed.last, boundaryTokens.contains(last) { keepBoundary(prompt + fed, logits: lastLogits) }
            stats.acceptedTokens += kept
            if done { break }
            if pending < 0 { pending = pick(logits[0, guesses.count]) }
            length = kept == guesses.count ? min(length + 2, maxDraft) : max(2, kept + 1)
            if !lookup {
                draftCost = average(draftCost, seconds(since: begin) / Double(kept + 1))
                draftChecks += 1
                if draftChecks % Self.probeEvery == 0 || (plainCost == nil && draftChecks == 4) { plainLeft = Self.probeSteps }
            }
        }
        return produced
    }

    /// Greedy guesses one model step at a time, each token fed to the next step without being read back first.
    static func stepDraft(_ draft: CausalLM, after token: Int32, count: Int, cache: [LayerVisitCache]) -> [Int32] {
        var t = MLXArray([token]).reshaped([1, 1])
        var out: [MLXArray] = []
        for _ in 0..<count {
            t = argMax(draft(t, cache: cache)[0, -1]).asType(.int32).reshaped([1, 1])
            asyncEval(t)
            out.append(t)
        }
        eval(out)
        return out.map { $0.item(Int32.self) }
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
