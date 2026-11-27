import Foundation

/// Every generation's stats, in order, for the app to sum up per reply (an agent reply runs one generation per
/// step: thinking, tool call, answer). Real chats, not the benchmark's repeated text.
public final class EngineStatsLog: @unchecked Sendable {
    public static let shared = EngineStatsLog()
    private let lock = NSLock()
    private var entries: [GenerationStats] = []
    private var dropped = 0
    public init() {}

    /// Position to pass to `since` later.
    public var count: Int { lock.withLock { dropped + entries.count } }

    public func append(_ stats: GenerationStats) {
        lock.withLock {
            entries.append(stats)
            if entries.count > 200 { entries.removeFirst(100); dropped += 100 }
        }
    }

    /// Stats appended after `mark` (`count` earlier).
    public func since(_ mark: Int) -> [GenerationStats] {
        lock.withLock { Array(entries.dropFirst(max(0, mark - dropped))) }
    }
}

/// One reply's engine numbers, summed over its steps.
public struct ReplyStats: Sendable, Equatable {
    public var steps = 0
    public var promptTokens = 0, computedTokens = 0
    public var firstTokenSeconds = 0.0
    public var generatedTokens = 0, decodeSeconds = 0.0
    public var drafted = 0, accepted = 0, checks = 0, plainSteps = 0

    public init(_ all: [GenerationStats]) {
        steps = all.count
        for s in all {
            promptTokens += s.promptTokens
            computedTokens += s.prefilledTokens
            generatedTokens += s.generatedTokens
            if s.decodeTokensPerSecond > 0, s.generatedTokens > 1 { decodeSeconds += Double(s.generatedTokens - 1) / s.decodeTokensPerSecond }
            drafted += s.draftedTokens
            accepted += s.acceptedTokens
            checks += s.checks
            plainSteps += s.plainSteps
        }
        firstTokenSeconds = all.first.map { $0.timeToFirstToken / .seconds(1) } ?? 0
    }

    /// Tokens per second while writing (all steps together).
    public var tokensPerSecond: Double? {
        let tokens = generatedTokens - steps
        return decodeSeconds > 0 && tokens > 0 ? Double(tokens) / decodeSeconds : nil
    }

    /// Plain lines for the speed panel.
    public var lines: [String] {
        var out = [String(format: "First token: %.1f s", firstTokenSeconds)]
        if let tps = tokensPerSecond { out.append(String(format: "Writing: %.1f tok/s (%d tokens)", tps, generatedTokens)) }
        out.append("Prompt: \(computedTokens) of \(promptTokens) tokens computed, rest reused" + (steps > 1 ? " (\(steps) steps)" : ""))
        if checks > 0 {
            out.append(String(format: "Guesses kept: %.0f%% (%.1f tokens per check, %d checks)",
                              Double(accepted) / Double(max(drafted, 1)) * 100, Double(accepted + checks) / Double(checks), checks))
        }
        if plainSteps > 0 { out.append("Plain steps: \(plainSteps)") }
        return out
    }
}
