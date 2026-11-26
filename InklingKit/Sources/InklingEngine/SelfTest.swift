import Foundation
import MLX
import InklingCore
import AgentCore

public struct KnownAnswer: Sendable, Equatable {
    public var prompt: String
    public var output: String
    public var passed: Bool
    public init(prompt: String, output: String, passed: Bool) { self.prompt = prompt; self.output = output; self.passed = passed }
}

public struct PathComparison: Sendable, Equatable {
    public var top1Agreement: Double
    public var meanKLD: Double
    public var maxAbsLogitDiff: Float
    public init(top1Agreement: Double, meanKLD: Double, maxAbsLogitDiff: Float) {
        self.top1Agreement = top1Agreement; self.meanKLD = meanKLD; self.maxAbsLogitDiff = maxAbsLogitDiff
    }
}

/// Fast path vs reference path, plus greedy known-answer prompts (B1, B2).
public struct SelfTestReport: Sendable {
    public var comparison: PathComparison
    public var knownAnswers: [KnownAnswer]
    public var top1Agreement: Double { comparison.top1Agreement }
    public var meanKLD: Double { comparison.meanKLD }
    public var maxAbsLogitDiff: Float { comparison.maxAbsLogitDiff }

    public init(comparison: PathComparison, knownAnswers: [KnownAnswer]) {
        self.comparison = comparison
        self.knownAnswers = knownAnswers
    }

    public var passed: Bool {
        comparison.top1Agreement >= SelfTest.minTop1 && comparison.meanKLD <= SelfTest.maxKLD && knownAnswers.allSatisfy(\.passed)
    }

    public var asText: String {
        var lines = [
            "Inkling Pocket self-test: \(passed ? "PASS" : "FAIL")",
            String(format: "Top-1 agreement: %.1f%% (need ≥ %.0f%%)", top1Agreement * 100, SelfTest.minTop1 * 100),
            String(format: "Mean KLD: %.5f (need ≤ %.2f)", meanKLD, SelfTest.maxKLD),
            String(format: "Max logit diff: %.4f", maxAbsLogitDiff),
        ]
        for a in knownAnswers {
            lines.append("\(a.passed ? "PASS" : "FAIL") \(a.prompt) → \(a.output.trimmingCharacters(in: .whitespacesAndNewlines).prefix(80))")
        }
        return lines.joined(separator: "\n")
    }
}

public enum SelfTest {
    public static let minTop1 = 0.97
    public static let maxKLD = 0.01
    public static let continuationTokens = 64

    public static let comparePrompts = [
        "Explain in a short paragraph why the sky is blue.",
        "Write a Swift function that reverses a string.",
        "List three tips for staying focused while studying.",
    ]

    /// (prompt, text the greedy answer must contain, case-insensitive).
    public static let knownAnswers: [(prompt: String, expected: String)] = [
        ("What is the capital of France? Answer in one word.", "paris"),
        ("What is 12 times 12? Reply with the number only.", "144"),
        ("Spell the word cat backwards. Reply with the result only.", "tac"),
    ]

    public static func check(_ output: String, expected: String) -> Bool {
        output.lowercased().contains(expected.lowercased())
    }

    /// Teacher-forced comparison, run the way decoding runs: each path reads the sequence up to the first scored
    /// position in one pass, then the remaining tokens one at a time (so the fast path's single-token kernels are used).
    /// Logits for the last `scored` positions are compared.
    public static func compare(reference: CausalLM, fast: CausalLM, sequences: [[Int32]], scored: Int) -> PathComparison {
        var agree = 0, total = 0
        var kld = 0.0
        var maxDiff: Float = 0
        func stepped(_ model: CausalLM, _ seq: [Int32], _ n: Int) -> MLXArray {
            let cache = model.makeCache()
            let split = seq.count - n + 1
            var rows = [model.lastLogits(MLXArray(Array(seq[..<split])).reshaped([1, -1]), cache: cache)]
            for token in seq[split...] {
                let row = model(MLXArray([token]).reshaped([1, 1]), cache: cache)[0, -1]
                eval(row)
                rows.append(row)
            }
            return stacked(rows, axis: 0).asType(.float32)
        }
        for seq in sequences {
            let n = min(scored, seq.count)
            guard n > 0 else { continue }
            let a = stepped(reference, seq, n)
            let b = stepped(fast, seq, n)
            agree += (argMax(a, axis: -1) .== argMax(b, axis: -1)).asType(.int32).sum().item(Int.self)
            total += n
            let la = a - logSumExp(a, axis: -1, keepDims: true)
            let lb = b - logSumExp(b, axis: -1, keepDims: true)
            kld += Double((exp(la) * (la - lb)).sum().item(Float.self))
            maxDiff = max(maxDiff, abs(a - b).max().item(Float.self))
        }
        return PathComparison(top1Agreement: total == 0 ? 0 : Double(agree) / Double(total),
                              meanKLD: total == 0 ? 0 : kld / Double(total), maxAbsLogitDiff: maxDiff)
    }
}

extension InklingGenerator {
    /// Runs the self-test on the loaded model: path comparison on 3 prompts × 64 tokens, then the known answers.
    public func selfTest(progress: @escaping @Sendable (String) -> Void = { _ in }) async throws -> SelfTestReport {
        let (reference, fast) = paths()
        var sequences: [[Int32]] = []
        var greedy = GenerateParameters()
        greedy.temperature = 0
        for (i, prompt) in SelfTest.comparePrompts.enumerated() {
            progress("Comparing paths \(i + 1)/\(SelfTest.comparePrompts.count)…")
            let tokens = tokenizer.encode(PromptFormatter.render(messages: [ChatMessage(role: .user, content: prompt)], tools: [], effort: .none))
            let gen = TokenGenerator(model: reference, eos: [])
            let continuation = try withError { gen.generate(prompt: tokens, sampler: Sampler(greedy), maxTokens: SelfTest.continuationTokens,
                                            shouldStop: { Task.isCancelled }, onToken: { _ in }) }
            try Task.checkCancellation()
            sequences.append(tokens + continuation)
        }
        let comparison = try withError {
            SelfTest.compare(reference: reference, fast: fast, sequences: sequences, scored: SelfTest.continuationTokens)
        }
        var answers: [KnownAnswer] = []
        // Room for a short thought first: at 48 tokens "cat backwards" ran out before answering.
        greedy.maxTokens = 160
        for (i, item) in SelfTest.knownAnswers.enumerated() {
            progress("Known answer \(i + 1)/\(SelfTest.knownAnswers.count)…")
            var text = "", thought = ""
            for try await event in generate([ChatMessage(role: .user, content: item.prompt)], tools: [], effort: .none, parameters: greedy) {
                if case .answer(let t) = event { text += t }
                if case .thinking(let t) = event { thought += t }
            }
            let shown = text.isEmpty && !thought.isEmpty ? "(no answer; thought: \(thought.prefix(120)))" : text
            answers.append(KnownAnswer(prompt: item.prompt, output: shown, passed: SelfTest.check(text, expected: item.expected)))
        }
        return SelfTestReport(comparison: comparison, knownAnswers: answers)
    }
}
