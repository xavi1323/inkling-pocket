import Foundation
import Testing
import MLX
@testable import InklingEngine
import InklingCore

/// Step 1 of the one-engine plan: faster prompt encoding, memory safety, reply stats.
@Suite(.serialized) struct StepOneTests {
    /// Special tokens split the text the same way as before (one scan now), including tricky spots.
    @Test func specialSplitMatchesWholeTokenLookups() throws {
        let tok = try TinyTokenizer.load()
        let cases = [
            "plain text only",
            "<|message_user|>hi<|end_message|><|message_model|>",
            "<|<|end_message|> <| almost |> <|message_tool|>x",
            "a<|end_message|b|end_message|><|end_message|>",
            "<|content_invoke_tool_json|>{\"a\": \"<|\"}<|end_message|>",
        ]
        let user = try #require(tok.specialTokenID("<|message_user|>"))
        let end = try #require(tok.specialTokenID("<|end_message|>"))
        #expect(tok.encode(cases[1]).first == user)
        #expect(tok.encode(cases[1]).filter { $0 == end }.count == 1)
        #expect(tok.encode(cases[2]).filter { $0 == end }.count == 1)
        #expect(tok.encode(cases[3]).filter { $0 == end }.count == 1)
        for text in cases {
            // Same tokens as encoding the pieces between specials on their own.
            let (tokens, cuts) = tok.encodeWithCuts(text)
            #expect(tokens == tok.encode(text))
            let bytes = Array(text.utf8)
            for cut in cuts {
                let head = String(bytes: bytes[..<cut.offset], encoding: .utf8)!
                #expect(Array(tokens.prefix(cut.count)) == tok.encode(head), "\(text) cut \(cut)")
            }
        }
    }

    /// Each turn reuses the conversation's tokens; the result equals encoding the whole prompt.
    @Test func promptEncoderReusesTheConversation() throws {
        let tok = try TinyTokenizer.load()
        let enc = PromptEncoder(tokenizer: tok)
        let turn1 = "<|message_system|>be nice<|end_message|><|message_user|>hello there<|end_message|><|message_model|>"
        let turn2 = turn1 + "hi! how can I help?<|end_message|><|message_user|>tell me more<|end_message|><|message_model|>"
        #expect(enc.encode(turn1) == tok.encode(turn1))
        #expect(enc.reusedTokens == 0)
        #expect(enc.encode(turn2) == tok.encode(turn2))
        #expect(enc.reusedTokens > 0)
        // A different conversation: correct, and nothing wrong reused.
        let other = "<|message_system|>be mean<|end_message|><|message_user|>yo<|end_message|>"
        #expect(enc.encode(other) == tok.encode(other))
        // Emoji and other multi-byte text around the cut.
        let emoji = other + "<|message_model|>😀 ok<|end_message|>"
        #expect(enc.encode(emoji) == tok.encode(emoji))
    }

    @Test func longContextsUseSmallerAttentionBlocks() {
        #expect(Attention.blockSize(heads: 24, keys: 1000) == 512)
        let long = Attention.blockSize(heads: 24, keys: 16_384)
        #expect(long == 64)
        #expect(24 * long * 16_384 * 2 <= Attention.maskBudget)
        #expect(Attention.blockSize(heads: 24, keys: 1_000_000) == 64)  // never below 64
    }

    /// The memory reserve covers K/V for an 8K chat on the real model's shape (0.5 GB did not).
    @Test func reserveCoversALongChat() throws {
        let text = try Tiny.config().text
        let r = InklingGenerator.reserve(for: text)
        #expect(r >= InklingGenerator.workingReserve)
    }

    /// Prefix states dropped on a memory warning: the next prompt still gives the same tokens.
    @Test func droppingSavedStatesKeepsOutputs() throws {
        let gen = TokenGenerator(model: try Tiny.model(), eos: [])
        let prompt: [Int32] = (0..<30).map { Int32(($0 * 13 + 5) % 500) }
        func run() -> [Int32] {
            gen.generate(prompt: prompt, sampler: Sampler(greedy()), maxTokens: 5, shouldStop: { false }, onToken: { _ in })
        }
        let first = run()
        gen.dropSavedStates()
        #expect(run() == first)
    }

    @Test func replyStatsSumTheSteps() {
        var a = GenerationStats(), b = GenerationStats()
        a.promptTokens = 100; a.prefilledTokens = 100; a.generatedTokens = 11; a.decodeTokensPerSecond = 10
        a.timeToFirstToken = .seconds(2); a.draftedTokens = 20; a.acceptedTokens = 10; a.checks = 5
        b.promptTokens = 150; b.prefilledTokens = 30; b.generatedTokens = 21; b.decodeTokensPerSecond = 20
        b.timeToFirstToken = .seconds(1)
        let r = ReplyStats([a, b])
        #expect(r.steps == 2 && r.computedTokens == 130 && r.promptTokens == 250 && r.firstTokenSeconds == 2)
        #expect(abs(r.tokensPerSecond! - 30.0 / 2.0) < 1e-9)  // 10 tokens in 1 s + 20 in 1 s
        #expect(r.lines.contains { $0.hasPrefix("Guesses kept: 50%") })

        let log = EngineStatsLog()
        log.append(a)
        let mark = log.count
        log.append(b)
        #expect(log.since(mark) == [b])
    }
}
