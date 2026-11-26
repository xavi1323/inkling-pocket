import MLX
import Testing
@testable import InklingEngine

/// Prompt-lookup drafting (longest match first) and self-tuning prefill chunk size.
@Suite(.serialized)
struct LookupChunkTests {
    init() { Device.setDefault(device: .cpu) }

    @Test func lookupPrefersLongerMatchOverMoreRecent() {
        // 5-token match [1,2,3,4,5] early (→ 60), more recent 3-token match [3,4,5] (→ 70).
        let ctx: [Int32] = [1, 2, 3, 4, 5, 60, 61, 9, 3, 4, 5, 70, 71, 8, 1, 2, 3, 4, 5]
        #expect(TokenGenerator.lookupDraft(ctx, max: 2) == [60, 61])
    }

    @Test func lookupPrefersLongerContinuation() {
        // Two 6-token matches: both continuations reach the cap of 4, so the most recent wins.
        let pat: [Int32] = [10, 11, 12, 13, 14, 15]
        let ctx: [Int32] = pat + [1, 2, 3, 4] + [99] + pat + pat
        #expect(TokenGenerator.lookupDraft(ctx, max: 4) == [10, 11, 12, 13])
        // The recent [7, 8, 9] has only 3 tokens after it; the earlier one has 4.
        let ctx2: [Int32] = [7, 8, 9, 50, 51, 52, 53, 1, 7, 8, 9, 7, 8, 9]
        #expect(TokenGenerator.lookupDraft(ctx2, max: 4) == [50, 51, 52, 53])
    }

    @Test func lookupEmptyWithoutMatchAndRespectsMax() {
        #expect(TokenGenerator.lookupDraft([1, 2, 3, 4, 5, 6, 7], max: 8).isEmpty)
        #expect(TokenGenerator.lookupDraft([1, 2, 3, 4, 5, 6], max: 0).isEmpty)
        let ctx: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3]
        #expect(TokenGenerator.lookupDraft(ctx, max: 2) == [4, 5])
    }

    @Test func speculativeOnRepeatedTextMatchesPlain() throws {
        let part: [Int32] = (0..<12).map { Int32(($0 * 37 + 11) % 500) }
        let prompt = part + [3, 9] + part + [3, 9] + part
        let model = try Tiny.model()
        let want = TokenGenerator(model: model, eos: [])
            .generate(prompt: prompt, sampler: Sampler(greedy()), maxTokens: 20, shouldStop: { false }, onToken: { _ in })
        let gen = TokenGenerator(model: FastPath(model), eos: [])
        gen.draft = model
        gen.exactStep = model
        let got = gen.generate(prompt: prompt, sampler: Sampler(greedy()), maxTokens: 20, shouldStop: { false }, onToken: { _ in })
        #expect(got == want)
    }

    @Test func chunkTuningTriesEveryChoiceAndKeepsOutput() throws {
        let gen = TokenGenerator(model: try Tiny.model(), eos: [])
        gen.prefillChunkChoices = [8, 16]
        #expect(gen.chosenPrefillChunk == gen.prefillChunk)
        for p in 0..<6 {
            let prompt: [Int32] = (0..<40).map { Int32(($0 * 37 + 11 + p * 101) % 500) }
            let got = gen.generate(prompt: prompt, sampler: Sampler(greedy()), maxTokens: 3, shouldStop: { false }, onToken: { _ in })
            let want = TokenGenerator(model: try Tiny.model(), eos: [])
                .generate(prompt: prompt, sampler: Sampler(greedy()), maxTokens: 3, shouldStop: { false }, onToken: { _ in })
            #expect(got == want)
        }
        #expect(gen.prefillChunkTries[8, default: 0] >= 2)
        #expect(gen.prefillChunkTries[16, default: 0] >= 2)
        #expect([8, 16].contains(gen.chosenPrefillChunk))
    }
}
