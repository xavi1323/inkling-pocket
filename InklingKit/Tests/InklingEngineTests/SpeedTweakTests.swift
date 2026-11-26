import Foundation
import Testing
import MLX
@testable import InklingEngine

/// The 2026-11-26 speed round: reusing what the model wrote, plain steps when guessing loses, read-ahead for checks
/// and prompt chunks, slots planned from expert heat, self-tuning read sizes.
@Suite(.serialized) struct SpeedTweakTests {
    init() { Device.setDefault(device: .cpu) }
    var prompt: [Int32] { (0..<40).map { Int32(($0 * 37 + 11) % 500) } }

    func run(_ gen: TokenGenerator, _ prompt: [Int32], tokens: Int) -> [Int32] {
        gen.generate(prompt: prompt, sampler: Sampler(greedy()), maxTokens: tokens, shouldStop: { false }, onToken: { _ in })
    }

    /// The next prompt starts with the last prompt and the reply (an agent step): only the new tokens are computed,
    /// and the result is what a fresh cache gives.
    @Test func replyIsNotComputedAgain() throws {
        for drafted in [false, true] {
            let model = try Tiny.model()
            let gen = TokenGenerator(model: model, eos: [])
            if drafted { gen.draft = FastPathTests.WrongDraft(model) }
            let reply = run(gen, prompt, tokens: 10)
            let next = prompt + reply + [5, 6, 7]
            let got = run(gen, next, tokens: 6)
            #expect(gen.stats.prefilledTokens <= 4)  // the last reply token may not be in the cache yet
            #expect(got == run(TokenGenerator(model: try Tiny.model(), eos: []), next, tokens: 6))
        }
    }

    /// A boundary token (end of a thought): the state right after it is kept, so a next prompt that differs after
    /// it still skips the thought.
    @Test func stateAfterABoundaryIsKept() throws {
        for drafted in [false, true] {
            let model = try Tiny.model()
            let gen = TokenGenerator(model: model, eos: [])
            if drafted { gen.draft = model }
            let reply = run(gen, prompt, tokens: 12)
            // Generate again with a reply token that occurs once (not last) as the boundary.
            let cut = try #require(reply.indices.dropLast().first { i in reply.filter { $0 == reply[i] }.count == 1 && i >= 2 })
            let gen2 = TokenGenerator(model: model, eos: [])
            if drafted { gen2.draft = model }
            gen2.boundaryTokens = [reply[cut]]
            #expect(run(gen2, prompt, tokens: 12) == reply)
            let next = prompt + reply[...cut] + [9, 9, 9, 9]
            let got = run(gen2, next, tokens: 4)
            #expect(gen2.stats.prefilledTokens == 4)
            #expect(got == run(TokenGenerator(model: try Tiny.model(), eos: []), next, tokens: 4))
        }
    }

    /// When checks keep rejecting guesses, plain exact steps take over; the output stays exact.
    @Test func plainStepsTakeOverWhenGuessingLoses() throws {
        let model = try Tiny.model()
        let gen = TokenGenerator(model: model, eos: [])
        gen.draft = FastPathTests.WrongDraft(model)
        let got = run(gen, prompt, tokens: 40)
        #expect(gen.stats.plainSteps > 0)
        #expect(got == run(TokenGenerator(model: try Tiny.model(), eos: []), prompt, tokens: 40))
    }

    @Test func checkGuessesAreReadBestRankFirst() {
        // 2 rows × k 3: row 0 → [4, 5, 6], row 1 → [5, 7, 8].
        let (experts, ranks) = SparseMoE.rankMajor([4, 5, 6, 5, 7, 8], rows: 2, k: 3, depth: 2)
        #expect(experts == [4, 5, 7])
        #expect(ranks == [0, 0, 1])
    }

    @Test func promptChunkReadsAheadMissingExperts() throws {
        let model = try ExpertStreamingTests.streamed(slots: 3)
        let cache = model.expertCaches[0]
        cache.readAheadMissing(2)
        cache.waitForLoads()
        #expect(cache.prefetched == 2)
        cache.readAheadMissing(0)
        cache.waitForLoads()
        #expect(cache.prefetched == 2)
    }

    /// Slots go to the layer whose routing is spread; a layer that uses two experts keeps only `topK`.
    @Test func heatPlanGivesSlotsWhereRoutingIsSpread() {
        let spread = Array(repeating: 10, count: 16)
        let focused = [100, 100] + Array(repeating: 0, count: 14)
        let plan = InklingGenerator.heatPlan(budget: 20, perLayer: [1, 1], heat: [spread, focused], experts: 16, topK: 2)
        #expect(plan == [16, 4])
        #expect(InklingGenerator.heatPlan(budget: 3, perLayer: [1, 1], heat: [spread, focused], experts: 16, topK: 2) == nil)
        #expect(InklingGenerator.heatPlan(budget: 20, perLayer: [1, 1], heat: [spread], experts: 16, topK: 2) == nil)
    }

    /// Every piece size is tried, then the fastest is used.
    @Test func readSizeSettlesOnTheFastest() {
        let reader = ExpertReader.shared
        var seen = Set<Int>()
        for _ in 0..<(ExpertReader.pieceChoices.count * ExpertReader.samplesPerChoice) {
            let (i, size) = reader.choosePieceSize()
            seen.insert(size)
            reader.record(choice: i, bytes: Int64(size), seconds: i == 1 ? 0.001 : 0.01)
        }
        #expect(seen.count == ExpertReader.pieceChoices.count)
        #expect(reader.preferredPieceSize == ExpertReader.pieceChoices[1])
    }

    /// Top-k / top-p now sample among a partial selection; with top-k it is the same candidate set.
    @Test func samplerKeepsTopKCandidates() {
        var p = greedy()
        p.temperature = 1
        p.topK = 3
        p.topP = 1
        let logits = MLXArray([0, 5, 1, 9, 2, 8, 3] as [Float])
        var picked = Set<Int32>()
        for _ in 0..<60 { picked.insert(Sampler(p).sample(logits, history: []).item(Int32.self)) }
        #expect(picked.isSubset(of: [3, 5, 1]))
        p.topK = 0
        p.topP = 0.5
        #expect(Sampler(p).sample(logits, history: []).item(Int32.self) == 3)
    }
}
