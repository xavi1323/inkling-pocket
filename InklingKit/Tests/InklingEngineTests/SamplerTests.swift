import Testing
import MLX
import AgentCore
@testable import InklingEngine

@Suite(.serialized) struct SamplerTests {
    init() { Device.setDefault(device: .cpu) }
    let logits = MLXArray([0.1, 3.0, 0.2, 2.9] as [Float])

    @Test func greedyWhenTemperatureZero() {
        var p = GenerateParameters()
        p.temperature = 0
        #expect(Sampler(p).sample(logits, history: []).item(Int.self) == 1)
    }

    @Test func topKOneIsGreedy() {
        var p = GenerateParameters()
        p.temperature = 1
        p.topK = 1
        let s = Sampler(p)
        for _ in 0..<30 { #expect(s.sample(logits, history: []).item(Int.self) == 1) }
    }

    @Test func tinyTopPKeepsOnlyTheBest() {
        var p = GenerateParameters()
        p.temperature = 1
        p.topP = 0.01
        let s = Sampler(p)
        for _ in 0..<30 { #expect(s.sample(logits, history: []).item(Int.self) == 1) }
    }

    @Test func topKTwoNeverPicksTheRest() {
        var p = GenerateParameters()
        p.temperature = 5
        p.topK = 2
        p.topP = 1
        let s = Sampler(p)
        let picks = Set((0..<60).map { _ in s.sample(logits, history: []).item(Int.self) })
        #expect(picks.isSubset(of: [1, 3]))
        #expect(picks.count == 2)
    }

    @Test func repetitionPenaltyLowersSeenTokens() {
        var p = GenerateParameters()
        p.temperature = 0
        p.repetitionPenalty = 1.5
        #expect(Sampler(p).sample(logits, history: [1]).item(Int.self) == 3)
    }

    @Test func acceptsBatchedLogits() {
        var p = GenerateParameters()
        p.temperature = 0
        #expect(Sampler(p).sample(logits.reshaped([1, 1, 4]), history: []).item(Int.self) == 1)
    }
}
