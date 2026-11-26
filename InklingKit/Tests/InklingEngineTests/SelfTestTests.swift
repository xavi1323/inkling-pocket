import Foundation
import Testing
import MLX
@testable import InklingEngine

@Suite(.serialized) struct SelfTestTests {
    init() { Device.setDefault(device: .cpu) }
    let sequence: [Int32] = [1, 17, 42, 99, 256, 3, 499, 7, 77, 128, 311, 64, 5, 6, 7, 8]

    @Test func selfTestPassesWhenPathsEqual() throws {
        let model = try Tiny.model()
        let c = SelfTest.compare(reference: model, fast: try Tiny.model(), sequences: [sequence], scored: 8)
        #expect(c.top1Agreement == 1)
        #expect(c.meanKLD < 1e-6)
        #expect(c.maxAbsLogitDiff < 1e-4)
        let report = SelfTestReport(comparison: c, knownAnswers: [KnownAnswer(prompt: "p", output: "Paris", passed: true)])
        #expect(report.passed)
    }

    @Test func selfTestFailsWhenPathsDiffer() throws {
        let other = ScriptModel(script: [3], vocab: 500)
        let c = SelfTest.compare(reference: try Tiny.model(), fast: other, sequences: [sequence], scored: 8)
        #expect(c.top1Agreement < 0.97)
        #expect(!SelfTestReport(comparison: c, knownAnswers: []).passed)
    }

    @Test func knownAnswerChecksAreCaseInsensitive() {
        #expect(SelfTest.knownAnswers.count == 3)
        #expect(SelfTest.check("The capital is PARIS.", expected: "paris"))
        #expect(SelfTest.check("tac", expected: "tac"))
        #expect(!SelfTest.check("143", expected: "144"))
    }

    @Test func reportTextContainsAllFields() {
        var bench = BenchmarkReport()
        bench.prefillTokensPerSecond = 120
        bench.decodeTokensPerSecond = 31.5
        bench.timeToFirstToken = .milliseconds(850)
        bench.availableMemoryBytes = 6_000_000_000
        bench.peakFootprintBytes = 2_000_000_000
        bench.mappedWeightBytes = 10_900_000_000
        bench.routing = "visit 0: 3, 5"
        let text = bench.asText
        for field in ["Prefill", "Decode", "TTFT", "Memory limit", "Peak footprint", "Mapped weights", "Expert hit rate", "Thermal", "Routing", "31.5"] {
            #expect(text.contains(field), "missing \(field)")
        }
        let st = SelfTestReport(comparison: .init(top1Agreement: 1, meanKLD: 0, maxAbsLogitDiff: 0),
                                knownAnswers: [KnownAnswer(prompt: "What is 12 times 12?", output: "144", passed: true)])
        for field in ["Top-1", "KLD", "Max logit diff", "144", "PASS"] { #expect(st.asText.contains(field), "missing \(field)") }
    }

    @Test func routingStatsCountsEveryVisit() throws {
        let model = try Tiny.model()
        let config = try Tiny.config().text
        let stats = RoutingStats(visits: config.layerVisits, numExperts: config.numExperts)
        model.routing = stats
        _ = model(MLXArray(sequence).reshaped([1, -1]), cache: model.makeCache())
        for visit in config.layerVisits.indices {
            #expect(stats.hitHistogram(visit: visit).reduce(0, +) == sequence.count * config.expertsPerToken, "visit \(visit)")
        }
        let overlap = stats.passOverlap(layer: 2)
        #expect(overlap >= 0 && overlap <= 1)
        #expect(stats.passOverlap(layer: 0) == 1)  // not looped: one pass overlaps with itself
        #expect(!stats.summary.isEmpty)
    }
}
