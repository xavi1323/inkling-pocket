import Foundation
import Testing
@testable import InklingEngine

/// Records which experts were asked to be paged in.
final class FakePager: ExpertPager, @unchecked Sendable {
    private let lock = NSLock()
    private var _wanted: [ExpertKey] = []
    var wanted: [ExpertKey] { lock.withLock { _wanted } }
    func willNeed(_ keys: [ExpertKey]) { lock.withLock { _wanted += keys } }
}

private func k(_ l: Int, _ e: Int) -> ExpertKey { ExpertKey(layer: l, expert: e) }

@Suite struct ResidencyTests {
    func make(budget: Int64, pager: FakePager = FakePager()) -> ExpertResidency {
        ExpertResidency(budget: budget, expertBytes: { _ in 100 }, pager: pager)
    }

    func recordUsage(_ r: ExpertResidency) {
        for _ in 0..<5 { r.record(layer: 0, experts: [1]) }
        for _ in 0..<3 { r.record(layer: 0, experts: [2]) }
        for _ in 0..<4 { r.record(layer: 1, experts: [7]) }
        r.record(layer: 1, experts: [3])
    }

    @Test func pinsHottestWithinBudget() {
        let pager = FakePager()
        let r = make(budget: 250, pager: pager)
        recordUsage(r)
        r.rebalance()
        #expect(r.pinned == [k(0, 1), k(1, 7)])
        #expect(r.pinnedBytes == 200)
        #expect(Set(pager.wanted) == [k(0, 1), k(1, 7)])
        // A second rebalance with the same counts asks for nothing new.
        r.rebalance()
        #expect(pager.wanted.count == 2)
    }

    @Test func memoryWarningHalvesPinned() {
        let r = make(budget: 1_000)
        recordUsage(r)
        r.rebalance()
        #expect(r.pinned.count == 4)
        r.handleMemoryWarning()
        #expect(r.pinnedBytes <= 200)
        #expect(r.pinned == [k(0, 1), k(1, 7)])
        r.rebalance()  // the lower budget sticks
        #expect(r.pinnedBytes <= 200)
    }

    @Test func hitRateCountsPinnedHits() {
        let r = make(budget: 100)
        recordUsage(r)
        r.rebalance()
        r.resetHitRate()
        r.record(layer: 0, experts: [1, 2])
        r.record(layer: 1, experts: [7, 1])
        #expect(r.hitRate == 0.25)
    }

    @Test func prefetchSkipsPinned() {
        let pager = FakePager()
        let r = make(budget: 100, pager: pager)
        recordUsage(r)
        r.rebalance()
        r.prefetch(layer: 0, experts: [1, 2])
        #expect(pager.wanted == [k(0, 1), k(0, 2)])
    }

    @Test func budgetOverrideWins() {
        #expect(MemoryBudget.current(overrideGB: 2.5) == 2_500_000_000)
        #expect(MemoryBudget.current(overrideGB: nil) >= 0)
    }

    @Test func expertBytesFromTinyWeights() throws {
        let weights = try MappedSafetensors(files: [Tiny.dir.appending(path: "model.safetensors")])
        let config = try Tiny.config().text
        let sizes = ExpertResidency.expertSizes(weights: weights, config: config)
        #expect(sizes.count == config.numHiddenLayers)
        // gate/up/down weight + scales + biases, one expert's slice of each.
        let name = "language_model.model.layers.0.mlp.switch_mlp."
        let expected = ["gate_proj", "up_proj", "down_proj"].flatMap { p in ["weight", "scales", "biases"].map { "\(name)\(p).\($0)" } }
            .compactMap { weights.entry($0) }.map { ($0.end - $0.start) / Int64(config.numExperts) }.reduce(0, +)
        #expect(sizes[0] == expected)
        #expect(expected > 0)
    }
}
