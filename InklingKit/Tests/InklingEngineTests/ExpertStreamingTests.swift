import Foundation
import Testing
import MLX
import InklingCore
@testable import InklingEngine

/// Experts streamed through a few fixed slots must give the same numbers as the full expert tensors.
@Suite(.serialized) struct ExpertStreamingTests {
    init() { Device.setDefault(device: .cpu) }

    static func streamed(slots: Int) throws -> Inkling2LanguageModel {
        let weights = try MappedSafetensors(files: [Tiny.dir.appending(path: "model.safetensors")])
        return try Inkling2LanguageModel.load(config: Tiny.config(), weights: weights, expertSlots: slots)
    }

    let tokens = (0..<40).map { Int32(($0 * 37 + 11) % 500) }

    @Test func allSlotsGiveIdenticalTokens() throws {
        let experts = try Tiny.config().text.numExperts
        let full = TokenGenerator(model: try Tiny.model(), eos: [])
        let streamed = TokenGenerator(model: try Self.streamed(slots: experts), eos: [])
        let a = full.generate(prompt: tokens, sampler: Sampler(greedy()), maxTokens: 12, shouldStop: { false }, onToken: { _ in })
        let b = streamed.generate(prompt: tokens, sampler: Sampler(greedy()), maxTokens: 12, shouldStop: { false }, onToken: { _ in })
        #expect(a == b)
    }

    /// One token per pass: each pass fits the slots, so results are exact while experts are evicted all the time.
    @Test func fewSlotsWithEvictionsAreExact() throws {
        let full = TokenGenerator(model: try Tiny.model(), eos: [])
        let model = try Self.streamed(slots: 3)
        let streamed = TokenGenerator(model: model, eos: [])
        full.prefillChunk = 1
        streamed.prefillChunk = 1
        let a = full.prefill(tokens), b = streamed.prefill(tokens)
        #expect(abs(a - b).max().item(Float.self) == 0)
        let caches = model.expertCaches
        #expect(!caches.isEmpty)
        #expect(caches.allSatisfy { $0.misses > 3 && $0.hits > 0 })
        #expect(caches.dropFirst().contains { $0.prefetched > 0 })  // next visit's experts are read ahead
    }

    /// A long chunk needs more experts than there are slots: they are run in batches.
    @Test func batchedPrefillMatches() throws {
        let a = TokenGenerator(model: try Tiny.model(), eos: []).prefill(tokens)
        let b = TokenGenerator(model: try Self.streamed(slots: 3), eos: []).prefill(tokens)
        let scale = abs(a).max().item(Float.self)
        #expect(abs(a - b).max().item(Float.self) <= 1e-2 * max(scale, 1))
    }

    @Test func slotsHoldTheRightExpertAndEvictLeastRecent() throws {
        let weights = try MappedSafetensors(files: [Tiny.dir.appending(path: "model.safetensors")])
        let config = try Tiny.config()
        let path = "model.layers.0.mlp"
        let cache = try ExpertCache(path: path, weights: weights, quant: config.quantization, prefix: "language_model.", slots: 3)
        let full = try weights.array("language_model.\(path).switch_mlp.down_proj.weight")
        var map = cache.ensure([0, 1])
        #expect(cache.misses == 2)
        map = cache.ensure([1, 2])
        #expect(cache.hits == 1 && cache.misses == 3)
        _ = cache.ensure([3])  // expert 0 was used longest ago
        #expect(cache.slot(of: 0) == nil && cache.slot(of: 1) != nil && cache.slot(of: 3) != nil)
        map = cache.ensure([0])
        let slot = Int(map[0]!)
        #expect(cache.misses == 5)
        #expect(arrayEqual(cache.down.weight[slot], full[0]).item(Bool.self))
        #expect(arrayEqual(cache.down.weight[Int(map[0]!)], full[0]).item(Bool.self))
        #expect(cache.bytes == Int64(cache.slots) * ExpertCache.bytesPerExpert(path: path, weights: weights, prefix: "language_model.", experts: config.text.numExperts))
    }

    /// An expert used often stays even when newer experts were each needed once.
    @Test func frequentExpertSurvivesOneOffs() throws {
        let weights = try MappedSafetensors(files: [Tiny.dir.appending(path: "model.safetensors")])
        let cache = try ExpertCache(path: "model.layers.0.mlp", weights: weights, quant: try Tiny.config().quantization,
                                    prefix: "language_model.", slots: 3)
        for _ in 0..<5 { _ = cache.ensure([0]) }
        _ = cache.ensure([1])
        _ = cache.ensure([2])
        _ = cache.ensure([3])
        #expect(cache.slot(of: 0) != nil)
        #expect(cache.slot(of: 1) == nil)
    }

    /// Layers the loop visits three times get three times the slots, within the byte budget.
    @Test func slotPlanFavoursRepeatedLayers() {
        let perLayer: [Int64] = Array(repeating: 3_200_000, count: 6)
        let visits = [1, 3, 3, 3, 1, 1]
        let plan = InklingGenerator.slotPlan(budget: 400_000_000, perLayer: perLayer, visits: visits, experts: 128, topK: 8)
        #expect(plan[1] == plan[2] && plan[1] >= 3 * plan[0] - 3)
        #expect(zip(plan, perLayer).map { Int64($0) * $1 }.reduce(0, +) <= 400_000_000)
        #expect(InklingGenerator.slotPlan(budget: 0, perLayer: perLayer, visits: visits, experts: 128, topK: 8).allSatisfy { $0 == 8 })
        #expect(InklingGenerator.slotPlan(budget: .max, perLayer: perLayer, visits: visits, experts: 128, topK: 8).allSatisfy { $0 == 128 })
        // Slots a repeated layer cannot use (all experts already) go to the others.
        let big = InklingGenerator.slotPlan(budget: 3_200_000 * 600, perLayer: perLayer, visits: visits, experts: 128, topK: 8)
        #expect(big[1] == 128 && big[0] == 72)
    }

    @Test func streamedModelHoldsOnlySlots() throws {
        let model = try Self.streamed(slots: 3)
        let slotsPerLayer = model.expertCaches.map { $0.down.weight.dim(0) }
        #expect(slotsPerLayer.allSatisfy { $0 == 3 })
    }

}

/// iRAM+ lifts the per-app limit, so `os_proc_available_memory` can report most of RAM; the iPad then runs out
/// system-wide (jetsam log: app at 9.9 GB of 11.6 GB). The budget is capped by physical memory.
@Test func budgetIsCappedByPhysicalMemory() {
    let ipad: Int64 = 12_434_000_000  // 11.58 GB
    #expect(MemoryBudget.budget(available: 10_500_000_000, physical: ipad, overrideGB: nil) == Int64(Double(ipad) * 0.6))
    #expect(MemoryBudget.budget(available: 5_000_000_000, physical: ipad, overrideGB: nil) == 5_000_000_000 - MemoryBudget.safetyMargin)
    #expect(MemoryBudget.budget(available: nil, physical: ipad, overrideGB: nil) == MemoryBudget.fallback)
    #expect(MemoryBudget.budget(available: 10_500_000_000, physical: ipad, overrideGB: 3) == 3_000_000_000)
}

/// A new chat with the same instructions skips them: the cache state after the stable prefix is kept.
@Suite(.serialized) struct PrefixReuseTests {
    init() { Device.setDefault(device: .cpu) }

    @Test func newPromptWithSamePrefixSkipsIt() throws {
        let prefix = (0..<20).map { Int32(($0 * 13 + 5) % 500) }
        let a = prefix + [7, 8, 9, 10], b = prefix + [11, 12, 13]
        let gen = TokenGenerator(model: try Tiny.model(), eos: [])
        gen.prefillChunk = 8
        gen.prefill(a, stablePrefix: prefix.count)
        let reused = gen.prefill(b, stablePrefix: prefix.count)
        #expect(gen.stats.prefilledTokens == b.count - prefix.count)
        let fresh = TokenGenerator(model: try Tiny.model(), eos: [])
        fresh.prefillChunk = 8
        let expected = fresh.prefill(b, stablePrefix: prefix.count)
        #expect(abs(reused - expected).max().item(Float.self) == 0)
    }

    @Test func differentPrefixStartsOver() throws {
        let gen = TokenGenerator(model: try Tiny.model(), eos: [])
        gen.prefill([1, 2, 3, 4, 5], stablePrefix: 3)
        gen.prefill([9, 2, 3, 4], stablePrefix: 3)
        #expect(gen.stats.prefilledTokens == 4)
    }
}

/// The instructions' cache state survives an app restart: saved to disk, loaded into a new engine.
@Suite(.serialized) struct PromptCacheFileTests {
    init() { Device.setDefault(device: .cpu) }

    @Test func savedPrefixIsReusedByANewEngine() throws {
        let prefix = (0..<300).map { Int32(($0 * 13 + 5) % 500) }  // longer than a ring buffer step
        let a = prefix + [7, 8, 9, 10], b = prefix + [11, 12, 13]
        let url = try tempDir().appending(path: "prompt-cache.safetensors")
        let first = TokenGenerator(model: try Tiny.model(), eos: [])
        first.prefillChunk = 64
        first.prefill(a, stablePrefix: prefix.count)
        try first.savePrefix(to: url, identity: "tiny")

        let second = TokenGenerator(model: try Tiny.model(), eos: [])
        second.prefillChunk = 64
        #expect(try second.loadPrefix(from: url, identity: "other", ifPrefixOf: b) == false)
        #expect(try second.loadPrefix(from: url, identity: "tiny", ifPrefixOf: [1, 2, 3]) == false)
        #expect(try second.loadPrefix(from: url, identity: "tiny", ifPrefixOf: b) == true)
        let reused = second.prefill(b, stablePrefix: prefix.count)
        #expect(second.stats.prefilledTokens == b.count - prefix.count)

        let fresh = TokenGenerator(model: try Tiny.model(), eos: [])
        fresh.prefillChunk = 64
        let expected = fresh.prefill(b, stablePrefix: prefix.count)
        #expect(abs(reused - expected).max().item(Float.self) == 0)
    }
}
