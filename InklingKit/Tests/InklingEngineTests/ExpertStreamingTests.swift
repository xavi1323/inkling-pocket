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

    @Test func streamedModelHoldsOnlySlots() throws {
        let model = try Self.streamed(slots: 3)
        let slotsPerLayer = model.expertCaches.map { $0.down.weight.dim(0) }
        #expect(slotsPerLayer.allSatisfy { $0 == 3 })
    }

    @Test func slotCountFitsTheBudget() throws {
        let config = try Tiny.config().text
        let perSlot: Int64 = 76_000_000  // real model: one expert in every layer
        #expect(InklingGenerator.slots(budget: 3_800_000_000, perSlot: perSlot, config: config) == min(50, config.numExperts))
        #expect(InklingGenerator.slots(budget: -1, perSlot: perSlot, config: config) == config.expertsPerToken)
        #expect(InklingGenerator.slots(budget: .max, perSlot: perSlot, config: config) == config.numExperts)
    }
}

/// iRAM+ lifts the per-app limit, so `os_proc_available_memory` can report most of RAM; the iPad then runs out
/// system-wide (jetsam log: app at 9.9 GB of 11.6 GB). The budget is capped by physical memory.
@Test func budgetIsCappedByPhysicalMemory() {
    let ipad: Int64 = 12_434_000_000  // 11.58 GB
    #expect(MemoryBudget.budget(available: 10_500_000_000, physical: ipad, overrideGB: nil) == Int64(Double(ipad) * MemoryBudget.physicalShare))
    #expect(MemoryBudget.budget(available: 5_000_000_000, physical: ipad, overrideGB: nil) == 5_000_000_000 - MemoryBudget.safetyMargin)
    #expect(MemoryBudget.budget(available: nil, physical: ipad, overrideGB: nil) == MemoryBudget.fallback)
    #expect(MemoryBudget.budget(available: 10_500_000_000, physical: ipad, overrideGB: 3) == 3_000_000_000)
}
