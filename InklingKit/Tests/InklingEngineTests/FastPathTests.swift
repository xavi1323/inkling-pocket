import Foundation
import Testing
import MLX
@testable import InklingEngine
#if canImport(Metal)
import Metal
#endif

@Suite(.serialized) struct FastPathTests {
    init() { Device.setDefault(device: .cpu) }
    let tokens: [Int32] = [1, 17, 42, 99, 256, 3, 499, 7, 77, 128, 311, 64]

    func input(_ model: Inkling2LanguageModel, seed: UInt64) -> MLXArray {
        let dtype = model.model.embedNorm.dtype
        return (MLXRandom.normal([1, 1, model.config.hiddenSize], key: MLXRandom.key(seed)) * 0.5).asType(dtype)
    }

    @Test func fusedMoEMatchesReference() throws {
        let model = try Tiny.model()
        for (i, layer) in model.model.layers.enumerated() {
            let x = input(model, seed: UInt64(i))
            let reference = layer.moe(x).asType(.float32)
            let fused = layer.moe(x, fast: true).asType(.float32)
            #expect(fused.shape == reference.shape)
            let diff = abs(fused - reference).max().item(Float.self)
            #expect(diff <= 2e-2, "layer \(i): \(diff)")
        }
    }

    /// The Metal kernel itself; only where a real Apple GPU is present (the iPad Self-test covers the real model).
    @Test func fusedKernelMatchesFallbackOnGPU() throws {
        #if canImport(Metal)
        guard let device = MTLCreateSystemDefaultDevice(), device.supportsFamily(.apple7) else { return }
        let model = try Tiny.model()
        let layer = model.model.layers[3]
        let x = input(model, seed: 9)
        let (ids, weights) = layer.moe.route(x)
        let cpu = layer.moe.fused.fallback(x.reshaped([-1]), ids.reshaped([-1]), weights.reshaped([-1])).asType(.float32)
        let gpu = Device.withDefaultDevice(.gpu) {
            layer.moe.fused.kernel(x.reshaped([-1]), ids.reshaped([-1]), weights.reshaped([-1])).asType(.float32)
        }
        #expect(abs(gpu - cpu).max().item(Float.self) <= 2e-2)
        #endif
    }

    /// Token-by-token logits of both paths. GPU only: on the CPU the reference `gather_qmm` rounds more than the FP32
    /// fallback (≈3.5× one bf16 step) and the tiny random model amplifies that into different logits. On the CPU the
    /// fast path is checked by `pipelinedGenerateMatchesGolden` instead.
    @Test func fastPathLogitsMatchReference() throws {
        #if canImport(Metal)
        guard let device = MTLCreateSystemDefaultDevice(), device.supportsFamily(.apple7) else { return }
        try Device.withDefaultDevice(.gpu) { try fastPathLogitsMatch() }
        #endif
    }

    func fastPathLogitsMatch() throws {
        let model = try Tiny.model()
        let fast = FastPath(model)
        let refCache = model.makeCache(), fastCache = fast.makeCache()
        for token in tokens {
            let t = MLXArray([token]).reshaped([1, 1])
            let a = model(t, cache: refCache)[0, -1]
            let b = fast(t, cache: fastCache)[0, -1]
            #expect(argMax(a).item(Int32.self) == argMax(b).item(Int32.self))
            #expect(abs(a - b).max().item(Float.self) < 0.1, "\(token): \(abs(a - b).max().item(Float.self))")
        }
    }

    /// Single-token steps through the compiled layer blocks (fewer, fused GPU kernels) give the reference logits.
    @Test func compiledStepsMatchReference() throws {
        DecoderLayer.splitOnCPU = true
        compile(enable: false)  // traced, not fused: no CPU JIT here (the iPad Self-test checks the fused kernels)
        defer {
            DecoderLayer.splitOnCPU = false
            compile(enable: true)
        }
        let model = try Tiny.model()
        let refCache = model.makeCache(), compiledCache = model.makeCache()
        let prompt = MLXArray(tokens.prefix(5)).reshaped([1, -1])
        _ = model(prompt, cache: refCache)
        _ = model(prompt, cache: compiledCache)
        for token in tokens.dropFirst(5) {
            let t = MLXArray([token]).reshaped([1, 1])
            let a = model(t, cache: refCache)[0, -1]
            let b = model(t, cache: compiledCache, fast: false, compiled: true)[0, -1]
            let scale = max(abs(a).max().item(Float.self), 1)
            #expect(abs(a - b).max().item(Float.self) <= 1e-3 * scale, "\(token): \(abs(a - b).max().item(Float.self))")
        }
        #expect(model.model.layers.allSatisfy { $0.compiledSteps > 0 })
    }

    /// Single-token attention straight over the cache buffers (no mask or sink copies), including the sliding
    /// window's ring after it wraps and after a prompt longer than the window.
    @Test func decodeAttentionOverCacheMatchesReference() throws {
        DecoderLayer.splitOnCPU = true
        compile(enable: false)
        defer {
            DecoderLayer.splitOnCPU = false
            compile(enable: true)
        }
        let model = try Tiny.model()
        let window = try Tiny.config().text.slidingWindowSize
        for promptLength in [3, window + 3] {
            let all = (0..<(promptLength + window + 6)).map { Int32(($0 * 53 + 7) % 500) }
            let refCache = model.makeCache(), testCache = model.makeCache()
            let prompt = MLXArray(Array(all.prefix(promptLength))).reshaped([1, -1])
            _ = model(prompt, cache: refCache)
            _ = model(prompt, cache: testCache)
            for token in all.dropFirst(promptLength) {
                let t = MLXArray([token]).reshaped([1, 1])
                let a = model(t, cache: refCache)[0, -1]
                let b = model(t, cache: testCache, fast: false, compiled: true)[0, -1]
                let scale = max(abs(a).max().item(Float.self), 1)
                #expect(abs(a - b).max().item(Float.self) <= 1e-3 * scale, "prompt \(promptLength), \(token)")
            }
        }
        #expect(model.model.layers.allSatisfy { $0.attention.cacheSteps > 0 })
    }

    @Test func gpuSamplerGreedyEqualsArgmax() {
        let logits = MLXRandom.normal([1, 1, 500], key: MLXRandom.key(4))
        let token = GPUSampler(Sampler(greedy())).sample(logits, history: nil)
        #expect(token.item(Int32.self) == argMax(logits.reshaped([-1])).item(Int32.self))
    }

    @Test func gpuSamplerPenaltyMatchesSampler() {
        var p = greedy()
        p.repetitionPenalty = 1.5
        let logits = MLXRandom.normal([500], key: MLXRandom.key(5)) * 0.1
        let top = argMax(logits).item(Int32.self)
        let history: [Int32] = [top, top, 3]
        let cpu = Sampler(p).sample(logits, history: history).item(Int32.self)
        let gpu = GPUSampler(Sampler(p)).sample(logits, history: MLXArray(history)).item(Int32.self)
        #expect(gpu == cpu)
    }

    @Test func pipelinedGenerateMatchesGolden() throws {
        let golden = try Tiny.golden()
        let gen = TokenGenerator(model: FastPath(try Tiny.model()), eos: [])
        gen.pipelined = true
        let out = gen.generate(prompt: golden.tokens.map(Int32.init), sampler: Sampler(greedy()), maxTokens: golden.greedy.count,
                               shouldStop: { false }, onToken: { _ in })
        #expect(out.map(Int.init) == golden.greedy)
        #expect(gen.stats.generatedTokens == golden.greedy.count)
    }

    @Test func pipelinedStopsAtEOSLikePlain() {
        func run(_ pipelined: Bool) -> [Int32] {
            let gen = TokenGenerator(model: ScriptModel(script: [5, 6, 7, 8, 9], vocab: 500), eos: [8])
            gen.pipelined = pipelined
            return gen.generate(prompt: [1, 2], sampler: Sampler(greedy()), maxTokens: 10, shouldStop: { false }, onToken: { _ in })
        }
        #expect(run(true) == run(false))
        #expect(run(true) == [5, 6, 7])
    }

    @Test func collectorKeepsInFlightPass() {
        let c = RouteCollector()
        c.add(visit: 0, experts: MLXArray([Int32(1), 2]))
        c.endPass()
        c.add(visit: 0, experts: MLXArray([Int32(3)]))
        c.endPass()
        let done = c.drain(keepingLatest: 1)
        #expect(done.map(\.experts) == [[1, 2]])
        #expect(c.drain(keepingLatest: 0).map(\.experts) == [[3]])
    }
}

/// Differs from its base only on single-token steps, like the fused MoE.
final class StepOnlyDifferent: CausalLM {
    let base: Inkling2LanguageModel
    init(_ base: Inkling2LanguageModel) { self.base = base }
    func callAsFunction(_ tokens: MLXArray, cache: [LayerVisitCache]?) -> MLXArray {
        let out = base(tokens, cache: cache)
        return tokens.dim(1) == 1 ? -out : out
    }
    func makeCache() -> [LayerVisitCache] { base.makeCache() }
}

@Suite(.serialized) struct SelfTestStepTests {
    init() { Device.setDefault(device: .cpu) }

    /// The Self-test must drive the fast path one token at a time, the way decoding does.
    @Test func selfTestComparesSingleTokenSteps() throws {
        let model = try Tiny.model()
        let seq: [Int32] = [1, 17, 42, 99, 256, 3, 499, 7, 77, 128, 311, 64]
        let c = SelfTest.compare(reference: model, fast: StepOnlyDifferent(model), sequences: [seq], scored: 8)
        #expect(c.top1Agreement < 0.97)
        let same = SelfTest.compare(reference: model, fast: model, sequences: [seq], scored: 8)
        #expect(same.top1Agreement == 1)
    }
}

/// Long prompt chunks attend in query blocks (bounded mask memory); same logits as one block.
extension FastPathTests {
    @Test func queryBlockedAttentionMatches() throws {
        let prompt = (0..<40).map { Int32(($0 * 37 + 11) % 500) }
        func logits(block: Int) throws -> MLXArray {
            let saved = Attention.queryBlock
            Attention.queryBlock = block
            defer { Attention.queryBlock = saved }
            let gen = TokenGenerator(model: try Tiny.model(), eos: [])
            gen.prefillChunk = 40
            return gen.prefill(prompt).asType(.float32)
        }
        let whole = try logits(block: 4096), blocked = try logits(block: 6)
        #expect(Attention.queryBlockedCalls > 0)
        #expect(abs(whole - blocked).max().item(Float.self) <= 1e-3 * max(abs(whole).max().item(Float.self), 1))
    }
}

/// A prompt chunk's last-position logits, without the LM head over every position.
extension FastPathTests {
    @Test func lastLogitsMatchFullLogits() throws {
        let prompt = MLXArray((0..<12).map { Int32(($0 * 37 + 11) % 500) }).reshaped([1, -1])
        let model = try Tiny.model()
        let full = model(prompt, cache: model.makeCache())[0, -1]
        let last = model.lastLogits(prompt, cache: model.makeCache())
        #expect(last.shape == full.shape)
        #expect(abs(full - last).max().item(Float.self) <= 1e-4)
        let fast = FastPath(model), turbo = TurboPath(model)
        #expect(abs(fast.lastLogits(prompt, cache: fast.makeCache()) - full).max().item(Float.self) <= 1e-3)
        #expect(abs(turbo.lastLogits(prompt, cache: turbo.makeCache()) - full).max().item(Float.self) <= 1e-3)
    }
}

/// Turbo's guard: a token that had to swap too much routing weight is rewound and run exact. With the limits at
/// zero every token is redone, so turbo gives exactly the fast path's tokens even with few slots.
extension FastPathTests {
    @Test func turboRedoneTokensMatchExact() throws {
        let prompt = (0..<40).map { Int32(($0 * 37 + 11) % 500) }
        let saved = (TurboPath.maxVisitMiss, TurboPath.maxTokenMiss)
        TurboPath.maxVisitMiss = -1
        TurboPath.maxTokenMiss = -1
        defer { (TurboPath.maxVisitMiss, TurboPath.maxTokenMiss) = saved }
        let exact = try ExpertStreamingTests.streamed(slots: 3)
        let turbo = try ExpertStreamingTests.streamed(slots: 3)
        let a = TokenGenerator(model: FastPath(exact), eos: [])
            .generate(prompt: prompt, sampler: Sampler(greedy()), maxTokens: 10, shouldStop: { false }, onToken: { _ in })
        let b = TokenGenerator(model: TurboPath(turbo), eos: [])
            .generate(prompt: prompt, sampler: Sampler(greedy()), maxTokens: 10, shouldStop: { false }, onToken: { _ in })
        #expect(a == b)
        #expect(turbo.turboRedone == 9)
    }

    /// A decode step can be undone: rewinding and repeating it gives the same logits (ring wrap included).
    @Test func decodeStepRewinds() throws {
        let model = try Tiny.model()
        let fast = FastPath(model)
        let cache = fast.makeCache()
        _ = fast.lastLogits(MLXArray((0..<20).map { Int32($0 * 7 % 500) }).reshaped([1, -1]), cache: cache)
        for t in [Int32(5), 9, 13] {
            let points = cache.map { ($0 as! InklingVisitCache).checkpoint() }
            let first = fast(MLXArray([t]).reshaped([1, 1]), cache: cache)[0, -1]
            eval(first)
            for (c, p) in zip(cache, points) { #expect((c as! InklingVisitCache).rewind(to: p)) }
            let again = fast(MLXArray([t]).reshaped([1, 1]), cache: cache)[0, -1]
            #expect(abs(first - again).max().item(Float.self) <= 1e-5)
        }
    }
}

/// Speculative decoding: a draft guesses ahead, the exact model checks every guess in one pass, and only tokens the
/// exact model would have produced are kept. The output is the exact path's whatever the draft guesses.
extension FastPathTests {
    /// Drafts the token after the one the model would pick: every guess is wrong, so every check rewinds.
    final class WrongDraft: CausalLM {
        let inner: CausalLM
        init(_ inner: CausalLM) { self.inner = inner }
        func callAsFunction(_ tokens: MLXArray, cache: [LayerVisitCache]?) -> MLXArray {
            let logits = inner(tokens, cache: cache)
            return concatenated([logits[.ellipsis, (logits.dim(-1) - 1)...], logits[.ellipsis, ..<(logits.dim(-1) - 1)]], axis: -1)
        }
        func makeCache() -> [LayerVisitCache] { inner.makeCache() }
    }

    func speculative(_ draft: (Inkling2LanguageModel) -> CausalLM, slots: Int?, prompt: [Int32], tokens: Int,
                     check: (TokenGenerator) -> Void = { _ in }) throws -> [Int32] {
        let model = try slots.map { try ExpertStreamingTests.streamed(slots: $0) } ?? Tiny.model()
        let gen = TokenGenerator(model: FastPath(model), eos: [])
        gen.draft = draft(model)
        let out = gen.generate(prompt: prompt, sampler: Sampler(greedy()), maxTokens: tokens, shouldStop: { false }, onToken: { _ in })
        check(gen)
        return out
    }

    /// The reference path: on the CPU its single-token steps match a whole-sequence pass, which the fused fast
    /// steps do not (see `fastPathLogitsMatchReference`); a check is a multi-token pass.
    func exact(slots: Int?, prompt: [Int32], tokens: Int) throws -> [Int32] {
        let model = try slots.map { try ExpertStreamingTests.streamed(slots: $0) } ?? Tiny.model()
        return TokenGenerator(model: model, eos: [])
            .generate(prompt: prompt, sampler: Sampler(greedy()), maxTokens: tokens, shouldStop: { false }, onToken: { _ in })
    }

    // 40 prompt tokens: past the tiny model's 8-token sliding window, so checks rewind a wrapped ring.
    var specPrompt: [Int32] { (0..<40).map { Int32(($0 * 37 + 11) % 500) } }

    @Test func speculativeWithPerfectDraftMatchesExactAndKeepsDrafts() throws {
        let want = try exact(slots: nil, prompt: specPrompt, tokens: 24)
        let got = try speculative({ $0 }, slots: nil, prompt: specPrompt, tokens: 24) { gen in
            #expect(gen.stats.draftedTokens > 0)
            #expect(gen.stats.acceptedTokens == gen.stats.draftedTokens)
        }
        #expect(got == want)
    }

    @Test func speculativeWithWrongDraftMatchesExact() throws {
        let want = try exact(slots: nil, prompt: specPrompt, tokens: 24)
        let got = try speculative({ WrongDraft(FastPath($0)) }, slots: nil, prompt: specPrompt, tokens: 24) { gen in
            #expect(gen.stats.draftedTokens > 0)
        }
        #expect(got == want)
    }

    @Test func speculativeTurboDraftMatchesExactWithFewSlots() throws {
        let want = try exact(slots: 3, prompt: specPrompt, tokens: 24)
        let got = try speculative({ TurboPath($0, redo: false) }, slots: 3, prompt: specPrompt, tokens: 24) { gen in
            #expect(gen.stats.draftedTokens > 0)
        }
        #expect(got == want)
    }

    /// Stops exactly at `maxTokens` even when a check accepts more.
    @Test func speculativeStopsAtMaxTokens() throws {
        for n in [1, 2, 5] {
            let got = try speculative({ $0 }, slots: nil, prompt: specPrompt, tokens: n)
            #expect(got == (try exact(slots: nil, prompt: specPrompt, tokens: n)))
        }
    }

    /// Text the model copies from the prompt is drafted by looking it up, with no draft model run.
    @Test func promptLookupDraftsRepeatedText() {
        let context: [Int32] = [5, 6, 7, 8, 9, 1, 2, 5, 6, 7]
        #expect(TokenGenerator.lookupDraft(context, max: 3) == [8, 9, 1])
        #expect(TokenGenerator.lookupDraft(context, max: 10) == [8, 9, 1, 2, 5, 6, 7])
        #expect(TokenGenerator.lookupDraft([1, 2, 3, 4], max: 4) == [])
    }
}
