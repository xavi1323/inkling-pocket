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
