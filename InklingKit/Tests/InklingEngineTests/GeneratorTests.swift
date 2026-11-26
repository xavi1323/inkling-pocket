import Foundation
import Testing
import MLX
import InklingCore
import AgentCore
@testable import InklingEngine

/// Emits a fixed token script (one-hot logits), counting forward passes.
final class ScriptModel: CausalLM, @unchecked Sendable {
    let script: [Int32]
    let vocab: Int
    private let lock = NSLock()
    private var step = 0
    private(set) var calls = 0
    var callCount: Int { lock.withLock { calls } }

    init(script: [Int32], vocab: Int) { self.script = script; self.vocab = vocab }

    func callAsFunction(_ tokens: MLXArray, cache: [LayerVisitCache]?) -> MLXArray {
        let next = lock.withLock { () -> Int32 in
            calls += 1
            defer { step += 1 }
            return script[min(step, script.count - 1)]
        }
        var row = [Float](repeating: -10, count: vocab)
        row[Int(next)] = 10
        let last = MLXArray(row).reshaped([1, 1, vocab])
        let length = tokens.dim(1)
        return length == 1 ? last : concatenated([MLXArray.zeros([1, length - 1, vocab]), last], axis: 1)
    }

    func makeCache() -> [LayerVisitCache] { [] }
}

enum TinyTokenizer {
    static func load() throws -> InklingTokenizer {
        try InklingTokenizer(modelFolder: Bundle.module.url(forResource: "Fixtures", withExtension: nil)!.appending(path: "tiny-tokenizer"))
    }
}

func greedy() -> GenerateParameters {
    var p = GenerateParameters()
    p.temperature = 0
    return p
}

@Suite(.serialized) struct GeneratorTests {
    init() { Device.setDefault(device: .cpu) }

    @Test func chunkedPrefillEqualsSinglePass() throws {
        let tokens: [Int32] = [1, 17, 42, 99, 256, 3, 499, 7, 77, 128, 311, 64]
        let whole = TokenGenerator(model: try Tiny.model(), eos: [])
        whole.prefillChunk = 12
        let chunked = TokenGenerator(model: try Tiny.model(), eos: [])
        chunked.prefillChunk = 4
        let a = whole.prefill(tokens), b = chunked.prefill(tokens)
        #expect(abs(a - b).max().item(Float.self) < 1e-3)
    }

    @Test func prefixReuseSkipsSharedTokens() throws {
        let first: [Int32] = [1, 17, 42, 99, 256, 3, 499, 7]
        let second = first + [77, 128, 311, 64]
        let gen = TokenGenerator(model: try Tiny.model(), eos: [])
        _ = gen.generate(prompt: first, sampler: Sampler(greedy()), maxTokens: 3, shouldStop: { false }, onToken: { _ in })
        #expect(gen.stats.prefilledTokens == first.count)
        let reused = gen.prefill(second)
        #expect(gen.stats.prefilledTokens == 4)
        let fresh = TokenGenerator(model: try Tiny.model(), eos: []).prefill(second)
        #expect(abs(reused - fresh).max().item(Float.self) < 1e-3)
    }

    @Test func differentPromptStartsOver() throws {
        let gen = TokenGenerator(model: try Tiny.model(), eos: [])
        _ = gen.prefill([1, 2, 3, 4])
        let other = gen.prefill([1, 2, 9, 4, 5])
        #expect(gen.stats.prefilledTokens == 5)
        let fresh = TokenGenerator(model: try Tiny.model(), eos: []).prefill([1, 2, 9, 4, 5])
        #expect(abs(other - fresh).max().item(Float.self) < 1e-3)
    }

    @Test func generateMatchesGolden() throws {
        let g = try Tiny.golden()
        let gen = TokenGenerator(model: try Tiny.model(), eos: [])
        let out = gen.generate(prompt: g.tokens.map(Int32.init), sampler: Sampler(greedy()), maxTokens: g.greedy.count,
                               shouldStop: { false }, onToken: { _ in })
        #expect(out.map(Int.init) == g.greedy)
        #expect(gen.stats.generatedTokens == g.greedy.count)
    }

    @Test func stopsAtEOS200006() async throws {
        let tokenizer = try TinyTokenizer.load()
        #expect(tokenizer.eosID == 200_006)
        let script = tokenizer.encode("<|message_model|><|content_text|>Hi<|end_message|>") + [200_006] + tokenizer.encode("more")
        let model = ScriptModel(script: script, vocab: tokenizer.vocabularySize)
        let gen = InklingGenerator(model: model, tokenizer: tokenizer)
        var events: [ModelEvent] = []
        for try await e in gen.generate([ChatMessage(role: .user, content: "hello")], tools: [], effort: .low, parameters: greedy()) {
            events.append(e)
        }
        let answer = events.compactMap { if case .answer(let t) = $0 { t } else { nil } }.joined()
        #expect(answer == "Hi")
        #expect(events.last == .end)
        #expect(model.callCount == 6)  // 5 tokens + EOS; "more" is never generated
        let stats = await gen.stats
        #expect(stats.generatedTokens == 5)
    }

    @Test func cancelStopsWithinOneToken() async throws {
        let tokenizer = try TinyTokenizer.load()
        let script = tokenizer.encode("<|message_model|><|content_text|>" + String(repeating: "word ", count: 200))
        let model = ScriptModel(script: script, vocab: tokenizer.vocabularySize)
        let gen = InklingGenerator(model: model, tokenizer: tokenizer)
        var seen = 0
        for try await e in gen.generate([ChatMessage(role: .user, content: "go")], tools: [], effort: .low, parameters: greedy()) {
            if case .answer = e { seen += 1 }
            if seen == 3 { break }
        }
        let atCancel = model.callCount
        try await Task.sleep(for: .milliseconds(300))
        #expect(model.callCount - atCancel <= 1)
        #expect(model.callCount < script.count)
    }

    @Test func cancelMidStreamSavesPartial() async throws {
        let tokenizer = try TinyTokenizer.load()
        let script = tokenizer.encode("<|message_model|><|content_text|>Partial answer that never ends")
        let gen = InklingGenerator(model: ScriptModel(script: script, vocab: tokenizer.vocabularySize), tokenizer: tokenizer)
        var state = ChatTurnState()
        for try await e in gen.generate([ChatMessage(role: .user, content: "go")], tools: [], effort: .low, parameters: greedy()) {
            if case .answer(let t) = e { ChatSessionLogic.apply(.answer(t), to: &state) }
            if state.answer.count >= 7 { break }
        }
        ChatSessionLogic.stop(&state)
        #expect(state.stopped)
        let saved = ChatSessionLogic.messagesToSave(state)
        #expect(saved.count == 1)
        #expect(saved.first?.content.hasPrefix("Partial") == true)
    }
}

@Suite(.serialized) struct PrefillStopTests {
    init() { Device.setDefault(device: .cpu) }

    @Test func stopDuringPrefillReturnsEarly() {
        let model = ScriptModel(script: [5], vocab: 500)
        let gen = TokenGenerator(model: model, eos: [])
        gen.prefillChunk = 256
        var checks = 0
        let out = gen.generate(prompt: Array(repeating: 3, count: 600), sampler: Sampler(greedy()), maxTokens: 4,
                               shouldStop: { checks += 1; return model.callCount >= 1 }, onToken: { _ in })
        #expect(out.isEmpty)
        #expect(model.callCount == 1)
        #expect(checks > 0)
    }
}

@Suite(.serialized) struct LongPromptTests {
    init() { Device.setDefault(device: .cpu) }

    /// ≥ 64 routed rows take the sorted-experts path (real model: any prompt of 8+ tokens). It must match the
    /// unsorted path, which short chunks use.
    @Test func longPromptSortedPathMatchesShortChunks() throws {
        let tokens = (0..<40).map { Int32(($0 * 37 + 11) % 500) }
        let whole = TokenGenerator(model: try Tiny.model(), eos: [])
        whole.prefillChunk = 40
        let chunked = TokenGenerator(model: try Tiny.model(), eos: [])
        chunked.prefillChunk = 8
        let a = whole.prefill(tokens), b = chunked.prefill(tokens)
        #expect(abs(a - b).max().item(Float.self) < 1e-2)
    }
}
