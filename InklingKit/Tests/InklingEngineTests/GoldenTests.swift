import Foundation
import Testing
import MLX
import InklingCore
@testable import InklingEngine

/// Outputs of the Python reference (mlx-vlm d998a62, CPU) for the tiny random model; see scripts/make_tiny_inkling.py.
struct Golden: Decodable {
    let tokens: [Int]
    let vocab: Int
    let logits: [[Float]]
    let greedy: [Int]
    let visits: Int
}

enum Tiny {
    static var dir: URL { Bundle.module.url(forResource: "Fixtures", withExtension: nil)!.appending(path: "tiny-inkling") }
    static func config() throws -> InklingModelConfig { try InklingModelConfig.decode(Data(contentsOf: dir.appending(path: "config.json"))) }
    static func golden() throws -> Golden { try JSONDecoder().decode(Golden.self, from: Data(contentsOf: dir.appending(path: "golden.json"))) }
    static func model() throws -> Inkling2LanguageModel {
        let weights = try MappedSafetensors(files: [dir.appending(path: "model.safetensors")])
        return try Inkling2LanguageModel.load(config: config(), weights: weights)
    }
}

func argmax(_ row: some Collection<Float>) -> Int { row.enumerated().max { $0.element < $1.element }!.offset }

@Suite(.serialized) struct GoldenTests {
    init() { Device.setDefault(device: .cpu) }

    @Test func tinyLogitsMatchPython() throws {
        let g = try Tiny.golden()
        let model = try Tiny.model()
        let logits = model(MLXArray(g.tokens.map(Int32.init)).reshaped([1, -1]), cache: model.makeCache()).asType(.float32)
        #expect(logits.shape == [1, g.tokens.count, g.vocab])
        let ours = logits.asArray(Float.self)
        var maxDiff: Float = 0
        var topMismatches: [Int] = []
        for p in 0..<g.tokens.count {
            let row = ours[(p * g.vocab)..<((p + 1) * g.vocab)]
            for (a, b) in zip(row, g.logits[p]) { maxDiff = max(maxDiff, abs(a - b)) }
            if argmax(row) != argmax(g.logits[p]) { topMismatches.append(p) }
        }
        #expect(maxDiff <= 1e-2, "max abs logit diff \(maxDiff)")
        #expect(topMismatches.isEmpty, "top-1 differs at positions \(topMismatches)")
    }

    @Test func tinyGreedyMatchesPython() throws {
        let g = try Tiny.golden()
        let model = try Tiny.model()
        let cache = model.makeCache()
        var logits = model(MLXArray(g.tokens.map(Int32.init)).reshaped([1, -1]), cache: cache)
        var produced: [Int] = []
        for _ in 0..<g.greedy.count {
            let next = argMax(logits[0, -1], axis: -1).item(Int.self)
            produced.append(next)
            logits = model(MLXArray([Int32(next)]).reshaped([1, 1]), cache: cache)
        }
        #expect(produced == g.greedy)
    }

    @Test func loopPassesUseSeparateCaches() throws {
        let config = try Tiny.config()
        let model = try Tiny.model()
        let cache = model.makeCache()
        #expect(cache.count == config.text.layerVisits.count)
        #expect(cache.count == 16)
        #expect(Set(cache.map { ObjectIdentifier($0) }).count == cache.count)
        #expect(try Tiny.golden().visits == cache.count)
    }
}
