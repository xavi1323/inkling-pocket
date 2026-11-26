import Foundation
import NaturalLanguage

public protocol TextEmbedding: Sendable {
    /// Mean-pooled sentence vector, or nil when the model isn't available (assets not downloaded yet).
    func embed(_ text: String) -> [Float]?
}

/// Apple's on-device contextual embedding (no network once its assets are installed).
public final class NLEmbedder: TextEmbedding, @unchecked Sendable {
    private let lock = NSLock()
    private let model: NLContextualEmbedding?
    private var loaded = false
    private var requested = false

    public init() { model = NLContextualEmbedding(language: .english) }

    public func embed(_ text: String) -> [Float]? {
        lock.withLock {
            guard let model, !text.isEmpty else { return nil }
            if !loaded {
                guard model.hasAvailableAssets else {
                    if !requested {
                        requested = true
                        model.requestAssets { _, _ in }
                    }
                    return nil
                }
                do { try model.load() } catch { return nil }
                loaded = true
            }
            guard let result = try? model.embeddingResult(for: text, language: .english) else { return nil }
            var sum = [Double](repeating: 0, count: model.dimension)
            var n = 0
            result.enumerateTokenVectors(in: text.startIndex..<text.endIndex) { vector, _ in
                for i in 0..<min(vector.count, sum.count) { sum[i] += vector[i] }
                n += 1
                return true
            }
            guard n > 0 else { return nil }
            return sum.map { Float($0 / Double(n)) }
        }
    }
}

extension Array where Element == Float {
    var data: Data { withUnsafeBufferPointer { Data(buffer: $0) } }
    init(data: Data) {
        self = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
}
