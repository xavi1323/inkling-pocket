import Foundation
import Testing
import MLX
@testable import InklingEngine

/// Keys whose single feature is the token position, so fetched order is easy to read.
private func positions(_ range: Range<Int>) -> MLXArray {
    MLXArray(range.map { Float($0) }).reshaped([1, 1, range.count, 1])
}
private func values(_ a: MLXArray) -> [Int] { a.reshaped([-1]).asArray(Float.self).map { Int($0) } }

@Suite(.serialized) struct CacheTests {
    init() { Device.setDefault(device: .cpu) }

    @Test func rotatingCacheKeepsLast1024() throws {
        // The tiny model's window is 8; the real one is 1024 (both come from sliding_window_size).
        let window = try Tiny.config().text.slidingWindowSize
        #expect(window == 8)
        let cache = RotatingKVCache(maxSize: window)
        let (k, _) = cache.update(keys: positions(0..<12), values: positions(0..<12))
        #expect(values(k) == Array(0..<12))  // a long prompt chunk sees all its own tokens
        for p in 12..<30 {
            let (k, v) = cache.update(keys: positions(p..<(p + 1)), values: positions(p..<(p + 1)))
            #expect(values(k) == Array((p - window + 1)...p), "position \(p)")
            #expect(values(v) == values(k))
        }
        #expect(cache.offset == 30)
    }

    @Test func rotatingCacheShortPromptThenDecode() {
        let cache = RotatingKVCache(maxSize: 8)
        _ = cache.update(keys: positions(0..<3), values: positions(0..<3))
        for p in 3..<11 {
            let (k, _) = cache.update(keys: positions(p..<(p + 1)), values: positions(p..<(p + 1)))
            #expect(values(k) == Array(max(0, p - 7)...p), "position \(p)")
        }
        // A second chunk after rotation keeps window - 1 old keys plus the chunk.
        let (k, _) = cache.update(keys: positions(11..<14), values: positions(11..<14))
        #expect(values(k) == Array(4..<14))
    }

    @Test func globalCacheGrowsInSteps() {
        let cache = KVCache()
        _ = cache.update(keys: positions(0..<200), values: positions(0..<200))
        let (k, _) = cache.update(keys: positions(200..<300), values: positions(200..<300))
        #expect(values(k) == Array(0..<300))
        #expect(cache.capacity == 512)
        for p in 300..<303 { _ = cache.update(keys: positions(p..<(p + 1)), values: positions(p..<(p + 1))) }
        #expect(cache.offset == 303)
    }

    @Test func snapshotRestoreRewinds() {
        let visit = InklingVisitCache(window: 8)
        let global = InklingVisitCache(window: nil)
        for c in [visit, global] {
            _ = c.updateAndFetch(keys: positions(0..<10), values: positions(0..<10))
            c.conv[0] = MLXArray([Float(1)])
        }
        let snaps = [visit.snapshot(), global.snapshot()]
        for c in [visit, global] {
            for p in 10..<13 { _ = c.updateAndFetch(keys: positions(p..<(p + 1)), values: positions(p..<(p + 1))) }
            c.conv[0] = MLXArray([Float(2)])
        }
        visit.restore(snaps[0])
        global.restore(snaps[1])
        #expect(visit.offset == 10 && global.offset == 10)
        #expect(values(visit.conv[0]!) == [1])
        let (kw, _) = visit.updateAndFetch(keys: positions(50..<51), values: positions(50..<51))
        #expect(values(kw) == [3, 4, 5, 6, 7, 8, 9, 50])
        let (kg, _) = global.updateAndFetch(keys: positions(50..<51), values: positions(50..<51))
        #expect(values(kg) == Array(0..<10) + [50])
    }
}
