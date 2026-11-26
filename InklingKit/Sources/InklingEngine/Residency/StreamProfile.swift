import Foundation

/// Where streaming time goes: waiting for the GPU to finish routing, waiting for experts to be read, and how many
/// expert reads (needed now vs. read ahead) there were. One per streamed model; read by the benchmark.
/// `@unchecked Sendable`: every field is behind `lock` (prefetch reads finish on background queues).
final class StreamProfile: @unchecked Sendable {
    struct Snapshot: Sendable, Equatable {
        var gpuWait: UInt64 = 0      // ns waiting for routing (all GPU work queued before it)
        var expertWait: UInt64 = 0   // ns `ensure` blocked (missing experts + a running read-ahead)
        var misses = 0               // experts read because a step needed them now
        var prefetched = 0           // experts read ahead
        var prefetchUsed = 0         // read-ahead experts a later step used
        var readBytes: Int64 = 0
        var readTime: UInt64 = 0     // ns spent in reads (summed over fills; fills of different layers overlap)

        static func - (a: Snapshot, b: Snapshot) -> Snapshot {
            Snapshot(gpuWait: a.gpuWait - b.gpuWait, expertWait: a.expertWait - b.expertWait, misses: a.misses - b.misses,
                     prefetched: a.prefetched - b.prefetched, prefetchUsed: a.prefetchUsed - b.prefetchUsed,
                     readBytes: a.readBytes - b.readBytes, readTime: a.readTime - b.readTime)
        }
    }

    private let lock = NSLock()
    private var s = Snapshot()

    var snapshot: Snapshot { lock.withLock { s } }
    func update(_ change: (inout Snapshot) -> Void) { lock.withLock { change(&s) } }

    static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

    /// Runs `body` and adds its duration to the field `key`.
    func time<T>(_ key: WritableKeyPath<Snapshot, UInt64>, _ body: () -> T) -> T {
        let start = Self.now()
        let result = body()
        let elapsed = Self.now() - start
        update { $0[keyPath: key] += elapsed }
        return result
    }
}

extension StreamProfile.Snapshot {
    /// One line: per token when `tokens` > 0, else totals.
    func summary(tokens: Int, seconds: Double) -> String {
        let n = Double(max(tokens, 1))
        let ms = { (ns: UInt64) in Double(ns) / 1e6 / n }
        let other = max(0, seconds * 1000 / n - ms(gpuWait) - ms(expertWait))
        let gbps = readTime > 0 ? Double(readBytes) / Double(readTime) : 0  // bytes per ns = GB/s
        let used = prefetched > 0 ? Double(prefetchUsed) / Double(prefetched) * 100 : 0
        return String(format: "GPU wait %.1f ms · expert wait %.1f ms · other %.1f ms · %.1f misses, %.1f read-ahead (%.0f%% used) · reads %.2f GB/s",
                      ms(gpuWait), ms(expertWait), other, Double(misses) / n, Double(prefetched) / n, used, gbps)
    }
}
