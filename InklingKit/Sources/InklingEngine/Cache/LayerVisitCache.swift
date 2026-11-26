import MLX

/// State for one layer visit (the looped core gets a separate cache per pass): attention K/V plus the four
/// short-convolution states (0 = k, 1 = v, 2 = attention output, 3 = MLP output).
public protocol LayerVisitCache: AnyObject {
    /// Tokens already stored (before the current call's update).
    var offset: Int { get }
    var conv: [MLXArray?] { get set }
    /// Appends `keys`/`values` [B,H,L,D] and returns everything attention should see, oldest first.
    func updateAndFetch(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
}

/// Saved cache position; restoring rewinds to it (used to reuse a chat's prompt prefix across turns).
public struct VisitSnapshot {
    let kv: KVSnapshot
    let conv: ConvStateCache
}

/// Sliding layers use a 1,024-token ring buffer; global layers a growing buffer.
public final class InklingVisitCache: LayerVisitCache {
    private let kv: KVStore
    private var convState = ConvStateCache()

    public init(window: Int?) {
        kv = window.map { RotatingKVCache(maxSize: $0) } ?? KVCache()
    }

    public var offset: Int { kv.offset }

    public var conv: [MLXArray?] {
        get { convState.slots }
        set { convState.slots = newValue }
    }

    public func updateAndFetch(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        kv.update(keys: keys, values: values)
    }

    public func snapshot() -> VisitSnapshot { VisitSnapshot(kv: kv.snapshot(), conv: convState) }

    public func restore(_ s: VisitSnapshot) {
        kv.restore(s.kv)
        convState = s.conv
    }

    /// Bytes held by K/V buffers (for the Benchmark screen).
    public var bytes: Int { kv.bytes }
}

protocol KVStore: AnyObject {
    var offset: Int { get }
    var bytes: Int { get }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    func snapshot() -> KVSnapshot
    func restore(_ s: KVSnapshot)
}

/// Rotating caches keep their (small) buffers; growing caches only remember the offset, because tokens before it are
/// never overwritten.
struct KVSnapshot {
    var keys: MLXArray?
    var values: MLXArray?
    var offset: Int
    var index: Int
}
