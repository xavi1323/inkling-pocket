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

/// Whole K/V buffers [1,H,capacity,D] after a single-token write. Slot `j < valid` holds the token
/// `(index - 1 - j + valid) % valid` steps before the newest; slots from `valid` on are unused.
struct DecodeKV {
    let keys, values: MLXArray
    let valid: Int
    let index: Int

    /// Steps back from the newest token for every slot (slots past `valid` get garbage; mask them).
    func distances() -> MLXArray {
        let j = MLXArray(Int32(0)..<Int32(keys.dim(2)))
        return MLXArray(Int32(index - 1)) - j + MLXArray(Int32(valid)) * (j .>= Int32(index)).asType(.int32)
    }
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

    /// Single token: writes `keys`/`values` [1,H,1,D] in place and returns the whole buffers (no copies).
    func updateForDecode(keys: MLXArray, values: MLXArray) -> DecodeKV {
        kv.updateForDecode(keys: keys, values: values)
    }

    public func snapshot() -> VisitSnapshot { VisitSnapshot(kv: kv.snapshot(), conv: convState) }

    public func restore(_ s: VisitSnapshot) {
        kv.restore(s.kv)
        convState = s.conv
    }

    /// Writes the state `s` stands for into `arrays`/`meta` under `key`.
    func export(_ s: VisitSnapshot, key: String, arrays: inout [String: MLXArray], meta: inout [String: String]) {
        let kv = self.kv.export(s.kv)
        if let k = kv.keys, let v = kv.values { arrays[key + ".k"] = k; arrays[key + ".v"] = v }
        for (i, c) in s.conv.slots.enumerated() { if let c { arrays[key + ".c\(i)"] = c } }
        meta[key] = "\(kv.offset),\(kv.index)"
    }

    /// Loads what `export` wrote; false when it is missing.
    func load(key: String, arrays: [String: MLXArray], meta: [String: String]) -> Bool {
        guard let m = meta[key]?.split(separator: ",").compactMap({ Int($0) }), m.count == 2 else { return false }
        kv.load(KVSnapshot(keys: arrays[key + ".k"], values: arrays[key + ".v"], offset: m[0], index: m[1]))
        var state = ConvStateCache()
        for i in 0..<state.slots.count { state.slots[i] = arrays[key + ".c\(i)"] }
        convState = state
        return true
    }

    /// Bytes held by K/V buffers (for the Benchmark screen).
    public var bytes: Int { kv.bytes }

    /// Before a single-token step: what `rewind` needs to undo it (no buffer copies).
    struct Checkpoint {
        let offset: Int
        let conv: ConvStateCache
    }

    func checkpoint() -> Checkpoint { Checkpoint(offset: kv.offset, conv: convState) }

    /// Whether the step after `p` can be undone (exactly one token written since).
    func canRewind(to p: Checkpoint) -> Bool { kv.offset == p.offset + 1 }

    /// After a multi-token call under `ConvContext.keepAll` that wrote `written` tokens: keeps the first `kept` of
    /// them (K/V and conv state as if only those had been fed).
    func keep(_ kept: Int, of written: Int) {
        precondition(kept >= 1 && kept <= written)
        kv.rewind(written - kept)
        for (i, slot) in convState.slots.enumerated() {
            guard let slot else { continue }
            let width = slot.dim(1) - written  // kernel - 1
            convState.slots[i] = slot[0..., kept..<(kept + width), 0...]
        }
    }

    /// Undoes the single-token step after `p`; false (nothing changed) when it is not exactly one step back.
    @discardableResult
    func rewind(to p: Checkpoint) -> Bool {
        guard canRewind(to: p) else { return false }
        kv.rewindOne()
        convState = p.conv
        return true
    }
}

protocol KVStore: AnyObject {
    var offset: Int { get }
    var bytes: Int { get }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    func updateForDecode(keys: MLXArray, values: MLXArray) -> DecodeKV
    func snapshot() -> KVSnapshot
    func restore(_ s: KVSnapshot)
    /// Forgets the last single-token write (the next write goes where it went).
    func rewindOne()
    /// Forgets the last `n` tokens of a multi-token write (`update` with more than one token).
    func rewind(_ n: Int)
    /// The K/V a snapshot stands for, as standalone arrays (for saving to disk).
    func export(_ s: KVSnapshot) -> KVSnapshot
    /// Replaces the contents with exported K/V.
    func load(_ s: KVSnapshot)
}

/// Rotating caches keep their (small) buffers; growing caches only remember the offset, because tokens before it are
/// never overwritten.
struct KVSnapshot {
    var keys: MLXArray?
    var values: MLXArray?
    var offset: Int
    var index: Int
}
