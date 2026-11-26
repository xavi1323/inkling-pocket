// Ported from mlx-vlm cache.py RotatingKVCache (MIT) at d998a62, with keep = 0.
import MLX

/// Sliding-window K/V: a ring of `maxSize` slots for single-token decode; multi-token chunks keep the last
/// `maxSize - 1` old entries so every new token still sees a full window. `update` returns keys in temporal order.
final class RotatingKVCache: KVStore {
    let maxSize: Int
    static let step = 256
    private var keys: MLXArray?
    private var values: MLXArray?
    private(set) var offset = 0
    private var index = 0

    init(maxSize: Int) { self.maxSize = maxSize }

    var bytes: Int { (keys?.nbytes ?? 0) + (values?.nbytes ?? 0) }

    func update(keys k: MLXArray, values v: MLXArray) -> (MLXArray, MLXArray) {
        let (rk, rv) = k.dim(2) == 1 ? updateInPlace(k, v) : updateConcat(k, v)
        return (temporalOrder(rk), temporalOrder(rv))
    }

    func updateForDecode(keys k: MLXArray, values v: MLXArray) -> DecodeKV {
        _ = updateInPlace(k, v)
        // Below the window the buffer may be padded past `offset`; once full it is exactly the window, a ring.
        return DecodeKV(keys: keys!, values: values!, valid: min(offset, keys!.dim(2)), index: index)
    }

    private func trim(_ size: Int, _ a: MLXArray, append: MLXArray? = nil) -> MLXArray {
        var parts = [size > 0 ? a[0..., 0..., size..., 0...] : a]
        if let append { parts.append(append) }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 2)
    }

    private func temporalOrder(_ a: MLXArray) -> MLXArray {
        if index == a.dim(2) { return a }
        if index < offset { return concatenated([a[0..., 0..., index..., 0...], a[0..., 0..., ..<index, 0...]], axis: 2) }
        return a[0..., 0..., ..<index, 0...]
    }

    private func updateConcat(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        if let oldK = keys, let oldV = values {
            let orderedK = temporalOrder(oldK), orderedV = temporalOrder(oldV)
            index = orderedK.dim(2)
            let trimSize = index - maxSize + 1
            keys = trim(trimSize, orderedK, append: k)
            values = trim(trimSize, orderedV, append: v)
        } else {
            keys = k
            values = v
        }
        offset += k.dim(2)
        index = keys!.dim(2)
        return (keys!, values!)
    }

    private func updateInPlace(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        let s = k.dim(2), previous = offset
        if keys == nil || (previous >= keys!.dim(2) && keys!.dim(2) < maxSize) {
            let size = min(Self.step, maxSize - previous)
            let newK = MLXArray.zeros([k.dim(0), k.dim(1), size, k.dim(3)], dtype: k.dtype)
            let newV = MLXArray.zeros([v.dim(0), v.dim(1), size, v.dim(3)], dtype: v.dtype)
            if let oldK = keys, let oldV = values {
                keys = concatenated([oldK, newK], axis: 2)
                values = concatenated([oldV, newV], axis: 2)
            } else {
                keys = newK
                values = newV
            }
            index = previous
        }
        let trimSize = keys!.dim(2) - maxSize
        if trimSize > 0 {
            keys = trim(trimSize, keys!)
            values = trim(trimSize, values!)
            index = maxSize
        }
        if index == maxSize { index = 0 }
        keys![0..., 0..., index..<(index + s), 0...] = k
        values![0..., 0..., index..<(index + s), 0...] = v
        offset += s
        index += s
        if offset < maxSize { return (keys![0..., 0..., ..<offset, 0...], values![0..., 0..., ..<offset, 0...]) }
        return (keys!, values!)
    }

    func snapshot() -> KVSnapshot {
        // Fresh array objects: later in-place writes replace `keys`' contents but not the snapshot's.
        KVSnapshot(keys: keys.map { $0[0...] }, values: values.map { $0[0...] }, offset: offset, index: index)
    }

    /// The last write went to slot `index - 1` (after any growth or trim), and the entry it replaced lies outside the
    /// window of the token that will be written there again.
    func rewindOne() {
        offset -= 1
        index -= 1
    }

    func restore(_ s: KVSnapshot) {
        keys = s.keys
        values = s.values
        offset = s.offset
        index = s.index
    }

    func export(_ s: KVSnapshot) -> KVSnapshot { s }
    func load(_ s: KVSnapshot) { restore(s) }
}
