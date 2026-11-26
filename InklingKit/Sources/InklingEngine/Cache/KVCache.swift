import MLX

/// Growing K/V buffer for global-attention visits; capacity grows in steps of 256 tokens and writes go in place.
final class KVCache: KVStore {
    static let step = 256
    private var keys: MLXArray?
    private var values: MLXArray?
    private(set) var offset = 0

    var capacity: Int { keys?.dim(2) ?? 0 }
    var bytes: Int { (keys?.nbytes ?? 0) + (values?.nbytes ?? 0) }

    func update(keys k: MLXArray, values v: MLXArray) -> (MLXArray, MLXArray) {
        let previous = offset, n = k.dim(2)
        if previous + n > capacity {
            let size = (previous + n + Self.step - 1) / Self.step * Self.step
            let (b, h) = (k.dim(0), k.dim(1))
            let newK = MLXArray.zeros([b, h, size, k.dim(3)], dtype: k.dtype)
            let newV = MLXArray.zeros([b, h, size, v.dim(3)], dtype: v.dtype)
            if let keys, let values, previous > 0 {
                newK[0..., 0..., ..<previous, 0...] = keys[0..., 0..., ..<previous, 0...]
                newV[0..., 0..., ..<previous, 0...] = values[0..., 0..., ..<previous, 0...]
            }
            keys = newK
            values = newV
        }
        keys![0..., 0..., previous..<(previous + n), 0...] = k
        values![0..., 0..., previous..<(previous + n), 0...] = v
        offset += n
        return (keys![0..., 0..., ..<offset, 0...], values![0..., 0..., ..<offset, 0...])
    }

    func snapshot() -> KVSnapshot { KVSnapshot(offset: offset, index: offset) }
    func restore(_ s: KVSnapshot) { offset = s.offset }

    func export(_ s: KVSnapshot) -> KVSnapshot {
        guard let keys, let values, s.offset > 0 else { return KVSnapshot(offset: 0, index: 0) }
        return KVSnapshot(keys: keys[0..., 0..., ..<s.offset, 0...], values: values[0..., 0..., ..<s.offset, 0...],
                          offset: s.offset, index: s.offset)
    }

    func load(_ s: KVSnapshot) {
        keys = s.keys
        values = s.values
        offset = s.keys == nil ? 0 : s.offset
    }
}
