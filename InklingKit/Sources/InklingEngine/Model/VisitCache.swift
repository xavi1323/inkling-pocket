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

/// Minimal reference cache: grows for global layers; for sliding layers keeps the last `window - 1` old entries plus
/// the new ones (same visible keys as mlx-vlm's RotatingKVCache, in temporal order). Task 17 adds the fast caches.
public final class SimpleVisitCache: LayerVisitCache {
    public let window: Int?
    public private(set) var offset = 0
    public var conv: [MLXArray?] = [nil, nil, nil, nil]
    private var keys: MLXArray?
    private var values: MLXArray?

    public init(window: Int?) { self.window = window }

    public func updateAndFetch(keys k: MLXArray, values v: MLXArray) -> (MLXArray, MLXArray) {
        if var oldK = keys, var oldV = values {
            if let window, oldK.dim(2) > window - 1 {
                let start = oldK.dim(2) - (window - 1)
                oldK = oldK[0..., 0..., start...]
                oldV = oldV[0..., 0..., start...]
            }
            keys = concatenated([oldK, k], axis: 2)
            values = concatenated([oldV, v], axis: 2)
        } else {
            keys = k
            values = v
        }
        offset += k.dim(2)
        return (keys!, values!)
    }
}
