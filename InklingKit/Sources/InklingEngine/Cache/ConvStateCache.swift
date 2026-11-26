import MLX

/// Last `kernel - 1` FP32 inputs of each short convolution in a layer visit (k, v, attention output, MLP output).
public struct ConvStateCache {
    public var slots: [MLXArray?] = [nil, nil, nil, nil]
    public init() {}
}
