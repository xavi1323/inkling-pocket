import Foundation
import MLX
import InklingCore
#if canImport(Darwin)
import Darwin
#endif

public enum MappedSafetensorsError: Error, Equatable {
    case cannotMap(String)
    case unknownTensor(String)
    case unsupportedDType(String)
}

/// Memory-maps every shard read-only. On repacked (16 KB-aligned) folders, `array(_:)` hands MLX
/// pointers into the mapping, which Metal wraps without copying; otherwise MLX copies the tensor.
public final class MappedSafetensors: @unchecked Sendable {
    private struct Mapping { let base: UnsafeMutableRawPointer; let size: Int }
    private var mappings: [Mapping] = []
    private var index: [String: (mapping: Int, entry: SafetensorsEntry)] = [:]
    /// Bytes mapped (sum of shard sizes).
    public private(set) var mappedBytes: Int64 = 0
    /// True when every shard was written by WeightRepacker (zero-copy capable).
    public private(set) var isAligned = true

    public convenience init(manifest: ModelManifest) throws {
        try self.init(files: manifest.shards)
    }

    public init(files: [URL]) throws {
        for url in files {
            let fd = open(url.path, O_RDONLY)
            guard fd >= 0 else { throw MappedSafetensorsError.cannotMap(url.lastPathComponent) }
            var st = stat()
            guard fstat(fd, &st) == 0, st.st_size > 0 else { close(fd); throw MappedSafetensorsError.cannotMap(url.lastPathComponent) }
            let size = Int(st.st_size)
            let ptr = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0)
            close(fd)
            guard let ptr, ptr != UnsafeMutableRawPointer(bitPattern: -1) else { throw MappedSafetensorsError.cannotMap(url.lastPathComponent) }
            let header = try SafetensorsHeader.read(url)
            if header.metadata[WeightRepacker.metadataKey] != "\(WeightRepacker.alignment)" { isAligned = false }
            mappings.append(Mapping(base: ptr, size: size))
            for e in header.entries { index[e.name] = (mappings.count - 1, e) }
            mappedBytes += Int64(size)
        }
    }

    deinit {
        for m in mappings { munmap(m.base, m.size) }
    }

    public var names: [String] { index.keys.sorted() }

    public func entry(_ name: String) -> SafetensorsEntry? { index[name]?.entry }

    /// Raw pointer to a tensor's bytes inside the mapping (for custom Metal kernels).
    public func pointer(_ name: String) throws -> UnsafeRawPointer {
        guard let (m, e) = index[name] else { throw MappedSafetensorsError.unknownTensor(name) }
        return UnsafeRawPointer(mappings[m].base + Int(e.start))
    }

    /// Whether `array(name)` can avoid copying (page-aligned start and whole pages inside the file).
    public func isZeroCopy(_ name: String) -> Bool {
        guard isAligned, let (m, e) = index[name] else { return false }
        return e.start % Int64(WeightRepacker.alignment) == 0 && e.start + e.paddedLength <= Int64(mappings[m].size)
    }

    public func array(_ name: String) throws -> MLXArray {
        guard let (m, e) = index[name] else { throw MappedSafetensorsError.unknownTensor(name) }
        let dtype = try Self.dtype(e.dtype)
        let ptr = mappings[m].base + Int(e.start)
        let count = e.shape.reduce(1, *)
        if isZeroCopy(name), Int(e.paddedLength) % dtype.size == 0 {
            // Wrap whole pages (Metal needs page-multiple lengths), then view the real elements.
            let whole = MLXArray(rawPointer: ptr, [Int(e.paddedLength) / dtype.size], dtype: dtype, finalizer: { [self] in _ = self })
            return whole[0..<count].reshaped(e.shape)
        }
        // Not repacked: MLX copies the bytes unless Metal can wrap them (a page-aligned start is enough on Apple), in
        // which case the array points into the mapping — so the finalizer must keep the mapping alive either way.
        return MLXArray(rawPointer: ptr, e.shape, dtype: dtype, finalizer: { [self] in _ = self })
    }

    static func dtype(_ name: String) throws -> DType {
        switch name {
        case "F32": .float32
        case "F16": .float16
        case "BF16": .bfloat16
        case "U32": .uint32
        case "I32": .int32
        case "U16": .uint16
        case "I16": .int16
        case "U8": .uint8
        case "I8": .int8
        case "I64": .int64
        case "U64": .uint64
        case "BOOL": .bool
        default: throw MappedSafetensorsError.unsupportedDType(name)
        }
    }
}
