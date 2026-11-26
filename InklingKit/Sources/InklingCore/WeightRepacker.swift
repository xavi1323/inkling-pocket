import Foundation

public struct SafetensorsEntry: Sendable, Equatable {
    public var name: String
    public var dtype: String
    public var shape: [Int]
    /// Absolute byte offsets in the file.
    public var start: Int64
    public var end: Int64
    public var length: Int64 { end - start }
    /// Bytes reserved for the tensor (rounded up to the repack alignment in aligned files).
    public var paddedLength: Int64 { (length + Int64(WeightRepacker.alignment) - 1) / Int64(WeightRepacker.alignment) * Int64(WeightRepacker.alignment) }
}

public enum SafetensorsHeader {
    public static func read(_ url: URL) throws -> (entries: [SafetensorsEntry], metadata: [String: String]) {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let lenData = try handle.read(upToCount: 8), lenData.count == 8 else { throw CocoaError(.fileReadCorruptFile) }
        let headerLen = Int(lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian)
        guard headerLen < 100_000_000, let headerData = try handle.read(upToCount: headerLen), headerData.count == headerLen else {
            throw CocoaError(.fileReadCorruptFile)
        }
        guard case .object(let items) = try JSONValue.parse(String(decoding: headerData, as: UTF8.self)) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let base = Int64(8 + headerLen)
        var entries: [SafetensorsEntry] = []
        var metadata: [String: String] = [:]
        for (name, info) in items {
            if name == "__metadata__" {
                if case .object(let m) = info { for (k, v) in m { metadata[k] = v.stringValue } }
                continue
            }
            guard let dtype = info["dtype"]?.stringValue, case .array(let shapeValues)? = info["shape"],
                  case .array(let offs)? = info["data_offsets"], offs.count == 2,
                  case .int(let s) = offs[0], case .int(let e) = offs[1] else { throw CocoaError(.fileReadCorruptFile) }
            let shape = shapeValues.compactMap { if case .int(let d) = $0 { Int(d) } else { nil } }
            entries.append(SafetensorsEntry(name: name, dtype: dtype, shape: shape, start: base + s, end: base + e))
        }
        return (entries.sorted { $0.start < $1.start }, metadata)
    }
}

/// Rewrites safetensors shards so every tensor starts on a 16 KB page and owns whole pages.
/// Metal can then wrap each tensor of the memory-mapped file without copying (see docs/Engine/Notes.md).
public enum WeightRepacker {
    public static let alignment = 16_384
    public static let metadataKey = "inkling_alignment"

    static func align(_ n: Int64) -> Int64 { (n + Int64(alignment) - 1) / Int64(alignment) * Int64(alignment) }

    /// Returns the new weight map (tensor name → aligned shard file name).
    public static func repack(shards: [URL], to directory: URL, skipPrefixes: [String] = ["vision_tower."],
                              chunkSize: Int = 8 << 20, progress: @Sendable (Double) -> Void) async throws -> [String: String] {
        var plans: [(URL, [SafetensorsEntry])] = []
        for shard in shards {
            let entries = try SafetensorsHeader.read(shard).entries.filter { e in !skipPrefixes.contains { e.name.hasPrefix($0) } }
            plans.append((shard, entries))
        }
        let total = max(1, plans.reduce(Int64(0)) { $0 + $1.1.reduce(0) { $0 + $1.length } })
        var done: Int64 = 0
        var map: [String: String] = [:]
        for (index, (shard, entries)) in plans.enumerated() {
            let name = "model-aligned-\(String(format: "%05d", index + 1))-of-\(String(format: "%05d", plans.count)).safetensors"
            let out = directory.appending(path: name)
            // Offsets are relative to the data start, which is itself aligned, so the header doesn't depend on its own size.
            var header: [(String, JSONValue)] = [("__metadata__", .object([("format", .string("mlx")), (metadataKey, .string("\(alignment)"))]))]
            var offset: Int64 = 0
            var placed: [(SafetensorsEntry, Int64)] = []
            for e in entries {
                header.append((e.name, .object([("dtype", .string(e.dtype)), ("shape", .array(e.shape.map { .int(Int64($0)) })),
                                                ("data_offsets", .array([.int(offset), .int(offset + e.length)]))])))
                placed.append((e, offset))
                offset = align(offset + e.length)
                map[e.name] = name
            }
            var json = Data(JSONValue.object(header).canonicalJSON(sortKeys: false).utf8)
            let dataStart = align(Int64(8 + json.count))
            json.append(Data(repeating: 0x20, count: Int(dataStart) - 8 - json.count))

            guard FileManager.default.createFile(atPath: out.path, contents: nil) else { throw CocoaError(.fileWriteUnknown) }
            let output = try FileHandle(forWritingTo: out)
            let input = try FileHandle(forReadingFrom: shard)
            defer { try? output.close(); try? input.close() }
            var prefix = Data()
            withUnsafeBytes(of: UInt64(json.count).littleEndian) { prefix.append(contentsOf: $0) }
            try output.write(contentsOf: prefix + json)
            for (e, rel) in placed {
                try Task.checkCancellation()
                try output.seek(toOffset: UInt64(dataStart + rel))
                try input.seek(toOffset: UInt64(e.start))
                var remaining = e.length
                while remaining > 0 {
                    try Task.checkCancellation()
                    let n = Int(min(Int64(chunkSize), remaining))
                    guard let chunk = try input.read(upToCount: n), chunk.count == n else { throw CocoaError(.fileReadCorruptFile) }
                    try output.write(contentsOf: chunk)
                    remaining -= Int64(n)
                    done += Int64(n)
                    progress(min(Double(done) / Double(total), 0.999_999))
                    await Task.yield()
                }
            }
            try output.truncate(atOffset: UInt64(dataStart + offset))
        }
        return map
    }
}

/// Copies a validated model folder into app storage with its weights repacked for zero-copy loading.
public enum ModelImporter {
    public static func importModel(_ manifest: ModelManifest, to destination: URL,
                                   progress: @escaping @Sendable (Double) -> Void) async throws {
        let fm = FileManager.default
        let partial = URL(fileURLWithPath: destination.path + ".partial")
        try? fm.removeItem(at: partial)
        try fm.createDirectory(at: partial, withIntermediateDirectories: true)
        // Small files first (config, tokenizer, template, tiktoken/…).
        let shardNames = Set(manifest.shards.map(\.lastPathComponent))
        for file in try ModelCopier.regularFiles(in: manifest.folder)
        where !shardNames.contains(file.relative) && file.relative != "model.safetensors.index.json" && !file.relative.hasSuffix(".safetensors") {
            let target = partial.appending(path: file.relative)
            try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try fm.copyItem(at: file.url, to: target)
        }
        let map = try await WeightRepacker.repack(shards: manifest.shards, to: partial, progress: progress)
        let index = JSONValue.object([
            ("metadata", .object([("format", .string("mlx")), (WeightRepacker.metadataKey, .string("\(WeightRepacker.alignment)"))])),
            ("weight_map", .object(map.keys.sorted().map { ($0, .string(map[$0]!)) })),
        ])
        try Data(index.canonicalJSON(sortKeys: false).utf8).write(to: partial.appending(path: "model.safetensors.index.json"))
        try Task.checkCancellation()
        if fm.fileExists(atPath: destination.path) { try fm.removeItem(at: destination) }
        try fm.moveItem(at: partial, to: destination)
        progress(1.0)
    }

    /// True when the folder's shards were written by `WeightRepacker`.
    public static func isAligned(_ folder: URL) throws -> Bool {
        let manifest = try ModelFolder.validate(folder)
        return try manifest.shards.allSatisfy { try SafetensorsHeader.read($0).metadata[WeightRepacker.metadataKey] == "\(WeightRepacker.alignment)" }
    }
}
