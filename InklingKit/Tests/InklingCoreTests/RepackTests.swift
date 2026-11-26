import Foundation
import Testing
@testable import InklingCore

/// Writes a plain safetensors file; tensors are (name, dtype, shape, bytes).
func writeSafetensors(_ url: URL, _ tensors: [(String, String, [Int], Data)]) throws {
    var header: [(String, JSONValue)] = []
    var data = Data()
    for (name, dtype, shape, bytes) in tensors {
        header.append((name, .object([("dtype", .string(dtype)), ("shape", .array(shape.map { .int(Int64($0)) })),
                                      ("data_offsets", .array([.int(Int64(data.count)), .int(Int64(data.count + bytes.count))]))])))
        data.append(bytes)
    }
    let json = Data(JSONValue.object(header).canonicalJSON(sortKeys: false).utf8)
    var out = Data()
    withUnsafeBytes(of: UInt64(json.count).littleEndian) { out.append(contentsOf: $0) }
    out.append(json)
    out.append(data)
    try out.write(to: url)
}

@Test func repackAlignsEveryTensorAndDropsVision() async throws {
    let dir = FileManager.default.temporaryDirectory.appending(path: "repack-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let a = Data((0..<24).map { UInt8($0) })            // F32 [2,3]
    let b = Data((0..<10).map { UInt8(100 + $0) })      // BF16 [5]
    let v = Data([9, 9, 9, 9])
    let src = dir.appending(path: "model-00001-of-00001.safetensors")
    try writeSafetensors(src, [("language_model.a", "F32", [2, 3], a), ("vision_tower.x", "F32", [1], v), ("language_model.b", "BF16", [5], b)])

    let out = dir.appending(path: "out")
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let map = try await WeightRepacker.repack(shards: [src], to: out, skipPrefixes: ["vision_tower."], progress: { _ in })
    #expect(map == ["language_model.a": "model-aligned-00001-of-00001.safetensors", "language_model.b": "model-aligned-00001-of-00001.safetensors"])

    let file = out.appending(path: "model-aligned-00001-of-00001.safetensors")
    let header = try SafetensorsHeader.read(file)
    #expect(header.entries.map(\.name) == ["language_model.a", "language_model.b"])
    let bytes = try Data(contentsOf: file)
    for e in header.entries {
        #expect(e.start % Int64(WeightRepacker.alignment) == 0)
        #expect(Int64(bytes.count) >= e.start + e.paddedLength)
        #expect(e.paddedLength % Int64(WeightRepacker.alignment) == 0)
    }
    #expect(bytes.subdata(in: Int(header.entries[0].start)..<Int(header.entries[0].end)) == a)
    #expect(bytes.subdata(in: Int(header.entries[1].start)..<Int(header.entries[1].end)) == b)
    #expect(header.entries[1].dtype == "BF16" && header.entries[1].shape == [5])
    #expect(try ModelFolder.expectedShardSize(file)! <= Int64(bytes.count))
}

@Test func importerProducesValidFolder() async throws {
    let src = FileManager.default.temporaryDirectory.appending(path: "import-src-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: src.appending(path: "tiktoken"), withIntermediateDirectories: true)
    for name in ModelFolder.requiredFiles where name != "model.safetensors.index.json" && name != "config.json" {
        try Data("x".utf8).write(to: src.appending(path: name))
    }
    try Data(#"{"model_type": "inkling2"}"#.utf8).write(to: src.appending(path: "config.json"))
    try writeSafetensors(src.appending(path: "model-00001-of-00002.safetensors"), [("language_model.a", "F32", [1], Data(count: 4))])
    try writeSafetensors(src.appending(path: "model-00002-of-00002.safetensors"), [("language_model.b", "F32", [2], Data(count: 8))])
    try Data(#"{"weight_map": {"language_model.a": "model-00001-of-00002.safetensors", "language_model.b": "model-00002-of-00002.safetensors"}}"#.utf8)
        .write(to: src.appending(path: "model.safetensors.index.json"))
    let manifest = try ModelFolder.validate(src)
    let dst = FileManager.default.temporaryDirectory.appending(path: "import-dst-\(UUID().uuidString)")
    let log = ProgressLog()
    try await ModelImporter.importModel(manifest, to: dst, progress: { log.add($0) })
    let imported = try ModelFolder.validate(dst)
    #expect(imported.shards.map(\.lastPathComponent) == ["model-aligned-00001-of-00002.safetensors", "model-aligned-00002-of-00002.safetensors"])
    #expect(log.all.last == 1.0)
    #expect(FileManager.default.fileExists(atPath: dst.appending(path: "tiktoken/tokenizer.model").path))
    #expect(try ModelImporter.isAligned(dst))
    #expect(try !ModelImporter.isAligned(src))
}
