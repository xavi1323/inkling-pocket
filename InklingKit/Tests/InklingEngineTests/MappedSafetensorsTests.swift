import Foundation
import Testing
import MLX
import InklingCore
@testable import InklingEngine

func writePlainSafetensors(_ url: URL, _ tensors: [(String, String, [Int], Data)]) throws {
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

func floats(_ v: [Float]) -> Data { v.withUnsafeBufferPointer { Data(buffer: $0) } }

func tempDir() throws -> URL {
    let dir = FileManager.default.temporaryDirectory.appending(path: "mapped-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

@Test func mapsTinySafetensors() throws {
    let file = try tempDir().appending(path: "w.safetensors")
    try writePlainSafetensors(file, [("a", "F32", [2, 3], floats([1, 2, 3, 4, 5, 6])), ("b", "U32", [2], Data([7, 0, 0, 0, 9, 0, 0, 0]))])
    let st = try MappedSafetensors(files: [file])
    #expect(st.names == ["a", "b"])
    #expect(!st.isAligned)
    let a = try st.array("a")
    #expect(a.shape == [2, 3])
    #expect(a.asArray(Float.self) == [1, 2, 3, 4, 5, 6])
    #expect(try st.array("b").asArray(UInt32.self) == [7, 9])
    #expect(throws: MappedSafetensorsError.unknownTensor("c")) { try st.array("c") }
}

@Test func mappedBytesEqualsFileSize() throws {
    let file = try tempDir().appending(path: "w.safetensors")
    try writePlainSafetensors(file, [("a", "F32", [4], floats([1, 2, 3, 4]))])
    let size = try FileManager.default.attributesOfItem(atPath: file.path)[.size] as! NSNumber
    #expect(try MappedSafetensors(files: [file]).mappedBytes == size.int64Value)
}

@Test func mapsRepackedSafetensorsZeroCopy() async throws {
    let dir = try tempDir()
    let src = dir.appending(path: "model-00001-of-00001.safetensors")
    let big = (0..<5000).map { Float($0) }
    try writePlainSafetensors(src, [("x", "F32", [5000], floats(big)), ("y", "BF16", [3], Data([0x80, 0x3F, 0x00, 0x40, 0x40, 0x40]))])
    _ = try await WeightRepacker.repack(shards: [src], to: dir, progress: { _ in })
    let st = try MappedSafetensors(files: [dir.appending(path: "model-aligned-00001-of-00001.safetensors")])
    #expect(st.isAligned)
    #expect(st.isZeroCopy("x") && st.isZeroCopy("y"))
    #expect(try st.array("x").asArray(Float.self) == big)
    #expect(try st.array("y").asType(.float32).asArray(Float.self) == [1, 2, 3])
}

/// Arrays must stay valid after the `MappedSafetensors` object is gone (macOS NaNs came from freed mappings).
@Test func arraysOutliveMapping() throws {
    let url = Tiny.dir.appending(path: "model.safetensors")
    let expected = try MappedSafetensors(files: [url]).names.sorted().map { name in
        (name, try MappedSafetensors(files: [url]).array(name).asType(.float32).sum().item(Float.self))
    }
    var arrays: [(String, MLXArray)] = []
    do {
        let weights = try MappedSafetensors(files: [url])
        arrays = try weights.names.sorted().map { ($0, try weights.array($0)) }
    }
    for _ in 0..<3 { _ = try MappedSafetensors(files: [url]).array(expected[0].0).sum().item(Float.self) }
    for ((name, array), (_, sum)) in zip(arrays, expected) {
        let value = array.asType(.float32).sum().item(Float.self)
        #expect(value == sum || (value.isNaN && sum.isNaN), "\(name)")
    }
}
