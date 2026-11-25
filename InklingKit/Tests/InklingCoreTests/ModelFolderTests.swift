import Foundation
import Testing
@testable import InklingCore

/// Builds a tiny fake model folder in a temp dir.
struct FakeModelFolder {
    let url: URL

    init(modelType: String = "inkling2_loop", shardDataBytes: Int = 16, truncateBy: Int = 0,
         skip: Set<String> = []) throws {
        url = FileManager.default.temporaryDirectory.appending(path: "fake-model-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url.appending(path: "tiktoken"), withIntermediateDirectories: true)
        func write(_ name: String, _ text: String) throws {
            if skip.contains(name) { return }
            try text.write(to: url.appending(path: name), atomically: true, encoding: .utf8)
        }
        try write("config.json", #"{"model_type":"\#(modelType)"}"#)
        try write("tokenizer.json", "{}")
        try write("tiktoken/tokenizer.model", "IQ== 0\n")
        try write("chat_template.jinja", "x")
        try write("tokenizer_config.json", "{}")
        let shard = "model-00001-of-00001.safetensors"
        try write("model.safetensors.index.json", #"{"weight_map":{"a":"\#(shard)","b":"\#(shard)"}}"#)
        if !skip.contains(shard) {
            let header = #"{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]},"b":{"dtype":"U8","shape":[\#(shardDataBytes - 8)],"data_offsets":[8,\#(shardDataBytes)]}}"#
            var data = Data()
            var len = UInt64(header.utf8.count).littleEndian
            data.append(Data(bytes: &len, count: 8))
            data.append(Data(header.utf8))
            data.append(Data(count: shardDataBytes - truncateBy))
            try data.write(to: url.appending(path: shard))
        }
    }
}

@Test func validFolderPasses() throws {
    let f = try FakeModelFolder()
    let m = try ModelFolder.validate(f.url)
    #expect(m.shards.count == 1)
    #expect(m.modelType == "inkling2_loop")
    #expect(m.totalBytes > 16)
}

@Test func missingConfigReported() throws {
    let f = try FakeModelFolder(skip: ["config.json"])
    #expect(throws: ModelFolderError.missingFile("config.json")) { try ModelFolder.validate(f.url) }
}

@Test func wrongModelTypeReported() throws {
    let f = try FakeModelFolder(modelType: "llama")
    #expect(throws: ModelFolderError.wrongArchitecture("llama")) { try ModelFolder.validate(f.url) }
}

@Test func missingShardReported() throws {
    let f = try FakeModelFolder(skip: ["model-00001-of-00001.safetensors"])
    #expect(throws: ModelFolderError.missingShard("model-00001-of-00001.safetensors")) { try ModelFolder.validate(f.url) }
}

@Test func truncatedShardDetected() throws {
    let f = try FakeModelFolder(truncateBy: 3)
    #expect(throws: ModelFolderError.truncatedShard("model-00001-of-00001.safetensors")) { try ModelFolder.validate(f.url) }
}

@Test func userMessagesAreReadable() {
    #expect(ModelFolderError.truncatedShard("x.safetensors").userMessage.contains("x.safetensors"))
    #expect(ModelFolderError.missingFile("config.json").userMessage.contains("config.json"))
}

@Test(.enabled(if: hasModelDir, "set INKLING_MODEL_DIR"))
func realModelFolderValidates() throws {
    let m = try ModelFolder.validate(URL(fileURLWithPath: modelDir!))
    #expect(m.shards.count == 11)
    #expect(m.modelType == "inkling2_loop")
    #expect(m.totalBytes > 11_000_000_000)
}
