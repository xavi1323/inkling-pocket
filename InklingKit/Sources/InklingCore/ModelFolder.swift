import Foundation

public struct ModelManifest: Sendable, Equatable {
    public let folder: URL
    public let shards: [URL]
    public let totalBytes: Int64
    public let modelType: String
}

public enum ModelFolderError: Error, Equatable {
    case missingFile(String)
    case wrongArchitecture(String)
    case unreadableIndex
    case missingShard(String)
    case truncatedShard(String)

    public var userMessage: String {
        switch self {
        case .missingFile(let f): "The folder is missing \(f). Pick the whole Inkling-2-Mini-abliterated-MLX-12GB folder."
        case .wrongArchitecture(let t): "This is a \(t) model, not Inkling 2. Pick the Inkling-2-Mini-abliterated-MLX-12GB folder."
        case .unreadableIndex: "model.safetensors.index.json can't be read. Copy the folder again."
        case .missingShard(let f): "\(f) is missing. The download or copy didn't finish."
        case .truncatedShard(let f): "\(f) is incomplete (file is shorter than it should be). Copy it again."
        }
    }
}

public enum ModelFolder {
    public static let requiredFiles = [
        "config.json", "model.safetensors.index.json", "tokenizer.json",
        "tiktoken/tokenizer.model", "chat_template.jinja", "tokenizer_config.json",
    ]

    /// Checks the folder without reading weights: required files, model type, every shard present and full length.
    public static func validate(_ url: URL) throws -> ModelManifest {
        let fm = FileManager.default
        for name in requiredFiles where !fm.fileExists(atPath: url.appending(path: name).path(percentEncoded: false)) {
            throw ModelFolderError.missingFile(name)
        }
        guard let configData = try? Data(contentsOf: url.appending(path: "config.json")),
              case let config? = try? JSONValue.parse(String(decoding: configData, as: UTF8.self)),
              let modelType = config["model_type"]?.stringValue else {
            throw ModelFolderError.missingFile("config.json")
        }
        guard modelType.hasPrefix("inkling2") else { throw ModelFolderError.wrongArchitecture(modelType) }

        guard let indexData = try? Data(contentsOf: url.appending(path: "model.safetensors.index.json")),
              let index = try? JSONSerialization.jsonObject(with: indexData) as? [String: Any],
              let weightMap = index["weight_map"] as? [String: String] else {
            throw ModelFolderError.unreadableIndex
        }
        var total: Int64 = 0
        var shards: [URL] = []
        for name in Set(weightMap.values).sorted() {
            let shardURL = url.appending(path: name)
            guard let attrs = try? fm.attributesOfItem(atPath: shardURL.path(percentEncoded: false)),
                  let size = (attrs[.size] as? NSNumber)?.int64Value else {
                throw ModelFolderError.missingShard(name)
            }
            guard let expected = try expectedShardSize(shardURL), size >= expected else {
                throw ModelFolderError.truncatedShard(name)
            }
            total += size
            shards.append(shardURL)
        }
        return ModelManifest(folder: url, shards: shards, totalBytes: total, modelType: modelType)
    }

    /// 8 + header length + largest data offset, read from the safetensors header only.
    static func expectedShardSize(_ url: URL) throws -> Int64? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let lenData = try handle.read(upToCount: 8), lenData.count == 8 else { return nil }
        let headerLen = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        guard headerLen < 100_000_000, let headerData = try handle.read(upToCount: Int(headerLen)),
              headerData.count == Int(headerLen),
              let header = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else { return nil }
        var maxEnd: Int64 = 0
        for (key, value) in header where key != "__metadata__" {
            if let info = value as? [String: Any], let offsets = info["data_offsets"] as? [NSNumber], offsets.count == 2 {
                maxEnd = max(maxEnd, offsets[1].int64Value)
            }
        }
        return 8 + Int64(headerLen) + maxEnd
    }
}
