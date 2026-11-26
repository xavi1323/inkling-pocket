import Foundation
import Testing
@testable import InklingCore

final class ProgressLog: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Double] = []
    func add(_ v: Double) { lock.withLock { values.append(v) } }
    var all: [Double] { lock.withLock { values } }
}

func makeSource(files: Int, bytesEach: Int) throws -> URL {
    let dir = FileManager.default.temporaryDirectory.appending(path: "copier-src-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir.appending(path: "tiktoken"), withIntermediateDirectories: true)
    for i in 0..<files {
        try Data(repeating: UInt8(i), count: bytesEach).write(to: dir.appending(path: "model-\(i).safetensors"))
    }
    try Data("ranks".utf8).write(to: dir.appending(path: "tiktoken/tokenizer.model"))
    return dir
}

@Test func copyReportsProgressToOne() async throws {
    let src = try makeSource(files: 3, bytesEach: 300_000)
    let dst = FileManager.default.temporaryDirectory.appending(path: "copier-dst-\(UUID().uuidString)")
    let log = ProgressLog()
    try await ModelCopier.copy(from: src, to: dst, chunkSize: 64 << 10, progress: { log.add($0) })
    #expect(log.all.last == 1.0)
    #expect(log.all == log.all.sorted())
    #expect(try Data(contentsOf: dst.appending(path: "model-2.safetensors")) == Data(repeating: 2, count: 300_000))
    #expect(try Data(contentsOf: dst.appending(path: "tiktoken/tokenizer.model")) == Data("ranks".utf8))
    #expect(!FileManager.default.fileExists(atPath: dst.path + ".partial"))
}

@Test func cancelLeavesNoFinalFolder() async throws {
    let src = try makeSource(files: 4, bytesEach: 1_000_000)
    let dst = FileManager.default.temporaryDirectory.appending(path: "copier-dst-\(UUID().uuidString)")
    await #expect(throws: CancellationError.self) {
        try await ModelCopier.copy(from: src, to: dst, chunkSize: 64 << 10, progress: { p in
            if p > 0.3 { withUnsafeCurrentTask { $0?.cancel() } }
        })
    }
    #expect(!FileManager.default.fileExists(atPath: dst.path))
    #expect(FileManager.default.fileExists(atPath: dst.path + ".partial"))
    ModelCopier.cleanupPartial(for: dst)
    #expect(!FileManager.default.fileExists(atPath: dst.path + ".partial"))
}
