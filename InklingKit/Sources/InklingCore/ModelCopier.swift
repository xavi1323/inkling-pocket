import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Copies the model folder into app storage: to `<dest>.partial` first, renamed when complete.
/// Cancel with Task cancellation; the `.partial` folder stays until `cleanupPartial(for:)`.
public enum ModelCopier {
    public static func copy(from source: URL, to destination: URL, chunkSize: Int = 8 << 20,
                            progress: @Sendable (Double) -> Void) async throws {
        let fm = FileManager.default
        let partial = URL(fileURLWithPath: destination.path + ".partial")
        try? fm.removeItem(at: partial)
        try fm.createDirectory(at: partial, withIntermediateDirectories: true)

        let files = try regularFiles(in: source)
        let total = max(1, files.reduce(Int64(0)) { $0 + $1.size })
        var done: Int64 = 0
        progress(0)
        for file in files {
            try Task.checkCancellation()
            let target = partial.appending(path: file.relative)
            try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            if try cloneIfPossible(file.url, target) {
                done += file.size
                progress(Double(done) / Double(total))
                continue
            }
            guard fm.createFile(atPath: target.path, contents: nil),
                  let input = try? FileHandle(forReadingFrom: file.url),
                  let output = try? FileHandle(forWritingTo: target) else {
                throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: target.path])
            }
            defer { try? input.close(); try? output.close() }
            while true {
                try Task.checkCancellation()
                guard let chunk = try input.read(upToCount: chunkSize), !chunk.isEmpty else { break }
                try output.write(contentsOf: chunk)
                done += Int64(chunk.count)
                progress(min(Double(done) / Double(total), 0.999_999))
                await Task.yield()
            }
        }
        try Task.checkCancellation()
        if fm.fileExists(atPath: destination.path) { try fm.removeItem(at: destination) }
        try fm.moveItem(at: partial, to: destination)
        progress(1.0)
    }

    /// Removes a leftover `.partial` copy (call on start).
    public static func cleanupPartial(for destination: URL) {
        try? FileManager.default.removeItem(at: URL(fileURLWithPath: destination.path + ".partial"))
    }

    struct SourceFile { let url: URL; let relative: String; let size: Int64 }

    static func regularFiles(in folder: URL) throws -> [SourceFile] {
        let base = folder.standardizedFileURL.resolvingSymlinksInPath().path
        guard let walker = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]) else {
            return []
        }
        var out: [SourceFile] = []
        for case let url as URL in walker {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            guard values.isRegularFile == true, !url.lastPathComponent.hasPrefix(".") else { continue }
            let full = url.standardizedFileURL.resolvingSymlinksInPath().path
            let relative = full.hasPrefix(base + "/") ? String(full.dropFirst(base.count + 1)) : url.lastPathComponent
            out.append(SourceFile(url: url, relative: relative, size: Int64(values.fileSize ?? 0)))
        }
        return out.sorted { $0.relative < $1.relative }
    }

    /// APFS clone (instant, no extra space) when source and destination share a volume.
    static func cloneIfPossible(_ src: URL, _ dst: URL) throws -> Bool {
        #if canImport(Darwin)
        return copyfile(src.path, dst.path, nil, copyfile_flags_t(COPYFILE_CLONE_FORCE)) == 0
        #else
        return false
        #endif
    }
}
