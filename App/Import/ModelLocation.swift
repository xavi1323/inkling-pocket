import Foundation
import InklingCore

/// Where the model lives: copied into Application Support/Model, or the picked folder via a bookmark.
@MainActor
final class ModelLocation {
    static let bookmarkKey = "modelBookmark"
    private let defaults: UserDefaults
    private var accessing: URL?

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        ModelCopier.cleanupPartial(for: Self.copiedFolder)
    }

    static var copiedFolder: URL {
        URL.applicationSupportDirectory.appending(path: "Model", directoryHint: .isDirectory)
    }

    func currentModelFolder() -> URL? {
        if let data = defaults.data(forKey: Self.bookmarkKey) {
            var stale = false
            if let url = try? URL(resolvingBookmarkData: data, options: [], relativeTo: nil, bookmarkDataIsStale: &stale) {
                if accessing != url {
                    accessing?.stopAccessingSecurityScopedResource()
                    _ = url.startAccessingSecurityScopedResource()
                    accessing = url
                }
                if stale, let fresh = try? url.bookmarkData() { defaults.set(fresh, forKey: Self.bookmarkKey) }
                if FileManager.default.fileExists(atPath: url.appending(path: "config.json").path) { return url }
            }
        }
        let copied = Self.copiedFolder
        return FileManager.default.fileExists(atPath: copied.appending(path: "config.json").path) ? copied : nil
    }

    /// Use the picked folder where it is (needs the picker's security scope, kept via bookmark).
    func useInPlace(_ url: URL) throws {
        defaults.set(try url.bookmarkData(), forKey: Self.bookmarkKey)
    }

    /// Copy into app storage; clears any in-place bookmark when done.
    func copyIn(_ url: URL, progress: @escaping @Sendable (Double) -> Void) async throws {
        try FileManager.default.createDirectory(at: URL.applicationSupportDirectory, withIntermediateDirectories: true)
        try await ModelCopier.copy(from: url, to: Self.copiedFolder, progress: progress)
        defaults.removeObject(forKey: Self.bookmarkKey)
    }

    func forget() {
        defaults.removeObject(forKey: Self.bookmarkKey)
        accessing?.stopAccessingSecurityScopedResource()
        accessing = nil
    }
}
