import SwiftUI
import Metal
import WebKit
import UniformTypeIdentifiers
import os

/// M0 smoke test: proves the pieces Inkling Pocket needs work inside LiveContainer.
struct SmokeCheck: Identifiable {
    enum Status { case pending, pass, fail }
    let id: String
    var status: Status = .pending
    var detail: String = "Not run yet"
}

@MainActor @Observable
final class SmokeModel {
    var checks: [SmokeCheck] = ["Metal", "WebKit", "Files", "Storage"].map { SmokeCheck(id: $0) }
    var availableMemoryGB: Double = 0
    private var probe: WebProbe?

    func set(_ id: String, _ status: SmokeCheck.Status, _ detail: String) {
        guard let i = checks.firstIndex(where: { $0.id == id }) else { return }
        checks[i].status = status
        checks[i].detail = detail
    }

    func runAutomatic() async {
        checkMetal()
        checkStorage()
        checkMemory()
        await checkWeb()
    }

    func checkMetal() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            set("Metal", .fail, "No Metal device")
            return
        }
        let gb = Double(device.recommendedMaxWorkingSetSize) / 1_073_741_824
        set("Metal", .pass, "\(device.name) · GPU working set \(String(format: "%.2f", gb)) GB")
    }

    func checkStorage() {
        do {
            let dir = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                                  appropriateFor: nil, create: true)
            let file = dir.appending(path: "smoke.txt")
            try "ok".write(to: file, atomically: true, encoding: .utf8)
            let read = try String(contentsOf: file, encoding: .utf8)
            set("Storage", read == "ok" ? .pass : .fail, dir.path(percentEncoded: false))
        } catch {
            set("Storage", .fail, error.localizedDescription)
        }
    }

    func checkMemory() {
        availableMemoryGB = Double(os_proc_available_memory()) / 1_073_741_824
    }

    func checkWeb() async {
        let probe = WebProbe()
        self.probe = probe
        do {
            let title = try await probe.load(URL(string: "https://example.com")!)
            set("WebKit", .pass, "Loaded page: \(title)")
        } catch {
            set("WebKit", .fail, error.localizedDescription)
        }
    }

    func checkFolder(_ url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            let bookmark = try url.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil)
            var stale = false
            let resolved = try URL(resolvingBookmarkData: bookmark, options: [], relativeTo: nil,
                                   bookmarkDataIsStale: &stale)
            let items = try FileManager.default.contentsOfDirectory(atPath: resolved.path(percentEncoded: false))
            set("Files", .pass, "Bookmark OK · \(items.count) items in \(resolved.lastPathComponent)")
        } catch {
            set("Files", .fail, error.localizedDescription)
        }
    }
}

/// Hidden WKWebView that loads one page and reports its title.
@MainActor
final class WebProbe: NSObject, WKNavigationDelegate {
    private let webView: WKWebView = {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        return WKWebView(frame: .zero, configuration: config)
    }()
    private var continuation: CheckedContinuation<String, Error>?

    func load(_ url: URL) async throws -> String {
        webView.navigationDelegate = self
        return try await withCheckedThrowingContinuation { cont in
            continuation = cont
            webView.load(URLRequest(url: url, timeoutInterval: 15))
        }
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        continuation?.resume(returning: webView.title ?? "(no title)")
        continuation = nil
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        continuation?.resume(throwing: error)
        continuation = nil
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        continuation?.resume(throwing: error)
        continuation = nil
    }
}

struct SmokeView: View {
    @State private var model = SmokeModel()
    @State private var pickingFolder = false

    var body: some View {
        NavigationStack {
            List {
                Section("LiveContainer checks") {
                    ForEach(model.checks) { check in
                        HStack(spacing: 12) {
                            Image(systemName: icon(for: check.status))
                                .foregroundStyle(color(for: check.status))
                            VStack(alignment: .leading) {
                                Text(check.id).font(.headline)
                                Text(check.detail).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        .accessibilityElement(children: .combine)
                    }
                }
                Section("Memory") {
                    Text("Available to app: \(String(format: "%.2f", model.availableMemoryGB)) GB")
                }
                Section {
                    Button("Pick a folder (Files test)") { pickingFolder = true }
                }
            }
            .navigationTitle("Inkling Pocket · Smoke Test")
            .toolbar {
                Button("Re-run") { Task { await model.runAutomatic() } }
            }
            .fileImporter(isPresented: $pickingFolder, allowedContentTypes: [.folder]) { result in
                switch result {
                case .success(let url): model.checkFolder(url)
                case .failure(let error): model.set("Files", .fail, error.localizedDescription)
                }
            }
            .task { await model.runAutomatic() }
        }
    }

    private func icon(for status: SmokeCheck.Status) -> String {
        switch status {
        case .pending: "circle.dashed"
        case .pass: "checkmark.circle.fill"
        case .fail: "xmark.octagon.fill"
        }
    }

    private func color(for status: SmokeCheck.Status) -> Color {
        switch status {
        case .pending: .secondary
        case .pass: .green
        case .fail: .red
        }
    }
}
