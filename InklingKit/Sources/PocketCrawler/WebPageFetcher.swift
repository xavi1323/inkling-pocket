import Foundation
import CrawlCore
#if canImport(WebKit)
import WebKit

/// Hidden, reused WKWebView that renders pages (runs their JavaScript) like crawl4ai's headless browser.
/// Requests are served one at a time. Non-HTML responses fall back to a plain HTTP fetch.
@MainActor
public final class WebPageFetcher: NSObject, PageFetcher, WKNavigationDelegate {
    private let webView: WKWebView
    private var navigation: CheckedContinuation<Void, Error>?
    private var nonHTMLType: String?
    private var queueTail: Task<Void, Never>?
    /// Extra wait after load so scripts can fill the page.
    var jsIdle: Duration = .milliseconds(600)

    public override init() {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        webView = WKWebView(frame: CGRect(x: 0, y: 0, width: 1024, height: 1366), configuration: config)
        webView.customUserAgent = CrawlDefaults.userAgent
        super.init()
        webView.navigationDelegate = self
    }

    public func fetch(_ url: URL, timeout: TimeInterval) async throws -> FetchedPage {
        let previous = queueTail
        let task = Task { @MainActor in
            await previous?.value
            return try await self.load(url, timeout: timeout)
        }
        queueTail = Task { _ = try? await task.value }
        return try await task.value
    }

    private func load(_ url: URL, timeout: TimeInterval) async throws -> FetchedPage {
        nonHTMLType = nil
        let timeoutTask = Task { @MainActor in
            try await Task.sleep(for: .seconds(timeout))
            self.webView.stopLoading()
            self.finishNavigation(.failure(FetchError.timeout))
        }
        defer { timeoutTask.cancel() }
        do {
            try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
                navigation = c
                if url.isFileURL {
                    webView.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
                } else {
                    webView.load(URLRequest(url: url, timeoutInterval: timeout))
                }
            }
        } catch {
            if let type = nonHTMLType {
                // Not a web page: fetch the bytes directly so the caller sees the real content type.
                _ = type
                return try await URLSessionFetcher().fetch(url, timeout: timeout)
            }
            if let e = error as? URLError, e.code == .notConnectedToInternet { throw FetchError.offline }
            throw error
        }
        try await Task.sleep(for: jsIdle)
        let html = try await webView.evaluateJavaScript("document.documentElement.outerHTML") as? String ?? ""
        return FetchedPage(html: html, finalURL: webView.url ?? url, contentType: "text/html", byteCount: html.utf8.count)
    }

    private func finishNavigation(_ result: Result<Void, Error>) {
        guard let c = navigation else { return }
        navigation = nil
        c.resume(with: result)
    }

    // MARK: WKNavigationDelegate

    public func webView(_ webView: WKWebView, decidePolicyFor navigationResponse: WKNavigationResponse) async -> WKNavigationResponsePolicy {
        let type = navigationResponse.response.mimeType?.lowercased() ?? "text/html"
        if navigationResponse.isForMainFrame, !["text/html", "application/xhtml+xml", "text/plain"].contains(type) {
            nonHTMLType = type
            return .cancel
        }
        return .allow
    }

    public func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        finishNavigation(.success(()))
    }

    public func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        finishNavigation(.failure(error))
    }

    public func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        finishNavigation(.failure(error))
    }
}
#endif
