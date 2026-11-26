import Foundation
import CrawlCore
#if canImport(WebKit)

/// Renders pages in the hidden WKWebView; falls back to a plain HTTP fetch when that fails (C1).
public struct HybridFetcher: PageFetcher {
    let web: WebPageFetcher
    let plain = URLSessionFetcher()

    @MainActor
    public init(web: WebPageFetcher = WebPageFetcher()) { self.web = web }

    public func fetch(_ url: URL, timeout: TimeInterval) async throws -> FetchedPage {
        do {
            return try await web.fetch(url, timeout: timeout)
        } catch FetchError.offline {
            throw FetchError.offline
        } catch {
            return try await plain.fetch(url, timeout: timeout)
        }
    }
}
#endif
