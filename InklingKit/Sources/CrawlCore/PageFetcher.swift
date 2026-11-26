import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public struct FetchedPage: Sendable {
    /// Decoded HTML/text, nil for binary content.
    public var html: String?
    public var finalURL: URL
    public var contentType: String
    public var byteCount: Int
    public init(html: String?, finalURL: URL, contentType: String, byteCount: Int) {
        self.html = html; self.finalURL = finalURL; self.contentType = contentType; self.byteCount = byteCount
    }
}

public enum FetchError: Error, Equatable, CustomStringConvertible {
    case offline, timeout
    case http(Int)
    case unsupported(String)
    public var description: String {
        switch self {
        case .offline: "offline"
        case .timeout: "timed out"
        case .http(let code): "HTTP \(code)"
        case .unsupported(let t): "unsupported content type: \(t)"
        }
    }
}

public protocol PageFetcher: Sendable {
    func fetch(_ url: URL, timeout: TimeInterval) async throws -> FetchedPage
}

public enum CrawlDefaults {
    public static let userAgent = "Mozilla/5.0 (iPad; CPU OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1"
    public static let timeout: TimeInterval = 15
    static let textTypes = ["text/html", "application/xhtml+xml", "text/plain"]
}

/// Plain HTTP fetch (no JavaScript). Works on Wi-Fi or cellular.
public struct URLSessionFetcher: PageFetcher {
    public init() {}

    public func fetch(_ url: URL, timeout: TimeInterval) async throws -> FetchedPage {
        var request = URLRequest(url: url, timeoutInterval: timeout)
        request.setValue(CrawlDefaults.userAgent, forHTTPHeaderField: "User-Agent")
        request.setValue("text/html,application/xhtml+xml;q=0.9,*/*;q=0.5", forHTTPHeaderField: "Accept")
        let data: Data, response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch let e as URLError {
            switch e.code {
            case .notConnectedToInternet, .networkConnectionLost, .dataNotAllowed, .internationalRoamingOff: throw FetchError.offline
            case .timedOut: throw FetchError.timeout
            default: throw e
            }
        }
        let http = response as? HTTPURLResponse
        if let code = http?.statusCode, code >= 400 { throw FetchError.http(code) }
        let type = (http?.value(forHTTPHeaderField: "Content-Type") ?? "text/html").split(separator: ";").first.map {
            $0.trimmingCharacters(in: .whitespaces).lowercased()
        } ?? "text/html"
        let text = CrawlDefaults.textTypes.contains(type)
            ? (String(data: data, encoding: .utf8) ?? String(decoding: data, as: UTF8.self))
            : nil
        return FetchedPage(html: text, finalURL: response.url ?? url, contentType: type, byteCount: data.count)
    }
}
