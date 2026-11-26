import Foundation
import SwiftSoup

public struct SearchResult: Sendable, Equatable {
    public var title: String
    public var url: URL
    public var snippet: String
    public init(title: String, url: URL, snippet: String) { self.title = title; self.url = url; self.snippet = snippet }
}

/// Parses DuckDuckGo's no-JavaScript results page (html.duckduckgo.com/html/).
public enum DuckDuckGoParser {
    public static func searchURL(_ query: String) -> URL {
        var c = URLComponents(string: "https://html.duckduckgo.com/html/")!
        c.queryItems = [URLQueryItem(name: "q", value: query)]
        return c.url!
    }

    public static func parse(html: String) -> [SearchResult] {
        guard let doc = try? SwiftSoup.parse(html), let blocks = try? doc.select("div.result") else { return [] }
        var out: [SearchResult] = []
        var seen = Set<String>()
        for block in blocks.array() {
            if block.hasClass("result--ad") { continue }
            guard let link = try? block.select("a.result__a").first(),
                  let href = try? link.attr("href"), let url = resolve(href) else { continue }
            if url.host?.hasSuffix("duckduckgo.com") == true { continue }
            if !seen.insert(url.absoluteString).inserted { continue }
            let title = (try? link.text()) ?? ""
            let snippet = (try? block.select(".result__snippet").first()?.text()) ?? ""
            out.append(SearchResult(title: title, url: url, snippet: snippet ?? ""))
        }
        return out
    }

    /// Unwraps //duckduckgo.com/l/?uddg=<encoded> redirect links.
    static func resolve(_ href: String) -> URL? {
        let full = href.hasPrefix("//") ? "https:" + href : href
        guard let comps = URLComponents(string: full) else { return URL(string: full) }
        if comps.host?.hasSuffix("duckduckgo.com") == true, comps.path.hasPrefix("/l/"),
           let target = comps.queryItems?.first(where: { $0.name == "uddg" })?.value {
            return URL(string: target)
        }
        return URL(string: full)
    }
}
