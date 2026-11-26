import Foundation
import InklingCore
import AgentCore

/// web_search: DuckDuckGo results + the cleaned text of the top N pages.
public struct WebSearchTool: Tool {
    let fetcher: any PageFetcher
    let crawlTopN: Int
    let perPageChars: Int

    /// Results listed (the rest are rarely used and cost reading time).
    static let maxResults = 5

    public init(fetcher: any PageFetcher, crawlTopN: Int = 2, perPageChars: Int = 1200) {
        self.fetcher = fetcher
        self.crawlTopN = crawlTopN
        self.perPageChars = perPageChars
    }

    public var spec: ToolSpec {
        ToolSpec(name: "web_search",
                 description: "Search the web. Returns the top results with titles, URLs, snippets and an excerpt of the top pages.",
                 parameters: objectSchema([("query", "What to search for")], required: ["query"]))
    }

    public func run(_ arguments: [(String, JSONValue)]) async throws -> String {
        let query = try stringArgument(arguments, "query")
        let page: FetchedPage
        do {
            page = try await fetcher.fetch(DuckDuckGoParser.searchURL(query), timeout: CrawlDefaults.timeout)
        } catch let e as FetchError {
            return "search unavailable: \(e)"
        } catch {
            return "search unavailable: \(error.localizedDescription)"
        }
        let results = Array(DuckDuckGoParser.parse(html: page.html ?? "").prefix(Self.maxResults))
        guard !results.isEmpty else { return "No results for \"\(query)\"." }

        let top = Array(results.prefix(crawlTopN))
        let excerpts = await withTaskGroup(of: (Int, String?).self) { group in
            for (i, r) in top.enumerated() {
                group.addTask {
                    guard let fetched = try? await fetcher.fetch(r.url, timeout: CrawlDefaults.timeout),
                          let html = fetched.html,
                          let md = try? MarkdownGenerator.generate(html: String(html.prefix(OpenURLTool.maxHTMLChars)),
                                                                  baseURL: fetched.finalURL, filter: PruningContentFilter())
                    else { return (i, nil) }
                    let text = md.fitMarkdown.isEmpty ? md.rawMarkdown : md.fitMarkdown
                    return (i, String(text.prefix(perPageChars)))
                }
            }
            var map: [Int: String] = [:]
            for await (i, text) in group { if let text { map[i] = text } }
            return map
        }

        var out = "Search results for \"\(query)\":\n"
        for (i, r) in results.enumerated() {
            out += "\n[\(i + 1)] \(r.title)\nURL: \(r.url.absoluteString)\n"
            if !r.snippet.isEmpty { out += r.snippet + "\n" }
            if let excerpt = excerpts[i] { out += "Page excerpt:\n" + excerpt.trimmingCharacters(in: .whitespacesAndNewlines) + "\n" }
        }
        return out
    }
}
