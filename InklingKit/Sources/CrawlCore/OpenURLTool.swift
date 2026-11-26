import Foundation
import InklingCore
import AgentCore

/// open_url: fetches a page and returns its main content as markdown with numbered sources.
public struct OpenURLTool: Tool {
    /// HTML beyond this is cut before parsing (keeps huge pages fast).
    static let maxHTMLChars = 1_500_000
    let fetcher: any PageFetcher
    let maxChars: Int

    public init(fetcher: any PageFetcher, maxChars: Int = 12_000) {
        self.fetcher = fetcher
        self.maxChars = maxChars
    }

    public var spec: ToolSpec {
        ToolSpec(name: "open_url", description: "Open a web page and read its main content as markdown, with numbered source links.",
                 parameters: objectSchema([("url", "The http(s) URL to open")], required: ["url"]))
    }

    public func run(_ arguments: [(String, JSONValue)]) async throws -> String {
        let raw = try stringArgument(arguments, "url")
        guard let url = URL(string: raw), let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https" else {
            throw ToolError.message("only http(s) URLs can be opened")
        }
        let page: FetchedPage
        do {
            page = try await fetcher.fetch(url, timeout: CrawlDefaults.timeout)
        } catch let e as FetchError {
            return "could not open \(raw): \(e)"
        }
        guard CrawlDefaults.textTypes.contains(page.contentType), let text = page.html else {
            return "unsupported content type: \(page.contentType)"
        }
        if page.contentType == "text/plain" { return String(text.prefix(maxChars)) }
        let md = try MarkdownGenerator.generate(html: String(text.prefix(Self.maxHTMLChars)), baseURL: page.finalURL,
                                                filter: PruningContentFilter())
        let body = md.fitMarkdown.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? md.rawMarkdown : md.fitMarkdown
        let (cited, references, _) = MarkdownGenerator.convertLinksToCitations(body, baseURL: page.finalURL.absoluteString)
        let budgetForBody = max(0, maxChars - min(references.count, maxChars / 4))
        var out = String(cited.prefix(budgetForBody))
        if cited.count > budgetForBody { out += "\n[trimmed]" }
        out += String(references.prefix(maxChars / 4))
        return "Source: \(page.finalURL.absoluteString)\n\n" + out
    }
}
