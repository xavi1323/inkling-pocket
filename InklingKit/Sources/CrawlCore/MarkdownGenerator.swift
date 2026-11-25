// Swift port of crawl4ai v0.9.6 DefaultMarkdownGenerator (Apache-2.0, Copyright UncleCode).
import Foundation

public struct MarkdownResult: Sendable, Equatable {
    public var rawMarkdown: String
    public var markdownWithCitations: String
    public var referencesMarkdown: String
    public var fitMarkdown: String
    /// Cited URLs in citation-number order (⟨1⟩ is references[0]).
    public var references: [URL]
}

public enum MarkdownGenerator {
    static let linkPattern = try! NSRegularExpression(
        pattern: #"!?\[((?:[^\[\]]|\[(?:[^\[\]]|\[[^\]]*\])*\])*)\]\(((?:[^()\s]|\([^()]*\))*)(?:\s+"([^"]*)")?\)"#)

    public static func generate(html: String, baseURL: URL, filter: PruningContentFilter?) throws -> MarkdownResult {
        let base = baseURL.absoluteString
        let raw = HTML2Markdown(baseURL: base).handle(html).replacingOccurrences(of: "    ```", with: "```")
        let (citations, references, urls) = convertLinksToCitations(raw, baseURL: base)
        var fit = ""
        if let filter {
            let blocks = try filter.filter(html: html)
            let filtered = blocks.map { "<div>\($0)</div>" }.joined(separator: "\n")
            fit = HTML2Markdown(baseURL: base).handle(filtered)
        }
        return MarkdownResult(rawMarkdown: raw, markdownWithCitations: citations, referencesMarkdown: references,
                              fitMarkdown: fit, references: urls)
    }

    static func fastURLJoin(_ base: String, _ url: String) -> String {
        if url.hasPrefix("http://") || url.hasPrefix("https://") || url.hasPrefix("mailto:") || url.hasPrefix("//") { return url }
        if url.hasPrefix("/") { return base.hasSuffix("/") ? String(base.dropLast()) + url : base + url }
        return HTML2Markdown.urlJoin(base, url)
    }

    /// Replaces [text](url "title") with text⟨n⟩ and builds the "## References" list.
    public static func convertLinksToCitations(_ markdown: String, baseURL: String) -> (String, String, [URL]) {
        let ns = markdown as NSString
        var linkMap: [String: (Int, String)] = [:]
        var order: [String] = []
        var parts = ""
        var lastEnd = 0
        var counter = 1
        for m in linkPattern.matches(in: markdown, range: NSRange(location: 0, length: ns.length)) {
            parts += ns.substring(with: NSRange(location: lastEnd, length: m.range.location - lastEnd))
            func group(_ i: Int) -> String? {
                let r = m.range(at: i)
                return r.location == NSNotFound ? nil : ns.substring(with: r)
            }
            let text = group(1) ?? ""
            var url = group(2) ?? ""
            let title = group(3)
            if !baseURL.isEmpty && !(url.hasPrefix("http://") || url.hasPrefix("https://") || url.hasPrefix("mailto:")) {
                url = fastURLJoin(baseURL, url)
            }
            if linkMap[url] == nil {
                var desc: [String] = []
                if let title, !title.isEmpty { desc.append(title) }
                if !text.isEmpty && text != title { desc.append(text) }
                linkMap[url] = (counter, desc.isEmpty ? "" : ": " + desc.joined(separator: " - "))
                order.append(url)
                counter += 1
            }
            let num = linkMap[url]!.0
            parts += ns.substring(with: m.range).hasPrefix("!") ? "![\(text)⟨\(num)⟩]" : "\(text)⟨\(num)⟩"
            lastEnd = m.range.location + m.range.length
        }
        parts += ns.substring(from: lastEnd)
        var refs = "\n\n## References\n\n"
        for url in order {
            let (num, desc) = linkMap[url]!
            refs += "⟨\(num)⟩ \(url)\(desc)\n"
        }
        return (parts, refs, order.compactMap { URL(string: $0.replacingOccurrences(of: "\\", with: "")) })
    }
}
