import Foundation
import Testing
import InklingCore
@testable import CrawlCore

/// Expected outputs come from crawl4ai v0.9.6 (scripts/crawl_fixtures.py).
struct CrawlExpected {
    let html: String
    let baseURL: String
    let raw: String, citations: String, references: String, fit: String

    init(dir: URL, name: String) throws {
        html = try String(contentsOf: dir.appending(path: "\(name).html"), encoding: .utf8)
        let j = try JSONValue.parse(String(contentsOf: dir.appending(path: "\(name).expected.json"), encoding: .utf8))
        baseURL = j["base_url"]!.stringValue!
        raw = j["raw"]!.stringValue!
        citations = j["citations"]!.stringValue!
        references = j["references"]!.stringValue!
        fit = j["fit"]!.stringValue!
    }
}

let syntheticDir = Bundle.module.url(forResource: "Fixtures/synthetic", withExtension: nil)!
let syntheticNames = ["basic", "lists", "table", "code", "quotes", "images", "boilerplate", "nested"]
/// Real pages are kept out of git (copyright); tests run only where they exist.
let localPagesDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "Fixtures/local-pages")
let localPageNames = ["wikipedia", "news", "docs", "github", "forum", "blog", "product", "tables"]
let hasLocalPages = FileManager.default.fileExists(atPath: localPagesDir.appending(path: "docs.expected.json").path)

/// Share of normalised non-empty lines in common (LCS / max line count).
func lineSimilarity(_ a: String, _ b: String) -> Double {
    func lines(_ s: String) -> [String] {
        s.split(separator: "\n").map { $0.split(whereSeparator: \.isWhitespace).joined(separator: " ") }.filter { !$0.isEmpty }
    }
    let x = lines(a), y = lines(b)
    if x.isEmpty && y.isEmpty { return 1 }
    var prev = [Int](repeating: 0, count: y.count + 1)
    for i in 0..<x.count {
        var cur = [Int](repeating: 0, count: y.count + 1)
        for j in 0..<y.count { cur[j + 1] = x[i] == y[j] ? prev[j] + 1 : max(prev[j + 1], cur[j]) }
        prev = cur
    }
    return Double(prev[y.count]) / Double(max(x.count, y.count))
}

/// The Markdown writer is our own, so crawl4ai's output is the reference for content, not for exact layout:
/// the same words (link targets aside), the same cited URLs, and the same cleaned-up text.
@Test(arguments: syntheticNames)
func rawMarkdownMatchesCrawl4AI(name: String) throws {
    let e = try CrawlExpected(dir: syntheticDir, name: name)
    let r = try MarkdownGenerator.generate(html: e.html, baseURL: URL(string: e.baseURL)!, filter: PruningContentFilter())
    #expect(wordSimilarity(r.rawMarkdown, e.raw) >= 0.9, "raw:\n\(r.rawMarkdown)\n--- crawl4ai:\n\(e.raw)")
    #expect(wordSimilarity(r.markdownWithCitations, e.citations) >= 0.9)
    let refs = { (s: String) in Set(s.split(separator: "\n").compactMap { $0.split(separator: " ").dropFirst().first.map(String.init) }) }
    #expect(refs(r.referencesMarkdown) == refs(e.references), "references:\n\(r.referencesMarkdown)\n--- crawl4ai:\n\(e.references)")
}

@Test(arguments: syntheticNames)
func fitMarkdownMatchesCrawl4AI(name: String) throws {
    let e = try CrawlExpected(dir: syntheticDir, name: name)
    let r = try MarkdownGenerator.generate(html: e.html, baseURL: URL(string: e.baseURL)!, filter: PruningContentFilter())
    #expect(wordSimilarity(r.fitMarkdown, e.fit) >= 0.9, "fit:\n\(r.fitMarkdown)\n--- crawl4ai:\n\(e.fit)")
}

@Test func writerFormatsBlocksAndMarks() throws {
    let html = """
        <h2>Title <em>here</em></h2><p>Some <strong>bold</strong> and <code>x()</code> text.</p>
        <ul><li>one</li><li>two<ul><li>deep</li></ul></li></ul><ol start="3"><li>three</li></ol>
        <blockquote><p>quoted</p></blockquote><p>a<br>b</p><script>evil()</script><hr>
        """
    let md = MarkdownWriter(baseURL: "https://a.b/").markdown(html)
    #expect(md.contains("## Title _here_"))
    #expect(md.contains("Some **bold** and `x()` text."))
    #expect(md.contains("* one\n* two\n  * deep"))
    #expect(md.contains("3. three"))
    #expect(md.contains("> quoted"))
    #expect(md.contains("a\nb"))
    #expect(md.contains("---"))
    #expect(!md.contains("evil"))
}

/// Same words, ignoring Markdown punctuation, layout and link targets: shared word count (multiset) / the larger
/// word count, accents folded (crawl4ai's html2text turns "Café" into "Cafe"; our writer keeps the real text).
func wordSimilarity(_ a: String, _ b: String) -> Double {
    func words(_ s: String) -> [String: Int] {
        var bag: [String: Int] = [:]
        let text = s.replacingOccurrences(of: #"\]\([^)\s]*(\s+"[^"]*")?\)"#, with: "]", options: .regularExpression)
        for w in text.lowercased().folding(options: .diacriticInsensitive, locale: nil).split(whereSeparator: { !$0.isLetter && !$0.isNumber }) { bag[String(w), default: 0] += 1 }
        return bag
    }
    let x = words(a), y = words(b)
    let shared = x.reduce(0) { $0 + min($1.value, y[$1.key] ?? 0) }
    let total = max(x.values.reduce(0, +), y.values.reduce(0, +))
    return total == 0 ? 1 : Double(shared) / Double(total)
}

/// Raw pages differ more than cleaned ones because the writer skips <noscript> fallbacks and SVG titles.
@Test(.enabled(if: hasLocalPages, "real pages not present (kept out of git)"), arguments: localPageNames)
func fitMarkdownSimilar(name: String) throws {
    let e = try CrawlExpected(dir: localPagesDir, name: name)
    let r = try MarkdownGenerator.generate(html: e.html, baseURL: URL(string: e.baseURL)!, filter: PruningContentFilter())
    let rawSim = wordSimilarity(r.rawMarkdown, e.raw)
    let fitSim = wordSimilarity(r.fitMarkdown, e.fit)
    #expect(rawSim >= 0.80, "raw similarity \(rawSim)")
    #expect(fitSim >= 0.85, "fit similarity \(fitSim)")
}

@Test func tablesBecomeMarkdownTables() throws {
    let html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    let md = try MarkdownGenerator.generate(html: html, baseURL: URL(string: "https://a.b/")!, filter: nil).rawMarkdown
    #expect(md.contains("| A | B |"))
    #expect(md.contains("| --- | --- |"))
    #expect(md.contains("| 1 | 2 |"))
}

@Test func codeBlocksFenced() throws {
    let html = "<pre><code>let x = 1\nlet y = 2</code></pre>"
    let md = try MarkdownGenerator.generate(html: html, baseURL: URL(string: "https://a.b/")!, filter: nil).rawMarkdown
    #expect(md.contains("```\nlet x = 1\nlet y = 2\n```"))
}

@Test func linksBecomeCitations() throws {
    let html = #"<p><a href="/one">One</a> and <a href="https://two.org">Two</a> and <a href="/one">again</a></p>"#
    let r = try MarkdownGenerator.generate(html: html, baseURL: URL(string: "https://a.b/c/")!, filter: nil)
    #expect(r.markdownWithCitations.contains("One⟨1⟩"))
    #expect(r.markdownWithCitations.contains("again⟨1⟩"))
    #expect(r.references.map(\.absoluteString) == ["https://a.b/one", "https://two.org"])
}
