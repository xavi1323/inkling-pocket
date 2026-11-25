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

@Test(arguments: syntheticNames)
func rawMarkdownMatchesCrawl4AI(name: String) throws {
    let e = try CrawlExpected(dir: syntheticDir, name: name)
    let r = try MarkdownGenerator.generate(html: e.html, baseURL: URL(string: e.baseURL)!, filter: PruningContentFilter())
    #expect(r.rawMarkdown == e.raw)
    #expect(r.markdownWithCitations == e.citations)
    #expect(r.referencesMarkdown == e.references)
}

@Test(arguments: syntheticNames)
func fitMarkdownMatchesCrawl4AI(name: String) throws {
    let e = try CrawlExpected(dir: syntheticDir, name: name)
    let r = try MarkdownGenerator.generate(html: e.html, baseURL: URL(string: e.baseURL)!, filter: PruningContentFilter())
    #expect(lineSimilarity(r.fitMarkdown, e.fit) >= 0.95, "fit:\n\(r.fitMarkdown)\n--- expected:\n\(e.fit)")
}

@Test(.enabled(if: hasLocalPages, "real pages not present (kept out of git)"), arguments: localPageNames)
func fitMarkdownSimilar(name: String) throws {
    let e = try CrawlExpected(dir: localPagesDir, name: name)
    let r = try MarkdownGenerator.generate(html: e.html, baseURL: URL(string: e.baseURL)!, filter: PruningContentFilter())
    let rawSim = lineSimilarity(r.rawMarkdown, e.raw)
    let fitSim = lineSimilarity(r.fitMarkdown, e.fit)
    #expect(rawSim >= 0.90, "raw similarity \(rawSim)")
    #expect(fitSim >= 0.90, "fit similarity \(fitSim)")
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
