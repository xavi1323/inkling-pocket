import Foundation
import Testing
import InklingCore
import AgentCore
@testable import CrawlCore

/// Scripted fetcher: maps URL prefixes to pages; records every fetched URL.
final class FakeFetcher: PageFetcher, @unchecked Sendable {
    private let lock = NSLock()
    private(set) var fetched: [URL] = []
    let pages: [String: FetchedPage]
    let error: FetchError?
    let delay: TimeInterval

    init(pages: [String: FetchedPage] = [:], error: FetchError? = nil, delay: TimeInterval = 0) {
        self.pages = pages; self.error = error; self.delay = delay
    }

    func fetch(_ url: URL, timeout: TimeInterval) async throws -> FetchedPage {
        lock.withLock { fetched.append(url) }
        if delay > 0 { try await Task.sleep(for: .seconds(delay)) }
        if let error { throw error }
        let key = pages.keys.sorted { $0.count > $1.count }.first { url.absoluteString.hasPrefix($0) }
        guard let key, let page = pages[key] else { throw FetchError.http(404) }
        return page
    }
}

func html(_ body: String, url: String) -> FetchedPage {
    FetchedPage(html: "<html><body>\(body)</body></html>", finalURL: URL(string: url)!, contentType: "text/html", byteCount: body.utf8.count)
}

let ddgFixture = try! String(contentsOf: Bundle.module.url(forResource: "Fixtures/ddg", withExtension: "html")!, encoding: .utf8)

@Test func parsesDDGResults() {
    let results = DuckDuckGoParser.parse(html: ddgFixture)
    #expect(results.count == 6)  // ad skipped
    #expect(results[0] == SearchResult(title: "Official site", url: URL(string: "https://www.swift.org/")!,
                                       snippet: "Swift is a general-purpose programming language that's approachable for newcomers and powerful for experts."))
    #expect(results[1].url.absoluteString == "https://en.wikipedia.org/wiki/Swift_(programming_language)")
    #expect(results.allSatisfy { !$0.url.absoluteString.contains("duckduckgo.com") })
    #expect(results[3].snippet == "The Swift book: a guided tour & language reference.")
}

@Test func searchCrawlsTopN() async throws {
    var pages: [String: FetchedPage] = ["https://html.duckduckgo.com/html/": FetchedPage(html: ddgFixture, finalURL: URL(string: "https://html.duckduckgo.com/html/")!, contentType: "text/html", byteCount: 1)]
    for p in ["https://www.swift.org/", "https://en.wikipedia.org/", "https://developer.apple.com/"] {
        pages[p] = html("<article><h1>Page \(p)</h1><p>Long enough body text about Swift so the pruning filter keeps this paragraph in the output.</p></article>", url: p)
    }
    let fetcher = FakeFetcher(pages: pages)
    let out = try await WebSearchTool(fetcher: fetcher, crawlTopN: 3).run([("query", .string("swift"))])
    #expect(fetcher.fetched.count == 4)  // results page + 3 crawls
    #expect(fetcher.fetched[0].absoluteString.hasPrefix("https://html.duckduckgo.com/html/?q=swift"))
    #expect(out.contains("[1] Official site"))
    #expect(out.contains("Long enough body text about Swift"))
}

@Test func searchFailsSoftWhenOffline() async throws {
    let start = Date()
    let out = try await WebSearchTool(fetcher: FakeFetcher(error: .offline), crawlTopN: 3).run([("query", .string("x"))])
    #expect(out.hasPrefix("search unavailable: offline"))
    #expect(Date().timeIntervalSince(start) < 3)
}

@Test func openURLRejectsPDF() async throws {
    let page = FetchedPage(html: nil, finalURL: URL(string: "https://a.b/f.pdf")!, contentType: "application/pdf", byteCount: 1000)
    let out = try await OpenURLTool(fetcher: FakeFetcher(pages: ["https://a.b/": page])).run([("url", .string("https://a.b/f.pdf"))])
    #expect(out == "unsupported content type: application/pdf")
}

@Test func openURLTrimsHugePage() async throws {
    let para = "<p>" + String(repeating: "word ", count: 200) + "</p>\n"
    let body = String(repeating: para, count: 20_000)  // ~20 MB
    let start = Date()
    let out = try await OpenURLTool(fetcher: FakeFetcher(pages: ["https://big.page/": html(body, url: "https://big.page/")]), maxChars: 8000)
        .run([("url", .string("https://big.page/"))])
    #expect(out.count <= 8000 + 200)
    #expect(Date().timeIntervalSince(start) < 5)
}

@Test func openURLReturnsMarkdownWithSources() async throws {
    let page = html(#"<article><h1>Title</h1><p>Main text with a <a href="/ref">reference link</a> inside a long enough paragraph to survive pruning.</p></article>"#, url: "https://site.org/a")
    let out = try await OpenURLTool(fetcher: FakeFetcher(pages: ["https://site.org/": page])).run([("url", .string("https://site.org/a"))])
    #expect(out.contains("# Title"))
    #expect(out.contains("reference link⟨1⟩"))
    #expect(out.contains("⟨1⟩ https://site.org/ref"))
}

@Test func openURLRejectsNonHTTP() async {
    await #expect(throws: ToolError.self) { try await OpenURLTool(fetcher: FakeFetcher()).run([("url", .string("file:///etc/passwd"))]) }
}
