import Foundation
import Testing
import CrawlCore
@testable import PocketCrawler

@MainActor
@Test func loadsLocalHTMLFileAndRunsJavaScript() async throws {
    let dir = FileManager.default.temporaryDirectory.appending(path: "wpf-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let file = dir.appending(path: "page.html")
    try """
    <html><body><p id="x">static</p>
    <script>setTimeout(function(){ document.getElementById('x').textContent = 'inserted by js'; }, 200);</script>
    </body></html>
    """.write(to: file, atomically: true, encoding: .utf8)
    let page = try await WebPageFetcher().fetch(file, timeout: 10)
    #expect(page.html?.contains("inserted by js") == true)
    #expect(page.contentType == "text/html")
}

@MainActor
@Test func unreachableHostFailsQuickly() async throws {
    let start = Date()
    await #expect(throws: (any Error).self) {
        _ = try await WebPageFetcher().fetch(URL(string: "http://10.255.255.1/")!, timeout: 1)
    }
    // Well under the 15 s default; WebKit start-up alone can take a few seconds on CI.
    #expect(Date().timeIntervalSince(start) < 10)
}
