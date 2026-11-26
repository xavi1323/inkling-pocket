import SwiftUI
import Testing
import OpenUICore
@testable import OpenUIViews

let everyComponent = """
root = Card([t, l, c1, c2, c3, c4, co, cb, s, b, f, i], "All")
t = Table(["A", "B"], [[1, "x"], [2.5, "y"]], "Caption")
l = List(["one", "**two**"], true)
c1 = Chart("bar", ["Mon", "Tue"], [{name: "S", values: [1, 2]}], "Bars")
c2 = Chart("line", ["Mon", "Tue"], [{name: "S", values: [1, 2]}])
c3 = Chart("pie", ["Mon", "Tue"], [{name: "S", values: [1, 2]}])
c4 = Chart("donut", ["Mon", "Tue"], [{name: "S", values: [1, 2]}])
co = Callout("Careful", "warning", "Note")
cb = CodeBlock("let x = 1", "swift")
s = Steps(["Plan", "Build"], 2)
b = Buttons([Button("Yes", null, "primary"), Button("No")])
f = Form("f", [Field("name", "Name"), Field("ok", "OK?", "toggle"), Field("pick", "Pick", "select", ["a", "b"]), Field("n", "Notes", "multiline")])
i = Image("https://example.com/x.png", "An image", "Caption")
"""

@MainActor
@Test func rendersEveryComponent() throws {
    let doc = OpenUIParser.parse(everyComponent)
    #expect(try #require(doc.root).nodeCount == 18)
    let renderer = ImageRenderer(content: OpenUIView(document: doc, onAction: { _ in }).frame(width: 700))
    #expect(renderer.cgImage != nil)
}

@MainActor
@Test func emptyDocumentRendersNothing() {
    let renderer = ImageRenderer(content: OpenUIView(document: OpenUIDocument(), onAction: { _ in }))
    #expect(renderer.cgImage == nil)
}
