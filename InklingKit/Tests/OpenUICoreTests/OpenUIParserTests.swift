import Foundation
import Testing
@testable import OpenUICore

let programNames = ["01-dashboard", "02-ternary-root", "03-buttons-actions", "04-form", "05-expressions", "06-recovery"]

func fixture(_ name: String, _ ext: String) throws -> String {
    let url = try #require(Bundle.module.url(forResource: "Fixtures/programs/\(name)", withExtension: ext))
    return try String(contentsOf: url, encoding: .utf8)
}

func nodeCount(_ doc: OpenUIDocument) -> Int { doc.root.map { $0.nodeCount } ?? 0 }

@Test func segmentsMarkdownAndOpenUI() {
    let text = "Here you go:\n```openui\nroot = Card([\"Hi\"])\n```\nDone, see `a ``` b` inline.\n```swift\nlet a = 1\n```\n"
    #expect(AnswerSegmenter.segments(text) == [
        .markdown("Here you go:\n"),
        .openUI(source: "root = Card([\"Hi\"])", complete: true),
        .markdown("Done, see `a ``` b` inline.\n```swift\nlet a = 1\n```\n"),
    ])
    #expect(AnswerSegmenter.segments("Look:\n```openui-lang\nroot = Card([") == [
        .markdown("Look:\n"), .openUI(source: "root = Card([", complete: false),
    ])
    #expect(AnswerSegmenter.segments("```openui\na = \"x\n```\ny\"\n```\nafter") == [
        .openUI(source: "a = \"x\n```\ny\"", complete: true), .markdown("after"),
    ])
    #expect(AnswerSegmenter.segments("plain") == [.markdown("plain")])
}

@Test(arguments: programNames)
func parsesSpecExamples(name: String) throws {
    let doc = OpenUIParser.parse(try fixture(name, "oui"))
    let expected = try fixture(name, "expected").trimmingCharacters(in: .whitespacesAndNewlines)
    #expect(doc.root?.summary == expected)
}

@Test(arguments: programNames)
func partialStreamRendersValidPrefix(name: String) throws {
    let source = try fixture(name, "oui")
    var parser = OpenUIParser()
    var last = 0
    for ch in source {
        let doc = parser.feed(String(ch))
        let count = nodeCount(doc)
        #expect(count >= last, "node count dropped from \(last) to \(count) in \(name)")
        last = count
        _ = doc.root?.summary  // every intermediate tree must be printable
    }
    #expect(parser.finish() == OpenUIParser.parse(source))
}

@Test func malformedStatementDropped() {
    let doc = OpenUIParser.parse("root = Card([a, b, c])\na = Callout(\"A\")\nb = Callout(\"B\" ) )\nc = )(\nthis line is prose\n")
    #expect(doc.root?.summary == #"Card(children: [Callout(text: "A"), Callout(text: "B")])"#)
}

@Test func unknownComponentDropped() {
    let doc = OpenUIParser.parse("root = Card([Text(\"a\"), Callout(\"b\")])")
    #expect(doc.root?.summary == #"Card(children: [Callout(text: "b")])"#)
}

@Test func pendingStatementDoesNotOverwriteCompleted() {
    var parser = OpenUIParser()
    _ = parser.feed("root = Card([a])\na = Callout(\"done\")\na = Callout(\"par")
    let doc = parser.feed("tial")
    #expect(doc.root?.summary == #"Card(children: [Callout(text: "done")])"#)
    #expect(parser.finish().root?.summary == #"Card(children: [Callout(text: "partial")])"#)
}

@Test func fallbackEntryIsFirstCardCall() {
    #expect(OpenUIParser.parse("page = Card([\"x\"])").root?.summary == #"Card(children: ["x"])"#)
    #expect(OpenUIParser.parse("t = Callout(\"x\")").root == nil)
}

@Test func systemPromptListsAllV1Components() {
    let prompt = ComponentRegistry.systemPrompt()
    for name in ["Card", "Table", "List", "Chart", "Callout", "CodeBlock", "Steps", "Buttons", "Form", "Image"] {
        #expect(prompt.contains("\n\(name)("), "missing \(name)")
    }
    #expect(prompt.contains("```openui"))
    #expect(ComponentRegistry.v1.map(\.name).prefix(10) == ["Card", "Table", "List", "Chart", "Callout", "CodeBlock", "Steps", "Buttons", "Form", "Image"])
}
