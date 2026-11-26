import Testing
@testable import OpenUICore

func button(_ src: String) throws -> OUINode {
    let doc = OpenUIParser.parse("root = Card([b])\nb = Buttons([\(src)])")
    let buttons = try #require(doc.root?["children"]?.arrayValue?.first?.nodeValue?["buttons"]?.arrayValue)
    return try #require(buttons.first?.nodeValue)
}

@Test func buttonActionSendsLabel() throws {
    #expect(buttonActions(try button(#"Button("Tell me more")"#)) == [.sendMessage("Tell me more")])
    #expect(buttonActions(try button(#"Button("Pro", @ToAssistant("I pick Pro"))"#)) == [.sendMessage("I pick Pro")])
    #expect(buttonActions(try button(#"Button("Docs", @OpenUrl("https://example.com"))"#)) == [.openURL("https://example.com")])
    #expect(buttonActions(try button(#"Button("Both", Action([@OpenUrl("https://a.b"), @ToAssistant("Opened")]))"#))
        == [.openURL("https://a.b"), .sendMessage("Opened")])
    #expect(buttonActions(try button(#"Button("Ctx", @ToAssistant("Pick", {id: 7}))"#)) == [.sendMessage("Pick\ncontext = {id: 7}")])
}

@Test func formSubmitMessageFormat() {
    let message = actionMessage(.submitForm(id: "trip", values: ["nights": "3", "destination": "Lisbon"]))
    #expect(message == "Form trip submitted: destination=Lisbon, nights=3")
    #expect(actionMessage(.sendMessage("hi")) == "hi")
}

@Test func unknownChartTypeFallsBackToTable() {
    #expect(chartKind("bar") == .bar)
    #expect(chartKind("line") == .line)
    #expect(chartKind("pie") == .pie)
    #expect(chartKind("donut") == .table)
    #expect(chartKind(nil) == .table)
}
