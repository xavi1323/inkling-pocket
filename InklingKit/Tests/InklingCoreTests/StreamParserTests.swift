import Testing
@testable import InklingCore

/// Merge adjacent same-kind text events so chunking doesn't matter.
func merged(_ events: [ModelEvent]) -> [ModelEvent] {
    var out: [ModelEvent] = []
    for e in events {
        switch (out.last, e) {
        case let (.thinking(a)?, .thinking(b)): out[out.count - 1] = .thinking(a + b)
        case let (.answer(a)?, .answer(b)): out[out.count - 1] = .answer(a + b)
        default: out.append(e)
        }
    }
    return out
}

func parseAll(_ text: String, charByChar: Bool) -> [ModelEvent] {
    var p = InklingStreamParser()
    var events: [ModelEvent] = []
    if charByChar {
        for ch in text { events += p.feed(String(ch)) }
    } else {
        events += p.feed(text)
    }
    events += p.finish()
    return merged(events)
}

let thinkThenAnswer = "<|message_model|><|content_thinking|>hm<|end_message|><|message_model|><|content_text|>Hi<|end_message|><|content_model_end_sampling|>"

@Test func thinkingThenAnswer() {
    #expect(parseAll(thinkThenAnswer, charByChar: false) == [.thinking("hm"), .answer("Hi"), .end])
}

@Test func markerSplitAcrossChunks() {
    #expect(parseAll(thinkThenAnswer, charByChar: true) == [.thinking("hm"), .answer("Hi"), .end])
}

@Test func toolCallParsed() {
    let text = #"<|message_model|>web_search<|content_invoke_tool_json|>{"name":"web_search","args":{"query":"x"}}<|end_message|><|content_model_end_sampling|>"#
    let expected: [ModelEvent] = [.toolCall(ToolCall(name: "web_search", arguments: [("query", .string("x"))])), .end]
    #expect(parseAll(text, charByChar: false) == expected)
    #expect(parseAll(text, charByChar: true) == expected)
}

@Test func badToolJSONReported() {
    let text = #"<|message_model|>web_search<|content_invoke_tool_json|>{"name":"web_search","args":{"query":<|end_message|>"#
    #expect(parseAll(text, charByChar: false) == [.toolCallError(raw: #"{"name":"web_search","args":{"query":"#)])
}

@Test func textNeverLeaksMarkers() {
    let text = "<|message_model|><|content_text|>a < b and <| not a marker |> ok<|end_message|><|content_model_end_sampling|>"
    let events = parseAll(text, charByChar: true)
    #expect(events == [.answer("a < b and <| not a marker |> ok"), .end])
    for e in events { if case .answer(let s) = e { #expect(!s.contains("<|content")) } }
}

@Test func toolErrorAndInvokeTextCountAsAnswer() {
    let text = "<|message_model|><|content_tool_error|>oops<|end_message|>"
    #expect(parseAll(text, charByChar: false) == [.answer("oops")])
}

@Test func unterminatedAnswerFlushedOnFinish() {
    #expect(parseAll("<|message_model|><|content_text|>partial <", charByChar: true) == [.answer("partial <")])
}
