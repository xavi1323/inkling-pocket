import Foundation
import Testing
import InklingCore
@testable import AgentCore

let t0 = Date(timeIntervalSince1970: 1_000_000)

@Test func thinkingAccumulates() {
    var state = ChatTurnState()
    ChatSessionLogic.apply(.thinking("Let me "), to: &state, at: t0)
    ChatSessionLogic.apply(.thinking("think."), to: &state, at: t0.addingTimeInterval(0.5))
    ChatSessionLogic.apply(.answer("Hi"), to: &state, at: t0.addingTimeInterval(1))
    #expect(state.thinking == "Let me think.")
    #expect(state.answer == "Hi")
    #expect(state.tokenCount == 3)
    #expect(state.tokensPerSecond == 2)  // 2 tokens after the first, over 1 s
}

@Test func toolChipLifecycle() {
    var state = ChatTurnState()
    ChatSessionLogic.apply(.answer("Looking."), to: &state, at: t0)
    ChatSessionLogic.apply(.toolStarted(ToolCall(name: "web_search", arguments: [("query", .string("ipad air m4"))])), to: &state, at: t0)
    #expect(state.chips == [ToolChipState(id: 0, title: "Searching: ipad air m4", status: .running, preview: nil)])
    ChatSessionLogic.apply(.toolFinished(name: "web_search", preview: "Search results…", isError: false), to: &state, at: t0)
    #expect(state.chips == [ToolChipState(id: 0, title: "Searched: ipad air m4", status: .done, preview: "Search results…")])
    ChatSessionLogic.apply(.answer("Found it."), to: &state, at: t0)
    #expect(state.answer == "Looking.\n\nFound it.")
    ChatSessionLogic.apply(.toolStarted(ToolCall(name: "open_url", arguments: [("url", .string("https://www.apple.com/ipad"))])), to: &state, at: t0)
    ChatSessionLogic.apply(.toolFinished(name: "open_url", preview: "Error: HTTP 404", isError: true), to: &state, at: t0)
    #expect(state.chips.last == ToolChipState(id: 1, title: "Read: www.apple.com", status: .failed, preview: "Error: HTTP 404"))
}

@Test func stopMarksTurnStopped() {
    var state = ChatTurnState()
    ChatSessionLogic.apply(.thinking("hmm"), to: &state, at: t0)
    ChatSessionLogic.apply(.answer("Partial ans"), to: &state, at: t0)
    ChatSessionLogic.apply(.toolStarted(ToolCall(name: "calculator", arguments: [("expression", .string("1+1"))])), to: &state, at: t0)
    ChatSessionLogic.stop(&state)
    #expect(state.stopped)
    #expect(state.isRunning == false)
    #expect(state.chips.first?.status == .failed)
    #expect(ChatSessionLogic.messagesToSave(state) == [ChatMessage(role: .assistant, content: "Partial ans", reasoning: "hmm")])
    // events after stop are ignored
    ChatSessionLogic.apply(.answer(" more"), to: &state, at: t0)
    #expect(state.answer == "Partial ans")
}

@Test func finishedKeepsAgentMessages() {
    var state = ChatTurnState()
    let msgs = [ChatMessage(role: .assistant, content: "Done")]
    ChatSessionLogic.apply(.answer("Done"), to: &state, at: t0)
    ChatSessionLogic.apply(.finished(msgs), to: &state, at: t0)
    #expect(state.isRunning == false)
    #expect(ChatSessionLogic.messagesToSave(state) == msgs)
}
