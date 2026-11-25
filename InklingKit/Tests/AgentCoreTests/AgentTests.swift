import Foundation
import Testing
import InklingCore
@testable import AgentCore

let userHi = [ChatMessage(role: .user, content: "hi")]

func makeAgent(_ model: ScriptedModel, tools: [any Tool] = [EchoTool()], maxSteps: Int = 8) -> Agent {
    var s = AgentSettings()
    s.maxSteps = maxSteps
    return Agent(model: model, tools: ToolRegistry(tools), settings: s)
}

func finishedMessages(_ events: [AgentEvent]) -> [ChatMessage] {
    for e in events { if case .finished(let m) = e { return m } }
    return []
}

@Test func answerWithoutTools() async throws {
    let model = ScriptedModel([[.thinking("t"), .answer("Hello"), .end]])
    let events = try await collect(await makeAgent(model).run(history: userHi, systemPrompt: "sys"))
    #expect(events.contains(.thinking("t")))
    #expect(events.contains(.answer("Hello")))
    #expect(finishedMessages(events) == [ChatMessage(role: .assistant, content: "Hello", reasoning: "t")])
    #expect(model.receivedMessages[0].first == ChatMessage(role: .system, content: "sys"))
}

@Test func toolCallThenAnswer() async throws {
    let call = ToolCall(name: "echo", arguments: [("text", .string("pong"))])
    let model = ScriptedModel([[.toolCall(call), .end], [.answer("Done"), .end]])
    let events = try await collect(await makeAgent(model).run(history: userHi, systemPrompt: "sys"))
    #expect(events.contains(.toolStarted(call)))
    #expect(events.contains(.toolFinished(name: "echo", preview: "pong", isError: false)))
    #expect(finishedMessages(events) == [
        ChatMessage(role: .assistant, content: "", toolCalls: [call]),
        ChatMessage(role: .tool, content: "pong", toolName: "echo"),
        ChatMessage(role: .assistant, content: "Done"),
    ])
    // second model call saw the tool result
    #expect(model.receivedMessages[1].last == ChatMessage(role: .tool, content: "pong", toolName: "echo"))
}

@Test func unknownToolReturnsError() async throws {
    let call = ToolCall(name: "nope", arguments: [])
    let model = ScriptedModel([[.toolCall(call), .end], [.answer("ok"), .end]])
    let events = try await collect(await makeAgent(model).run(history: userHi, systemPrompt: "sys"))
    let toolMsg = finishedMessages(events)[1]
    #expect(toolMsg.isToolError)
    #expect(toolMsg.content.hasPrefix("Error: unknown tool nope"))
}

@Test func badToolJSONFedBack() async throws {
    let model = ScriptedModel([[.toolCallError(raw: "{bad"), .end], [.answer("fixed"), .end]])
    let events = try await collect(await makeAgent(model).run(history: userHi, systemPrompt: "sys"))
    let msgs = finishedMessages(events)
    #expect(msgs.contains { $0.role == .tool && $0.isToolError && $0.content.contains("{bad") })
    #expect(msgs.last == ChatMessage(role: .assistant, content: "fixed"))
}

@Test func stopsAtMaxSteps() async throws {
    let call = ToolCall(name: "echo", arguments: [("text", .string("x"))])
    let model = ScriptedModel(Array(repeating: [.toolCall(call), .end], count: 10))
    let events = try await collect(await makeAgent(model, maxSteps: 3).run(history: userHi, systemPrompt: "sys"))
    #expect(events.contains(.stepLimitReached))
    #expect(model.receivedMessages.count == 3)
}

@Test func toolResultTrimmed() async throws {
    let call = ToolCall(name: "huge", arguments: [])
    let model = ScriptedModel([[.toolCall(call), .end], [.answer("ok"), .end]])
    let events = try await collect(await makeAgent(model, tools: [HugeTool()]).run(history: userHi, systemPrompt: "sys"))
    let toolMsg = finishedMessages(events)[1]
    #expect(model.countTokens(toolMsg.content) <= 2000)
    #expect(toolMsg.content.hasSuffix("[trimmed]"))
}
