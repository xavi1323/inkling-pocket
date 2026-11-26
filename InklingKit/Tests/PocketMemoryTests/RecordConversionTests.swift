import Foundation
import Testing
import InklingCore
@testable import PocketMemory

@Test func messageRecordRoundTrips() {
    let m = ChatMessage(role: .assistant, content: "Looking", reasoning: "think",
                        toolCalls: [ToolCall(name: "web_search", arguments: [("query", .string("x")), ("n", .int(2))])])
    let r = MessageRecord(m, order: 3, stopped: true)
    #expect(r.role == "assistant")
    #expect(r.order == 3)
    #expect(r.stopped)
    #expect(r.chatMessage == m)
    let tool = ChatMessage(role: .tool, content: "Error: x", toolName: "open_url", isToolError: true)
    #expect(MessageRecord(tool, order: 0).chatMessage == tool)
}

@MainActor
@Test func chatSearchMatchesAssistantMessages() async throws {
    let container = try PocketSchema.container(inMemory: true)
    let chat = ChatRecord(title: "Trip")
    container.mainContext.insert(chat)
    let m = MessageRecord(ChatMessage(role: .assistant, content: "Lisbon has great pastel de nata."), order: 0)
    container.mainContext.insert(m)
    m.chat = chat
    try container.mainContext.save()
    let hits = try await SwiftDataChatSearch(container: container).search("pastel", limit: 3)
    #expect(hits.map(\.chatTitle) == ["Trip"])
}
