import Foundation
import InklingCore

extension MessageRecord {
    public convenience init(_ message: ChatMessage, order: Int, stopped: Bool = false, created: Date = Date()) {
        let calls = message.toolCalls.isEmpty ? nil
            : (try? JSONEncoder().encode(message.toolCalls)).flatMap { String(data: $0, encoding: .utf8) }
        self.init(role: message.role.rawValue, content: message.content, reasoning: message.reasoning,
                  toolName: message.toolName, toolCallsJSON: calls, isToolError: message.isToolError,
                  stopped: stopped, created: created, order: order)
    }

    /// The stored message as the agent sees it (nil for an unknown role).
    public var chatMessage: ChatMessage? {
        guard let role = Role(rawValue: role) else { return nil }
        let calls = toolCallsJSON.flatMap { $0.data(using: .utf8) }.flatMap { try? JSONDecoder().decode([ToolCall].self, from: $0) } ?? []
        return ChatMessage(role: role, content: content, reasoning: reasoning, toolCalls: calls,
                           toolName: toolName, isToolError: isToolError)
    }
}
