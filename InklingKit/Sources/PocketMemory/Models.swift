import Foundation
import SwiftData

@Model
public final class ChatRecord {
    @Attribute(.unique) public var id: UUID
    public var title: String
    public var created: Date
    public var updated: Date
    public var effort: String
    @Relationship(deleteRule: .cascade, inverse: \MessageRecord.chat)
    public var messages: [MessageRecord] = []

    public init(id: UUID = UUID(), title: String, effort: String = "low", created: Date = Date()) {
        self.id = id
        self.title = title
        self.created = created
        self.updated = created
        self.effort = effort
    }

    /// Messages in the order they were written.
    public var orderedMessages: [MessageRecord] { messages.sorted { ($0.created, $0.order) < ($1.created, $1.order) } }
}

@Model
public final class MessageRecord {
    @Attribute(.unique) public var id: UUID
    public var role: String
    public var content: String
    public var reasoning: String?
    public var toolName: String?
    public var toolCallsJSON: String?
    public var isToolError: Bool
    public var stopped: Bool
    public var created: Date
    /// Tie-breaker for messages written in the same instant.
    public var order: Int
    public var chat: ChatRecord?

    public init(id: UUID = UUID(), role: String, content: String, reasoning: String? = nil, toolName: String? = nil,
                toolCallsJSON: String? = nil, isToolError: Bool = false, stopped: Bool = false, created: Date = Date(), order: Int = 0) {
        self.id = id
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.toolName = toolName
        self.toolCallsJSON = toolCallsJSON
        self.isToolError = isToolError
        self.stopped = stopped
        self.created = created
        self.order = order
    }
}

@Model
public final class FactRecord {
    @Attribute(.unique) public var id: UUID
    public var text: String
    public var created: Date
    public var sourceChat: UUID?
    public var pinned: Bool
    public var embedding: Data?

    public init(id: UUID = UUID(), text: String, created: Date = Date(), sourceChat: UUID? = nil, pinned: Bool = false, embedding: Data? = nil) {
        self.id = id
        self.text = text
        self.created = created
        self.sourceChat = sourceChat
        self.pinned = pinned
        self.embedding = embedding
    }
}

public enum PocketSchema {
    public static let models: [any PersistentModel.Type] = [ChatRecord.self, MessageRecord.self, FactRecord.self]

    public static func container(inMemory: Bool = false) throws -> ModelContainer {
        try ModelContainer(for: Schema(models), configurations: ModelConfiguration(isStoredInMemoryOnly: inMemory))
    }
}
