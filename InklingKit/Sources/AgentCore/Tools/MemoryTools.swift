import Foundation
import InklingCore

public struct MemorySaveTool: Tool {
    let store: any FactStoring
    let currentChat: UUID?
    public init(store: any FactStoring, currentChat: UUID? = nil) { self.store = store; self.currentChat = currentChat }
    public var spec: ToolSpec {
        ToolSpec(name: "memory_save",
                 description: "Remember a lasting fact about the user (preferences, names, plans). Use short, self-contained sentences.",
                 parameters: objectSchema([("fact", "The fact to remember")], required: ["fact"]))
    }
    public func run(_ arguments: [(String, JSONValue)]) async throws -> String {
        let fact = try stringArgument(arguments, "fact")
        let id = try await store.save(fact, sourceChat: currentChat)
        return "Saved memory \(id.uuidString): \(fact)"
    }
}

public struct MemorySearchTool: Tool {
    let store: any FactStoring
    public init(store: any FactStoring) { self.store = store }
    public var spec: ToolSpec {
        ToolSpec(name: "memory_search", description: "Search what you remember about the user.",
                 parameters: objectSchema([("query", "What to look for")], required: ["query"]))
    }
    public func run(_ arguments: [(String, JSONValue)]) async throws -> String {
        let query = try stringArgument(arguments, "query")
        let facts = try await store.search(query, limit: 8)
        guard !facts.isEmpty else { return "No memories match \"\(query)\"." }
        return facts.map { "- [\($0.id.uuidString)] \($0.text)" }.joined(separator: "\n")
    }
}

public struct MemoryForgetTool: Tool {
    let store: any FactStoring
    public init(store: any FactStoring) { self.store = store }
    public var spec: ToolSpec {
        ToolSpec(name: "memory_forget", description: "Forget a remembered fact by its id (from memory_search).",
                 parameters: objectSchema([("id", "The memory id")], required: ["id"]))
    }
    public func run(_ arguments: [(String, JSONValue)]) async throws -> String {
        let raw = try stringArgument(arguments, "id")
        guard let id = UUID(uuidString: raw) else { throw ToolError.message("'\(raw)' is not a memory id") }
        try await store.forget(id)
        return "Forgot memory \(raw)."
    }
}

public struct ChatSearchTool: Tool {
    let search: any ChatSearching
    public init(search: any ChatSearching) { self.search = search }
    public var spec: ToolSpec {
        ToolSpec(name: "chat_search", description: "Search earlier chats with the user.",
                 parameters: objectSchema([("query", "Words to look for")], required: ["query"]))
    }
    public func run(_ arguments: [(String, JSONValue)]) async throws -> String {
        let query = try stringArgument(arguments, "query")
        let hits = try await search.search(query, limit: 6)
        guard !hits.isEmpty else { return "No past chats mention \"\(query)\"." }
        return hits.map { "From \"\($0.chatTitle)\": \($0.snippet)" }.joined(separator: "\n")
    }
}
