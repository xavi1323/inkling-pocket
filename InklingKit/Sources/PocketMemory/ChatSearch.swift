import Foundation
import SwiftData
import AgentCore

/// Keyword search over saved user/model messages of earlier chats.
public actor SwiftDataChatSearch: ChatSearching, ModelActor {
    public nonisolated let modelContainer: ModelContainer
    public nonisolated let modelExecutor: any ModelExecutor
    private let excludeChat: UUID?
    static let scanLimit = 5000

    public init(container: ModelContainer, excludeChat: UUID? = nil) {
        modelContainer = container
        modelExecutor = DefaultSerialModelExecutor(modelContext: ModelContext(container))
        self.excludeChat = excludeChat
    }

    public func search(_ query: String, limit: Int) async throws -> [ChatSnippet] {
        var descriptor = FetchDescriptor<MessageRecord>(sortBy: [SortDescriptor(\.created, order: .reverse)])
        descriptor.fetchLimit = Self.scanLimit
        let messages = try modelContext.fetch(descriptor)
        let words = KeywordScorer.words(query)
        guard !words.isEmpty else { return [] }
        var hits: [(ChatSnippet, Double)] = []
        for m in messages where m.role == "user" || m.role == "model" {
            guard let chat = m.chat, chat.id != excludeChat else { continue }
            let score = KeywordScorer.score(query: query, text: m.content)
            if score > 0 { hits.append((ChatSnippet(chatTitle: chat.title, snippet: Self.snippet(m.content, around: words)), score)) }
        }
        return hits.sorted { $0.1 > $1.1 }.prefix(limit).map(\.0)
    }

    /// ~240 characters around the first matching word.
    static func snippet(_ text: String, around words: [String]) -> String {
        let lower = text.lowercased()
        guard let hit = words.compactMap({ lower.range(of: $0) }).min(by: { $0.lowerBound < $1.lowerBound }) else {
            return String(text.prefix(240))
        }
        let offset = lower.distance(from: lower.startIndex, to: hit.lowerBound)
        let start = text.index(text.startIndex, offsetBy: max(0, offset - 80), limitedBy: text.endIndex) ?? text.startIndex
        let end = text.index(start, offsetBy: 240, limitedBy: text.endIndex) ?? text.endIndex
        var s = String(text[start..<end]).replacingOccurrences(of: "\n", with: " ")
        if start > text.startIndex { s = "…" + s }
        if end < text.endIndex { s += "…" }
        return s
    }
}
