import Foundation
import Testing
import InklingCore
@testable import AgentCore

struct FakeChatSearch: ChatSearching {
    func search(_ query: String, limit: Int) async throws -> [ChatSnippet] {
        query.contains("pizza") ? [ChatSnippet(chatTitle: "Dinner plans", snippet: "We talked about pizza dough.")] : []
    }
}

@Test func saveThenSearchFindsFact() async throws {
    let store = InMemoryFactStore()
    let out = try await MemorySaveTool(store: store).run([("fact", .string("The owner's favorite color is green"))])
    #expect(out.hasPrefix("Saved memory"))
    let found = try await MemorySearchTool(store: store).run([("query", .string("favorite color"))])
    #expect(found.contains("favorite color is green"))
}

@Test func forgetRemoves() async throws {
    let store = InMemoryFactStore()
    let id = try await store.save("Lives in Lisbon", sourceChat: nil)
    _ = try await MemoryForgetTool(store: store).run([("id", .string(id.uuidString))])
    #expect(try await store.all().isEmpty)
}

@Test func memoryBlockRespectsBudget() {
    let facts = (0..<100).map { StoredFact(id: UUID(), text: "Fact number \($0) with some extra words", created: Date()) }
    let block = memoryBlock(facts: facts, budget: 300, count: { ($0.count + 3) / 4 })
    #expect((block.count + 3) / 4 <= 300)
    #expect(block.hasPrefix("Things you remember about the user:"))
    #expect(memoryBlock(facts: [], budget: 300, count: { $0.count }) == "")
}

@Test func keywordFallbackWhenNoEmbedding() {
    #expect(KeywordScorer.score(query: "color", text: "favorite color is green") > 0)
    #expect(KeywordScorer.score(query: "colour preference", text: "lives in Lisbon") == 0)
}

@Test func chatSearchFindsOldMessage() async throws {
    let out = try await ChatSearchTool(search: FakeChatSearch()).run([("query", .string("pizza"))])
    #expect(out.contains("Dinner plans"))
    #expect(out.contains("pizza dough"))
    let none = try await ChatSearchTool(search: FakeChatSearch()).run([("query", .string("cars"))])
    #expect(none == "No past chats mention \"cars\".")
}

@Test func cosineSimilarityBasics() {
    #expect(abs(cosineSimilarity([1, 0], [1, 0]) - 1) < 1e-6)
    #expect(abs(cosineSimilarity([1, 0], [0, 1])) < 1e-6)
    #expect(cosineSimilarity([0, 0], [1, 0]) == 0)
    #expect(cosineSimilarity([1, 2, 3], [1, 2]) == 0)  // size mismatch
}
