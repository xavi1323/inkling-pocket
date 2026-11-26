import Foundation
import Testing
import SwiftData
import AgentCore
@testable import PocketMemory

struct NilEmbedder: TextEmbedding { func embed(_ text: String) -> [Float]? { nil } }

/// Deterministic bag-of-letters embedding for tests.
struct LetterEmbedder: TextEmbedding {
    func embed(_ text: String) -> [Float]? {
        var v = [Float](repeating: 0, count: 26)
        for s in text.lowercased().unicodeScalars where s.value >= 97 && s.value <= 122 { v[Int(s.value - 97)] += 1 }
        return v
    }
}

@Test func swiftDataSaveThenSearchFindsFact() async throws {
    let store = SwiftDataFactStore(container: try PocketSchema.container(inMemory: true), embedder: LetterEmbedder())
    _ = try await store.save("The owner's favorite color is green", sourceChat: nil)
    _ = try await store.save("Has a dog called Pixel", sourceChat: nil)
    let hits = try await store.search("favorite color", limit: 1)
    #expect(hits.first?.text == "The owner's favorite color is green")
}

@Test func swiftDataForgetRemoves() async throws {
    let store = SwiftDataFactStore(container: try PocketSchema.container(inMemory: true), embedder: nil)
    let id = try await store.save("Lives in Lisbon", sourceChat: nil)
    try await store.forget(id)
    #expect(try await store.all().isEmpty)
}

@Test func swiftDataKeywordFallbackWhenNoEmbedding() async throws {
    let store = SwiftDataFactStore(container: try PocketSchema.container(inMemory: true), embedder: NilEmbedder())
    _ = try await store.save("favorite color is green", sourceChat: nil)
    _ = try await store.save("works as a nurse", sourceChat: nil)
    let hits = try await store.search("color", limit: 5)
    #expect(hits.map(\.text) == ["favorite color is green"])
}

@MainActor
@Test func swiftDataChatSearchFindsOldMessage() async throws {
    let container = try PocketSchema.container(inMemory: true)
    let chat = ChatRecord(title: "Dinner plans")
    container.mainContext.insert(chat)
    let message = MessageRecord(role: "user", content: "Can you help me with pizza dough hydration?")
    container.mainContext.insert(message)
    message.chat = chat
    try container.mainContext.save()
    let hits = try await SwiftDataChatSearch(container: container).search("pizza", limit: 5)
    #expect(hits == [ChatSnippet(chatTitle: "Dinner plans", snippet: "Can you help me with pizza dough hydration?")])
    #expect(try await SwiftDataChatSearch(container: container).search("cars", limit: 5).isEmpty)
}
