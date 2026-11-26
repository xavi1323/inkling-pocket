import Foundation

public struct StoredFact: Sendable, Equatable, Identifiable {
    public var id: UUID
    public var text: String
    public var created: Date
    public init(id: UUID, text: String, created: Date) { self.id = id; self.text = text; self.created = created }
}

public struct ChatSnippet: Sendable, Equatable {
    public var chatTitle: String
    public var snippet: String
    public init(chatTitle: String, snippet: String) { self.chatTitle = chatTitle; self.snippet = snippet }
}

public protocol FactStoring: Sendable {
    func save(_ text: String, sourceChat: UUID?) async throws -> UUID
    func search(_ query: String, limit: Int) async throws -> [StoredFact]
    func forget(_ id: UUID) async throws
    func all() async throws -> [StoredFact]
}

public protocol ChatSearching: Sendable {
    func search(_ query: String, limit: Int) async throws -> [ChatSnippet]
}

/// Share of the query's words (3+ letters, case-insensitive) found in the text.
public enum KeywordScorer {
    public static func words(_ s: String) -> [String] {
        s.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init).filter { $0.count >= 3 }
    }
    public static func score(query: String, text: String) -> Double {
        let q = Set(words(query))
        guard !q.isEmpty else { return 0 }
        let t = Set(words(text))
        return Double(q.intersection(t).count) / Double(q.count)
    }
}

/// Simple store used in tests and as a fallback when SwiftData isn't available.
public actor InMemoryFactStore: FactStoring {
    private var facts: [StoredFact] = []
    public init() {}
    public func save(_ text: String, sourceChat: UUID?) async throws -> UUID {
        let f = StoredFact(id: UUID(), text: text, created: Date())
        facts.append(f)
        return f.id
    }
    public func search(_ query: String, limit: Int) async throws -> [StoredFact] {
        facts.map { ($0, KeywordScorer.score(query: query, text: $0.text)) }
            .filter { $0.1 > 0 }.sorted { $0.1 > $1.1 }.prefix(limit).map(\.0)
    }
    public func forget(_ id: UUID) async throws { facts.removeAll { $0.id == id } }
    public func all() async throws -> [StoredFact] { facts }
}

/// The block of remembered facts injected into the system prompt.
public func memoryBlock(facts: [StoredFact], budget: Int = 300, count: (String) -> Int) -> String {
    guard !facts.isEmpty else { return "" }
    var block = "Things you remember about the user:"
    for f in facts {
        let line = "\n- " + f.text
        if count(block + line) > budget { break }
        block += line
    }
    return block
}

/// Cosine similarity; 0 when sizes differ or a vector is all zeros.
public func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count, !a.isEmpty else { return 0 }
    var dot: Float = 0, na: Float = 0, nb: Float = 0
    for i in 0..<a.count {
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    }
    guard na > 0, nb > 0 else { return 0 }
    return dot / (na.squareRoot() * nb.squareRoot())
}
