import Foundation
import SwiftData
import AgentCore

/// Facts in SwiftData. Search score = 0.7 × embedding cosine + 0.3 × keyword overlap;
/// keyword-only when either side has no embedding.
public actor SwiftDataFactStore: FactStoring, ModelActor {
    public nonisolated let modelContainer: ModelContainer
    public nonisolated let modelExecutor: any ModelExecutor
    private let embedder: (any TextEmbedding)?
    /// Hits scoring below this are not returned.
    static let minScore: Double = 0.3

    public init(container: ModelContainer, embedder: (any TextEmbedding)?) {
        modelContainer = container
        modelExecutor = DefaultSerialModelExecutor(modelContext: ModelContext(container))
        self.embedder = embedder
    }

    public func save(_ text: String, sourceChat: UUID?) async throws -> UUID {
        let record = FactRecord(text: text, sourceChat: sourceChat, embedding: embedder?.embed(text)?.data)
        modelContext.insert(record)
        try modelContext.save()
        return record.id
    }

    public func search(_ query: String, limit: Int) async throws -> [StoredFact] {
        let records = try modelContext.fetch(FetchDescriptor<FactRecord>())
        let queryVector = embedder?.embed(query)
        var scored: [(FactRecord, Double)] = []
        for r in records {
            let keyword = KeywordScorer.score(query: query, text: r.text)
            var score = keyword
            if let q = queryVector, let data = r.embedding {
                score = 0.7 * Double(cosineSimilarity(q, [Float](data: data))) + 0.3 * keyword
            }
            if score >= Self.minScore || keyword > 0 { scored.append((r, score)) }
        }
        return scored.sorted { $0.1 > $1.1 }.prefix(limit).map { Self.stored($0.0) }
    }

    public func forget(_ id: UUID) async throws {
        try modelContext.delete(model: FactRecord.self, where: #Predicate { $0.id == id })
        try modelContext.save()
    }

    public func all() async throws -> [StoredFact] {
        try modelContext.fetch(FetchDescriptor<FactRecord>(sortBy: [SortDescriptor(\.created)])).map(Self.stored)
    }

    /// Pinned facts first, then newest — the ones injected into the system prompt.
    public func promptFacts(limit: Int = 20) throws -> [StoredFact] {
        let all = try modelContext.fetch(FetchDescriptor<FactRecord>(sortBy: [SortDescriptor(\.created, order: .reverse)]))
        return (all.filter(\.pinned) + all.filter { !$0.pinned }).prefix(limit).map(Self.stored)
    }

    public func update(_ id: UUID, text: String) throws {
        guard let r = try modelContext.fetch(FetchDescriptor<FactRecord>(predicate: #Predicate { $0.id == id })).first else { return }
        r.text = text
        r.embedding = embedder?.embed(text)?.data
        try modelContext.save()
    }

    static func stored(_ r: FactRecord) -> StoredFact { StoredFact(id: r.id, text: r.text, created: r.created) }
}
