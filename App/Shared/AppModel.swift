import SwiftUI
import SwiftData
import InklingCore
import AgentCore
import CrawlCore
import PocketCrawler
import PocketMemory
import OpenUICore

enum SidebarItem: Hashable {
    case chat(UUID)
    case memory, benchmark, settings, diagnostics
}

/// App-wide services: storage, settings, tools, and one ChatSession per open chat.
@Observable @MainActor
final class AppModel {
    let container: ModelContainer
    let settings = AppSettings()
    var selection: SidebarItem?
    @ObservationIgnored let factStore: SwiftDataFactStore
    @ObservationIgnored let fetcher: any PageFetcher
    @ObservationIgnored private var sessions: [UUID: ChatSession] = [:]
    @ObservationIgnored var model: any LanguageModel = EchoModel()

    init() {
        let container = (try? PocketSchema.container()) ?? (try! PocketSchema.container(inMemory: true))
        self.container = container
        factStore = SwiftDataFactStore(container: container, embedder: NLEmbedder())
        fetcher = HybridFetcher()
    }

    func newChat() {
        let chat = ChatRecord(title: "New chat", effort: settings.effort.rawValue)
        container.mainContext.insert(chat)
        try? container.mainContext.save()
        selection = .chat(chat.id)
    }

    func delete(_ chat: ChatRecord) {
        sessions[chat.id]?.stop()
        sessions[chat.id] = nil
        if selection == .chat(chat.id) { selection = nil }
        container.mainContext.delete(chat)
        try? container.mainContext.save()
    }

    func session(for id: UUID) -> ChatSession? {
        if let s = sessions[id] { return s }
        let ctx = container.mainContext
        guard let chat = try? ctx.fetch(FetchDescriptor<ChatRecord>(predicate: #Predicate { $0.id == id })).first else { return nil }
        let s = ChatSession(chat: chat, context: ctx,
                            agentFactory: { [unowned self] in self.makeAgent(chatID: id, effort: chat.effort) },
                            systemPrompt: { [unowned self] in await self.systemPrompt() })
        sessions[id] = s
        return s
    }

    func makeAgent(chatID: UUID, effort: String) -> Agent {
        let tools = ToolRegistry([
            WebSearchTool(fetcher: fetcher, crawlTopN: settings.crawlTopN),
            OpenURLTool(fetcher: fetcher),
            MemorySaveTool(store: factStore, currentChat: chatID),
            MemorySearchTool(store: factStore),
            MemoryForgetTool(store: factStore),
            ChatSearchTool(search: SwiftDataChatSearch(container: container, excludeChat: chatID)),
            CurrentTimeTool(),
            CalculatorTool(),
        ])
        var s = AgentSettings()
        s.maxSteps = settings.maxSteps
        s.contextCap = settings.contextCap
        s.effort = ReasoningEffort(rawValue: effort) ?? settings.effort
        s.parameters.temperature = Float(settings.temperature)
        s.parameters.topP = Float(settings.topP)
        return Agent(model: model, tools: tools, settings: s)
    }

    func systemPrompt() async -> String {
        let facts = (try? await factStore.promptFacts()) ?? []
        let model = self.model
        return SystemPrompt.build(now: Date(), memory: memoryBlock(facts: facts, budget: 300, count: { model.countTokens($0) }))
    }
}

enum SystemPrompt {
    static func build(now: Date, memory: String) -> String {
        let date = now.formatted(date: .complete, time: .shortened)
        var parts = [
            """
            You are Inkling, a helpful assistant that runs entirely on the user's iPad. It is \(date).
            Use tools when they help: web_search for current or unknown facts, open_url to read a page, \
            memory_save for lasting facts about the user (preferences, names, plans), memory_search and chat_search to recall. \
            When you use web results, cite them with their ⟨n⟩ numbers.
            """,
        ]
        if !memory.isEmpty { parts.append(memory) }
        parts.append(ComponentRegistry.systemPrompt())
        return parts.joined(separator: "\n\n")
    }
}
