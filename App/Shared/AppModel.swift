import SwiftUI
import SwiftData
import InklingCore
import AgentCore
import CrawlCore
import PocketCrawler
import PocketMemory
import OpenUICore
import InklingEngine

/// Whether the real model is loaded; chats use the echo model until it is.
enum EngineStatus: Equatable {
    case echo, loading, ready, failed(String)

    var text: String {
        switch self {
        case .echo: "Not imported (echo model)"
        case .loading: "Loading…"
        case .ready: "Inkling 2 Mini, on device"
        case .failed(let why): "Failed: \(why)"
        }
    }
}

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
    var modelFolder: URL?
    var skippedImport = false
    @ObservationIgnored let location = ModelLocation()
    @ObservationIgnored let factStore: SwiftDataFactStore
    @ObservationIgnored let fetcher: any PageFetcher
    @ObservationIgnored private var sessions: [UUID: ChatSession] = [:]
    @ObservationIgnored var model: any LanguageModel = EchoModel()
    var engineStatus: EngineStatus = .echo
    @ObservationIgnored private var loadedEngine: (folder: URL, path: EnginePath)?

    init() {
        let container = (try? PocketSchema.container()) ?? (try! PocketSchema.container(inMemory: true))
        self.container = container
        factStore = SwiftDataFactStore(container: container, embedder: NLEmbedder())
        fetcher = HybridFetcher()
        modelFolder = location.currentModelFolder()
    }

    func refreshModelFolder() {
        modelFolder = location.currentModelFolder()
    }

    /// Loads the real model for the current folder and engine setting (weights are memory-mapped, so this is quick).
    func loadEngineIfNeeded() async {
        guard let folder = modelFolder else {
            model = EchoModel()
            engineStatus = .echo
            loadedEngine = nil
            return
        }
        let path = settings.enginePath
        if let loaded = loadedEngine, loaded.folder == folder, loaded.path == path, engineStatus == .ready { return }
        // A newer load (folder or engine changed) cancels this one; its result is then thrown away.
        engineStatus = .loading
        do {
            let override = settings.memoryBudgetOverrideGB
            let engine = try await InklingGenerator(modelFolder: folder, path: path, memoryBudgetGB: override > 0 ? override : nil)
            await engine.setContextLimit(settings.contextCap)
            guard !Task.isCancelled else { return }
            model = engine
            loadedEngine = (folder, path)
            engineStatus = .ready
        } catch {
            guard !Task.isCancelled else { return }
            model = EchoModel()
            loadedEngine = nil
            engineStatus = .failed(String(describing: error))
        }
    }

    /// iOS is short of memory: let the engine drop part of its pinned experts (acts even mid-answer).
    func handleMemoryWarning() {
        (model as? InklingGenerator)?.handleMemoryWarning()
    }

    /// Settings → context size or memory budget changed: pass them to the loaded engine.
    func applyEngineSettings() {
        guard let engine = model as? InklingGenerator else { return }
        let cap = settings.contextCap, override = settings.memoryBudgetOverrideGB
        engine.setMemoryBudget(gb: override > 0 ? override : nil)
        Task { await engine.setContextLimit(cap) }
    }

    /// Stops every running generation (e.g. when the app goes to the background).
    func stopAll() {
        for session in sessions.values { session.stop() }
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
