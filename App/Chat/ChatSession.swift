import SwiftUI
import SwiftData
import InklingCore
import AgentCore
import PocketMemory
import InklingEngine

/// One chat: sends messages through the agent, streams the reply, stores the result.
@Observable @MainActor
final class ChatSession {
    let chat: ChatRecord
    @ObservationIgnored private let context: ModelContext
    @ObservationIgnored private let agentFactory: @MainActor () -> Agent
    @ObservationIgnored private let systemPrompt: @MainActor () async -> String
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var generation = 0

    /// The reply being written (nil when idle).
    private(set) var turn: ChatTurnState?
    private(set) var lastSpeed: Double?
    /// The engine's numbers for the last finished reply (all its steps), shown from the speed button.
    private(set) var lastReply: ReplyStats?
    @ObservationIgnored private var statsMark = 0
    var errorMessage: String?

    var isRunning: Bool { turn?.isRunning ?? false }
    var streamingAnswer: String { turn?.answer ?? "" }
    var streamingThinking: String { turn?.thinking ?? "" }
    var toolChips: [ToolChipState] { turn?.chips ?? [] }
    var tokensPerSecond: Double? { turn?.tokensPerSecond ?? lastSpeed }

    init(chat: ChatRecord, context: ModelContext, agentFactory: @escaping @MainActor () -> Agent,
         systemPrompt: @escaping @MainActor () async -> String) {
        self.chat = chat
        self.context = context
        self.agentFactory = agentFactory
        self.systemPrompt = systemPrompt
    }

    var messages: [MessageRecord] { chat.orderedMessages }

    func send(_ text: String) {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, task == nil else { return }
        errorMessage = nil
        append([ChatMessage(role: .user, content: text)])
        if chat.title == "New chat" { chat.title = String(text.prefix(48)) }
        let history = chat.orderedMessages.compactMap(\.chatMessage)
        let agent = agentFactory()
        turn = ChatTurnState()
        statsMark = EngineStatsLog.shared.count
        generation += 1
        let gen = generation
        task = Task { [weak self] in
            guard let self else { return }
            let prompt = await self.systemPrompt()
            do {
                for try await event in await agent.run(history: history, systemPrompt: prompt) {
                    if Task.isCancelled || gen != self.generation { break }
                    guard self.turn != nil else { break }
                    ChatSessionLogic.apply(event, to: &self.turn!)
                }
            } catch is CancellationError {
            } catch {
                if gen == self.generation { self.errorMessage = "\(error)" }
            }
            if gen == self.generation { self.finishTurn() }
        }
    }

    func stop() {
        guard turn != nil else { return }
        ChatSessionLogic.stop(&turn!)
        task?.cancel()
        finishTurn()
    }

    /// Re-sends the text of a tapped OpenUI button or submitted form.
    func sendFollowUp(_ text: String) {
        if isRunning { return }
        send(text)
    }

    private func finishTurn() {
        guard let state = turn else { return }
        lastSpeed = state.tokensPerSecond ?? lastSpeed
        let steps = EngineStatsLog.shared.since(statsMark)
        if !steps.isEmpty { lastReply = ReplyStats(steps) }
        let messages = ChatSessionLogic.messagesToSave(state)
        append(messages, stoppedLast: state.stopped)
        turn = nil
        task = nil
    }

    private func append(_ messages: [ChatMessage], stoppedLast: Bool = false) {
        var order = (chat.messages.map(\.order).max() ?? -1) + 1
        for (i, m) in messages.enumerated() {
            let record = MessageRecord(m, order: order, stopped: stoppedLast && i == messages.count - 1)
            context.insert(record)
            record.chat = chat
            order += 1
        }
        chat.updated = Date()
        try? context.save()
    }
}
