import Foundation
import InklingCore

public struct ToolChipState: Equatable, Sendable, Identifiable {
    public enum Status: Equatable, Sendable { case running, done, failed }
    public var id: Int
    public var title: String
    public var status: Status
    public var preview: String?
    public init(id: Int, title: String, status: Status, preview: String?) {
        self.id = id; self.title = title; self.status = status; self.preview = preview
    }
}

/// Everything the UI shows for the reply in progress.
public struct ChatTurnState: Equatable, Sendable {
    public var thinking = ""
    public var answer = ""
    public var chips: [ToolChipState] = []
    public var stopped = false
    public var stepLimitReached = false
    public var finishedMessages: [ChatMessage]?
    public var tokenCount = 0
    public var firstTokenAt: Date?
    public var lastTokenAt: Date?
    var pendingSeparator = false
    var runningTools: [String: Int] = [:]
    var runningTitles: [Int: (String, String)] = [:]

    public init() {}

    public var isRunning: Bool { !stopped && finishedMessages == nil }

    /// Tokens per second after the first token (nil until there are two).
    public var tokensPerSecond: Double? {
        guard tokenCount > 1, let a = firstTokenAt, let b = lastTokenAt, b > a else { return nil }
        return Double(tokenCount - 1) / b.timeIntervalSince(a)
    }

    public static func == (l: ChatTurnState, r: ChatTurnState) -> Bool {
        l.thinking == r.thinking && l.answer == r.answer && l.chips == r.chips && l.stopped == r.stopped
            && l.stepLimitReached == r.stepLimitReached && l.finishedMessages == r.finishedMessages && l.tokenCount == r.tokenCount
    }
}

public enum ChatSessionLogic {
    /// Folds one agent event into the turn state. Events after stop/finish are ignored.
    public static func apply(_ event: AgentEvent, to state: inout ChatTurnState, at now: Date = Date()) {
        guard state.isRunning else { return }
        switch event {
        case .thinking(let t):
            countToken(&state, now)
            state.thinking += t
        case .answer(let a):
            countToken(&state, now)
            if state.pendingSeparator && !state.answer.isEmpty && !a.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                state.answer += "\n\n"
                state.pendingSeparator = false
            }
            if state.pendingSeparator && state.answer.isEmpty { state.pendingSeparator = false }
            state.answer += state.pendingSeparator ? "" : a
        case .toolStarted(let call):
            let (running, done) = titles(for: call)
            let id = state.chips.count
            state.chips.append(ToolChipState(id: id, title: running, status: .running, preview: nil))
            state.runningTools[call.name] = id
            state.runningTitles[id] = (running, done)
            state.pendingSeparator = true
        case .toolFinished(let name, let preview, let isError):
            if let id = state.runningTools.removeValue(forKey: name) {
                state.chips[id].status = isError ? .failed : .done
                state.chips[id].title = state.runningTitles[id]?.1 ?? state.chips[id].title
                state.chips[id].preview = preview
            } else {
                let title = name == "invalid_tool_call" ? "Invalid tool call" : "Ran \(name)"
                state.chips.append(ToolChipState(id: state.chips.count, title: title, status: isError ? .failed : .done, preview: preview))
            }
            state.pendingSeparator = true
        case .stepLimitReached:
            state.stepLimitReached = true
        case .finished(let messages):
            state.finishedMessages = messages
            failRunningChips(&state)
        }
    }

    /// Stop pressed: running tools are marked failed and nothing more is applied.
    public static func stop(_ state: inout ChatTurnState) {
        guard state.isRunning else { return }
        failRunningChips(&state)
        state.stopped = true
    }

    /// Messages to store for this turn: the agent's own list, or the partial reply when stopped early.
    public static func messagesToSave(_ state: ChatTurnState) -> [ChatMessage] {
        if let finished = state.finishedMessages { return finished }
        guard !state.answer.isEmpty || !state.thinking.isEmpty else { return [] }
        return [ChatMessage(role: .assistant, content: state.answer, reasoning: state.thinking.isEmpty ? nil : state.thinking)]
    }

    static func countToken(_ state: inout ChatTurnState, _ now: Date) {
        state.tokenCount += 1
        if state.firstTokenAt == nil { state.firstTokenAt = now }
        state.lastTokenAt = now
    }

    static func failRunningChips(_ state: inout ChatTurnState) {
        for id in state.runningTools.values { state.chips[id].status = .failed }
        state.runningTools = [:]
    }

    /// (while running, when done) chip titles.
    public static func titles(for call: ToolCall) -> (String, String) {
        let arg = { (key: String) in call.argument(key)?.stringValue ?? "" }
        switch call.name {
        case "web_search": return ("Searching: \(arg("query"))", "Searched: \(arg("query"))")
        case "open_url":
            let host = URL(string: arg("url"))?.host ?? arg("url")
            return ("Reading: \(host)", "Read: \(host)")
        case "memory_save": return ("Remembering", "Remembered")
        case "memory_search": return ("Recalling: \(arg("query"))", "Recalled: \(arg("query"))")
        case "memory_forget": return ("Forgetting", "Forgot")
        case "chat_search": return ("Searching chats: \(arg("query"))", "Searched chats: \(arg("query"))")
        case "current_time": return ("Checking the time", "Checked the time")
        case "calculator": return ("Calculating: \(arg("expression"))", "Calculated: \(arg("expression"))")
        default: return ("Running \(call.name)", "Ran \(call.name)")
        }
    }
}
