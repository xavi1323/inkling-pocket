import Foundation
import InklingCore

public struct AgentSettings: Sendable {
    public var maxSteps = 8
    public var toolResultTokenBudget = 2000
    public var contextCap = 16_384
    public var effort: ReasoningEffort = .low
    public var parameters = GenerateParameters()
    public init() {}
}

public enum AgentEvent: Sendable, Equatable {
    case thinking(String)
    case answer(String)
    case toolStarted(ToolCall)
    case toolFinished(name: String, preview: String, isError: Bool)
    case stepLimitReached
    /// Messages added during this run (assistant turns and tool results), in order.
    case finished([ChatMessage])
}

/// Runs the generate → tool → generate loop.
public actor Agent {
    private let model: any LanguageModel
    private let tools: ToolRegistry
    private let settings: AgentSettings

    public init(model: any LanguageModel, tools: ToolRegistry, settings: AgentSettings) {
        self.model = model
        self.tools = tools
        self.settings = settings
    }

    public func run(history: [ChatMessage], systemPrompt: String) -> AsyncThrowingStream<AgentEvent, Error> {
        let model = self.model, tools = self.tools, settings = self.settings
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await Agent.loop(model: model, tools: tools, settings: settings,
                                         history: history, systemPrompt: systemPrompt, emit: { continuation.yield($0) })
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    static func loop(model: any LanguageModel, tools: ToolRegistry, settings: AgentSettings,
                     history: [ChatMessage], systemPrompt: String,
                     emit: @Sendable (AgentEvent) -> Void) async throws {
        var added: [ChatMessage] = []
        let count: (String) -> Int = { model.countTokens($0) }
        for _ in 0..<settings.maxSteps {
            try Task.checkCancellation()
            let window = ContextWindow.fit(system: systemPrompt, history: history + added,
                                           cap: settings.contextCap, count: count)
            let messages = [ChatMessage(role: .system, content: systemPrompt)] + window
            var thinking = "", answer = ""
            var calls: [ToolCall] = []
            var badCalls: [String] = []
            for try await event in model.generate(messages, tools: tools.specs, effort: settings.effort,
                                                  parameters: settings.parameters) {
                switch event {
                case .thinking(let t): thinking += t; emit(.thinking(t))
                case .answer(let a): answer += a; emit(.answer(a))
                case .toolCall(let c): calls.append(c)
                case .toolCallError(let raw): badCalls.append(raw)
                case .end: break
                }
            }
            added.append(ChatMessage(role: .assistant, content: answer,
                                     reasoning: thinking.isEmpty ? nil : thinking, toolCalls: calls))
            if calls.isEmpty && badCalls.isEmpty {
                emit(.finished(added))
                return
            }
            for raw in badCalls {
                let text = "Error: could not parse tool call JSON: \(raw)"
                added.append(ChatMessage(role: .tool, content: text, toolName: "invalid_tool_call", isToolError: true))
                emit(.toolFinished(name: "invalid_tool_call", preview: String(text.prefix(200)), isError: true))
            }
            for call in calls {
                try Task.checkCancellation()
                emit(.toolStarted(call))
                var isError = false
                var result: String
                if let tool = tools.tool(named: call.name) {
                    do {
                        result = try await tool.run(call.arguments)
                    } catch {
                        isError = true
                        result = "Error: \(error)"
                    }
                } else {
                    isError = true
                    result = "Error: unknown tool \(call.name)"
                }
                result = trimToBudget(result, budget: settings.toolResultTokenBudget, count: count)
                added.append(ChatMessage(role: .tool, content: result, toolName: call.name, isToolError: isError))
                emit(.toolFinished(name: call.name, preview: String(result.prefix(200)), isError: isError))
            }
        }
        emit(.stepLimitReached)
        emit(.finished(added))
    }
}
