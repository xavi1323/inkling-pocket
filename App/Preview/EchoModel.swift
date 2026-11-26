import Foundation
import InklingCore
import AgentCore

/// Stand-in model until the engine lands: echoes the user, `/tool` calls current_time, `/ui` shows OpenUI.
struct EchoModel: LanguageModel {
    func countTokens(_ text: String) -> Int { max(1, text.utf8.count / 4) }

    func generate(_ messages: [ChatMessage], tools: [ToolSpec], effort: ReasoningEffort,
                  parameters: GenerateParameters) -> AsyncThrowingStream<ModelEvent, Error> {
        let last = messages.last
        return AsyncThrowingStream { continuation in
            let task = Task {
                func stream(_ text: String, thinking: Bool = false) async throws {
                    for word in text.split(separator: " ", omittingEmptySubsequences: false) {
                        try await Task.sleep(for: .milliseconds(25))
                        continuation.yield(thinking ? .thinking(String(word) + " ") : .answer(String(word) + " "))
                    }
                }
                do {
                    if last?.role == .tool {
                        try await stream("The \(last?.toolName ?? "tool") said: \(last?.content ?? "")")
                    } else {
                        let text = last?.content ?? ""
                        try await stream("The user wrote \(text.count) characters. I will echo them.", thinking: true)
                        if text.hasPrefix("/tool") {
                            continuation.yield(.toolCall(ToolCall(name: "current_time", arguments: [])))
                        } else if text.hasPrefix("/ui") {
                            try await stream("Here is a sample:\n```openui\nroot = Card([t, b], \"Echo UI\")\nt = Table([\"Word\", \"Length\"], [[\"echo\", 4], [\"pocket\", 6]])\nb = Buttons([Button(\"Again\", @ToAssistant(\"/ui again\"))])\n```\nDone.")
                        } else {
                            try await stream("Echo: " + text)
                        }
                    }
                    continuation.yield(.end)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
