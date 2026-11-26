import Foundation
import InklingCore
@testable import AgentCore

/// Fake model: each generate() call replays the next scripted event list and records its input.
final class ScriptedModel: LanguageModel, @unchecked Sendable {
    private let lock = NSLock()
    private var scripts: [[ModelEvent]]
    private(set) var receivedMessages: [[ChatMessage]] = []
    private(set) var receivedParameters: [GenerateParameters] = []

    init(_ scripts: [[ModelEvent]]) { self.scripts = scripts }

    func generate(_ messages: [ChatMessage], tools: [ToolSpec], effort: ReasoningEffort,
                  parameters: GenerateParameters) -> AsyncThrowingStream<ModelEvent, Error> {
        lock.lock()
        receivedMessages.append(messages)
        receivedParameters.append(parameters)
        let events = scripts.isEmpty ? [.answer("(no script)"), .end] : scripts.removeFirst()
        lock.unlock()
        return AsyncThrowingStream { c in
            for e in events { c.yield(e) }
            c.finish()
        }
    }

    /// 1 token per 4 characters, like a rough BPE.
    func countTokens(_ text: String) -> Int { (text.count + 3) / 4 }
}

struct EchoTool: Tool {
    var spec: ToolSpec { ToolSpec(name: "echo", description: "Echo", parameters: .object([("type", .string("object"))])) }
    func run(_ arguments: [(String, JSONValue)]) async throws -> String {
        arguments.first { $0.0 == "text" }?.1.stringValue ?? ""
    }
}

struct HugeTool: Tool {
    var spec: ToolSpec { ToolSpec(name: "huge", description: "Huge", parameters: .object([])) }
    func run(_ arguments: [(String, JSONValue)]) async throws -> String { String(repeating: "word ", count: 4000) }
}

func collect(_ stream: AsyncThrowingStream<AgentEvent, Error>) async throws -> [AgentEvent] {
    var out: [AgentEvent] = []
    for try await e in stream { out.append(e) }
    return out
}
