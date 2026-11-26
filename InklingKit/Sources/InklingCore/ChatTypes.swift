import Foundation

public enum Role: String, Codable, Sendable { case system, user, assistant, tool }

public struct ToolCall: Codable, Sendable, Equatable {
    public var name: String
    /// Top-level arguments in the order the model wrote them.
    public var arguments: [(String, JSONValue)]

    public init(name: String, arguments: [(String, JSONValue)]) {
        self.name = name
        self.arguments = arguments
    }

    public func argument(_ key: String) -> JSONValue? { arguments.first { $0.0 == key }?.1 }

    public static func == (a: ToolCall, b: ToolCall) -> Bool {
        a.name == b.name && JSONValue.object(a.arguments) == JSONValue.object(b.arguments)
    }

    enum CodingKeys: String, CodingKey { case name, arguments }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        name = try c.decode(String.self, forKey: .name)
        guard case .object(let pairs) = try c.decode(JSONValue.self, forKey: .arguments) else {
            arguments = []
            return
        }
        arguments = pairs
    }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(name, forKey: .name)
        try c.encode(JSONValue.object(arguments), forKey: .arguments)
    }
}

public struct ChatMessage: Codable, Sendable, Equatable {
    public var role: Role
    public var content: String
    public var reasoning: String?
    public var toolCalls: [ToolCall]
    /// Tool messages: the tool's name (rendered after `<|message_tool|>`).
    public var toolName: String?
    /// UI-only flag. The chat template has no error form, so errors render as text.
    public var isToolError: Bool

    public init(role: Role, content: String, reasoning: String? = nil, toolCalls: [ToolCall] = [],
                toolName: String? = nil, isToolError: Bool = false) {
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.toolName = toolName
        self.isToolError = isToolError
    }
}

public struct ToolSpec: Codable, Sendable, Equatable {
    public var name: String
    public var description: String
    public var parameters: JSONValue

    public init(name: String, description: String, parameters: JSONValue) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

public enum ReasoningEffort: String, CaseIterable, Codable, Sendable {
    case none, minimal, low, medium, high, xhigh, max

    /// Same numbers as the chat template's `effort_map`.
    public var value: Double {
        switch self {
        case .none: 0.0
        case .minimal: 0.1
        case .low: 0.2
        case .medium: 0.7
        case .high: 0.9
        case .xhigh, .max: 0.99
        }
    }

    /// Most thinking tokens per reply before the thought is closed and the model answers (nil: no cap). Thinking is
    /// most of the wait for an answer; low effort keeps it short.
    public var thinkingTokenCap: Int? {
        switch self {
        case .none, .minimal: 128
        case .low: 384
        case .medium: 1024
        case .high: 2048
        case .xhigh, .max: nil
        }
    }
}
