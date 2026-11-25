import Foundation

/// Native re-implementation of Inkling 2's `chat_template.jinja`.
/// Output must be byte-identical to the Jinja render (see PromptFormatterTests).
public enum PromptFormatter {
    public enum Token {
        public static let user = "<|message_user|>"
        public static let model = "<|message_model|>"
        public static let system = "<|message_system|>"
        public static let tool = "<|message_tool|>"
        public static let text = "<|content_text|>"
        public static let thinking = "<|content_thinking|>"
        public static let invokeToolJSON = "<|content_invoke_tool_json|>"
        public static let xml = "<|content_xml|>"
        public static let endMessage = "<|end_message|>"
        public static let endSampling = "<|content_model_end_sampling|>"
    }

    public static func render(messages: [ChatMessage], tools: [ToolSpec], effort: ReasoningEffort) -> String {
        var out = ""
        if !tools.isEmpty {
            let specs = JSONValue.array(tools.map { t in
                .object([
                    ("description", .string(t.description)),
                    ("name", .string(t.name)),
                    ("parameters", t.parameters),
                    ("type", .string("function")),
                ])
            })
            out += Token.system + "tool_declare" + Token.xml + specs.canonicalJSON(sortKeys: true) + Token.endMessage
        }
        var effortEmitted = false
        for m in messages {
            if !effortEmitted && m.role != .system {
                out += effortLine(effort)
                effortEmitted = true
            }
            out += render(m)
        }
        if !effortEmitted { out += effortLine(effort) }
        return out
    }

    static func effortLine(_ effort: ReasoningEffort) -> String {
        let v = effort.value
        let num = v == 0 ? "0" : JSONValue.pythonFloat(v)
        return Token.system + Token.text + "Thinking effort level: " + num + Token.endMessage
    }

    static func roleToken(_ role: Role) -> String {
        switch role {
        case .user: Token.user
        case .assistant: Token.model
        case .system: Token.system
        case .tool: Token.tool
        }
    }

    static func render(_ m: ChatMessage) -> String {
        var header = roleToken(m.role)
        if m.role == .tool, let name = m.toolName, !name.isEmpty { header += name }
        var out = ""
        let reasoning = m.reasoning ?? ""
        if m.role == .assistant, !reasoning.isEmpty {
            out += header + Token.thinking + reasoning + Token.endMessage
        }
        let isAssistantWithExtras = m.role == .assistant && (!m.toolCalls.isEmpty || !reasoning.isEmpty)
        if !m.content.isEmpty || !isAssistantWithExtras {
            out += header + Token.text + m.content + Token.endMessage
        }
        if m.role == .assistant {
            for call in m.toolCalls {
                let args = call.arguments.sorted { $0.0.unicodeScalars.lexicographicallyPrecedes($1.0.unicodeScalars) }
                    .map { JSONValue.string($0.0).canonicalJSON(sortKeys: false) + ":" + $0.1.canonicalJSON(sortKeys: false) }
                    .joined(separator: ",")
                out += Token.model + call.name + Token.invokeToolJSON
                out += "{\"name\":" + JSONValue.string(call.name).canonicalJSON(sortKeys: false) + ",\"args\":{" + args + "}}"
                out += Token.endMessage
            }
            out += Token.endSampling
        }
        return out
    }
}
