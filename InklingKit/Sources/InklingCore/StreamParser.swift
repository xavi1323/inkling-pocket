import Foundation

/// What the model is saying, split out of the raw token text.
public enum ModelEvent: Sendable, Equatable {
    case thinking(String)
    case answer(String)
    case toolCall(ToolCall)
    case toolCallError(raw: String)
    case end
}

/// Streaming parser for Inkling 2 output. Feed decoded text in any chunk sizes;
/// markers split across chunks are held back until complete.
public struct InklingStreamParser: Sendable {
    private enum State: Equatable {
        case outside          // between messages
        case header           // after <|message_model|>, reading an optional name
        case thinking
        case answer
        case toolJSON
        case done
    }

    private static let messageModel = "<|message_model|>"
    private static let endMessage = "<|end_message|>"
    private static let endSampling = "<|content_model_end_sampling|>"
    private static let contentMarkers: [(String, State)] = [
        ("<|content_thinking|>", .thinking),
        ("<|content_text|>", .answer),
        ("<|content_tool_error|>", .answer),
        ("<|content_invoke_tool_text|>", .answer),
        ("<|content_invoke_tool_json|>", .toolJSON),
    ]
    private static let allMarkers: [String] = [messageModel, endMessage, endSampling] + contentMarkers.map(\.0)

    private var state: State = .outside
    private var buffer = ""
    private var headerName = ""
    private var toolBuffer = ""

    public init() {}

    public mutating func feed(_ text: String) -> [ModelEvent] {
        buffer += text
        return drain(final: false)
    }

    public mutating func finish() -> [ModelEvent] {
        var events = drain(final: true)
        if !buffer.isEmpty { events += emitText(buffer); buffer = "" }
        if state == .toolJSON, !toolBuffer.isEmpty { events.append(.toolCallError(raw: toolBuffer)); toolBuffer = "" }
        return events
    }

    private mutating func drain(final: Bool) -> [ModelEvent] {
        var events: [ModelEvent] = []
        while !buffer.isEmpty, state != .done {
            guard let open = buffer.range(of: "<|") else {
                // No marker start. Keep a trailing "<" (could begin a marker) unless final.
                if !final, buffer.hasSuffix("<") {
                    events += emitText(String(buffer.dropLast()))
                    buffer = "<"
                } else {
                    events += emitText(buffer)
                    buffer = ""
                }
                break
            }
            if open.lowerBound > buffer.startIndex {
                events += emitText(String(buffer[buffer.startIndex..<open.lowerBound]))
                buffer.removeSubrange(buffer.startIndex..<open.lowerBound)
            }
            if let marker = Self.allMarkers.first(where: { buffer.hasPrefix($0) }) {
                buffer.removeFirst(marker.count)
                events += handle(marker)
                continue
            }
            if !final, Self.allMarkers.contains(where: { $0.hasPrefix(buffer) }) {
                break  // partial marker: wait for more text
            }
            // "<|" that is not a marker: plain text.
            events += emitText("<|")
            buffer.removeFirst(2)
        }
        return events
    }

    private mutating func handle(_ marker: String) -> [ModelEvent] {
        switch marker {
        case Self.messageModel:
            state = .header
            headerName = ""
            return []
        case Self.endMessage:
            defer { state = .outside }
            if state == .toolJSON {
                let raw = toolBuffer
                toolBuffer = ""
                return [parseToolCall(raw)]
            }
            return []
        case Self.endSampling:
            state = .done
            return [.end]
        default:
            if let (_, next) = Self.contentMarkers.first(where: { $0.0 == marker }) {
                state = next
                toolBuffer = ""
            }
            return []
        }
    }

    private mutating func emitText(_ text: String) -> [ModelEvent] {
        guard !text.isEmpty else { return [] }
        switch state {
        case .thinking: return [.thinking(text)]
        case .answer, .outside: return [.answer(text)]
        case .header: headerName += text; return []
        case .toolJSON: toolBuffer += text; return []
        case .done: return []
        }
    }

    private func parseToolCall(_ raw: String) -> ModelEvent {
        guard let json = try? JSONValue.parse(raw),
              let name = json["name"]?.stringValue ?? (headerName.isEmpty ? nil : headerName) else {
            return .toolCallError(raw: raw)
        }
        var args: [(String, JSONValue)] = []
        if case .object(let pairs)? = json["args"] { args = pairs }
        return .toolCall(ToolCall(name: name, arguments: args))
    }
}
