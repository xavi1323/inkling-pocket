import Foundation

/// What a tap or form submit asks the host to do.
public enum OpenUIAction: Equatable, Sendable {
    case sendMessage(String)
    case submitForm(id: String, values: [String: String])
    case openURL(String)
}

/// Text of the follow-up user message for an action (`openURL` has none and returns "").
public func actionMessage(_ action: OpenUIAction) -> String {
    switch action {
    case .sendMessage(let text): text
    case .submitForm(let id, let values):
        "Form \(id) submitted: " + values.keys.sorted().map { "\($0)=\(values[$0]!)" }.joined(separator: ", ")
    case .openURL: ""
    }
}

/// Actions for a Button node: its action steps, or "send the label" when it has none.
public func buttonActions(_ node: OUINode) -> [OpenUIAction] {
    guard case .action(let steps)? = node["action"], !steps.isEmpty else {
        return [.sendMessage(node["label"]?.stringValue ?? "")]
    }
    return steps.compactMap { step in
        switch step {
        case .toAssistant(let message, let context):
            return .sendMessage(context.map { message + "\ncontext = " + $0.summary } ?? message)
        case .openURL(let url): return .openURL(url)
        case .other: return nil
        }
    }
}

public enum ChartKind: Equatable, Sendable { case bar, line, pie, table }

/// Chart type from the program; anything unknown is shown as a table.
public func chartKind(_ type: String?) -> ChartKind {
    switch type?.lowercased() {
    case "bar": .bar
    case "line": .line
    case "pie": .pie
    default: .table
    }
}
