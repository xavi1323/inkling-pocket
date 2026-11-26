// Swift implementation of OpenUI Lang 1.1 (spec: github.com/thesysdev/openui, MIT License, Copyright (c) Thesys Inc.).
import Foundation

public enum OUIActionStep: Equatable, Sendable {
    case toAssistant(message: String, context: OUIValue?)
    case openURL(String)
    /// @Set, @Reset, @Run and custom actions: kept so they round-trip, not run by this client.
    case other(name: String, args: [OUIValue])
}

public indirect enum OUIValue: Equatable, Sendable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case null
    case array([OUIValue])
    case object([String: OUIValue])
    case node(OUINode)
    case ref(String)
    case action([OUIActionStep])

    public var stringValue: String? { if case .string(let s) = self { s } else { nil } }
    public var numberValue: Double? { if case .number(let n) = self { n } else { nil } }
    public var arrayValue: [OUIValue]? { if case .array(let a) = self { a } else { nil } }
    public var nodeValue: OUINode? { if case .node(let n) = self { n } else { nil } }
    public var boolValue: Bool? { if case .bool(let b) = self { b } else { nil } }

    /// Text shown when a value sits in a component position (numbers in JavaScript form).
    public var displayText: String {
        switch self {
        case .string(let s): s
        case .number(let n): OUIValue.formatNumber(n)
        case .bool(let b): b ? "true" : "false"
        case .null, .node, .action, .ref: ""
        case .array(let a): a.map(\.displayText).joined(separator: ",")
        case .object: "[object Object]"
        }
    }

    /// JavaScript Number::toString for the common cases.
    public static func formatNumber(_ n: Double) -> String {
        if n.isNaN { return "NaN" }
        if n.isInfinite { return n > 0 ? "Infinity" : "-Infinity" }
        if n == n.rounded(), abs(n) < 1e21 { return String(Int64(n)) }
        return "\(n)"
    }

    var summary: String {
        switch self {
        case .string(let s): OUIValue.quote(s)
        case .number(let n): OUIValue.formatNumber(n)
        case .bool(let b): b ? "true" : "false"
        case .null: "null"
        case .array(let a): "[" + a.map(\.summary).joined(separator: ", ") + "]"
        case .object(let o): "{" + o.keys.sorted().map { "\($0): \(o[$0]!.summary)" }.joined(separator: ", ") + "}"
        case .node(let n): n.summary
        case .ref(let r): r
        case .action(let steps): steps.map(\.summary).joined(separator: " + ")
        }
    }

    static func quote(_ s: String) -> String {
        var out = "\""
        for ch in s.unicodeScalars {
            switch ch {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\t": out += "\\t"
            case "\r": out += "\\r"
            default: out.unicodeScalars.append(ch)
            }
        }
        return out + "\""
    }
}

extension OUIActionStep {
    var summary: String {
        switch self {
        case .toAssistant(let m, let c): "@ToAssistant(" + ([OUIValue.quote(m)] + (c.map { [$0.summary] } ?? [])).joined(separator: ", ") + ")"
        case .openURL(let u): "@OpenUrl(\(OUIValue.quote(u)))"
        case .other(let n, let a): "@\(n)(" + a.map(\.summary).joined(separator: ", ") + ")"
        }
    }
}

public struct OUINode: Equatable, Sendable {
    public var component: String
    /// Validated positional arguments (`.null` where an optional one was left out).
    public var args: [OUIValue]
    /// The same arguments keyed by parameter name (left-out ones absent).
    public var named: [String: OUIValue]

    public init(component: String, args: [OUIValue], named: [String: OUIValue]) {
        self.component = component
        self.args = args
        self.named = named
    }

    public subscript(_ name: String) -> OUIValue? { named[name] }

    /// Compact text form: `Card(children: [...], title: "x")`, params in schema order.
    public var summary: String {
        let order = ComponentRegistry.spec(component)?.params.map(\.name) ?? named.keys.sorted()
        return component + "(" + order.compactMap { k in named[k].map { "\(k): \($0.summary)" } }.joined(separator: ", ") + ")"
    }

    /// This node plus every node inside its arguments.
    public var nodeCount: Int {
        func count(_ v: OUIValue) -> Int {
            switch v {
            case .node(let n): n.nodeCount
            case .array(let a): a.reduce(0) { $0 + count($1) }
            case .object(let o): o.values.reduce(0) { $0 + count($1) }
            default: 0
            }
        }
        return 1 + named.values.reduce(0) { $0 + count($1) }
    }
}

public struct OpenUIDocument: Equatable, Sendable {
    public var root: OUINode?
    /// Evaluated value statements and state defaults (`$name`).
    public var bindings: [String: OUIValue]
    /// True when the input ended inside an unclosed string or bracket.
    public var incomplete: Bool

    public init(root: OUINode? = nil, bindings: [String: OUIValue] = [:], incomplete: Bool = false) {
        self.root = root
        self.bindings = bindings
        self.incomplete = incomplete
    }
}
