// Component library + system prompt for OpenUI Lang. Prompt wording adapted from
// @openuidev/lang-core prompt.ts (MIT License, Copyright (c) Thesys Inc.).
import Foundation

public indirect enum ParamType: Equatable, Sendable {
    case string, number, boolean, any
    case enumeration([String])
    case array(ParamType)
    /// Allowed component names; empty = any component.
    case component([String])
    case action
    case object([(String, ParamType)])
    case oneOf([ParamType])

    public static func == (a: ParamType, b: ParamType) -> Bool { a.text == b.text }

    /// Signature text (OpenUI prompt spec §5.1).
    public var text: String {
        switch self {
        case .string: "string"
        case .number: "number"
        case .boolean: "boolean"
        case .any: "any"
        case .enumeration(let v): v.map { "\"\($0)\"" }.joined(separator: " | ")
        case .array(let t):
            if case .oneOf = t { "(\(t.text))[]" } else if case .enumeration = t { "(\(t.text))[]" } else { "\(t.text)[]" }
        case .component(let names): names.isEmpty ? "Component" : names.joined(separator: " | ")
        case .action: "ActionExpression"
        case .object(let fields): "{" + fields.map { "\($0.0): \($0.1.text)" }.joined(separator: ", ") + "}"
        case .oneOf(let types): types.map(\.text).joined(separator: " | ")
        }
    }
}

public struct ComponentParam: Sendable, Equatable {
    public var name: String
    public var type: ParamType
    public var required: Bool
}

public struct ComponentSpec: Sendable, Equatable {
    public var name: String
    public var summary: String
    public var params: [ComponentParam]

    public var signature: String {
        name + "(" + params.map { "\($0.name)\($0.required ? "" : "?"): \($0.type.text)" }.joined(separator: ", ") + ")"
    }
}

public enum ComponentRegistry {
    static func p(_ name: String, _ type: ParamType, _ required: Bool = true) -> ComponentParam {
        ComponentParam(name: name, type: type, required: required)
    }

    static let content: ParamType = .oneOf([.component([]), .string, .number])

    /// The v1 components (first ten) plus the child-only Button and Field.
    public static let v1: [ComponentSpec] = [
        ComponentSpec(name: "Card", summary: "Container; children stack vertically. Strings render as text paragraphs.",
                      params: [p("children", .array(content)), p("title", .string, false)]),
        ComponentSpec(name: "Table", summary: "Rows of data under column headers. Use for comparisons and specs.",
                      params: [p("columns", .array(.string)), p("rows", .array(.array(.any))), p("caption", .string, false)]),
        ComponentSpec(name: "List", summary: "Bulleted list, or numbered when ordered is true.",
                      params: [p("items", .array(content)), p("ordered", .boolean, false)]),
        ComponentSpec(name: "Chart", summary: "bar/line: one value per label per series. pie: first series only.",
                      params: [p("type", .enumeration(["bar", "line", "pie"])), p("labels", .array(.string)),
                               p("series", .array(.object([("name", .string), ("values", .array(.number))]))), p("title", .string, false)]),
        ComponentSpec(name: "Callout", summary: "Highlighted note. kind sets the color and icon.",
                      params: [p("text", .string), p("kind", .enumeration(["info", "success", "warning", "error"]), false), p("title", .string, false)]),
        ComponentSpec(name: "CodeBlock", summary: "Monospaced code with a copy button.",
                      params: [p("code", .string), p("language", .string, false)]),
        ComponentSpec(name: "Steps", summary: "Numbered steps; current is the 1-based step in progress.",
                      params: [p("items", .array(.string)), p("current", .number, false)]),
        ComponentSpec(name: "Buttons", summary: "Row of buttons the user can tap to reply.",
                      params: [p("buttons", .array(.component(["Button"])))]),
        ComponentSpec(name: "Form", summary: "Input form. Submitting sends the values back to you as the next user message.",
                      params: [p("id", .string), p("fields", .array(.component(["Field"]))), p("submitLabel", .string, false)]),
        ComponentSpec(name: "Image", summary: "Image from an https URL.",
                      params: [p("url", .string), p("alt", .string, false), p("caption", .string, false)]),
        ComponentSpec(name: "Button", summary: "Only inside Buttons. Without an action, tapping sends the label as the user's reply.",
                      params: [p("label", .string), p("action", .action, false), p("style", .enumeration(["primary", "secondary"]), false)]),
        ComponentSpec(name: "Field", summary: "Only inside Form. options are for kind \"select\".",
                      params: [p("name", .string), p("label", .string),
                               p("kind", .enumeration(["text", "number", "email", "multiline", "toggle", "select"]), false),
                               p("options", .array(.string), false), p("placeholder", .string, false)]),
    ]

    /// Component used as the entry when there is no `root` statement.
    public static let rootComponent = "Card"

    private static let byName: [String: ComponentSpec] = Dictionary(uniqueKeysWithValues: v1.map { ($0.name, $0) })

    public static func spec(_ name: String) -> ComponentSpec? { byName[name] }

    /// The OpenUI section of the system prompt (inline mode: markdown with ```openui blocks).
    public static func systemPrompt() -> String {
        let components = v1.map { "\($0.signature) — \($0.summary)" }.joined(separator: "\n")
        return """
        ## Rich UI (OpenUI)

        Answer in normal markdown. When a table, chart, list of options, steps, or a form would help, add a block of openui code inside a ```openui fence. Questions that need only words get only words.

        ### Syntax
        1. Each statement is on its own line: `identifier = Expression`
        2. `root = Card(...)` is the entry point and the FIRST line of every block
        3. Expressions: strings ("..."), numbers, true/false, null, arrays ([...]), objects ({key: value}), or component calls Name(arg1, arg2, ...)
        4. Arguments are POSITIONAL (order matters, not names). Optional arguments can be left out from the end; write null to skip one in the middle
        5. Use references: define `name = ...` on its own line and use `name` in its parent. Every name other than root must be used by a parent, or it will not show
        6. Strings use double quotes with backslash escaping. `+` joins strings

        ### Components
        \(components)

        ### Actions
        @ToAssistant("message") — send a message to you as the user's reply
        @OpenUrl("https://...") — open a web page
        Action([@ToAssistant("a"), @OpenUrl("b")]) — several actions in order

        ### Example
        ```openui
        root = Card([intro, table, pick], "Laptops")
        intro = "Two good options:"
        table = Table(["Model", "Price"], [["Air", 999], ["Pro", 1599]])
        pick = Buttons([Button("Tell me more about Air"), Button("Compare batteries", @ToAssistant("Compare their battery life"))])
        ```

        Write root first, then its children, then the data, so the UI appears while you write.
        """
    }
}
