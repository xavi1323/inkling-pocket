import Foundation
import Testing
@testable import InklingCore

/// Fixtures are rendered from the model's chat_template.jinja by scripts/render_template_fixtures.py.
/// Parsed with JSONValue.parse so argument key order is preserved.
struct PromptFixture {
    let root: JSONValue
    var effort: String { root["effort"]!.stringValue! }
    var expected: String { root["expected"]!.stringValue! }

    var chatMessages: [ChatMessage] {
        guard case .array(let items) = root["messages"]! else { return [] }
        return items.map { m in
            var calls: [ToolCall] = []
            if case .array(let tcs)? = m["tool_calls"] {
                for tc in tcs {
                    let fn = tc["function"]!
                    var args: [(String, JSONValue)] = []
                    if case .object(let pairs)? = fn["arguments"] { args = pairs }
                    calls.append(ToolCall(name: fn["name"]!.stringValue!, arguments: args))
                }
            }
            return ChatMessage(role: Role(rawValue: m["role"]!.stringValue!)!,
                               content: m["content"]?.stringValue ?? "",
                               reasoning: m["reasoning_content"]?.stringValue,
                               toolCalls: calls,
                               toolName: m["name"]?.stringValue)
        }
    }
    var toolSpecs: [ToolSpec] {
        guard case .array(let items) = root["tools"]! else { return [] }
        return items.map { t in
            let fn = t["function"]!
            return ToolSpec(name: fn["name"]!.stringValue!, description: fn["description"]!.stringValue!,
                            parameters: fn["parameters"]!)
        }
    }
}

let promptFixtureNames = [
    "a_user_only", "b_system_user", "c_tools_user", "d_reasoning_history",
    "e_tool_roundtrip", "f_none_unicode", "g_xhigh_error_tool",
]

func loadPromptFixture(_ name: String) throws -> PromptFixture {
    let url = Bundle.module.url(forResource: "Fixtures/prompts/\(name)", withExtension: "json")!
    return PromptFixture(root: try JSONValue.parse(String(contentsOf: url, encoding: .utf8)))
}

@Test(arguments: promptFixtureNames)
func matchesJinja(name: String) throws {
    let f = try loadPromptFixture(name)
    let rendered = PromptFormatter.render(
        messages: f.chatMessages, tools: f.toolSpecs, effort: ReasoningEffort(rawValue: f.effort)!)
    #expect(rendered == f.expected)
}

@Test func effortValues() {
    #expect(ReasoningEffort.none.value == 0)
    #expect(ReasoningEffort.low.value == 0.2)
    #expect(ReasoningEffort.max.value == 0.99)
}

@Test func canonicalJSONSortsOnlyWhenAsked() {
    let v: JSONValue = .object([("z", .int(1)), ("a", .array([.double(1.5), .bool(true), .null]))])
    #expect(v.canonicalJSON(sortKeys: false) == #"{"z":1,"a":[1.5,true,null]}"#)
    #expect(v.canonicalJSON(sortKeys: true) == #"{"a":[1.5,true,null],"z":1}"#)
    #expect(JSONValue.string("é\"\n\u{01}/").canonicalJSON(sortKeys: false) == #""é\"\n\u0001/""#)
}

@Test func jsonValueParsesPreservingOrderAndIntVsDouble() throws {
    let v = try JSONValue.parse(#"{"b":2,"a":2.0}"#)
    #expect(v == .object([("b", .int(2)), ("a", .double(2.0))]))
}
