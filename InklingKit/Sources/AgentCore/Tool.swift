import InklingCore

public enum ToolError: Error, Equatable, CustomStringConvertible {
    case message(String)
    case missingArgument(String)
    public var description: String {
        switch self {
        case .message(let m): m
        case .missingArgument(let a): "missing argument '\(a)'"
        }
    }
}

public protocol Tool: Sendable {
    var spec: ToolSpec { get }
    func run(_ arguments: [(String, JSONValue)]) async throws -> String
}

extension Tool {
    public var name: String { spec.name }
}

/// Reads a required string argument.
public func stringArgument(_ args: [(String, JSONValue)], _ key: String) throws -> String {
    guard let v = args.first(where: { $0.0 == key })?.1.stringValue, !v.isEmpty else {
        throw ToolError.missingArgument(key)
    }
    return v
}

/// JSON schema helper: object with string properties.
public func objectSchema(_ props: [(name: String, description: String)], required: [String]) -> JSONValue {
    .object([
        ("type", .string("object")),
        ("properties", .object(props.map { ($0.name, .object([("type", .string("string")), ("description", .string($0.description))])) })),
        ("required", .array(required.map { .string($0) })),
    ])
}

public struct ToolRegistry: Sendable {
    private let tools: [any Tool]
    public init(_ tools: [any Tool]) { self.tools = tools }
    public var specs: [ToolSpec] { tools.map(\.spec) }
    public func tool(named name: String) -> (any Tool)? { tools.first { $0.spec.name == name } }
}
