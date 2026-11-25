import Foundation
import InklingCore

public struct CurrentTimeTool: Tool {
    public init() {}
    public var spec: ToolSpec {
        ToolSpec(name: "current_time", description: "Get the current local date and time.",
                 parameters: objectSchema([], required: []))
    }
    public func run(_ arguments: [(String, JSONValue)]) async throws -> String {
        let now = Date()
        let iso = ISO8601DateFormatter()
        iso.timeZone = .current
        let human = DateFormatter()
        human.dateFormat = "EEEE d MMMM yyyy, HH:mm zzz"
        return iso.string(from: now) + "\n" + human.string(from: now)
    }
}
