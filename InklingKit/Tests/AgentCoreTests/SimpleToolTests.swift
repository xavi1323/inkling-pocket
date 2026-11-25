import Foundation
import Testing
import InklingCore
@testable import AgentCore

@Test(arguments: [("2*(3+4)", "14"), ("1.5+1.5", "3"), ("2^3^2", "512"), ("-3+10/4", "-0.5"), ("(1+2)*-2", "-6")])
func calculatorEvaluates(expr: String, expected: String) async throws {
    #expect(try await CalculatorTool().run([("expression", .string(expr))]) == expected)
}

@Test func calculatorDivisionByZeroThrows() async {
    await #expect(throws: ToolError.self) { try await CalculatorTool().run([("expression", .string("1/0"))]) }
}

@Test func calculatorRejectsGarbage() async {
    await #expect(throws: ToolError.self) { try await CalculatorTool().run([("expression", .string("2+*"))]) }
}

@Test func currentTimeISO8601() async throws {
    let out = try await CurrentTimeTool().run([])
    let iso = out.split(separator: "\n").first.map(String.init) ?? ""
    #expect(ISO8601DateFormatter().date(from: iso) != nil)
}

@Test func registryFindsToolsByName() {
    let r = ToolRegistry([CalculatorTool(), CurrentTimeTool()])
    #expect(r.tool(named: "calculator") != nil)
    #expect(r.tool(named: "missing") == nil)
    #expect(r.specs.map(\.name) == ["calculator", "current_time"])
}
