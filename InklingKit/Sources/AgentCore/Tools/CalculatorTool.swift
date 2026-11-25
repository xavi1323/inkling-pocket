import Foundation
import InklingCore

/// Safe arithmetic: + - * / ^ ( ), decimals, unary minus. No NSExpression.
public struct CalculatorTool: Tool {
    public init() {}
    public var spec: ToolSpec {
        ToolSpec(name: "calculator", description: "Evaluate an arithmetic expression exactly. Supports + - * / ^ and parentheses.",
                 parameters: objectSchema([("expression", "The expression, e.g. 2*(3+4)")], required: ["expression"]))
    }

    public func run(_ arguments: [(String, JSONValue)]) async throws -> String {
        var p = ExpressionParser(try stringArgument(arguments, "expression"))
        let value = try p.parse()
        guard value.isFinite else { throw ToolError.message("result is not a finite number") }
        if value == value.rounded(), abs(value) < 1e15 { return String(Int64(value)) }
        return "\(value)"
    }
}

struct ExpressionParser {
    private let chars: [Character]
    private var i = 0
    init(_ text: String) { chars = Array(text.filter { !$0.isWhitespace }.replacingOccurrences(of: "×", with: "*").replacingOccurrences(of: "÷", with: "/")) }

    mutating func parse() throws -> Double {
        let v = try expression()
        guard i == chars.count else { throw ToolError.message("unexpected '\(chars[i])' in expression") }
        return v
    }
    private mutating func expression() throws -> Double {
        var v = try term()
        while i < chars.count, chars[i] == "+" || chars[i] == "-" {
            let op = chars[i]; i += 1
            let r = try term()
            v = op == "+" ? v + r : v - r
        }
        return v
    }
    private mutating func term() throws -> Double {
        var v = try power()
        while i < chars.count, chars[i] == "*" || chars[i] == "/" {
            let op = chars[i]; i += 1
            let r = try power()
            if op == "/" {
                guard r != 0 else { throw ToolError.message("division by zero") }
                v /= r
            } else { v *= r }
        }
        return v
    }
    private mutating func power() throws -> Double {
        let base = try unary()
        if i < chars.count, chars[i] == "^" {
            i += 1
            return pow(base, try power())  // right associative
        }
        return base
    }
    private mutating func unary() throws -> Double {
        if i < chars.count, chars[i] == "-" { i += 1; return -(try unary()) }
        if i < chars.count, chars[i] == "+" { i += 1; return try unary() }
        return try primary()
    }
    private mutating func primary() throws -> Double {
        guard i < chars.count else { throw ToolError.message("expression ends too early") }
        if chars[i] == "(" {
            i += 1
            let v = try expression()
            guard i < chars.count, chars[i] == ")" else { throw ToolError.message("missing ')'") }
            i += 1
            return v
        }
        let start = i
        while i < chars.count, chars[i].isNumber || chars[i] == "." { i += 1 }
        guard i > start, let v = Double(String(chars[start..<i])) else {
            throw ToolError.message("expected a number at position \(start + 1)")
        }
        return v
    }
}
