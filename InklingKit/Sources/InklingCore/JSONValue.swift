import Foundation

/// JSON value that keeps object key order and the int/double distinction,
/// so prompts can be rendered byte-identical to Python's `json.dumps`.
public indirect enum JSONValue: Sendable, Equatable {
    case null
    case bool(Bool)
    case int(Int64)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([(String, JSONValue)])

    public static func == (lhs: JSONValue, rhs: JSONValue) -> Bool {
        switch (lhs, rhs) {
        case (.null, .null): true
        case let (.bool(a), .bool(b)): a == b
        case let (.int(a), .int(b)): a == b
        case let (.double(a), .double(b)): a == b
        case let (.string(a), .string(b)): a == b
        case let (.array(a), .array(b)): a == b
        case let (.object(a), .object(b)):
            a.count == b.count && zip(a, b).allSatisfy { $0.0 == $1.0 && $0.1 == $1.1 }
        default: false
        }
    }

    /// Value for `key` in an object (first match), else nil.
    public subscript(key: String) -> JSONValue? {
        guard case .object(let pairs) = self else { return nil }
        return pairs.first { $0.0 == key }?.1
    }

    public var stringValue: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    /// Compact JSON (`,` and `:` separators, non-ASCII kept), like
    /// `json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=sortKeys)`.
    public func canonicalJSON(sortKeys: Bool) -> String {
        var out = ""
        write(into: &out, sortKeys: sortKeys)
        return out
    }

    private func write(into out: inout String, sortKeys: Bool) {
        switch self {
        case .null: out += "null"
        case .bool(let b): out += b ? "true" : "false"
        case .int(let i): out += String(i)
        case .double(let d): out += JSONValue.pythonFloat(d)
        case .string(let s): JSONValue.writeString(s, into: &out)
        case .array(let items):
            out += "["
            for (i, item) in items.enumerated() {
                if i > 0 { out += "," }
                item.write(into: &out, sortKeys: sortKeys)
            }
            out += "]"
        case .object(let pairs):
            let ordered = sortKeys ? pairs.sorted { $0.0.unicodeScalars.lexicographicallyPrecedes($1.0.unicodeScalars) } : pairs
            out += "{"
            for (i, (key, value)) in ordered.enumerated() {
                if i > 0 { out += "," }
                JSONValue.writeString(key, into: &out)
                out += ":"
                value.write(into: &out, sortKeys: sortKeys)
            }
            out += "}"
        }
    }

    static func pythonFloat(_ d: Double) -> String {
        if d.isNaN { return "NaN" }
        if d.isInfinite { return d > 0 ? "Infinity" : "-Infinity" }
        var s = "\(d)"  // shortest round-trip, like Python repr
        if s.contains("e") {
            // Swift: 1e-05 / 1e+20 ; Python: 1e-05 / 1e+20 — same, but Swift may print "1e-5".
            let parts = s.split(separator: "e")
            var exp = String(parts[1])
            let sign = exp.hasPrefix("-") ? "-" : "+"
            exp = exp.trimmingCharacters(in: CharacterSet(charactersIn: "+-"))
            if exp.count < 2 { exp = "0" + exp }
            s = "\(parts[0])e\(sign)\(exp)"
        }
        return s
    }

    static func writeString(_ s: String, into out: inout String) {
        out += "\""
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        out += "\""
    }
}

// MARK: - Parsing (order-preserving)

public enum JSONParseError: Error, Equatable {
    case unexpectedEnd
    case unexpectedCharacter(Int)
    case trailingData(Int)
}

extension JSONValue {
    /// Parses JSON text, keeping object key order and int-vs-double.
    public static func parse(_ text: String) throws -> JSONValue {
        var p = JSONParser(Array(text.utf8))
        let value = try p.parseValue()
        p.skipWhitespace()
        if p.i != p.bytes.count { throw JSONParseError.trailingData(p.i) }
        return value
    }
}

private struct JSONParser {
    let bytes: [UInt8]
    var i = 0
    init(_ bytes: [UInt8]) { self.bytes = bytes }

    mutating func skipWhitespace() {
        while i < bytes.count, [0x20, 0x0A, 0x0D, 0x09].contains(bytes[i]) { i += 1 }
    }

    mutating func parseValue() throws -> JSONValue {
        skipWhitespace()
        guard i < bytes.count else { throw JSONParseError.unexpectedEnd }
        switch bytes[i] {
        case UInt8(ascii: "{"): return try parseObject()
        case UInt8(ascii: "["): return try parseArray()
        case UInt8(ascii: "\""): return .string(try parseString())
        case UInt8(ascii: "t"): try expect("true"); return .bool(true)
        case UInt8(ascii: "f"): try expect("false"); return .bool(false)
        case UInt8(ascii: "n"): try expect("null"); return .null
        default: return try parseNumber()
        }
    }

    mutating func expect(_ word: String) throws {
        for b in word.utf8 {
            guard i < bytes.count else { throw JSONParseError.unexpectedEnd }
            guard bytes[i] == b else { throw JSONParseError.unexpectedCharacter(i) }
            i += 1
        }
    }

    mutating func parseObject() throws -> JSONValue {
        i += 1
        var pairs: [(String, JSONValue)] = []
        skipWhitespace()
        if i < bytes.count, bytes[i] == UInt8(ascii: "}") { i += 1; return .object(pairs) }
        while true {
            skipWhitespace()
            guard i < bytes.count else { throw JSONParseError.unexpectedEnd }
            guard bytes[i] == UInt8(ascii: "\"") else { throw JSONParseError.unexpectedCharacter(i) }
            let key = try parseString()
            skipWhitespace()
            guard i < bytes.count else { throw JSONParseError.unexpectedEnd }
            guard bytes[i] == UInt8(ascii: ":") else { throw JSONParseError.unexpectedCharacter(i) }
            i += 1
            pairs.append((key, try parseValue()))
            skipWhitespace()
            guard i < bytes.count else { throw JSONParseError.unexpectedEnd }
            if bytes[i] == UInt8(ascii: ",") { i += 1; continue }
            if bytes[i] == UInt8(ascii: "}") { i += 1; return .object(pairs) }
            throw JSONParseError.unexpectedCharacter(i)
        }
    }

    mutating func parseArray() throws -> JSONValue {
        i += 1
        var items: [JSONValue] = []
        skipWhitespace()
        if i < bytes.count, bytes[i] == UInt8(ascii: "]") { i += 1; return .array(items) }
        while true {
            items.append(try parseValue())
            skipWhitespace()
            guard i < bytes.count else { throw JSONParseError.unexpectedEnd }
            if bytes[i] == UInt8(ascii: ",") { i += 1; continue }
            if bytes[i] == UInt8(ascii: "]") { i += 1; return .array(items) }
            throw JSONParseError.unexpectedCharacter(i)
        }
    }

    mutating func parseString() throws -> String {
        i += 1
        var scalars = String.UnicodeScalarView()
        var raw: [UInt8] = []
        func flush() {
            if !raw.isEmpty {
                scalars.append(contentsOf: String(decoding: raw, as: UTF8.self).unicodeScalars)
                raw.removeAll()
            }
        }
        while i < bytes.count {
            let b = bytes[i]
            if b == UInt8(ascii: "\"") { i += 1; flush(); return String(scalars) }
            if b == UInt8(ascii: "\\") {
                flush()
                i += 1
                guard i < bytes.count else { throw JSONParseError.unexpectedEnd }
                let e = bytes[i]
                i += 1
                switch e {
                case UInt8(ascii: "\""): scalars.append("\"")
                case UInt8(ascii: "\\"): scalars.append("\\")
                case UInt8(ascii: "/"): scalars.append("/")
                case UInt8(ascii: "b"): scalars.append("\u{08}")
                case UInt8(ascii: "f"): scalars.append("\u{0C}")
                case UInt8(ascii: "n"): scalars.append("\n")
                case UInt8(ascii: "r"): scalars.append("\r")
                case UInt8(ascii: "t"): scalars.append("\t")
                case UInt8(ascii: "u"):
                    var code = try parseHex4()
                    if (0xD800...0xDBFF).contains(code), i + 1 < bytes.count,
                       bytes[i] == UInt8(ascii: "\\"), bytes[i + 1] == UInt8(ascii: "u") {
                        i += 2
                        let low = try parseHex4()
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                    }
                    scalars.append(Unicode.Scalar(code) ?? "\u{FFFD}")
                default: throw JSONParseError.unexpectedCharacter(i - 1)
                }
            } else {
                raw.append(b)
                i += 1
            }
        }
        throw JSONParseError.unexpectedEnd
    }

    mutating func parseHex4() throws -> UInt32 {
        guard i + 4 <= bytes.count else { throw JSONParseError.unexpectedEnd }
        guard let v = UInt32(String(decoding: bytes[i..<i + 4], as: UTF8.self), radix: 16) else {
            throw JSONParseError.unexpectedCharacter(i)
        }
        i += 4
        return v
    }

    mutating func parseNumber() throws -> JSONValue {
        let start = i
        var isDouble = false
        while i < bytes.count {
            let b = bytes[i]
            if (b >= UInt8(ascii: "0") && b <= UInt8(ascii: "9")) || b == UInt8(ascii: "-") || b == UInt8(ascii: "+") {
                i += 1
            } else if b == UInt8(ascii: ".") || b == UInt8(ascii: "e") || b == UInt8(ascii: "E") {
                isDouble = true
                i += 1
            } else {
                break
            }
        }
        let text = String(decoding: bytes[start..<i], as: UTF8.self)
        if !isDouble, let v = Int64(text) { return .int(v) }
        if let d = Double(text), !text.isEmpty { return .double(d) }
        throw JSONParseError.unexpectedCharacter(start)
    }
}

// MARK: - Codable (stored as JSON text so key order survives)

extension JSONValue: Codable {
    public init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        self = try JSONValue.parse(try c.decode(String.self))
    }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        try c.encode(canonicalJSON(sortKeys: false))
    }
}
