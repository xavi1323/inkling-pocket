// Lexer for OpenUI Lang 1.1 §1.3 (MIT, Thesys Inc.). Lexing never fails.
import Foundation

enum OUIToken: Equatable {
    case ident(String)
    case state(String)  // includes "$"
    case function(String)  // without "@"
    case string(String)
    case number(Double)
    case op(String)
    case newline

    var isValue: Bool {
        switch self {
        case .ident, .state, .string, .number: true
        case .op(let o): o == ")" || o == "]"
        default: false
        }
    }
}

struct OUILexer {
    /// Tokens, and whether the text ended inside a string.
    static func lex(_ text: String) -> (tokens: [OUIToken], unterminatedString: Bool) {
        let s = Array(text.unicodeScalars)
        var i = 0
        var out: [OUIToken] = []
        var unterminated = false
        func isIdentStart(_ c: Unicode.Scalar) -> Bool { (c >= "a" && c <= "z") || (c >= "A" && c <= "Z") || c == "_" }
        func isDigit(_ c: Unicode.Scalar) -> Bool { c >= "0" && c <= "9" }
        func isIdent(_ c: Unicode.Scalar) -> Bool { isIdentStart(c) || isDigit(c) }
        func identifier(from start: Int) -> (String, Int) {
            var j = start
            while j < s.count && isIdent(s[j]) { j += 1 }
            var v = String.UnicodeScalarView()
            v.append(contentsOf: s[start..<j])
            return (String(v), j)
        }
        while i < s.count {
            let c = s[i]
            if c == " " || c == "\t" { i += 1; continue }
            if c == "\r" && i + 1 < s.count && s[i + 1] == "\n" { i += 1; continue }
            if c == "\n" { out.append(.newline); i += 1; continue }
            if c == "#" || (c == "/" && i + 1 < s.count && s[i + 1] == "/") {
                while i < s.count && s[i] != "\n" { i += 1 }
                continue
            }
            if c == "\"" || c == "'" {
                let quote = c
                var value = String.UnicodeScalarView()
                i += 1
                var closed = false
                while i < s.count {
                    let d = s[i]
                    if d == quote { closed = true; i += 1; break }
                    if d == "\\" {
                        guard i + 1 < s.count else { i += 1; break }  // ends mid-escape: dropped
                        let e = s[i + 1]
                        i += 2
                        switch e {
                        case "n": value.append("\n")
                        case "t": value.append("\t")
                        case "r" where quote == "\"": value.append("\r")
                        case "b" where quote == "\"": value.append("\u{08}")
                        case "f" where quote == "\"": value.append("\u{0C}")
                        case "u" where quote == "\"":
                            var hex = ""
                            while hex.count < 4 && i < s.count, s[i].properties.isASCIIHexDigit { hex.unicodeScalars.append(s[i]); i += 1 }
                            if hex.count == 4, let code = UInt32(hex, radix: 16), let scalar = Unicode.Scalar(code) { value.append(scalar) }
                        default: value.append(e)
                        }
                        continue
                    }
                    value.append(d)
                    i += 1
                }
                if !closed { unterminated = true }
                out.append(.string(String(value)))
                continue
            }
            let prevIsValue = out.last?.isValue ?? false
            if isDigit(c) || (c == "-" && i + 1 < s.count && isDigit(s[i + 1]) && !prevIsValue) {
                var j = i
                if s[j] == "-" { j += 1 }
                while j < s.count && isDigit(s[j]) { j += 1 }
                if j + 1 < s.count && s[j] == "." && isDigit(s[j + 1]) {
                    j += 1
                    while j < s.count && isDigit(s[j]) { j += 1 }
                }
                if j < s.count && (s[j] == "e" || s[j] == "E") {
                    var k = j + 1
                    if k < s.count && (s[k] == "+" || s[k] == "-") { k += 1 }
                    while k < s.count && isDigit(s[k]) { k += 1 }
                    j = k
                }
                var v = String.UnicodeScalarView()
                v.append(contentsOf: s[i..<j])
                let text = String(v)
                out.append(.number(Double(text) ?? Double(text.prefix { $0 != "e" && $0 != "E" }) ?? 0))
                i = j
                continue
            }
            if isIdentStart(c) {
                let (name, j) = identifier(from: i)
                out.append(.ident(name))
                i = j
                continue
            }
            if (c == "$" || c == "@") && i + 1 < s.count && isIdentStart(s[i + 1]) {
                let (name, j) = identifier(from: i + 1)
                out.append(c == "$" ? .state("$" + name) : .function(name))
                i = j
                continue
            }
            let two = i + 1 < s.count ? String(c) + String(s[i + 1]) : ""
            if ["==", "!=", ">=", "<=", "&&", "||"].contains(two) {
                out.append(.op(two))
                i += 2
                continue
            }
            switch c {
            case "&": out.append(.op("&&"))
            case "|": out.append(.op("||"))
            case "=", ">", "<", "+", "-", "*", "/", "%", "!", "?", ":", ".", ",", "(", ")", "[", "]", "{", "}":
                out.append(.op(String(c)))
            default: break  // skipped
            }
            i += 1
        }
        return (out, unterminated)
    }
}
