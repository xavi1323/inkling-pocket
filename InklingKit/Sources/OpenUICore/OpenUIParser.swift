// Streaming parser + evaluator for OpenUI Lang 1.1 (spec: github.com/thesysdev/openui, MIT License, Copyright (c) Thesys Inc.).
// Covers the full grammar. Evaluation covers literals, references, operators, member access, the numeric built-ins and
// actions; queries, mutations, live state and @Filter/@Sort/@Each evaluate to null (this client renders chat answers).
import Foundation

indirect enum OUIExpr {
    case literal(OUIValue)
    case ref(String)
    case state(String)
    case array([OUIExpr])
    case object([(String, OUIExpr)])
    case component(String, [OUIExpr])
    case function(String, [OUIExpr])
    case member(OUIExpr, String)
    case index(OUIExpr, OUIExpr)
    case unary(String, OUIExpr)
    case binary(String, OUIExpr, OUIExpr)
    case ternary(OUIExpr, OUIExpr, OUIExpr)
    case assign(String, OUIExpr)
}

struct OUIStatement {
    var name: String
    var expr: OUIExpr
    /// Ended by a newline whose next line is known (or by the end of the stream).
    var complete: Bool
    /// Needed implicit closing (unterminated string or bracket).
    var closedImplicitly: Bool
}

public struct OpenUIParser {
    private var buffer = ""
    private var finished = false

    public init() {}

    /// Batch parse of a whole program (equals streaming it and calling `finish()`).
    public static func parse(_ source: String) -> OpenUIDocument {
        var p = OpenUIParser()
        _ = p.feed(source)
        return p.finish()
    }

    /// Adds a chunk and returns the best document for the text so far.
    public mutating func feed(_ chunk: String) -> OpenUIDocument {
        if finished { buffer = ""; finished = false }
        buffer += chunk
        return Self.build(buffer, final: false)
    }

    /// Ends the stream: the pending tail becomes final.
    public mutating func finish() -> OpenUIDocument {
        finished = true
        return Self.build(buffer, final: true)
    }

    // MARK: statements

    static func build(_ text: String, final: Bool) -> OpenUIDocument {
        let (tokens, unterminated) = OUILexer.lex(text.trimmingCharacters(in: .whitespacesAndNewlines))
        let (statements, incomplete) = split(tokens, unterminated: unterminated, final: final)
        var table: [String: OUIStatement] = [:]
        var order: [String] = []
        for st in statements {
            if let existing = table[st.name], existing.complete, !st.complete, st.closedImplicitly { continue }
            if table[st.name] == nil { order.append(st.name) }
            table[st.name] = st
        }
        var evaluator = OUIEvaluator(statements: table, final: final)
        var bindings: [String: OUIValue] = [:]
        for name in order {
            if let v = evaluator.evaluateStatement(name) { bindings[name] = v }
        }
        var root: OUINode?
        if let r = table["root"] {
            _ = r
            root = evaluator.evaluateStatement("root")?.nodeValue
        } else if let first = statements.first(where: {
            if !$0.name.hasPrefix("$"), case .component(let c, _) = $0.expr, c == ComponentRegistry.rootComponent { return true }
            return false
        }) {
            root = evaluator.evaluateStatement(first.name)?.nodeValue
        }
        return OpenUIDocument(root: root, bindings: bindings, incomplete: incomplete)
    }

    /// Cuts the token stream into statements (OpenUI Lang §1.4 / §4).
    static func split(_ tokens: [OUIToken], unterminated: Bool, final: Bool) -> ([OUIStatement], Bool) {
        var result: [OUIStatement] = []
        var current: [OUIToken] = []
        var depth = 0
        var openTernaries = 0
        var incomplete = false
        var i = 0

        func flush(complete: Bool, implicit: Bool) {
            defer { current = []; depth = 0; openTernaries = 0 }
            guard current.count >= 2, case .op("=") = current[1] else { return }
            let name: String
            switch current[0] {
            case .ident(let n), .state(let n): name = n
            default: return
            }
            var parser = OUIExprParser(tokens: Array(current[2...]))
            guard let expr = parser.parseExpression(), parser.ok else { return }
            result.append(OUIStatement(name: name, expr: expr, complete: complete, closedImplicitly: implicit))
        }

        while i < tokens.count {
            let t = tokens[i]
            if t == .newline {
                if current.isEmpty { i += 1; continue }
                if depth > 0 { i += 1; continue }
                if case .op(let o) = current.last, o == "?" || o == ":" { i += 1; continue }
                if openTernaries > 0 { i += 1; continue }
                var j = i + 1
                while j < tokens.count && tokens[j] == .newline { j += 1 }
                if j < tokens.count {
                    if tokens[j] == .op("?") { i += 1; continue }
                    flush(complete: true, implicit: false)
                    i = j
                    continue
                }
                // Next line hasn't arrived: provisional end.
                break
            }
            switch t {
            case .op("("), .op("["), .op("{"): depth += 1
            case .op(")"), .op("]"), .op("}"): if depth > 0 { depth -= 1 }
            case .op("?") where depth == 0: openTernaries += 1
            case .op(":") where depth == 0 && openTernaries > 0: openTernaries -= 1
            default: break
            }
            current.append(t)
            i += 1
        }
        if !current.isEmpty {
            let implicit = depth > 0 || unterminated
            incomplete = implicit
            flush(complete: final, implicit: implicit)
        }
        return (result, incomplete)
    }
}

// MARK: - expression parser

struct OUIExprParser {
    let tokens: [OUIToken]
    var pos = 0
    var ok = true

    init(tokens: [OUIToken]) { self.tokens = tokens.filter { $0 != .newline } }

    var atEnd: Bool { pos >= tokens.count }
    func peek() -> OUIToken? { pos < tokens.count ? tokens[pos] : nil }
    func isOp(_ o: String) -> Bool { peek() == .op(o) }
    mutating func eat(_ o: String) -> Bool {
        if isOp(o) { pos += 1; return true }
        return false
    }

    mutating func parseExpression() -> OUIExpr? { ternary() }

    mutating func ternary() -> OUIExpr? {
        guard let cond = binary(0) else { return nil }
        guard eat("?") else { return cond }
        guard let a = ternary() else { return atEnd ? cond : fail() }
        guard eat(":") else { return atEnd ? cond : fail() }
        guard let b = ternary() else { return atEnd ? cond : fail() }
        return .ternary(cond, a, b)
    }

    static let levels: [[String]] = [["||"], ["&&"], ["==", "!="], [">", "<", ">=", "<="], ["+", "-"], ["*", "/", "%"]]

    mutating func binary(_ level: Int) -> OUIExpr? {
        if level == Self.levels.count { return unary() }
        guard var left = binary(level + 1) else { return nil }
        while let t = peek(), case .op(let o) = t, Self.levels[level].contains(o) {
            pos += 1
            guard let right = binary(level + 1) else { return atEnd ? left : fail() }
            left = .binary(o, left, right)
        }
        return left
    }

    mutating func unary() -> OUIExpr? {
        if isOp("!") || isOp("-") {
            guard case .op(let o) = peek()! else { return nil }
            pos += 1
            guard let e = unary() else { return fail() }
            return .unary(o, e)
        }
        return postfix()
    }

    mutating func postfix() -> OUIExpr? {
        guard var e = primary() else { return nil }
        while true {
            if eat(".") {
                guard let t = peek() else { return e }  // field not arrived yet
                pos += 1
                switch t {
                case .ident(let n): e = .member(e, n)
                case .string(let s): e = .member(e, s)
                case .number(let n): e = .member(e, OUIValue.formatNumber(n))
                case .state(let s): e = .member(e, String(s.dropFirst()))
                default: e = .member(e, "?")
                }
            } else if eat("[") {
                guard let idx = parseExpression() else { return atEnd ? e : fail() }
                skipMismatched(until: "]")
                if !eat("]") && !atEnd { return fail() }
                e = .index(e, idx)
            } else {
                return e
            }
        }
    }

    mutating func fail() -> OUIExpr? {
        ok = false
        return nil
    }

    /// Mismatched closing brackets are skipped (§4 rule 2).
    mutating func skipMismatched(until closer: String) {
        while let t = peek(), case .op(let o) = t, [")", "]", "}"].contains(o), o != closer { pos += 1 }
    }

    mutating func list(closer: String) -> [OUIExpr]? {
        var items: [OUIExpr] = []
        while true {
            while eat(",") {}
            skipMismatched(until: closer)
            if atEnd || eat(closer) { return items }
            guard let e = parseExpression() else { return atEnd ? items : nil }
            items.append(e)
        }
    }

    mutating func primary() -> OUIExpr? {
        guard let t = peek() else { return nil }
        pos += 1
        switch t {
        case .string(let s): return .literal(.string(s))
        case .number(let n): return .literal(.number(n))
        case .state(let s):
            if isOp("="), !(pos + 1 < tokens.count && tokens[pos + 1] == .op("=")) {
                pos += 1
                guard let v = parseExpression() else { return fail() }
                return .assign(s, v)
            }
            return .state(s)
        case .ident(let n):
            switch n {
            case "true": return .literal(.bool(true))
            case "false": return .literal(.bool(false))
            case "null": return .literal(.null)
            default: break
            }
            if n.first!.isUppercase, eat("(") {
                guard let args = list(closer: ")") else { return fail() }
                return .component(n, args)
            }
            return .ref(n)
        case .function(let n):
            guard eat("(") else { return atEnd ? .function(n, []) : fail() }
            guard let args = list(closer: ")") else { return fail() }
            return .function(n, args)
        case .op("("):
            guard let e = parseExpression() else { return atEnd ? nil : fail() }
            skipMismatched(until: ")")
            if !eat(")") && !atEnd { return fail() }
            return e
        case .op("["):
            guard let items = list(closer: "]") else { return fail() }
            return .array(items)
        case .op("{"):
            var entries: [(String, OUIExpr)] = []
            while true {
                while eat(",") {}
                skipMismatched(until: "}")
                if atEnd || eat("}") { return .object(entries) }
                guard let k = peek() else { return .object(entries) }
                pos += 1
                let key: String
                switch k {
                case .ident(let n): key = n
                case .string(let s): key = s
                case .number(let n): key = OUIValue.formatNumber(n)
                case .state(let s): key = String(s.dropFirst())
                default: key = "?"
                }
                guard eat(":") else { if atEnd { return .object(entries) }; return fail() }
                guard let v = parseExpression() else { return atEnd ? .object(entries) : fail() }
                entries.append((key, v))
            }
        default:
            return fail()
        }
    }
}

// MARK: - evaluation

struct OUIEvaluator {
    let statements: [String: OUIStatement]
    let final: Bool
    private var visiting: Set<String> = []
    private var cache: [String: OUIValue?] = [:]

    init(statements: [String: OUIStatement], final: Bool) {
        self.statements = statements
        self.final = final
    }

    /// nil = unresolved / renders nothing.
    mutating func evaluateStatement(_ name: String) -> OUIValue? {
        if let cached = cache[name] { return cached }
        guard let st = statements[name], !visiting.contains(name) else { return nil }
        if case .component(let c, _) = st.expr, c == "Query" || c == "Mutation" { return nil }
        visiting.insert(name)
        let v = eval(st.expr, valuePosition: true)
        visiting.remove(name)
        if visiting.isEmpty { cache[name] = .some(v) }
        return v
    }

    mutating func eval(_ e: OUIExpr, valuePosition: Bool = false) -> OUIValue? {
        switch e {
        case .literal(let v): return v
        case .ref(let n): return evaluateStatement(n)
        case .state(let n): return evaluateStatement(n) ?? .null
        case .assign(_, let v): return eval(v)
        case .array(let items): return .array(items.compactMap { eval($0, valuePosition: true) })
        case .object(let entries):
            var o: [String: OUIValue] = [:]
            for (k, v) in entries { o[k] = eval(v) ?? .null }
            return .object(o)
        case .component(let name, let args):
            if name == "Query" || name == "Mutation" { return valuePosition ? nil : .null }
            if name == "Action" { return actionList(args) }
            return validate(name, args.map { eval($0, valuePosition: true) })
        case .function(let name, let args): return function(name, args)
        case .member(let base, let field):
            let b = eval(base) ?? .null
            switch b {
            case .object(let o): return o[field] ?? .null
            case .array(let a):
                if field == "length" { return .number(Double(a.count)) }
                return .array(a.map { if case .object(let o) = $0 { o[field] ?? .null } else { .null } })
            case .string(let s) where field == "length": return .number(Double(s.utf16.count))
            default: return .null
            }
        case .index(let base, let idx):
            let b = eval(base) ?? .null
            let i = eval(idx) ?? .null
            switch (b, i) {
            case (.array(let a), _):
                let n = Int(Self.toNumber(i))
                return n >= 0 && n < a.count ? a[n] : .null
            case (.object(let o), _): return o[Self.toKey(i)] ?? .null
            default: return .null
            }
        case .unary(let op, let x):
            let v = eval(x) ?? .null
            return op == "!" ? .bool(!Self.truthy(v)) : .number(-Self.toNumber(v))
        case .binary(let op, let l, let r):
            let a = eval(l) ?? .null
            if op == "&&" { return Self.truthy(a) ? (eval(r) ?? .null) : a }
            if op == "||" { return Self.truthy(a) ? a : (eval(r) ?? .null) }
            let b = eval(r) ?? .null
            return Self.binary(op, a, b)
        case .ternary(let c, let a, let b):
            return Self.truthy(eval(c) ?? .null) ? eval(a, valuePosition: valuePosition) : eval(b, valuePosition: valuePosition)
        }
    }

    mutating func actionList(_ args: [OUIExpr]) -> OUIValue? {
        var steps: [OUIActionStep] = []
        for a in args {
            switch eval(a) {
            case .array(let items)?: for case .action(let s) in items { steps += s }
            case .action(let s)?: steps += s
            default: break
            }
        }
        return .action(steps)
    }

    mutating func function(_ name: String, _ args: [OUIExpr]) -> OUIValue? {
        switch name {
        case "Action": return actionList(args)
        case "ToAssistant":
            guard let first = args.first, let msg = eval(first), msg != .null else { return nil }
            let ctx = args.count > 1 ? eval(args[1]) : nil
            return .action([.toAssistant(message: msg.displayText, context: ctx == .null ? nil : ctx)])
        case "OpenUrl":
            guard let first = args.first, case .string(let url)? = eval(first) else { return nil }
            return .action([.openURL(url)])
        case "Set", "Reset", "Run":
            return .action([.other(name: name, args: args.map { a in
                if case .state(let s) = a { return .ref(s) }
                if case .ref(let r) = a { return .ref(r) }
                return eval(a) ?? .null
            })])
        default: break
        }
        let values = args.map { eval($0) ?? .null }
        let first = values.first ?? .null
        let numbers: [Double]? = first.arrayValue?.map(Self.toNumber)
        switch name {
        case "Count": return .number(Double(first.arrayValue?.count ?? 0))
        case "First": return first.arrayValue?.first ?? .null
        case "Last": return first.arrayValue?.last ?? .null
        case "Sum": return .number(numbers?.reduce(0, +) ?? 0)
        case "Avg": return .number(numbers.map { $0.isEmpty ? 0 : $0.reduce(0, +) / Double($0.count) } ?? 0)
        case "Min": return .number(numbers?.min() ?? 0)
        case "Max": return .number(numbers?.max() ?? 0)
        case "Round":
            let places = values.count > 1 ? Self.toNumber(values[1]) : 0
            let f = pow(10, places)
            return Self.mapNumbers(first) { ($0 * f + 0.5).rounded(.down) / f }
        case "Abs": return Self.mapNumbers(first) { abs($0) }
        case "Floor": return Self.mapNumbers(first) { $0.rounded(.down) }
        case "Ceil": return Self.mapNumbers(first) { $0.rounded(.up) }
        default: return .null  // @Filter/@Sort/@Each and unknown functions
        }
    }

    static func mapNumbers(_ v: OUIValue, _ f: (Double) -> Double) -> OUIValue {
        if case .array(let a) = v { return .array(a.map { mapNumbers($0, f) }) }
        return .number(f(toNumber(v)))
    }

    // MARK: validation against the registry (§8.4)

    func validate(_ name: String, _ args: [OUIValue?]) -> OUIValue? {
        guard let spec = ComponentRegistry.spec(name) else { return nil }
        var positional: [OUIValue] = []
        var named: [String: OUIValue] = [:]
        for (i, param) in spec.params.enumerated() {
            let raw = i < args.count ? args[i] : nil
            guard let value = raw, value != .null else {
                if param.required { return nil }
                positional.append(.null)
                continue
            }
            guard let checked = check(value, param.type) else {
                if param.required { return nil }
                positional.append(.null)
                continue
            }
            positional.append(checked)
            named[param.name] = checked
        }
        while positional.last == .null { positional.removeLast() }
        return .node(OUINode(component: name, args: positional, named: named))
    }

    func check(_ v: OUIValue, _ type: ParamType) -> OUIValue? {
        switch type {
        case .any: return v
        case .string: return v.stringValue != nil ? v : nil
        case .number: return v.numberValue != nil ? v : nil
        case .boolean: return v.boolValue != nil ? v : nil
        case .enumeration(let values):
            guard let s = v.stringValue else { return nil }
            return (!final || values.contains(s)) ? v : nil
        case .array(let element):
            guard case .array(let items) = v else { return nil }
            return .array(items.compactMap { $0 == .null ? nil : check($0, element) })
        case .component(let names):
            guard case .node(let n) = v, names.isEmpty || names.contains(n.component) else { return nil }
            return v
        case .action:
            if case .action = v { return v }
            if case .array(let items) = v {
                let steps: [OUIActionStep] = items.flatMap { item -> [OUIActionStep] in if case .action(let s) = item { return s }; return [] }
                return steps.isEmpty ? nil : .action(steps)
            }
            return nil
        case .object(let fields):
            guard case .object(var o) = v else { return nil }
            for (k, t) in fields {
                if let fv = o[k] { o[k] = check(fv, t) ?? .null }
            }
            return .object(o)
        case .oneOf(let types):
            for t in types { if let ok = check(v, t) { return ok } }
            return nil
        }
    }

    // MARK: JavaScript-style coercion (§3.1, §3.2)

    static func toNumber(_ v: OUIValue) -> Double {
        switch v {
        case .number(let n): return n.isNaN ? 0 : n
        case .bool(let b): return b ? 1 : 0
        case .string(let s):
            let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
            if t.isEmpty { return 0 }
            if let d = Double(t), d.isFinite || t.lowercased().contains("infinity") { return d }
            if t.hasPrefix("0x") || t.hasPrefix("0X"), let i = Int64(t.dropFirst(2), radix: 16) { return Double(i) }
            return 0
        default: return 0
        }
    }

    static func truthy(_ v: OUIValue) -> Bool {
        switch v {
        case .null: false
        case .bool(let b): b
        case .number(let n): n != 0 && !n.isNaN
        case .string(let s): !s.isEmpty
        default: true
        }
    }

    static func toKey(_ v: OUIValue) -> String {
        if case .string(let s) = v { return s }
        return v.displayText
    }

    static func looselyEqual(_ a: OUIValue, _ b: OUIValue) -> Bool {
        switch (a, b) {
        case (.null, .null): return true
        case (.null, _), (_, .null): return false
        case (.string(let x), .string(let y)): return x == y
        case (.number, _), (_, .number), (.bool, _), (_, .bool):
            if case .string = a, case .string = b { return false }
            if isPrimitive(a) && isPrimitive(b) { return toNumber(a) == toNumber(b) }
            return false
        default: return a == b
        }
    }

    static func isPrimitive(_ v: OUIValue) -> Bool {
        switch v {
        case .string, .number, .bool: true
        default: false
        }
    }

    static func binary(_ op: String, _ a: OUIValue, _ b: OUIValue) -> OUIValue {
        switch op {
        case "+":
            if case .string = a { return .string(a.displayText + b.displayText) }
            if case .string = b { return .string(a.displayText + b.displayText) }
            return .number(toNumber(a) + toNumber(b))
        case "-": return .number(toNumber(a) - toNumber(b))
        case "*": return .number(toNumber(a) * toNumber(b))
        case "/": let d = toNumber(b); return .number(d == 0 ? 0 : toNumber(a) / d)
        case "%": let d = toNumber(b); return .number(d == 0 ? 0 : toNumber(a).truncatingRemainder(dividingBy: d))
        case "==": return .bool(looselyEqual(a, b))
        case "!=": return .bool(!looselyEqual(a, b))
        case ">": return .bool(toNumber(a) > toNumber(b))
        case "<": return .bool(toNumber(a) < toNumber(b))
        case ">=": return .bool(toNumber(a) >= toNumber(b))
        case "<=": return .bool(toNumber(a) <= toNumber(b))
        default: return .null
        }
    }
}
