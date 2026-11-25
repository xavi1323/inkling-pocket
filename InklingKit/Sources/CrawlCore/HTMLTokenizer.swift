// Streaming HTML tokenizer modelled on Python's html.parser.HTMLParser (convert_charrefs=False),
// so the html2text port sees the same event sequence crawl4ai does.

enum HTMLEvent: Equatable {
    case start(String, [(String, String?)])
    case end(String)
    case startEnd(String, [(String, String?)])
    case data(String)
    case entity(String)
    case charRef(String)

    static func == (a: HTMLEvent, b: HTMLEvent) -> Bool {
        switch (a, b) {
        case let (.start(x, xa), .start(y, ya)), let (.startEnd(x, xa), .startEnd(y, ya)):
            x == y && xa.count == ya.count && zip(xa, ya).allSatisfy { $0.0 == $1.0 && $0.1 == $1.1 }
        case let (.end(x), .end(y)), let (.data(x), .data(y)), let (.entity(x), .entity(y)), let (.charRef(x), .charRef(y)):
            x == y
        default: false
        }
    }
}

struct HTMLTokenizer {
    private let s: [Unicode.Scalar]
    private var i = 0
    private var cdataElement: String?
    private(set) var events: [HTMLEvent] = []

    static func tokenize(_ html: String) -> [HTMLEvent] {
        var t = HTMLTokenizer(Array(html.unicodeScalars))
        t.run()
        return t.events
    }

    private init(_ scalars: [Unicode.Scalar]) { s = scalars }

    private func str(_ a: Int, _ b: Int) -> String {
        var v = String.UnicodeScalarView()
        v.append(contentsOf: s[a..<min(b, s.count)])
        return String(v)
    }

    private static func isAlpha(_ c: Unicode.Scalar) -> Bool { (c >= "a" && c <= "z") || (c >= "A" && c <= "Z") }
    private static func isDigit(_ c: Unicode.Scalar) -> Bool { c >= "0" && c <= "9" }
    private static func isHex(_ c: Unicode.Scalar) -> Bool { isDigit(c) || (c >= "a" && c <= "f") || (c >= "A" && c <= "F") }
    private static func isSpace(_ c: Unicode.Scalar) -> Bool { c == " " || c == "\t" || c == "\n" || c == "\r" || c == "\u{0C}" }

    private mutating func emitData(_ text: String) {
        guard !text.isEmpty else { return }
        events.append(.data(text))
    }

    private mutating func run() {
        let n = s.count
        while i < n {
            if let cdata = cdataElement {
                // Raw text until </cdata followed by whitespace, '/' or '>'.
                var j = i
                var found = -1
                while j + 1 < n {
                    if s[j] == "<" && s[j + 1] == "/" && matchesWord(at: j + 2, cdata) {
                        let after = j + 2 + cdata.unicodeScalars.count
                        if after >= n || HTMLTokenizer.isSpace(s[after]) || s[after] == "/" || s[after] == ">" {
                            found = j
                            break
                        }
                    }
                    j += 1
                }
                if found < 0 {
                    emitData(str(i, n))
                    i = n
                    break
                }
                emitData(str(i, found))
                i = found
                cdataElement = nil
                parseEndTag()
                continue
            }
            var j = i
            while j < n && s[j] != "<" && s[j] != "&" { j += 1 }
            if j > i { emitData(str(i, j)) }
            i = j
            if i >= n { break }
            if s[i] == "<" {
                if i + 1 < n && HTMLTokenizer.isAlpha(s[i + 1]) {
                    parseStartTag()
                } else if i + 1 < n && s[i + 1] == "/" {
                    parseEndTag()
                } else if startsWith("<!--") {
                    if let end = find("-->", from: i + 4) { i = end + 3 } else { i = n }
                } else if startsWith("<?") || startsWith("<!") {
                    if let end = find(">", from: i + 2) { i = end + 1 } else { i = n }
                } else {
                    emitData("<")
                    i += 1
                }
            } else {  // '&'
                if i + 1 < n && s[i + 1] == "#" {
                    var k = i + 2
                    let hex = k < n && (s[k] == "x" || s[k] == "X")
                    if hex { k += 1 }
                    let digitsStart = k
                    while k < n && (hex ? HTMLTokenizer.isHex(s[k]) : HTMLTokenizer.isDigit(s[k])) { k += 1 }
                    if k > digitsStart && k < n {
                        events.append(.charRef(str(i + 2, k)))
                        i = s[k] == ";" ? k + 1 : k
                    } else {
                        emitData(str(i, min(i + 2, n)))
                        i = min(i + 2, n)
                    }
                } else if i + 1 < n && HTMLTokenizer.isAlpha(s[i + 1]) {
                    var k = i + 2
                    while k < n && (HTMLTokenizer.isAlpha(s[k]) || HTMLTokenizer.isDigit(s[k]) || s[k] == "-" || s[k] == ".") { k += 1 }
                    if k < n {
                        events.append(.entity(str(i + 1, k)))
                        i = s[k] == ";" ? k + 1 : k
                    } else {
                        emitData(str(i, n))
                        i = n
                    }
                } else {
                    emitData("&")
                    i += 1
                }
            }
        }
    }

    private func startsWith(_ p: String) -> Bool {
        var k = i
        for c in p.unicodeScalars {
            guard k < s.count, s[k] == c else { return false }
            k += 1
        }
        return true
    }

    private func matchesWord(at k: Int, _ word: String) -> Bool {
        var k = k
        for c in word.unicodeScalars {
            guard k < s.count, String(s[k]).lowercased() == String(c) else { return false }
            k += 1
        }
        return true
    }

    private func find(_ p: String, from: Int) -> Int? {
        let pat = Array(p.unicodeScalars)
        var k = from
        while k + pat.count <= s.count {
            if s[k] == pat[0] && Array(s[k..<k + pat.count]) == pat { return k }
            k += 1
        }
        return nil
    }

    private mutating func parseStartTag() {
        let n = s.count
        var k = i + 1
        let nameStart = k
        while k < n && !HTMLTokenizer.isSpace(s[k]) && s[k] != "/" && s[k] != ">" && s[k] != "\0" { k += 1 }
        let name = str(nameStart, k).lowercased()
        var attrs: [(String, String?)] = []
        var selfClosing = false
        while k < n {
            while k < n && (HTMLTokenizer.isSpace(s[k]) || s[k] == "/") {
                if s[k] == "/" && k + 1 < n && s[k + 1] == ">" { selfClosing = true }
                k += 1
            }
            if k >= n { break }
            if s[k] == ">" { k += 1; break }
            let aStart = k
            k += 1  // first char may be '='
            while k < n && !HTMLTokenizer.isSpace(s[k]) && s[k] != "/" && s[k] != "=" && s[k] != ">" { k += 1 }
            let attrName = str(aStart, k).lowercased()
            var p = k
            while p < n && HTMLTokenizer.isSpace(s[p]) { p += 1 }
            if p < n && s[p] == "=" {
                p += 1
                while p < n && s[p] == "=" { p += 1 }
                while p < n && HTMLTokenizer.isSpace(s[p]) { p += 1 }
                var value = ""
                if p < n && (s[p] == "\"" || s[p] == "'") {
                    let q = s[p]
                    var e = p + 1
                    while e < n && s[e] != q { e += 1 }
                    value = str(p + 1, e)
                    p = min(e + 1, n)
                } else {
                    var e = p
                    while e < n && !HTMLTokenizer.isSpace(s[e]) && s[e] != ">" { e += 1 }
                    value = str(p, e)
                    p = e
                }
                attrs.append((attrName, HTMLTokenizer.unescape(value)))
                k = p
            } else {
                attrs.append((attrName, nil))
            }
        }
        i = k
        if selfClosing {
            events.append(.startEnd(name, attrs))
        } else {
            events.append(.start(name, attrs))
            if name == "script" || name == "style" { cdataElement = name }
        }
    }

    private mutating func parseEndTag() {
        let n = s.count
        var k = i + 2
        while k < n && HTMLTokenizer.isSpace(s[k]) { k += 1 }
        let nameStart = k
        while k < n && !HTMLTokenizer.isSpace(s[k]) && s[k] != ">" { k += 1 }
        let name = str(nameStart, k).lowercased()
        while k < n && s[k] != ">" { k += 1 }
        i = min(k + 1, n)
        if !name.isEmpty { events.append(.end(name)) }
    }

    /// html.unescape for attribute values (full html5 table, no html2text substitutions).
    static func unescape(_ text: String) -> String {
        guard text.contains("&") else { return text }
        var out = ""
        var events = HTMLTokenizer(Array(text.unicodeScalars))
        events.run()
        for e in events.events {
            switch e {
            case .data(let d): out += d
            case .entity(let name): out += HTMLEntities.html5[name] ?? "&\(name);"
            case .charRef(let ref): out += charRefString(ref, unifiable: false)
            default: break
            }
        }
        return out
    }

    static func charRefString(_ ref: String, unifiable: Bool) -> String {
        let code: UInt32?
        if ref.hasPrefix("x") || ref.hasPrefix("X") { code = UInt32(ref.dropFirst(), radix: 16) } else { code = UInt32(ref) }
        guard let code else { return "" }
        if unifiable, let u = HTMLEntities.unifiableByCodePoint[code] { return u }
        guard let scalar = Unicode.Scalar(code) else { return "" }
        return String(Character(scalar))
    }
}
