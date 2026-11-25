// Port of html2text 2024.2.26 as configured by crawl4ai v0.9.6 (CustomHTML2Text + DefaultMarkdownGenerator options):
// body_width 0, inline links, single_line_break, mark_code, escape_snob off, ignore_mailto_links, skip_internal_links off.
// Original: Copyright (C) Aaron Swartz and contributors, GPL-3.0 (html2text); crawl4ai changes Apache-2.0.
import Foundation

final class HTML2Markdown {
    private final class ListElement {
        let name: String
        var num: Int
        init(_ name: String, _ num: Int) { self.name = name; self.num = num }
    }
    private enum Force { case none, yes, end }

    private var baseURL: String
    private var outText: [String] = []
    private var lastWasNL = false
    private var quiet = 0
    private var pP = 0
    private var outcount = 0
    private var start = true
    private var space = false
    private var astack: [[(String, String?)]?] = []
    private var maybeAutomaticLink: String?
    private var emptyLink = false
    private var list: [ListElement] = []
    private var blockquote = 0
    private var code = false
    private var quote = false
    private var brToggle = ""
    private var lastWasList = false
    private var style = 0
    private var splitNextTD = false
    private var tdCount = 0
    private var tableStart = false
    private var inheader = false
    private var abbrTitle: String?
    private var abbrData: String?
    private var abbrList: [(String, String)] = []
    private var stressed = false
    private var precedingStressed = false
    private var precedingData = ""
    private var currentTag = ""
    // CustomHTML2Text state
    private var insidePre = false
    private var insideCode = false
    private var insideLink = false

    init(baseURL: String) { self.baseURL = baseURL }

    func handle(_ html: String) -> String {
        start = true
        for event in HTMLTokenizer.tokenize(html) {
            switch event {
            case .start(let t, let a): handleTag(t, a, start: true)
            case .end(let t): handleTag(t, [], start: false)
            case .startEnd(let t, let a): handleTag(t, a, start: true); handleTag(t, [], start: false)
            case .data(let d): handleData(d, entityChar: false)
            case .entity(let name):
                let ref = entityRef(name)
                if !ref.isEmpty { handleData(ref, entityChar: true) }
            case .charRef(let ref): handleData(HTMLTokenizer.charRefString(ref, unifiable: true), entityChar: true)
            }
        }
        pbr()
        o("", force: .end)
        let text = outText.joined().replacingOccurrences(of: "&nbsp_place_holder;", with: " ")
        outText = []
        return text
    }

    private func entityRef(_ name: String) -> String {
        if name == "nbsp" { return "&nbsp_place_holder;" }
        if let u = HTMLEntities.unifiable[name] { return u }
        return HTMLEntities.html5[name] ?? "&\(name);"
    }

    private static func attr(_ attrs: [(String, String?)], _ key: String) -> String?? {
        guard let pair = attrs.first(where: { $0.0 == key }) else { return nil }
        return .some(pair.1)
    }

    private static func hn(_ tag: String) -> Int {
        let c = Array(tag)
        guard c.count == 2, c[0] == "h", let d = c[1].wholeNumberValue, d > 0 else { return 0 }
        return d
    }

    // MARK: output

    private func out(_ s: String) {
        outText.append(s)
        if let last = s.last { lastWasNL = last == "\n" }
    }

    private func pbr() { if pP == 0 { pP = 1 } }
    private func p() { pP = 1 }  // single_line_break
    private func softBR() { pbr(); brToggle = "  " }

    private func o(_ input: String, puredata: Bool = false, force: Force = .none) {
        var data = input
        if abbrData != nil { abbrData! += data }
        guard quiet == 0 else { return }
        if puredata {
            data = HTML2Markdown.collapseWhitespace(data)
            if data.first == " " { space = true; data.removeFirst() }
        }
        if data.isEmpty && force == .none { return }
        var bq = String(repeating: ">", count: blockquote)
        if !(force != .none && data.first == ">") && blockquote > 0 { bq += " " }
        if start { space = false; pP = 0; start = false }
        if force == .end { pP = 0; out("\n"); space = false }
        if pP > 0 {
            out(String(repeating: brToggle + "\n" + bq, count: pP))
            space = false
            brToggle = ""
        }
        if space {
            if !lastWasNL { out(" ") }
            space = false
        }
        if !abbrList.isEmpty && force == .end {
            for (abbr, def) in abbrList { out("  *[" + abbr + "]: " + def + "\n") }
        }
        pP = 0
        out(data)
        outcount += 1
    }

    /// Python re.sub(r"\s+", " ", data)
    static func collapseWhitespace(_ s: String) -> String {
        var out = ""
        var inSpace = false
        for ch in s {
            if ch.isWhitespace {
                if !inSpace { out.append(" "); inSpace = true }
            } else {
                out.append(ch)
                inSpace = false
            }
        }
        return out
    }

    /// escape_md: backslash before \ [ ] ( )
    static func escapeMD(_ s: String) -> String {
        var out = ""
        for ch in s {
            if ch == "\\" || ch == "[" || ch == "]" || ch == "(" || ch == ")" { out.append("\\") }
            out.append(ch)
        }
        return out
    }

    /// escape_md_section with only escape_backslash on: "\" followed by a markdown char gets doubled.
    static func escapeMDSection(_ s: String) -> String {
        guard s.contains("\\") else { return s }
        let special: Set<Character> = ["\\", "`", "*", "_", "{", "}", "[", "]", "(", ")", "#", "+", "-", ".", "!"]
        let chars = Array(s)
        var out = ""
        for (k, ch) in chars.enumerated() {
            if ch == "\\", k + 1 < chars.count, special.contains(chars[k + 1]) { out.append("\\") }
            out.append(ch)
        }
        return out
    }

    static func urlJoin(_ base: String, _ ref: String) -> String {
        if base.isEmpty { return ref }
        if let b = URL(string: base), let u = URL(string: ref, relativeTo: b) { return u.absoluteString }
        if ref.contains("://") { return ref }
        if let enc = ref.addingPercentEncoding(withAllowedCharacters: .urlFragmentAllowed),
           let b = URL(string: base), let u = URL(string: enc, relativeTo: b) {
            return u.absoluteString.removingPercentEncoding ?? u.absoluteString
        }
        return ref
    }

    // MARK: data

    private func handleData(_ input: String, entityChar: Bool) {
        // CustomHTML2Text overrides
        if insidePre { o(input); return }
        if insideCode { o(input.replacingOccurrences(of: "\n", with: " ")); return }

        var data = input
        if data.isEmpty { return }
        if stressed {
            data = data.trimmingCharacters(in: .whitespacesAndNewlines)
            stressed = false
            precedingStressed = true
        } else if precedingStressed {
            if let first = data.first, !"][(){}.!?".contains(first), !first.isWhitespace,
               HTML2Markdown.hn(currentTag) == 0, !["a", "code", "pre"].contains(currentTag) {
                data = " " + data
            }
            precedingStressed = false
        }
        if let href = maybeAutomaticLink {
            if href == data, HTML2Markdown.isAbsoluteURL(href) {
                o("<" + data + ">")
                emptyLink = false
                return
            } else {
                o("[")
                maybeAutomaticLink = nil
                emptyLink = false
            }
        }
        if !code && !entityChar { data = HTML2Markdown.escapeMDSection(data) }
        precedingData = data
        o(data, puredata: true)
    }

    private static func isAbsoluteURL(_ s: String) -> Bool {
        guard let r = s.range(of: "://") else { return false }
        let scheme = s[s.startIndex..<r.lowerBound]
        return !scheme.isEmpty && scheme.allSatisfy { $0.isLetter && $0.isASCII || $0 == "+" }
    }

    // MARK: tags

    private func handleTag(_ tag: String, _ attrs: [(String, String?)], start isStart: Bool) {
        // CustomHTML2Text.handle_tag
        if tag == "base" && isStart { baseHandleTag(tag, attrs, isStart); return }
        if tag == "pre" {
            if isStart {
                let lang = (HTML2Markdown.attr(attrs, "data-language") ?? nil) ?? ""
                o("\n```\(lang)\n")
                insidePre = true
            } else {
                o("\n```\n")
                insidePre = false
            }
        } else if tag == "code" {
            if insidePre { return }
            if isStart {
                if !insideLink { o("`") }
                insideCode = true
            } else {
                if !insideLink { o("`") }
                insideCode = false
            }
            if insideLink { baseHandleTag(tag, attrs, isStart) }
        } else {
            baseHandleTag(tag, attrs, isStart)
        }
    }

    private func baseHandleTag(_ tag: String, _ attrs: [(String, String?)], _ isStart: Bool) {
        currentTag = tag
        if tag == "base" && isStart, let href = HTML2Markdown.attr(attrs, "href") ?? nil, !href.isEmpty {
            baseURL = HTML2Markdown.urlJoin(baseURL, href)
        }
        if isStart, maybeAutomaticLink != nil, !["p", "div", "style", "dl", "dt"].contains(tag), tag != "img" {
            o("[")
            maybeAutomaticLink = nil
            emptyLink = false
        }
        let h = HTML2Markdown.hn(tag)
        if h > 0 {
            if !astack.isEmpty {
                if isStart {
                    inheader = true
                    if outText.last == "[" {
                        outText.removeLast()
                        space = false
                        o(String(repeating: "#", count: h) + " ")
                        o("[")
                    }
                } else {
                    pP = 0
                    inheader = false
                    return
                }
            } else {
                p()
                if isStart {
                    inheader = true
                    o(String(repeating: "#", count: h) + " ")
                } else {
                    inheader = false
                    return
                }
            }
        }
        if tag == "p" || tag == "div" {
            if !astack.isEmpty || splitNextTD {
                // pass
            } else {
                p()
            }
        }
        if tag == "br" && isStart { o(blockquote > 0 ? "  \n> " : "  \n") }
        if tag == "hr" && isStart { p(); o("* * *"); p() }
        if tag == "head" || tag == "style" || tag == "script" { quiet += isStart ? 1 : -1 }
        if tag == "style" { style += isStart ? 1 : -1 }
        if tag == "body" { quiet = 0 }
        if tag == "blockquote" {
            if isStart {
                p()
                o("> ", force: .yes)
                start = true
                blockquote += 1
            } else {
                blockquote -= 1
                p()
            }
        }
        if tag == "em" || tag == "i" || tag == "u" {
            var emphasis = "_"
            if isStart, let last = precedingData.last, !last.isWhitespace, !HTML2Markdown.isPunctuation(last) {
                emphasis = " _"
                precedingData += " "
            }
            o(emphasis)
            if isStart { stressed = true }
        }
        if tag == "strong" || tag == "b" {
            var strong = "**"
            if isStart, precedingData.last == "*" {
                strong = " **"
                precedingData += " "
            }
            o(strong)
            if isStart { stressed = true }
        }
        if tag == "del" || tag == "strike" || tag == "s" {
            var strike = "~~"
            if isStart, precedingData.last == "~" {
                strike = " ~~"
                precedingData += " "
            }
            o(strike)
            if isStart { stressed = true }
        }
        if tag == "kbd" || tag == "code" || tag == "tt" {
            o("`")
            code.toggle()
        }
        if tag == "abbr" {
            if isStart {
                abbrTitle = HTML2Markdown.attr(attrs, "title") ?? nil
                abbrData = ""
            } else {
                if let t = abbrTitle, let d = abbrData {
                    if let idx = abbrList.firstIndex(where: { $0.0 == d }) { abbrList[idx].1 = t } else { abbrList.append((d, t)) }
                    abbrTitle = nil
                }
                abbrData = nil
            }
        }
        if tag == "q" {
            o("\"")
            quote.toggle()
        }
        if tag == "a" {
            if isStart {
                insideLink = true
                if let href = HTML2Markdown.attr(attrs, "href") ?? nil, !href.hasPrefix("mailto:") {
                    astack.append(attrs)
                    maybeAutomaticLink = href
                    emptyLink = true
                } else {
                    astack.append(nil)
                }
            } else {
                insideLink = false
                if !astack.isEmpty {
                    let a = astack.removeLast()
                    if maybeAutomaticLink != nil && !emptyLink {
                        maybeAutomaticLink = nil
                    } else if let a {
                        if emptyLink {
                            o("[")
                            emptyLink = false
                            maybeAutomaticLink = nil
                        }
                        pP = 0
                        let href = (HTML2Markdown.attr(a, "href") ?? nil) ?? ""
                        let rawTitle = (HTML2Markdown.attr(a, "title") ?? nil) ?? ""
                        let title = HTML2Markdown.escapeMD(rawTitle)
                        let url = HTML2Markdown.urlJoin(baseURL, href)
                        let titlePart = title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "" : " \"\(title)\""
                        o("](" + HTML2Markdown.escapeMD(url) + titlePart + ")")
                    }
                }
            }
        }
        if tag == "img" && isStart, let src = HTML2Markdown.attr(attrs, "src") ?? nil {
            let alt = (HTML2Markdown.attr(attrs, "alt") ?? nil) ?? ""
            if maybeAutomaticLink != nil {
                o("[")
                maybeAutomaticLink = nil
                emptyLink = false
            }
            o("![" + HTML2Markdown.escapeMD(alt) + "]")
            o("(" + HTML2Markdown.escapeMD(HTML2Markdown.urlJoin(baseURL, src)) + ")")
        }
        if tag == "dl" && isStart {
            p()
            pP = 0
        } else if tag == "dt" && isStart {
            if pP == 0 { o("\n\n") }
            pP = 0
        } else if tag == "dt" && !isStart {
            o("\n")
        } else if tag == "dd" && isStart {
            o("    ")
        } else if tag == "dd" && !isStart {
            pP = 0
        }
        if tag == "ol" || tag == "ul" {
            if list.isEmpty && !lastWasList { p() }
            if isStart {
                var startNum = 0
                if let s = HTML2Markdown.attr(attrs, "start") ?? nil, let v = Int(s.trimmingCharacters(in: .whitespaces)) { startNum = v - 1 }
                list.append(ListElement(tag, startNum))
            } else if !list.isEmpty {
                list.removeLast()
                if list.isEmpty { o("\n") }
            }
            lastWasList = true
        } else {
            lastWasList = false
        }
        if tag == "li" {
            pbr()
            if isStart {
                let li = list.last ?? ListElement("ul", 0)
                var parent: String?
                for l in list {
                    o(parent == "ol" && l.name == "ul" ? "   " : "  ")
                    parent = l.name
                }
                if li.name == "ul" {
                    o("* ")
                } else if li.name == "ol" {
                    li.num += 1
                    o("\(li.num). ")
                }
                start = true
            }
        }
        if tag == "caption" && !isStart { softBR() }
        if tag == "table" || tag == "tr" || tag == "td" || tag == "th" {
            if tag == "table" && isStart {
                tableStart = true
                softBR()
            }
            if (tag == "td" || tag == "th") && isStart {
                o(splitNextTD ? " | " : "| ")
                splitNextTD = true
            }
            if tag == "tr" && isStart { tdCount = 0 }
            if tag == "tr" && !isStart {
                o(" |")
                splitNextTD = false
                softBR()
            }
            if tag == "tr" && !isStart && tableStart {
                o("| " + Array(repeating: "---", count: tdCount).joined(separator: " | ") + " |")
                softBR()
                tableStart = false
            }
            if (tag == "td" || tag == "th") && isStart { tdCount += 1 }
        }
    }

    /// Python string.punctuation (ASCII only).
    private static func isPunctuation(_ c: Character) -> Bool {
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".contains(c)
    }
}
