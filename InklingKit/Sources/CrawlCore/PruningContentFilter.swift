// Swift port of crawl4ai v0.9.6 PruningContentFilter (Apache-2.0, Copyright UncleCode).
import Foundation
import SwiftSoup

public struct PruningContentFilter: Sendable {
    public enum ThresholdType: Sendable { case fixed, dynamic }

    public var threshold: Double
    public var thresholdType: ThresholdType
    public var minWordThreshold: Int?

    static let excludedTags: Set<String> = ["nav", "footer", "header", "aside", "script", "style", "form", "iframe", "noscript"]
    static let tagImportance: [String: Double] = ["article": 1.5, "main": 1.4, "section": 1.3, "p": 1.2, "h1": 1.4, "h2": 1.3, "h3": 1.2, "div": 0.7, "span": 0.6]
    static let tagWeights: [String: Double] = ["div": 0.5, "p": 1.0, "article": 1.5, "section": 1.0, "span": 0.3, "li": 0.5, "ul": 0.5, "ol": 0.5,
                                               "h1": 1.2, "h2": 1.1, "h3": 1.0, "h4": 0.9, "h5": 0.8, "h6": 0.7]
    static let inlineTags: Set<String> = ["a", "abbr", "b", "bdi", "bdo", "cite", "code", "data", "del", "dfn", "em", "i", "ins", "kbd", "mark",
                                          "q", "ruby", "s", "samp", "small", "span", "strong", "sub", "time", "u", "var"]
    static let negativePattern = try! NSRegularExpression(pattern: "^(nav|footer|header|sidebar|ads|comment|promo|advert|social|share)", options: [.caseInsensitive])

    public init(threshold: Double = 0.48, thresholdType: ThresholdType = .fixed, minWordThreshold: Int? = nil) {
        self.threshold = threshold
        self.thresholdType = thresholdType
        self.minWordThreshold = minWordThreshold
    }

    /// Returns the surviving top-level blocks of <body> as HTML strings.
    public func filter(html: String) throws -> [String] {
        guard !html.isEmpty else { return [] }
        let doc = try SwiftSoup.parse(html)
        guard let body = doc.body() else { return [] }
        try removeComments(doc)
        for tag in Self.excludedTags {
            for el in try doc.getElementsByTag(tag).array() { try el.remove() }
        }
        if try prune(body) { return [] }  // body itself removed
        var blocks: [String] = []
        for child in body.getChildNodes() {
            guard let el = child as? Element else { continue }
            if Self.strippedTextLength(el) > 0 { blocks.append(Self.serialize(el)) }
        }
        return blocks
    }

    private func removeComments(_ node: Node) throws {
        for child in node.getChildNodes() {
            if child is Comment { try child.remove() } else { try removeComments(child) }
        }
    }

    /// Returns true when `node` was removed.
    @discardableResult
    private func prune(_ node: Element) throws -> Bool {
        let tag = node.tagNameNormal()
        let textLen = Self.strippedTextLength(node)
        let tagLen = Self.serializeChildren(node).unicodeScalars.count
        var linkTextLen = 0
        for child in node.getChildNodes() {
            if let a = child as? Element, a.tagNameNormal() == "a", let s = Self.bsString(a) {
                linkTextLen += Self.pyStrip(s).unicodeScalars.count
            }
        }
        let score = compositeScore(node: node, tag: tag, textLen: textLen, tagLen: tagLen, linkTextLen: linkTextLen)
        var limit = threshold
        if thresholdType == .dynamic {
            let importance = Self.tagImportance[tag] ?? 0.7
            let textRatio = tagLen > 0 ? Double(textLen) / Double(tagLen) : 0
            let linkRatio = textLen > 0 ? Double(linkTextLen) / Double(textLen) : 1
            if importance > 1 { limit *= 0.8 }
            if textRatio > 0.4 { limit *= 0.9 }
            if linkRatio > 0.6 { limit *= 1.2 }
        }
        if score < limit {
            try node.remove()
            return true
        }
        let inSentence = node.getChildNodes().contains { ($0 as? TextNode).map { !Self.pyStrip($0.getWholeText()).isEmpty } ?? false }
        for child in node.getChildNodes() {
            guard let el = child as? Element else { continue }
            if inSentence && Self.inlineTags.contains(el.tagNameNormal()) { continue }
            try prune(el)
        }
        return false
    }

    private func compositeScore(node: Element, tag: String, textLen: Int, tagLen: Int, linkTextLen: Int) -> Double {
        if let minWords = minWordThreshold, minWords > 0 {
            let text = Self.strippedText(node)
            if text.filter({ $0 == " " }).count + 1 < minWords { return -1 }
        }
        var score = 0.0
        score += 0.4 * (tagLen > 0 ? Double(textLen) / Double(tagLen) : 0)
        score += 0.2 * (1 - (textLen > 0 ? Double(linkTextLen) / Double(textLen) : 0))
        score += 0.2 * (Self.tagWeights[tag] ?? 0.5)
        score += 0.1 * max(0, classIDWeight(node))
        score += 0.1 * log(Double(textLen) + 1)
        return score / 1.0
    }

    private func classIDWeight(_ node: Element) -> Double {
        var w = 0.0
        for key in ["class", "id"] {
            if node.hasAttr(key), let v = try? node.attr(key) {
                let joined = v.split(whereSeparator: \.isWhitespace).joined(separator: " ")
                let ns = joined as NSString
                if Self.negativePattern.firstMatch(in: joined, range: NSRange(location: 0, length: ns.length)) != nil { w -= 0.5 }
            }
        }
        return w
    }

    // MARK: BeautifulSoup-compatible helpers

    /// Python str.strip()
    static func pyStrip(_ s: String) -> String { s.trimmingCharacters(in: .whitespacesAndNewlines) }

    /// get_text(strip=True): each text node stripped, empty ones dropped, joined with "".
    static func strippedText(_ node: Node) -> String {
        var out = ""
        func walk(_ n: Node) {
            for c in n.getChildNodes() {
                if let t = c as? TextNode {
                    out += pyStrip(t.getWholeText())
                } else if c is Element {
                    walk(c)
                }
            }
        }
        walk(node)
        return out
    }

    static func strippedTextLength(_ node: Node) -> Int { strippedText(node).unicodeScalars.count }

    /// BeautifulSoup `.string`: the text when there is exactly one child (recursively), else nil.
    static func bsString(_ el: Element) -> String? {
        let kids = el.getChildNodes()
        guard kids.count == 1 else { return nil }
        if let t = kids[0] as? TextNode { return t.getWholeText() }
        if let e = kids[0] as? Element { return bsString(e) }
        return nil
    }

    static let voidTags: Set<String> = ["area", "base", "br", "col", "embed", "hr", "img", "input", "keygen", "link", "menuitem", "meta",
                                        "param", "source", "track", "wbr", "basefont", "bgsound", "command", "frame", "image", "isindex", "nextid", "spacer"]

    static func escapeText(_ s: String) -> String {
        s.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: ">", with: "&gt;")
    }

    /// str(element) in BeautifulSoup's minimal formatter.
    static func serialize(_ node: Node) -> String {
        if let t = node as? TextNode { return escapeText(t.getWholeText()) }
        if let d = node as? DataNode { return d.getWholeData() }
        guard let el = node as? Element else { return "" }
        let tag = el.tagNameNormal()
        var s = "<" + tag
        if let attrs = el.getAttributes() {
            for a in attrs {
                let v = escapeText(a.getValue()).replacingOccurrences(of: "\"", with: "&quot;")
                s += " \(a.getKey())=\"\(v)\""
            }
        }
        if voidTags.contains(tag) { return s + "/>" }
        return s + ">" + serializeChildren(el) + "</" + tag + ">"
    }

    static func serializeChildren(_ el: Element) -> String {
        el.getChildNodes().map(serialize).joined()
    }
}
