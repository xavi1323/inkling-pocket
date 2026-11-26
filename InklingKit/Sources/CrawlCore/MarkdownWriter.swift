// Inkling Pocket's own HTML → Markdown writer (written from scratch; not derived from html2text).
// Builds a small element tree, then renders blocks (headings, paragraphs, lists, quotes, code, tables) and inline
// marks (links, images, emphasis, code spans). Output shape follows crawl4ai's defaults: inline links, no line
// wrapping, `*` bullets, fenced code, mailto links dropped.
import Foundation

final class MarkdownWriter {
    final class Node {
        let name: String  // "" for text
        let attrs: [String: String]
        var text: String
        var children: [Node] = []
        init(name: String, attrs: [String: String] = [:], text: String = "") { self.name = name; self.attrs = attrs; self.text = text }
    }

    static let void: Set<String> = ["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"]
    static let skipped: Set<String> = ["script", "style", "noscript", "template", "head", "iframe", "svg", "canvas", "select", "textarea", "object"]
    static let blocks: Set<String> = [
        "address", "article", "aside", "blockquote", "body", "center", "dd", "details", "dialog", "div", "dl", "dt",
        "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hgroup",
        "hr", "html", "li", "main", "nav", "ol", "p", "pre", "section", "summary", "table", "ul", "caption",
    ]
    /// Opening one of these closes an open element of the same kind (HTML's implied end tags).
    static let selfClosingSiblings: Set<String> = ["p", "li", "dt", "dd", "tr", "td", "th", "option"]

    let baseURL: String
    init(baseURL: String) { self.baseURL = baseURL }

    func markdown(_ html: String) -> String {
        let root = Self.parse(html)
        let out = renderBlocks(root.children, depth: 0).joined(separator: "\n\n")
        return out.isEmpty ? "" : out + "\n"
    }

    // MARK: Tree

    static func parse(_ html: String) -> Node {
        let root = Node(name: "#root")
        var stack = [root]
        var skipDepth = 0
        func append(text: String) {
            guard skipDepth == 0, !text.isEmpty else { return }
            let parent = stack.last!
            if let last = parent.children.last, last.name.isEmpty { last.text += text } else { parent.children.append(Node(name: "", text: text)) }
        }
        func open(_ name: String, _ attrs: [(String, String?)], selfClosing: Bool) {
            if skipped.contains(name) {
                if !selfClosing { skipDepth += 1 }
                return
            }
            guard skipDepth == 0 else { return }
            if selfClosingSiblings.contains(name), let i = stack.lastIndex(where: { $0.name == name }),
               !stack[(i + 1)...].contains(where: { ["ul", "ol", "table", "dl"].contains($0.name) }) {
                stack.removeSubrange(i...)
            }
            if blocks.contains(name), stack.last?.name == "p" { stack.removeLast() }
            var dict: [String: String] = [:]
            for (k, v) in attrs where dict[k] == nil { dict[k] = HTMLTokenizer.unescape(v ?? "") }
            let node = Node(name: name, attrs: dict)
            stack.last!.children.append(node)
            if !selfClosing && !void.contains(name) { stack.append(node) }
        }
        for event in HTMLTokenizer.tokenize(html) {
            switch event {
            case .start(let n, let a): open(n.lowercased(), a, selfClosing: false)
            case .startEnd(let n, let a): open(n.lowercased(), a, selfClosing: true)
            case .end(let raw):
                let name = raw.lowercased()
                if skipped.contains(name) { skipDepth = max(0, skipDepth - 1); continue }
                guard skipDepth == 0, let i = stack.lastIndex(where: { $0.name == name }), i > 0 else { continue }
                stack.removeSubrange(i...)
            case .data(let d): append(text: d)
            case .entity(let e): append(text: HTMLEntities.html5[e] ?? "&\(e);")
            case .charRef(let r): append(text: HTMLTokenizer.charRefString(r))
            }
        }
        return root
    }

    // MARK: Blocks

    /// Renders children as Markdown blocks; runs of inline content become paragraphs.
    func renderBlocks(_ nodes: [Node], depth: Int) -> [String] {
        var out: [String] = []
        var inline: [Node] = []
        func flush() {
            let text = Self.tidy(inline.map(renderInline).joined())
            if !text.isEmpty { out.append(text) }
            inline = []
        }
        for node in nodes {
            if node.name.isEmpty || !Self.blocks.contains(node.name) {
                inline.append(node)
                continue
            }
            flush()
            out += renderBlock(node, depth: depth)
        }
        flush()
        return out
    }

    func renderBlock(_ node: Node, depth: Int) -> [String] {
        switch node.name {
        case "h1", "h2", "h3", "h4", "h5", "h6":
            let text = Self.tidy(node.children.map(renderInline).joined()).replacingOccurrences(of: "\n", with: " ")
            return text.isEmpty ? [] : [String(repeating: "#", count: Int(node.name.dropFirst())!) + " " + text]
        case "hr":
            return ["---"]
        case "pre":
            return [codeBlock(node)]
        case "ul", "ol":
            let list = renderList(node, depth: depth)
            return list.isEmpty ? [] : [list]
        case "blockquote":
            let inner = renderBlocks(node.children, depth: depth).joined(separator: "\n\n")
            guard !inner.isEmpty else { return [] }
            return [inner.split(separator: "\n", omittingEmptySubsequences: false).map { $0.isEmpty ? ">" : "> " + $0 }.joined(separator: "\n")]
        case "table":
            let table = renderTable(node)
            return table.isEmpty ? [] : [table]
        case "dd":
            return renderBlocks(node.children, depth: depth).map { "  " + $0.replacingOccurrences(of: "\n", with: "\n  ") }
        default:
            return renderBlocks(node.children, depth: depth)
        }
    }

    func renderList(_ list: Node, depth: Int) -> String {
        var number = Int(list.attrs["start"] ?? "") ?? 1
        var items: [String] = []
        for item in list.children {
            let children: [Node]
            if item.name == "li" { children = item.children }
            else if item.name.isEmpty { if Self.tidy(item.text).isEmpty { continue }; children = [item] }
            else { children = [item] }
            let marker = list.name == "ol" ? "\(number). " : "* "
            number += 1
            let body = renderBlocks(children, depth: depth + 1).joined(separator: "\n")
            let pad = String(repeating: " ", count: marker.count)
            let lines = body.split(separator: "\n", omittingEmptySubsequences: false)
            guard let first = lines.first else { items.append(marker.trimmingCharacters(in: .whitespaces)); continue }
            items.append(([marker + first] + lines.dropFirst().map { $0.isEmpty ? "" : pad + $0 }).joined(separator: "\n"))
        }
        return items.joined(separator: "\n")
    }

    func codeBlock(_ pre: Node) -> String {
        var language = ""
        for node in [pre] + pre.children where !node.name.isEmpty {
            if let cls = node.attrs["class"], let lang = cls.split(separator: " ").first(where: { $0.hasPrefix("language-") || $0.hasPrefix("lang-") }) {
                language = String(lang.split(separator: "-", maxSplits: 1).last ?? "")
                break
            }
        }
        var code = Self.rawText(pre)
        if code.hasPrefix("\n") { code.removeFirst() }
        while code.hasSuffix("\n") { code.removeLast() }
        let fence = code.contains("```") ? "````" : "```"
        return "\(fence)\(language)\n\(code)\n\(fence)"
    }

    func renderTable(_ table: Node) -> String {
        var rows: [[String]] = []
        func collect(_ node: Node) {
            for child in node.children {
                switch child.name {
                case "tr":
                    let cells = child.children.filter { $0.name == "td" || $0.name == "th" }.map { cell in
                        Self.tidy(cell.children.map(renderInline).joined())
                            .replacingOccurrences(of: "\n", with: " ").replacingOccurrences(of: "|", with: "\\|")
                    }
                    if !cells.isEmpty { rows.append(cells) }
                case "thead", "tbody", "tfoot": collect(child)
                default: break
                }
            }
        }
        collect(table)
        let caption = table.children.first { $0.name == "caption" }.map { Self.tidy($0.children.map(renderInline).joined()) } ?? ""
        guard let width = rows.map(\.count).max(), width > 0 else { return caption }
        let padded = rows.map { $0 + Array(repeating: "", count: width - $0.count) }
        func line(_ cells: [String]) -> String { "| " + cells.joined(separator: " | ") + " |" }
        let body = ([line(padded[0]), line(Array(repeating: "---", count: width))] + padded.dropFirst().map(line)).joined(separator: "\n")
        return caption.isEmpty ? body : caption + "\n\n" + body
    }

    // MARK: Inline

    func renderInline(_ node: Node) -> String {
        if node.name.isEmpty { return Self.collapse(node.text) }
        let inner = { node.children.map(self.renderInline).joined() }
        switch node.name {
        case "br": return "\n"
        case "img":
            guard let src = node.attrs["src"], !src.isEmpty, !src.hasPrefix("data:") else { return "" }
            let title = node.attrs["title"].map { " \"\($0)\"" } ?? ""
            return "![\(Self.collapse(node.attrs["alt"] ?? "").trimmingCharacters(in: .whitespaces))](\(absolute(src))\(title))"
        case "a":
            // A link label is one line even when the link wraps blocks (cards, headings).
            let text = inner().split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }.map { $0.drop { $0 == "#" }.trimmingCharacters(in: .whitespaces) }.joined(separator: " ")
            guard let href = node.attrs["href"], !href.isEmpty, !href.lowercased().hasPrefix("javascript:"),
                  !href.lowercased().hasPrefix("mailto:") else { return text }
            let label = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !label.isEmpty else { return text }
            if label == href || label == absolute(href) { return text }  // bare URL: the text already is the link
            let title = node.attrs["title"].map { " \"\($0)\"" } ?? ""
            return Self.wrap(text, "[", "](\(absolute(href).replacingOccurrences(of: " ", with: "%20"))\(title))")
        case "strong", "b": return Self.wrap(inner(), "**", "**")
        case "em", "i", "cite", "dfn": return Self.wrap(inner(), "_", "_")
        case "del", "s", "strike": return Self.wrap(inner(), "~~", "~~")
        case "code", "kbd", "samp", "tt":
            let code = Self.collapse(Self.rawText(node))
            let trimmed = code.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { return code }
            let tick = trimmed.contains("`") ? "``" : "`"
            return Self.wrap(code, tick, tick)
        case "input", "button", "label", "option": return node.name == "input" ? "" : inner()
        default:
            // Block elements met inside inline content (e.g. a <div> inside a <span>) still break the line.
            if Self.blocks.contains(node.name) { return "\n" + renderBlocks(node.children, depth: 0).joined(separator: "\n") + "\n" }
            return inner()
        }
    }

    /// Links and images are written with absolute URLs (relative to the page).
    func absolute(_ url: String) -> String {
        baseURL.isEmpty ? url : Self.urlJoin(baseURL, url)
    }

    /// Puts marks around the text but keeps its outer spaces outside: " a " → " **a** ".
    static func wrap(_ text: String, _ open: String, _ close: String) -> String {
        let core = text.trimmingCharacters(in: .whitespaces)
        guard !core.isEmpty else { return text }
        let lead = text.prefix { $0 == " " }, trail = String(text.reversed().prefix { $0 == " " })
        return lead + open + core + close + trail
    }

    static func rawText(_ node: Node) -> String {
        if node.name.isEmpty { return node.text }
        if node.name == "br" { return "\n" }
        return node.children.map(rawText).joined()
    }

    /// HTML whitespace rules: any run of spaces, tabs or newlines is one space (non-breaking spaces become spaces).
    static func collapse(_ s: String) -> String {
        var out = ""
        var lastSpace = false
        for ch in s {
            if ch.isWhitespace && ch != "\u{00A0}" || ch == "\u{00A0}" {
                if !lastSpace { out.append(" ") }
                lastSpace = true
            } else {
                out.append(ch)
                lastSpace = false
            }
        }
        return out
    }

    /// Trims each line and drops blank lines inside a paragraph.
    static func tidy(_ s: String) -> String {
        s.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }.joined(separator: "\n")
    }

    static func urlJoin(_ base: String, _ url: String) -> String {
        guard let b = URL(string: base), let joined = URL(string: url, relativeTo: b) else { return url }
        return joined.absoluteString
    }
}
