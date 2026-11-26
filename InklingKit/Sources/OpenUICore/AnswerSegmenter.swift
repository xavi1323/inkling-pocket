// Fence extraction per OpenUI Lang 1.1 §1.2 (MIT, Thesys Inc.), adapted for a chat answer that mixes markdown and OpenUI.
import Foundation

public enum AnswerSegment: Equatable, Sendable {
    case markdown(String)
    case openUI(source: String, complete: Bool)
}

public enum AnswerSegmenter {
    static let tags: Set<String> = ["openui", "openui-lang"]

    /// Splits a model answer into markdown and ```openui fenced programs (an unterminated block runs to the end).
    public static func segments(_ text: String) -> [AnswerSegment] {
        var out: [AnswerSegment] = []
        var markdown = ""
        var lines = text.components(separatedBy: "\n")
        let endsWithNewline = text.hasSuffix("\n")
        if endsWithNewline { lines.removeLast() }
        var i = 0
        var otherFence = false  // inside a non-OpenUI fenced block (kept as markdown)
        while i < lines.count {
            let line = lines[i]
            let isLast = i == lines.count - 1
            let nl = (isLast && !endsWithNewline) ? "" : "\n"
            if let fence = fenceInfo(line) {
                if otherFence {
                    otherFence = false
                    markdown += line + nl
                    i += 1
                    continue
                }
                if tags.contains(fence.tag) {
                    if !markdown.isEmpty { out.append(.markdown(markdown)); markdown = "" }
                    var code: [String] = []
                    if !fence.rest.isEmpty { code.append(fence.rest) }
                    var inString = quoteOpen(fence.rest, false)
                    var closed = false
                    i += 1
                    while i < lines.count {
                        let l = lines[i]
                        if !inString, let f = fenceInfo(l), f.tag.isEmpty, f.rest.isEmpty {
                            closed = true
                            i += 1
                            break
                        }
                        code.append(l)
                        inString = quoteOpen(l, inString)
                        i += 1
                    }
                    out.append(.openUI(source: code.joined(separator: "\n"), complete: closed))
                    continue
                }
                otherFence = true
            }
            markdown += line + nl
            i += 1
        }
        if !markdown.isEmpty { out.append(.markdown(markdown)) }
        return out
    }

    /// A fence line: optional indent (≤3 spaces), ``` (not closed again on the line), a tag, maybe code after it.
    static func fenceInfo(_ line: String) -> (tag: String, rest: String)? {
        var s = Substring(line)
        var indent = 0
        while s.first == " " && indent < 4 { s = s.dropFirst(); indent += 1 }
        guard indent <= 3, s.hasPrefix("```") else { return nil }
        var body = s.dropFirst(3)
        while body.first == "`" { body = body.dropFirst() }
        if body.contains("```") { return nil }  // inline code
        let trimmed = body.drop { $0 == " " || $0 == "\t" }
        let tag = trimmed.prefix { !$0.isWhitespace }
        let rest = trimmed.dropFirst(tag.count).trimmingCharacters(in: .whitespaces)
        return (String(tag).lowercased(), rest)
    }

    /// Tracks whether a double-quoted string is still open after `line`.
    static func quoteOpen(_ line: String, _ open: Bool) -> Bool {
        let c = Array(line)
        var inString = open
        var single = false
        var i = 0
        while i < c.count {
            let ch = c[i]
            if (inString || single) && ch == "\\" { i += 2; continue }
            if single {
                if ch == "'" { single = false }
            } else if inString {
                if ch == "\"" { inString = false }
            } else {
                if ch == "\"" { inString = true }
                else if ch == "'" { single = true }
                else if ch == "#" || (ch == "/" && i + 1 < c.count && c[i + 1] == "/") { break }
            }
            i += 1
        }
        return inString
    }
}
