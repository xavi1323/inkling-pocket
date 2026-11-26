import SwiftUI

/// Block-level markdown for answers: headings, paragraphs, lists, quotes, fenced code and pipe tables.
struct MarkdownView: View {
    let text: String

    enum Block: Hashable {
        case heading(Int, String)
        case paragraph(String)
        case bullet(String, ordered: String?, indent: Int)
        case quote(String)
        case code(String, language: String)
        case table([String], [[String]])
        case rule
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(Self.blocks(text).enumerated()), id: \.offset) { _, block in
                view(for: block)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func view(for block: Block) -> some View {
        switch block {
        case .heading(let level, let t):
            inline(t).font(level == 1 ? .title2.bold() : level == 2 ? .title3.bold() : .headline)
                .accessibilityAddTraits(.isHeader)
                .padding(.top, 4)
        case .paragraph(let t):
            inline(t)
        case .bullet(let t, let ordered, let indent):
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(ordered ?? "•").foregroundStyle(.secondary).monospacedDigit()
                inline(t)
            }
            .padding(.leading, CGFloat(indent) * 16)
        case .quote(let t):
            HStack(spacing: 10) {
                RoundedRectangle(cornerRadius: 2).fill(.tertiary).frame(width: 3)
                inline(t).foregroundStyle(.secondary)
            }
        case .code(let code, let language):
            VStack(alignment: .leading, spacing: 4) {
                if !language.isEmpty { Text(language).font(.caption).foregroundStyle(.secondary) }
                ScrollView(.horizontal) {
                    Text(code).font(.system(.callout, design: .monospaced)).textSelection(.enabled)
                }
            }
            .padding(12)
            .background(.background.tertiary, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        case .table(let header, let rows):
            ScrollView(.horizontal) {
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                    GridRow { ForEach(Array(header.enumerated()), id: \.offset) { _, h in inline(h).bold() } }
                    Divider()
                    ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                        GridRow { ForEach(Array(row.enumerated()), id: \.offset) { _, c in inline(c) } }
                    }
                }
            }
        case .rule:
            Divider()
        }
    }

    private func inline(_ s: String) -> Text {
        let a = (try? AttributedString(markdown: s, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(s)
        return Text(a)
    }

    static func blocks(_ text: String) -> [Block] {
        var out: [Block] = []
        var paragraph: [String] = []
        let lines = text.components(separatedBy: "\n")
        var i = 0
        func flush() {
            if !paragraph.isEmpty { out.append(.paragraph(paragraph.joined(separator: "\n"))); paragraph = [] }
        }
        func cells(_ line: String) -> [String] {
            var t = line.trimmingCharacters(in: .whitespaces)
            if t.hasPrefix("|") { t.removeFirst() }
            if t.hasSuffix("|") { t.removeLast() }
            return t.components(separatedBy: "|").map { $0.trimmingCharacters(in: .whitespaces) }
        }
        while i < lines.count {
            let line = lines[i]
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("```") {
                flush()
                let lang = String(trimmed.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                var code: [String] = []
                i += 1
                while i < lines.count && !lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                    code.append(lines[i]); i += 1
                }
                out.append(.code(code.joined(separator: "\n"), language: lang))
                i += 1
                continue
            }
            if trimmed.isEmpty { flush(); i += 1; continue }
            if trimmed.hasPrefix("|"), i + 1 < lines.count,
               lines[i + 1].trimmingCharacters(in: .whitespaces).allSatisfy({ "|-: ".contains($0) }),
               lines[i + 1].contains("-") {
                flush()
                let header = cells(trimmed)
                var rows: [[String]] = []
                i += 2
                while i < lines.count, lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("|") {
                    rows.append(cells(lines[i])); i += 1
                }
                out.append(.table(header, rows))
                continue
            }
            if let level = Optional(trimmed.prefix { $0 == "#" }.count), level > 0, level <= 6,
               trimmed.dropFirst(level).first == " " {
                flush()
                out.append(.heading(level, String(trimmed.dropFirst(level + 1))))
                i += 1
                continue
            }
            if trimmed == "---" || trimmed == "***" || trimmed == "* * *" { flush(); out.append(.rule); i += 1; continue }
            let indent = (line.prefix { $0 == " " }.count) / 2
            if trimmed.hasPrefix("- ") || trimmed.hasPrefix("* ") || trimmed.hasPrefix("+ ") {
                flush()
                out.append(.bullet(String(trimmed.dropFirst(2)), ordered: nil, indent: indent))
                i += 1
                continue
            }
            if let dot = trimmed.firstIndex(of: "."), trimmed[..<dot].allSatisfy(\.isNumber), !trimmed[..<dot].isEmpty,
               trimmed[trimmed.index(after: dot)...].first == " " {
                flush()
                out.append(.bullet(String(trimmed[trimmed.index(dot, offsetBy: 2)...]), ordered: String(trimmed[...dot]), indent: indent))
                i += 1
                continue
            }
            if trimmed.hasPrefix(">") {
                flush()
                out.append(.quote(String(trimmed.dropFirst()).trimmingCharacters(in: .whitespaces)))
                i += 1
                continue
            }
            paragraph.append(line)
            i += 1
        }
        flush()
        return out
    }
}
