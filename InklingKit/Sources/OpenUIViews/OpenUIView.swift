import SwiftUI
import OpenUICore

/// Renders an OpenUI document. Content only — no glass here (Liquid Glass is for controls/navigation).
public struct OpenUIView: View {
    let document: OpenUIDocument
    let onAction: @MainActor @Sendable (OpenUIAction) -> Void

    public init(document: OpenUIDocument, onAction: @escaping @MainActor @Sendable (OpenUIAction) -> Void) {
        self.document = document
        self.onAction = onAction
    }

    public var body: some View {
        if let root = document.root {
            NodeView(node: root)
                .environment(\.openUIAction, OpenUIActionHandler(perform: onAction))
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

struct OpenUIActionHandler: Sendable {
    var perform: @MainActor @Sendable (OpenUIAction) -> Void = { _ in }
}

extension EnvironmentValues {
    @Entry var openUIAction = OpenUIActionHandler()
}

struct NodeView: View {
    let node: OUINode

    var body: some View {
        switch node.component {
        case "Card": CardView(node: node)
        case "Table": TableView(node: node)
        case "List": ListView(node: node)
        case "Chart": ChartView(node: node)
        case "Callout": CalloutView(node: node)
        case "CodeBlock": CodeBlockView(node: node)
        case "Steps": StepsView(node: node)
        case "Buttons": ButtonsView(node: node)
        case "Form": FormView(node: node)
        case "Image": RemoteImageView(node: node)
        case "Button": ButtonsView.single(node)
        default: EmptyView()
        }
    }
}

/// A child slot: a component, or a primitive shown as (inline markdown) text.
struct ContentView: View {
    let value: OUIValue

    var body: some View {
        if case .node(let n) = value {
            NodeView(node: n)
        } else if !value.displayText.isEmpty {
            InlineMarkdown(text: value.displayText)
        }
    }
}

struct InlineMarkdown: View {
    let text: String
    var body: some View {
        let attributed = (try? AttributedString(markdown: text, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
            ?? AttributedString(text)
        Text(attributed)
            .font(.body)
            .fixedSize(horizontal: false, vertical: true)
            .textSelection(.enabled)
    }
}

extension OUINode {
    func string(_ key: String) -> String? { self[key]?.stringValue }
    func array(_ key: String) -> [OUIValue] { self[key]?.arrayValue ?? [] }
}

/// Wraps children onto new lines when they don't fit (used for button rows).
struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = arrange(width: proposal.width ?? .infinity, subviews: subviews)
        let height = rows.map(\.height).reduce(0, +) + spacing * CGFloat(max(rows.count - 1, 0))
        let width = rows.map(\.width).max() ?? 0
        return CGSize(width: proposal.width ?? width, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var y = bounds.minY
        for row in arrange(width: bounds.width, subviews: subviews) {
            var x = bounds.minX
            for index in row.indices {
                let size = subviews[index].sizeThatFits(.unspecified)
                subviews[index].place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += row.height + spacing
        }
    }

    private struct Row { var indices: [Int] = []; var width: CGFloat = 0; var height: CGFloat = 0 }

    private func arrange(width: CGFloat, subviews: Subviews) -> [Row] {
        var rows: [Row] = [Row()]
        for (i, sub) in subviews.enumerated() {
            let size = sub.sizeThatFits(.unspecified)
            let needed = rows[rows.count - 1].indices.isEmpty ? size.width : rows[rows.count - 1].width + spacing + size.width
            if needed > width && !rows[rows.count - 1].indices.isEmpty {
                rows.append(Row())
            }
            var row = rows[rows.count - 1]
            row.width = row.indices.isEmpty ? size.width : row.width + spacing + size.width
            row.height = max(row.height, size.height)
            row.indices.append(i)
            rows[rows.count - 1] = row
        }
        return rows.filter { !$0.indices.isEmpty }
    }
}
