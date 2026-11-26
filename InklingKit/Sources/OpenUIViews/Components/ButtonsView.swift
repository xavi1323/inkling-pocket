import SwiftUI
import OpenUICore

struct ButtonsView: View {
    let node: OUINode

    var body: some View {
        FlowLayout(spacing: 8) {
            ForEach(Array(node.array("buttons").enumerated()), id: \.offset) { _, b in
                if let n = b.nodeValue { ButtonsView.single(n) }
            }
        }
    }

    @ViewBuilder
    static func single(_ node: OUINode) -> some View {
        ActionButton(node: node)
    }
}

struct ActionButton: View {
    let node: OUINode
    @Environment(\.openUIAction) private var handler

    var body: some View {
        let label = node.string("label") ?? ""
        let button = Button(label) {
            for action in buttonActions(node) { handler.perform(action) }
        }
        .accessibilityHint("Sends a reply")
        if node.string("style") == "primary" {
            button.buttonStyle(.borderedProminent)
        } else {
            button.buttonStyle(.bordered)
        }
    }
}
