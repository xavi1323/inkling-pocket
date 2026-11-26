import SwiftUI

struct ThinkingSection: View {
    let text: String
    let isStreaming: Bool
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            Text(text)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 4)
        } label: {
            Label(isStreaming ? "Thinking…" : "Thought", systemImage: "brain")
                .font(.callout.weight(.medium))
                .foregroundStyle(.secondary)
        }
        .accessibilityHint(expanded ? "Hides the model's reasoning" : "Shows the model's reasoning")
    }
}
