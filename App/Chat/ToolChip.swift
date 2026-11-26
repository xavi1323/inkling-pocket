import SwiftUI
import AgentCore

struct ToolChip: View {
    let chip: ToolChipState
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button {
                if chip.preview != nil { expanded.toggle() }
            } label: {
                HStack(spacing: 6) {
                    switch chip.status {
                    case .running: ProgressView().controlSize(.small)
                    case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                    case .failed: Image(systemName: "exclamationmark.circle.fill").foregroundStyle(.orange)
                    }
                    Text(chip.title).lineLimit(1)
                    if chip.preview != nil {
                        Image(systemName: expanded ? "chevron.up" : "chevron.down").font(.caption2)
                    }
                }
                .font(.callout)
                .padding(.horizontal, 10).padding(.vertical, 6)
                .background(.quaternary.opacity(0.6), in: Capsule())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("\(chip.title), \(chip.status == .running ? "running" : chip.status == .done ? "done" : "failed")")
            if expanded, let preview = chip.preview {
                Text(preview).font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                    .padding(.leading, 10)
            }
        }
    }
}
