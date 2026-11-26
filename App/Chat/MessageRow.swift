import SwiftUI
import InklingCore
import AgentCore
import PocketMemory
import OpenUICore
import OpenUIViews

struct MessageRow: View {
    let message: MessageRecord
    let onAction: @MainActor (String) -> Void

    var body: some View {
        switch message.role {
        case "user":
            HStack {
                Spacer(minLength: 80)
                Text(message.content)
                    .textSelection(.enabled)
                    .padding(.horizontal, 14).padding(.vertical, 10)
                    .background(Color.accentColor.opacity(0.15), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .accessibilityLabel("You said: \(message.content)")
            }
        case "tool":
            let chip = ToolChipState(id: 0, title: toolTitle, status: message.isToolError ? .failed : .done,
                                     preview: String(message.content.prefix(200)))
            ToolChip(chip: chip)
        default:
            if !message.content.isEmpty || message.reasoning != nil {
                AssistantReply(thinking: message.reasoning ?? "", answer: message.content, chips: [],
                               streaming: false, stopped: message.stopped, onAction: onAction)
            }
        }
    }

    private var toolTitle: String {
        switch message.toolName {
        case "web_search": "Searched the web"
        case "open_url": "Read a page"
        case "memory_save": "Remembered"
        case "memory_search": "Recalled"
        case "memory_forget": "Forgot"
        case "chat_search": "Searched chats"
        case "current_time": "Checked the time"
        case "calculator": "Calculated"
        case let name?: "Ran \(name)"
        case nil: "Tool"
        }
    }
}

/// Thinking + tool chips + answer (markdown and OpenUI blocks).
struct AssistantReply: View {
    let thinking: String
    let answer: String
    let chips: [ToolChipState]
    let streaming: Bool
    let stopped: Bool
    let onAction: @MainActor (String) -> Void
    @Environment(\.openURL) private var openURL

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if !thinking.isEmpty {
                ThinkingSection(text: thinking, isStreaming: streaming && answer.isEmpty)
            }
            ForEach(chips) { ToolChip(chip: $0) }
            ForEach(Array(AnswerSegmenter.segments(answer).enumerated()), id: \.offset) { _, segment in
                switch segment {
                case .markdown(let text):
                    MarkdownView(text: text)
                case .openUI(let source, let complete):
                    OpenUIView(document: complete ? OpenUIParser.parse(source) : partial(source), onAction: { handle($0) })
                }
            }
            if streaming && answer.isEmpty && thinking.isEmpty && chips.isEmpty {
                ProgressView().accessibilityLabel("Writing reply")
            }
            if stopped {
                Label("Stopped", systemImage: "stop.circle").font(.caption).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func partial(_ source: String) -> OpenUIDocument {
        var parser = OpenUIParser()
        return parser.feed(source)
    }

    private func handle(_ action: OpenUIAction) {
        if case .openURL(let s) = action {
            if let url = URL(string: s), url.scheme == "https" || url.scheme == "http" { openURL(url) }
            return
        }
        onAction(actionMessage(action))
    }
}
