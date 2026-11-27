import SwiftUI
import InklingCore
import PocketMemory

struct ChatView: View {
    @Bindable var session: ChatSession
    @State private var draft = ""

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 18) {
                    ForEach(session.messages, id: \.id) { message in
                        MessageRow(message: message, onAction: session.sendFollowUp)
                    }
                    if let turn = session.turn {
                        AssistantReply(thinking: turn.thinking, answer: turn.answer, chips: turn.chips,
                                       streaming: true, stopped: false, onAction: session.sendFollowUp)
                    }
                    if let error = session.errorMessage {
                        Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.red).font(.callout)
                    }
                    Color.clear.frame(height: 90).id("bottom")
                }
                .padding(.horizontal, 24)
                .padding(.top, 16)
                .frame(maxWidth: 860)
                .frame(maxWidth: .infinity)
            }
            .scrollDismissesKeyboard(.interactively)
            .onChange(of: session.streamingAnswer.count + session.messages.count) {
                proxy.scrollTo("bottom", anchor: .bottom)
            }
        }
        .overlay(alignment: .bottom) {
            ComposerBar(draft: $draft, isRunning: session.isRunning, speed: session.tokensPerSecond,
                        details: session.lastReply?.lines,
                        send: { session.send(draft); draft = "" }, stop: session.stop)
                .padding(.horizontal, 24)
                .padding(.bottom, 12)
                .frame(maxWidth: 860)
        }
        .navigationTitle(session.chat.title)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Menu {
                    Picker("Thinking effort", selection: Binding(get: { session.chat.effort }, set: { session.chat.effort = $0 })) {
                        ForEach(ReasoningEffort.allCases, id: \.self) { Text($0.rawValue.capitalized).tag($0.rawValue) }
                    }
                } label: {
                    Label("Thinking: \(session.chat.effort.capitalized)", systemImage: "brain")
                }
                .accessibilityLabel("Thinking effort, \(session.chat.effort)")
            }
        }
    }
}
