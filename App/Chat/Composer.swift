import SwiftUI

/// Floating composer + speed HUD. Liquid Glass on controls only; plain material with Reduce Transparency.
struct ComposerBar: View {
    @Binding var draft: String
    let isRunning: Bool
    let speed: Double?
    let send: @MainActor () -> Void
    let stop: @MainActor () -> Void
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @FocusState private var focused: Bool

    var body: some View {
        GlassEffectContainer(spacing: 12) {
            HStack(alignment: .bottom, spacing: 12) {
                SpeedHUD(tokensPerSecond: speed, isRunning: isRunning)
                    .modifier(GlassOrMaterial(shape: Capsule(), reduce: reduceTransparency))
                HStack(alignment: .bottom, spacing: 8) {
                    TextField("Message Inkling", text: $draft, axis: .vertical)
                        .lineLimit(1...8)
                        .focused($focused)
                        .padding(.vertical, 10)
                        .padding(.leading, 16)
                        .onSubmit { if !isRunning { send() } }
                        .accessibilityLabel("Message")
                    if isRunning {
                        Button(action: stop) {
                            Image(systemName: "stop.fill").font(.body.weight(.semibold)).frame(width: 36, height: 36)
                        }
                        .buttonStyle(.borderedProminent)
                        .buttonBorderShape(.circle)
                        .keyboardShortcut(".", modifiers: .command)
                        .accessibilityLabel("Stop")
                    } else {
                        Button(action: send) {
                            Image(systemName: "arrow.up").font(.body.weight(.semibold)).frame(width: 36, height: 36)
                        }
                        .buttonStyle(.borderedProminent)
                        .buttonBorderShape(.circle)
                        .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        .keyboardShortcut(.return, modifiers: .command)
                        .accessibilityLabel("Send")
                    }
                }
                .padding(4)
                .modifier(GlassOrMaterial(shape: RoundedRectangle(cornerRadius: 24, style: .continuous), reduce: reduceTransparency))
            }
        }
        .onAppear { focused = true }
    }
}

struct GlassOrMaterial<S: Shape>: ViewModifier {
    let shape: S
    let reduce: Bool

    func body(content: Content) -> some View {
        if reduce {
            content.background(.thickMaterial, in: shape)
        } else {
            content.glassEffect(.regular.interactive(), in: shape)
        }
    }
}
