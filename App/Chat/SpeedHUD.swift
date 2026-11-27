import SwiftUI

/// tok/s of the reply; tap for the engine's numbers of the last finished reply (first token, writing speed,
/// reused prompt, guesses kept).
struct SpeedHUD: View {
    let tokensPerSecond: Double?
    let isRunning: Bool
    var details: [String]? = nil
    @State private var showing = false

    var body: some View {
        Button { showing = true } label: {
            HStack(spacing: 4) {
                Image(systemName: "speedometer")
                Text(tokensPerSecond.map { String(format: "%.1f tok/s", $0) } ?? "– tok/s").monospacedDigit()
            }
            .font(.caption.weight(.medium))
            .foregroundStyle(isRunning ? .primary : .secondary)
            .padding(.horizontal, 12).padding(.vertical, 8)
        }
        .buttonStyle(.plain)
        .disabled(details == nil)
        .popover(isPresented: $showing) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Last reply").font(.headline)
                ForEach(details ?? [], id: \.self) { Text($0).font(.callout).monospacedDigit() }
            }
            .padding()
            .presentationCompactAdaptation(.popover)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(tokensPerSecond.map { String(format: "Speed %.1f tokens per second", $0) } ?? "Speed not measured yet")
        .accessibilityHint(details == nil ? "" : "Shows the last reply's engine numbers")
    }
}
