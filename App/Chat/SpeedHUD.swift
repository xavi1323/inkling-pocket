import SwiftUI

struct SpeedHUD: View {
    let tokensPerSecond: Double?
    let isRunning: Bool

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "speedometer")
            Text(tokensPerSecond.map { String(format: "%.1f tok/s", $0) } ?? "– tok/s").monospacedDigit()
        }
        .font(.caption.weight(.medium))
        .foregroundStyle(isRunning ? .primary : .secondary)
        .padding(.horizontal, 12).padding(.vertical, 8)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(tokensPerSecond.map { String(format: "Speed %.1f tokens per second", $0) } ?? "Speed not measured yet")
    }
}
