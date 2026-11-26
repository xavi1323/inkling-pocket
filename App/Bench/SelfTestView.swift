import SwiftUI
import InklingEngine

/// B1–B2: compares the fast engine against the reference and checks three known answers.
struct SelfTestSection: View {
    let engine: InklingGenerator?

    var body: some View {
        RunReportView(
            title: "Self-test",
            explanation: "Checks that the fast engine gives the same answers as the plain one, then asks three questions with known answers. Takes about a minute.",
            runLabel: "Run self-test",
            isAvailable: engine != nil
        ) { progress in
            guard let engine else { return "No model loaded." }
            return try await engine.selfTest(progress: progress).asText
        }
    }
}
