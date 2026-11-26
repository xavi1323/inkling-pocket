import SwiftUI
import InklingEngine

/// B3–B4: speed, memory and routing numbers, copyable as text.
struct BenchmarkView: View {
    @Environment(AppModel.self) private var app

    private var engine: InklingGenerator? { app.model as? InklingGenerator }

    var body: some View {
        Form {
            if engine == nil {
                Section {
                    Label(app.engineStatus.text, systemImage: "exclamationmark.triangle")
                    Text("Import the model first (Settings shows its status).").foregroundStyle(.secondary)
                }
            }
            SelfTestSection(engine: engine)
            RunReportView(
                title: "Benchmark",
                explanation: "Reads a 512-token prompt, writes 256 tokens, then records which experts each layer uses. Keep the iPad plugged in and the app open.",
                runLabel: "Run benchmark",
                isAvailable: engine != nil
            ) { progress in
                guard let engine else { return "No model loaded." }
                return try await engine.benchmark(progress: progress).asText
            }
        }
        .navigationTitle("Benchmark")
    }
}
