import SwiftUI
import UIKit

/// Shared layout for a long-running check: a run button, live status, then the plain-text report with Copy.
struct RunReportView: View {
    let title: String
    let explanation: String
    let runLabel: String
    let isAvailable: Bool
    let run: (@escaping @Sendable (String) -> Void) async throws -> String

    @State private var state = RunState()

    var body: some View {
        Section {
            Text(explanation).font(.callout).foregroundStyle(.secondary)
            if let task = state.task, let status = state.status {
                HStack {
                    ProgressView()
                    Text(status).font(.callout)
                    Spacer()
                    Button("Cancel", role: .cancel) { task.cancel() }
                }
            } else {
                Button(runLabel) { start() }
                    .disabled(!isAvailable)
            }
            if let report = state.report {
                Text(report)
                    .font(.system(.footnote, design: .monospaced))
                    .textSelection(.enabled)
                Button("Copy results", systemImage: "doc.on.doc") { UIPasteboard.general.string = report }
            }
        } header: {
            Text(title)
        }
    }

    private func start() {
        let state = self.state
        let run = self.run
        state.report = nil
        state.status = "Starting…"
        state.task = Task {
            do {
                state.report = try await run { message in Task { @MainActor in state.status = message } }
            } catch is CancellationError {
                state.report = "Cancelled."
            } catch {
                state.report = "Failed: \(error)"
            }
            state.status = nil
            state.task = nil
        }
    }
}

@Observable @MainActor
private final class RunState {
    var status: String?
    var report: String?
    var task: Task<Void, Never>?
}
