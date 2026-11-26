import SwiftUI
import InklingCore
import InklingEngine

struct SettingsView: View {
    @Environment(AppSettings.self) private var settings
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var settings = settings
        Form {
            Section("Thinking") {
                Picker("Default thinking effort", selection: $settings.effort) {
                    ForEach(ReasoningEffort.allCases, id: \.self) { Text($0.rawValue.capitalized).tag($0) }
                }
                LabeledContent("Temperature") {
                    Slider(value: $settings.temperature, in: 0...1.5, step: 0.05) { Text("Temperature") }
                    Text(settings.temperature, format: .number.precision(.fractionLength(2))).monospacedDigit().frame(width: 44)
                }
                LabeledContent("Top-p") {
                    Slider(value: $settings.topP, in: 0.1...1, step: 0.05) { Text("Top-p") }
                    Text(settings.topP, format: .number.precision(.fractionLength(2))).monospacedDigit().frame(width: 44)
                }
            }
            Section("Agent") {
                Stepper("Max tool steps: \(settings.maxSteps)", value: $settings.maxSteps, in: 1...20)
                Picker("Context size", selection: $settings.contextCap) {
                    ForEach([4096, 8192, 16_384, 32_768], id: \.self) { Text("\($0 / 1024)K tokens").tag($0) }
                }
                Stepper("Pages read per search: \(settings.crawlTopN)", value: $settings.crawlTopN, in: 0...6)
            }
            Section {
                Picker("Engine", selection: $settings.enginePath) {
                    ForEach(EnginePath.allCases) { Text($0.title).tag($0) }
                }
                Toggle("Smart expert picks (faster, near-exact)", isOn: $settings.smartRouting)
                LabeledContent("Model", value: app.engineStatus.text)
                LabeledContent("Memory budget") {
                    Stepper(settings.memoryBudgetOverrideGB == 0 ? "Automatic" : String(format: "%.1f GB", settings.memoryBudgetOverrideGB),
                            value: $settings.memoryBudgetOverrideGB, in: 0...12, step: 0.5)
                }
                Toggle("Use model folder in place (don't copy)", isOn: $settings.keepModelInPlace)
            } header: {
                Text("Engine")
            } footer: {
                Text("Smart expert picks: each word keeps its top experts and swaps lower-ranked ones for experts already in memory, so far fewer are read from storage. Tool calls always run exact. Reference is slower but simple; use it to check answers if the fast engine looks wrong.")
            }
            Section("About") {
                Text("Built on mlx-swift, mlx-vlm, OpenUI and SwiftSoup. This product includes software developed by UncleCode (https://x.com/unclecode) as part of the Crawl4AI project (https://github.com/unclecode/crawl4ai).")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Settings")
    }
}
