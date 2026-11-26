import SwiftUI
import InklingEngine

@main
struct InklingPocketApp: App {
    @State private var app = AppModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(app)
                .environment(app.settings)
                .modelContainer(app.container)
                .task(id: EngineKey(folder: app.modelFolder, path: app.settings.enginePath, budget: app.settings.memoryBudgetOverrideGB)) { await app.loadEngineIfNeeded() }
                .onChange(of: app.settings.contextCap) { app.applyEngineSettings() }
                .onChange(of: app.settings.smartRouting) { app.applyEngineSettings() }
                .onChange(of: app.settings.memoryBudgetOverrideGB) { app.applyEngineSettings() }
                .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in
                    app.handleMemoryWarning()
                }
        }
        .onChange(of: scenePhase) { _, phase in
            // iOS does not allow GPU work in the background; stop cleanly and keep the partial answer.
            if phase == .background { app.stopAll() }
        }
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Chat") { app.newChat() }
                    .keyboardShortcut("n", modifiers: .command)
            }
        }
    }
}

private struct EngineKey: Equatable {
    var folder: URL?
    var path: EnginePath
    var budget: Double
}

struct RootView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        if app.modelFolder == nil && !app.skippedImport {
            ImportView()
        } else {
            mainView
        }
    }

    private var mainView: some View {
        NavigationSplitView {
            SidebarView()
        } detail: {
            switch app.selection {
            case .chat(let id):
                if let session = app.session(for: id) {
                    ChatView(session: session).id(id)
                } else {
                    ContentUnavailableView("Chat not found", systemImage: "bubble.left.and.exclamationmark.bubble.right")
                }
            case .memory: MemoryView()
            case .benchmark: BenchmarkView()
            case .settings: SettingsView()
            case .diagnostics: SmokeView()
            case nil:
                ContentUnavailableView {
                    Label("Inkling Pocket", systemImage: "sparkles")
                } description: {
                    Text("Start a chat. Everything runs on this iPad.")
                } actions: {
                    Button("New Chat") { app.newChat() }.buttonStyle(.borderedProminent)
                }
            }
        }
    }
}
