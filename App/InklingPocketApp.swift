import SwiftUI

@main
struct InklingPocketApp: App {
    @State private var app = AppModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(app)
                .environment(app.settings)
                .modelContainer(app.container)
        }
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Chat") { app.newChat() }
                    .keyboardShortcut("n", modifiers: .command)
            }
        }
    }
}

struct RootView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
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
            case .benchmark:
                ContentUnavailableView("Benchmark", systemImage: "speedometer",
                                       description: Text("Arrives with the engine. It will measure tokens per second, memory and expert routing."))
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
