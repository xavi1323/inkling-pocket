import SwiftUI
import SwiftData
import PocketMemory

struct SidebarView: View {
    @Environment(AppModel.self) private var app
    @Query(sort: \ChatRecord.updated, order: .reverse) private var chats: [ChatRecord]

    var body: some View {
        @Bindable var app = app
        List(selection: $app.selection) {
            Section("Chats") {
                ForEach(chats) { chat in
                    NavigationLink(value: SidebarItem.chat(chat.id)) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(chat.title).lineLimit(1)
                            Text(chat.updated, format: .relative(presentation: .named)).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .contextMenu {
                        Button("Delete", systemImage: "trash", role: .destructive) { app.delete(chat) }
                    }
                }
                .onDelete { offsets in
                    for chat in offsets.map({ chats[$0] }) { app.delete(chat) }
                }
            }
            Section {
                NavigationLink(value: SidebarItem.memory) { Label("Memory", systemImage: "brain.head.profile") }
                NavigationLink(value: SidebarItem.benchmark) { Label("Benchmark", systemImage: "speedometer") }
                NavigationLink(value: SidebarItem.settings) { Label("Settings", systemImage: "gearshape") }
                NavigationLink(value: SidebarItem.diagnostics) { Label("Diagnostics", systemImage: "stethoscope") }
            }
        }
        .navigationTitle("Inkling")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button("New Chat", systemImage: "square.and.pencil") { app.newChat() }
                    .accessibilityLabel("New chat")
            }
        }
    }
}
