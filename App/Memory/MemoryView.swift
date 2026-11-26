import SwiftUI
import SwiftData
import PocketMemory

/// Lists remembered facts; edit, pin or delete them (R2).
struct MemoryView: View {
    @Environment(\.modelContext) private var context
    @Query(sort: \FactRecord.created, order: .reverse) private var facts: [FactRecord]
    @State private var newFact = ""

    var body: some View {
        List {
            Section {
                HStack {
                    TextField("Add something to remember", text: $newFact)
                        .onSubmit(add)
                    Button("Add", action: add).disabled(newFact.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            } footer: {
                Text("Inkling saves facts here when you tell it something worth remembering. Pinned facts are always included.")
            }
            Section("Remembered") {
                if facts.isEmpty {
                    Text("Nothing yet.").foregroundStyle(.secondary)
                }
                ForEach(facts) { fact in
                    FactRow(fact: fact)
                }
                .onDelete { offsets in
                    for i in offsets { context.delete(facts[i]) }
                    try? context.save()
                }
            }
        }
        .navigationTitle("Memory")
    }

    private func add() {
        let text = newFact.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        context.insert(FactRecord(text: text))
        try? context.save()
        newFact = ""
    }
}

struct FactRow: View {
    @Bindable var fact: FactRecord
    @Environment(\.modelContext) private var context

    var body: some View {
        HStack {
            TextField("Fact", text: $fact.text, axis: .vertical)
                .onSubmit {
                    fact.embedding = nil  // re-embedded on next search miss; keyword search still works
                    try? context.save()
                }
            Button {
                fact.pinned.toggle()
                try? context.save()
            } label: {
                Image(systemName: fact.pinned ? "pin.fill" : "pin")
            }
            .buttonStyle(.borderless)
            .accessibilityLabel(fact.pinned ? "Unpin" : "Pin")
        }
    }
}
