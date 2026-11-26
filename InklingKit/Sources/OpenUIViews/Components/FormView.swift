import SwiftUI
import OpenUICore

struct FormView: View {
    let node: OUINode
    @Environment(\.openUIAction) private var handler
    @State private var values: [String: String] = [:]
    @State private var submitted = false

    var body: some View {
        let fields = node.array("fields").compactMap(\.nodeValue)
        VStack(alignment: .leading, spacing: 12) {
            ForEach(Array(fields.enumerated()), id: \.offset) { _, field in
                FieldView(field: field, value: binding(for: field.string("name") ?? ""))
            }
            Button(submitted ? "Sent" : (node.string("submitLabel") ?? "Submit")) {
                var out: [String: String] = [:]
                for f in fields {
                    let name = f.string("name") ?? ""
                    out[name] = values[name] ?? (f.string("kind") == "toggle" ? "false" : "")
                }
                submitted = true
                handler.perform(.submitForm(id: node.string("id") ?? "form", values: out))
            }
            .buttonStyle(.borderedProminent)
            .disabled(submitted)
        }
        .padding(14)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private func binding(for name: String) -> Binding<String> {
        Binding(get: { values[name] ?? "" }, set: { values[name] = $0 })
    }
}

struct FieldView: View {
    let field: OUINode
    @Binding var value: String

    var body: some View {
        let label = field.string("label") ?? field.string("name") ?? ""
        let placeholder = field.string("placeholder") ?? ""
        VStack(alignment: .leading, spacing: 4) {
            switch field.string("kind") ?? "text" {
            case "toggle":
                Toggle(label, isOn: Binding(get: { value == "true" }, set: { value = $0 ? "true" : "false" }))
            case "select":
                Picker(label, selection: $value) {
                    Text("Choose…").tag("")
                    ForEach(field.array("options").map(\.displayText), id: \.self) { Text($0).tag($0) }
                }
            case "multiline":
                Text(label).font(.subheadline)
                TextEditor(text: $value)
                    .frame(minHeight: 80)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                    .accessibilityLabel(label)
            case let kind:
                Text(label).font(.subheadline)
                TextField(placeholder, text: $value)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel(label)
                    #if os(iOS)
                    .keyboardType(kind == "number" ? .decimalPad : kind == "email" ? .emailAddress : .default)
                    .textInputAutocapitalization(kind == "email" ? .never : .sentences)
                    #endif
            }
        }
    }
}
