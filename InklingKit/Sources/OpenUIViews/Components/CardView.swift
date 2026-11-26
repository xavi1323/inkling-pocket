import SwiftUI
import OpenUICore

struct CardView: View {
    let node: OUINode

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let title = node.string("title") {
                Text(title).font(.headline).accessibilityAddTraits(.isHeader)
            }
            ForEach(Array(node.array("children").enumerated()), id: \.offset) { _, child in
                ContentView(value: child)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

struct ListView: View {
    let node: OUINode

    var body: some View {
        let ordered = node["ordered"]?.boolValue ?? false
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(node.array("items").enumerated()), id: \.offset) { index, item in
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(ordered ? "\(index + 1)." : "•").foregroundStyle(.secondary).monospacedDigit()
                    ContentView(value: item)
                }
                .accessibilityElement(children: .combine)
            }
        }
    }
}

struct CalloutView: View {
    let node: OUINode

    var body: some View {
        let kind = node.string("kind") ?? "info"
        let (icon, color): (String, Color) = switch kind {
        case "success": ("checkmark.circle.fill", .green)
        case "warning": ("exclamationmark.triangle.fill", .orange)
        case "error": ("xmark.octagon.fill", .red)
        default: ("info.circle.fill", .blue)
        }
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon).foregroundStyle(color).accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 4) {
                if let title = node.string("title") { Text(title).font(.subheadline.weight(.semibold)) }
                InlineMarkdown(text: node.string("text") ?? "")
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(color.opacity(0.12), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(kind.capitalized): \(node.string("title").map { $0 + ". " } ?? "")\(node.string("text") ?? "")")
    }
}

struct StepsView: View {
    let node: OUINode

    var body: some View {
        let current = Int(node["current"]?.numberValue ?? 0)
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(node.array("items").enumerated()), id: \.offset) { index, item in
                let number = index + 1
                let done = current > 0 && number < current
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    ZStack {
                        Circle().fill(number == current ? Color.accentColor : Color.secondary.opacity(0.2))
                        if done {
                            Image(systemName: "checkmark").font(.caption.bold())
                        } else {
                            Text("\(number)").font(.caption.bold())
                                .foregroundStyle(number == current ? Color.white : Color.primary)
                        }
                    }
                    .frame(width: 24, height: 24)
                    InlineMarkdown(text: item.displayText)
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel("Step \(number)\(done ? ", done" : number == current ? ", current" : ""): \(item.displayText)")
            }
        }
    }
}
