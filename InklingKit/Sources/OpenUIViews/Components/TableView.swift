import SwiftUI
import OpenUICore

struct TableView: View {
    let node: OUINode

    var body: some View {
        DataTable(columns: node.array("columns").map(\.displayText),
                  rows: node.array("rows").map { ($0.arrayValue ?? [$0]).map(\.displayText) },
                  caption: node.string("caption"))
    }
}

struct DataTable: View {
    let columns: [String]
    let rows: [[String]]
    let caption: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ScrollView(.horizontal) {
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                    GridRow {
                        ForEach(Array(columns.enumerated()), id: \.offset) { _, c in
                            Text(c).font(.subheadline.weight(.semibold))
                        }
                    }
                    .accessibilityAddTraits(.isHeader)
                    Divider()
                    ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                        GridRow {
                            ForEach(0..<max(columns.count, row.count), id: \.self) { i in
                                Text(i < row.count ? row[i] : "").font(.subheadline).textSelection(.enabled)
                            }
                        }
                        .accessibilityElement(children: .combine)
                    }
                }
                .padding(.vertical, 4)
            }
            if let caption { Text(caption).font(.caption).foregroundStyle(.secondary) }
        }
    }
}
