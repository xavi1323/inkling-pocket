import SwiftUI
import Charts
import OpenUICore

struct ChartView: View {
    let node: OUINode

    struct Point: Identifiable {
        let id: Int
        let label: String
        let series: String
        let value: Double
    }

    var body: some View {
        let labels = node.array("labels").map(\.displayText)
        let series: [(name: String, values: [Double])] = node.array("series").enumerated().map { i, s in
            guard case .object(let o) = s else { return ("Series \(i + 1)", []) }
            return (o["name"]?.displayText ?? "Series \(i + 1)", o["values"]?.arrayValue?.compactMap(\.numberValue) ?? [])
        }
        var points: [Point] = []
        for s in series {
            for (i, v) in s.values.enumerated() where i < labels.count {
                points.append(Point(id: points.count, label: labels[i], series: s.name, value: v))
            }
        }
        let kind = chartKind(node.string("type"))
        return VStack(alignment: .leading, spacing: 8) {
            if let title = node.string("title") { Text(title).font(.subheadline.weight(.semibold)) }
            switch kind {
            case .bar:
                Chart(points) { p in
                    BarMark(x: .value("Label", p.label), y: .value("Value", p.value))
                        .foregroundStyle(by: .value("Series", p.series))
                        .position(by: .value("Series", p.series))
                }
                .frame(minHeight: 220)
            case .line:
                Chart(points) { p in
                    LineMark(x: .value("Label", p.label), y: .value("Value", p.value))
                        .foregroundStyle(by: .value("Series", p.series))
                    PointMark(x: .value("Label", p.label), y: .value("Value", p.value))
                        .foregroundStyle(by: .value("Series", p.series))
                }
                .frame(minHeight: 220)
            case .pie:
                Chart(points.filter { $0.series == series.first?.name }) { p in
                    SectorMark(angle: .value("Value", p.value), innerRadius: .ratio(0.5), angularInset: 1)
                        .foregroundStyle(by: .value("Label", p.label))
                }
                .frame(minHeight: 220)
            case .table:
                DataTable(columns: ["Label"] + series.map(\.name),
                          rows: labels.enumerated().map { i, l in
                              [l] + series.map { i < $0.values.count ? OUIValue.formatNumber($0.values[i]) : "" }
                          },
                          caption: nil)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(node.string("title") ?? "Chart")
    }
}
