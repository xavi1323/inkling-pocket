import SwiftUI
import OpenUICore

struct RemoteImageView: View {
    let node: OUINode

    var body: some View {
        let url = node.string("url").flatMap(URL.init(string:)).flatMap { $0.scheme == "https" ? $0 : nil }
        VStack(alignment: .leading, spacing: 6) {
            AsyncImage(url: url) { phase in
                switch phase {
                case .success(let image): image.resizable().scaledToFit()
                case .failure: Label("Image unavailable", systemImage: "photo").foregroundStyle(.secondary)
                default: ProgressView().frame(maxWidth: .infinity, minHeight: 120)
                }
            }
            .frame(maxHeight: 360)
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .accessibilityLabel(node.string("alt") ?? node.string("caption") ?? "Image")
            if let caption = node.string("caption") { Text(caption).font(.caption).foregroundStyle(.secondary) }
        }
    }
}
