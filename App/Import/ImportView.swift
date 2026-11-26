import SwiftUI
import UniformTypeIdentifiers
import InklingCore

/// First run: pick the Inkling-2-Mini 12GB folder, check it, then copy it in or use it in place (M1–M4).
struct ImportView: View {
    @Environment(AppModel.self) private var app
    @Environment(AppSettings.self) private var settings
    @State private var picking = false
    @State private var error: String?
    @State private var progress: Double?
    @State private var copyTask: Task<Void, Never>?
    @State private var manifest: ModelManifest?

    var body: some View {
        @Bindable var settings = settings
        VStack(spacing: 24) {
            Image(systemName: "shippingbox").font(.system(size: 56)).foregroundStyle(.tint)
            Text("Import the model").font(.largeTitle.bold())
            Text("Choose the **Inkling-2-Mini-abliterated-MLX** 12GB folder in Files. Nothing is downloaded or uploaded.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .frame(maxWidth: 520)

            if let progress {
                VStack(spacing: 8) {
                    ProgressView(value: progress) {
                        Text("Copying… \(Int(progress * 100))%")
                    }
                    .frame(maxWidth: 420)
                    Button("Cancel", role: .cancel) { copyTask?.cancel() }
                }
            } else {
                Toggle("Use the folder where it is (don't copy 12 GB)", isOn: $settings.keepModelInPlace)
                    .frame(maxWidth: 420)
                Button("Choose Folder…") { picking = true }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
            }
            if let manifest {
                Label("\(manifest.shards.count) weight files, \(ByteCountFormatter.string(fromByteCount: manifest.totalBytes, countStyle: .file))",
                      systemImage: "checkmark.seal")
                    .foregroundStyle(.green)
            }
            if let error {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.leading)
                    .frame(maxWidth: 520)
            }
            Button("Skip for now (try the app with an echo model)") { app.skippedImport = true }
                .buttonStyle(.borderless)
                .padding(.top, 24)
        }
        .padding(40)
        .fileImporter(isPresented: $picking, allowedContentTypes: [.folder]) { result in
            switch result {
            case .success(let url): handle(url)
            case .failure(let e): error = e.localizedDescription
            }
        }
    }

    private func handle(_ url: URL) {
        error = nil
        manifest = nil
        let scoped = url.startAccessingSecurityScopedResource()
        do {
            manifest = try ModelFolder.validate(url)
        } catch let e as ModelFolderError {
            error = e.userMessage
            if scoped { url.stopAccessingSecurityScopedResource() }
            return
        } catch let e {
            error = e.localizedDescription
            if scoped { url.stopAccessingSecurityScopedResource() }
            return
        }
        if settings.keepModelInPlace {
            do {
                try app.location.useInPlace(url)
                app.refreshModelFolder()
            } catch let e { error = e.localizedDescription }
            return
        }
        progress = 0
        copyTask = Task {
            do {
                try await app.location.copyIn(url) { p in Task { @MainActor in progress = p } }
                app.refreshModelFolder()
            } catch is CancellationError {
                error = "Copy cancelled."
            } catch let e {
                error = "Copy failed: \(e.localizedDescription)"
            }
            if scoped { url.stopAccessingSecurityScopedResource() }
            progress = nil
            copyTask = nil
        }
    }
}
