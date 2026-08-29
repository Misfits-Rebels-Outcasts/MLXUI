import AppKit
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Display toggle

enum ImageDisplay: String, CaseIterable {
    case original = "Original"
    case upscaled = "Upscaled"
}

// MARK: - Run model

@MainActor
@Observable
final class SeedVR2RunModel {
    var sourceImage: CGImage?
    var upscaledImage: CGImage?
    var display: ImageDisplay = .original
    var scale: Int = 2
    var isRunning = false
    var progress = 0.0
    var progressLabel = ""
    var errorText: String?

    func setImage(_ cg: CGImage) {
        sourceImage = cg
        upscaledImage = nil
        display = .original
        errorText = nil
    }

    func upscale(modelID: String) {
        guard let source = sourceImage else { return }
        isRunning = true
        progress = 0.0
        progressLabel = "Preparing image…"
        errorText = nil
        upscaledImage = nil

        let chosenScale = scale

        Task {
            do {
                let result = try await SeedVR2Engine.upscale(
                    image: source,
                    scale: chosenScale,
                    seed: nil,
                    modelID: modelID,
                    progress: { [weak self] p in
                        Task { @MainActor [weak self] in
                            guard let self else { return }
                            self.progress = p
                            self.progressLabel = Self.label(for: p)
                        }
                    }
                )
                upscaledImage = result
                display = .upscaled
                progress = 1.0
                progressLabel = "Done"
                isRunning = false
            } catch {
                errorText = error.localizedDescription
                isRunning = false
            }
        }
    }

    private static func label(for p: Double) -> String {
        switch p {
        case ..<0.10: return "Preparing image…"
        case ..<0.32: return "VAE encoding…"
        case ..<0.82: return "Denoising (int8 transformer)…"
        case ..<1.00: return "VAE decoding…"
        default:      return "Done"
        }
    }
}

// MARK: - Run view

/// Bespoke Run surface for SeedVR2 3B image super-resolution (SV-AM5).
/// Workflow: load image → pick scale → Upscale → before/after toggle → Save PNG.
struct SeedVR2RunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let modelID: String

    @State private var model = SeedVR2RunModel()
    @State private var showImporter = false
    @State private var isTargeted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            imageCanvas
            controls
            if model.isRunning { progressRow }
            if let error = model.errorText { RunErrorBar(message: error) }
            Spacer(minLength: 0)
            footer
        }
        .padding(18)
        .frame(width: 660, height: 780)
        .fileImporter(
            isPresented: $showImporter,
            allowedContentTypes: [.image],
            allowsMultipleSelection: false
        ) { result in
            if case let .success(urls) = result, let url = urls.first {
                loadImage(url)
            }
        }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "sparkle.magnifyingglass")
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Image → super-resolution").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var imageCanvas: some View {
        VStack(alignment: .leading, spacing: 6) {
            if model.upscaledImage != nil {
                Picker("", selection: $model.display) {
                    ForEach(ImageDisplay.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(maxWidth: 200)
            } else {
                Text("Image").font(.subheadline.bold())
            }

            ZStack {
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(
                        isTargeted ? Color.accentColor : .secondary.opacity(0.4),
                        style: StrokeStyle(lineWidth: 1.5, dash: [6])
                    )
                    .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 10))

                if let shown = displayedImage {
                    Image(nsImage: NSImage(cgImage: shown, size: .zero))
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "square.and.arrow.down").font(.title2)
                        Text("Drop an image here, or choose one")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .frame(height: 360)
            .onDrop(of: [.image], isTargeted: $isTargeted) { loadDropped($0) }
        }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                Button("Choose Image…") { showImporter = true }
                    .disabled(model.isRunning)

                Spacer()

                Text("Scale")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Picker("", selection: $model.scale) {
                    Text("2×").tag(2)
                    Text("4×").tag(4)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 80)
                .disabled(model.isRunning)
            }

            if let result = model.upscaledImage {
                HStack(spacing: 12) {
                    Label("Upscaled \(model.scale)×", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green).font(.subheadline.bold())
                    let outW = result.width, outH = result.height
                    Text("\(outW) × \(outH)")
                        .font(.caption).foregroundStyle(.secondary)
                    Button {
                        savePNG(result)
                    } label: {
                        Label("Save PNG…", systemImage: "square.and.arrow.down")
                    }
                }
            }
        }
    }

    private var progressRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                ProgressView(value: model.progress)
                    .progressViewStyle(.linear)
                Text("\(Int(model.progress * 100))%")
                    .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
            }
            Text(model.progressLabel)
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            Button(model.isRunning ? "Upscaling…" : "Upscale") {
                model.upscale(modelID: modelID)
            }
            .keyboardShortcut(.defaultAction)
            .disabled(model.sourceImage == nil || model.isRunning)
        }
    }

    // MARK: - Helpers

    private var displayedImage: CGImage? {
        switch model.display {
        case .original: return model.sourceImage
        case .upscaled: return model.upscaledImage ?? model.sourceImage
        }
    }

    private func loadImage(_ url: URL) {
        Task {
            let cg = await Task.detached(priority: .userInitiated) {
                ImageLoader.decodedCGImage(fromSecurityScoped: url)
            }.value
            if let cg {
                model.setImage(cg)
            } else {
                model.errorText = "Could not read image."
            }
        }
    }

    private func loadDropped(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
            guard let data, let cg = ImageLoader.decodedCGImage(from: data) else { return }
            Task { @MainActor in model.setImage(cg) }
        }
        return true
    }

    private func savePNG(_ image: CGImage) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "seedvr2-upscaled.png"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let rep = NSBitmapImageRep(cgImage: image)
        guard let data = rep.representation(using: .png, properties: [:]) else {
            model.errorText = "Could not encode image as PNG."
            return
        }
        do {
            try data.write(to: url)
        } catch {
            model.errorText = "Save failed: \(error.localizedDescription)"
        }
    }
}
