import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// Minimal run UI for WAN 2.1 T2V. Accepts a text prompt, runs the engine,
/// shows denoising progress, and previews the first decoded frame.
/// Full video playback + MP4 export is WAN-AM5.
struct WanVideoRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let modelID: String

    @State private var model = WanRunModel()
    @State private var prompt = "A cat walks on the grass, realistic style."

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            promptField
            if model.isRunning { progressRow }
            if let image = model.image { resultView(image) }
            if let err   = model.errorText { RunErrorBar(message: err) }
            Spacer(minLength: 0)
            footer
        }
        .padding(18)
        .frame(width: 620, height: 680)
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "film.stack").foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Text → video (first frame preview)").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var promptField: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Prompt").font(.subheadline.bold())
            TextEditor(text: $prompt)
                .font(.body)
                .frame(minHeight: 90)
                .padding(6)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private var progressRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                ProgressView(value: model.progress).progressViewStyle(.linear)
                Text("\(Int(model.progress * 100))%")
                    .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
            }
            Text(progressLabel).font(.caption).foregroundStyle(.secondary)
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private var progressLabel: String {
        switch model.progress {
        case 0 ..< 0.20: return "Encoding text…"
        case 0.20 ..< 0.40: return "Loading DiT weights…"
        case 0.40 ..< 0.82: return "Denoising (\(WanVideoEngine.numSteps)-step Euler)…"
        case 0.82 ..< 0.95: return "Loading VAE…"
        default:             return "Decoding frames…"
        }
    }

    private func resultView(_ image: CGImage) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(nsImage: NSImage(cgImage: image, size: .zero))
                .resizable().scaledToFit()
                .frame(maxHeight: 280)
                .background(.black.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
            HStack(spacing: 12) {
                Label("First frame ready", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green).font(.subheadline.bold())
                Button { savePNG(image) } label: { Label("Save PNG…", systemImage: "square.and.arrow.down") }
                Spacer()
            }
        }
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            Button(model.isRunning ? "Generating…" : "Run") { run() }
                .keyboardShortcut(.defaultAction)
                .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || model.isRunning)
        }
    }

    private func run() {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let stage = WanVideoStage(modelID: modelID)
        model.start(prompt: trimmed, stage: stage)
    }

    private func savePNG(_ image: CGImage) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "wan-frame.png"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let rep  = NSBitmapImageRep(cgImage: image)
        guard let data = rep.representation(using: .png, properties: [:]) else { return }
        try? data.write(to: url)
    }
}

// MARK: - View model

@MainActor @Observable
final class WanRunModel {
    var isRunning = false
    var progress  = 0.0
    var image:     CGImage?
    var errorText: String?

    func start(prompt: String, stage: any PipelineStage) {
        isRunning = true; progress = 0; image = nil; errorText = nil
        Task {
            do {
                let result = try await stage.run(.text(prompt)) { fraction in
                    Task { @MainActor in self.progress = fraction }
                }
                if case let .image(m) = result { self.image = m.cgImage }
                isRunning = false; progress = 1.0
            } catch {
                errorText = "\(error)"; isRunning = false
            }
        }
    }
}
