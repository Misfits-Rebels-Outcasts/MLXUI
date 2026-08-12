import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// `ModelUI` for SDXL-Turbo text → image. Presents `SDXLTurboRunView` (prompt + seed + Run →
/// image preview + Save PNG). Same claim predicate as `SDXLTurboSDK`. SD-UI1.
struct SDXLTurboUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .image, model.source == .mlx else { return .no }
        let haystack = (model.id + " " + model.hfModelId).lowercased()
        guard haystack.contains("sdxl-turbo") || haystack.contains("stable-diffusion-xl-turbo") else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(SDXLTurboRunView(modelDisplayName: model.displayName,
                                 license: model.license,
                                 modelID: model.id))
    }
}

/// The run surface for SDXL-Turbo text → image: a prompt, a seed (for reproducible output), Run →
/// live progress → image preview → Save PNG. No size picker — SDXL-Turbo's 4-step turbo path is
/// fixed at 512×512 (64×64 latent → 512×512 via the VAE ×8 upscale).
struct SDXLTurboRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let modelID: String

    @State private var model = SDXLTurboRunModel()
    @State private var prompt = "A photorealistic cat sitting on a moon-lit rooftop."
    @State private var seedText = "42"

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            promptField
            options
            if model.isRunning { progressRow }
            if let image = model.image { resultView(image) }
            if let error = model.errorText { errorBar(error) }
            Spacer(minLength: 0)
            footer
        }
        .padding(18)
        .frame(width: 620, height: 720)
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "photo.artframe").foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Text → image (4-step DDIM)").font(.caption).foregroundStyle(.secondary)
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

    private var options: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Seed").frame(width: 60, alignment: .leading)
                TextField("Seed (blank = random)", text: $seedText)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 160)
                    .disabled(model.isRunning)
                Spacer()
            }
            Text(license ?? "License: see model page")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var progressRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                ProgressView(value: model.progress)
                    .progressViewStyle(.linear)
                Text("\(Int(model.progress * 100))%")
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Text(model.progress < 0.5 ? "Loading weights…" : "Generating (4-step DDIM denoise)…")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private func resultView(_ image: CGImage) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(nsImage: NSImage(cgImage: image, size: .zero))
                .resizable()
                .scaledToFit()
                .frame(maxHeight: 320)
                .background(.black.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
            HStack(spacing: 12) {
                Label("Ready", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green).font(.subheadline.bold())
                Button { save(image) } label: {
                    Label("Save PNG…", systemImage: "square.and.arrow.down")
                }
                Spacer()
            }
        }
    }

    private func errorBar(_ message: String) -> some View {
        RunErrorBar(message: message)
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

    // MARK: - Actions

    private func run() {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let seed: UInt64? = {
            let t = seedText.trimmingCharacters(in: .whitespacesAndNewlines)
            return t.isEmpty ? nil : UInt64(t)
        }()
        let stage = SDXLTurboStage(modelID: modelID, seed: seed)
        model.start(prompt: trimmed, stage: stage)
    }

    private func save(_ image: CGImage) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "sdxl-turbo.png"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let rep = NSBitmapImageRep(cgImage: image)
        guard let data = rep.representation(using: .png, properties: [:]) else {
            model.errorText = "Could not encode the image as PNG."
            return
        }
        do {
            try data.write(to: url)
        } catch {
            model.errorText = "Save failed: \(error.localizedDescription)"
        }
    }
}

/// View-model driving one SDXL-Turbo run: runs the stage off-actor and folds progress + the image
/// back onto the main actor. Mirrors `FluxRunModel`.
@MainActor
@Observable
final class SDXLTurboRunModel {
    var isRunning = false
    var progress = 0.0
    var image: CGImage?
    var errorText: String?

    func start(prompt: String, stage: any PipelineStage) {
        isRunning = true
        progress = 0.0
        image = nil
        errorText = nil

        Task {
            do {
                let result = try await stage.run(.text(prompt)) { fraction in
                    Task { @MainActor in self.progress = fraction }
                }
                if case let .image(media) = result { self.image = media.cgImage }
                self.isRunning = false
                self.progress = 1.0
            } catch {
                self.errorText = "\(error)"
                self.isRunning = false
            }
        }
    }
}
