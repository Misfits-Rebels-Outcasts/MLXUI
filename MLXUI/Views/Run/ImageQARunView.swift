import SwiftUI
import AppKit
import ImageIO
import UniformTypeIdentifiers

/// Reusable run surface for any image+prompt→text (VLM) model. The model module supplies
/// the model id; this view drops/picks an image, takes a prompt, builds a fresh `VLMStage`
/// per run, and shows the answer. Mirrors `ASRRunView` / `TTSRunView` (`plan-nonchat-aiui.md`
/// VL2). No Apple Vision — the chosen `CGImage` flows straight into the MLX VLM.
struct ImageQARunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let modelID: String

    @State private var model = ImageQARunModel()
    @State private var image: CGImage?
    @State private var imageName: String?
    @State private var prompt = "Describe this image in detail."
    @State private var isTargeted = false
    @State private var showImporter = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            imageWell
            promptField
            Text(license ?? "License: see model page")
                .font(.caption).foregroundStyle(.secondary)
            if model.isRunning { runningRow }
            if let answer = model.answer { resultView(answer) }
            if let error = model.errorText { errorBar(error) }
            Spacer(minLength: 0)
            footer
        }
        .padding(18)
        .frame(width: 560, height: 720)
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.image],
                      allowsMultipleSelection: false) { result in
            if case let .success(urls) = result, let url = urls.first { load(url) }
        }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "photo")
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Image + prompt → text").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var imageWell: some View {
        VStack(alignment: .leading, spacing: 6) {
            ZStack {
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(isTargeted ? Color.accentColor : .secondary.opacity(0.4),
                                  style: StrokeStyle(lineWidth: 1.5, dash: [6]))
                    .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 10))
                if let image {
                    Image(nsImage: NSImage(cgImage: image, size: .zero))
                        .resizable().scaledToFit()
                        .padding(6)
                } else {
                    VStack(spacing: 6) {
                        Image(systemName: "square.and.arrow.down").font(.title2)
                        Text("Drop an image here, or choose one").foregroundStyle(.secondary)
                    }
                }
            }
            .frame(height: 220)
            .onDrop(of: [.image], isTargeted: $isTargeted) { providers in loadDropped(providers) }
            HStack {
                Button("Choose Image…") { showImporter = true }
                Text(imageName ?? "No image selected")
                    .font(.callout).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            }
        }
    }

    private var promptField: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Prompt").font(.subheadline.bold())
            TextEditor(text: $prompt)
                .font(.body)
                .frame(minHeight: 70)
                .padding(6)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private var runningRow: some View {
        HStack(spacing: 8) {
            ProgressView().scaleEffect(0.6).frame(width: 16, height: 16)
            Text("Thinking…").font(.callout).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private func resultView(_ answer: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Label("Answer", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green).font(.subheadline.bold())
                Spacer()
                Button("Copy") { copy(answer) }
            }
            ScrollView {
                Text(answer).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 180)
            .padding(8)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private func errorBar(_ message: String) -> some View {
        RunErrorBar(message: message)
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            Button(model.isRunning ? "Running…" : "Run") { run() }
                .keyboardShortcut(.defaultAction)
                .disabled(image == nil || model.isRunning)
        }
    }

    // MARK: - Actions

    private func run() {
        guard let image else { return }
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let stage = VLMStage(modelID: modelID,
                             prompt: trimmed.isEmpty ? VLMSDK.defaultPrompt : trimmed)
        model.start(image: image, stage: stage)
    }

    /// Load a picked file URL into a fully-decoded, memory-resident `CGImage` (sandbox-safe:
    /// the scope is only open here, so the image must not stay file-mapped — see `ImageLoader`).
    private func load(_ url: URL) {
        guard let cg = ImageLoader.decodedCGImage(fromSecurityScoped: url) else {
            model.errorText = "Could not read image."
            return
        }
        image = cg
        imageName = url.lastPathComponent
    }

    /// Load dropped image data directly.
    private func loadDropped(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
            guard let data, let cg = ImageLoader.decodedCGImage(from: data) else { return }
            Task { @MainActor in
                self.image = cg
                self.imageName = "Dropped image"
            }
        }
        return true
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

/// View-model driving one VLM run: runs the stage off-actor and folds the answer back onto
/// the main actor. Mirrors `ASRRunModel`.
@MainActor
@Observable
final class ImageQARunModel {
    var isRunning = false
    var answer: String?
    var errorText: String?

    func start(image: CGImage, stage: any PipelineStage) {
        isRunning = true
        answer = nil
        errorText = nil

        Task {
            do {
                let result = try await stage.run(.image(ImageMedia(cgImage: image))) { _ in }
                if case let .text(text) = result { self.answer = text }
                self.isRunning = false
            } catch {
                self.errorText = "\(error)"
                self.isRunning = false
            }
        }
    }
}
