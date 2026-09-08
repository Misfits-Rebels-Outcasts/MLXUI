import SwiftUI
import AppKit
import ImageIO
import UniformTypeIdentifiers

/// Reusable run surface for any image→text (OCR) stage. The model module supplies the
/// concrete `stage`; this view drops/picks an image, runs the stage, and shows the
/// extracted text (monospaced, copyable). Most OCR models take a pre-built stage (their
/// prompt is frozen). When `promptSupport` is `.modes` (OCP-0: PaddleOCR-VL), a mode
/// picker appears and `rebuildStage` produces a fresh stage for the chosen mode per run,
/// the way `ImageQARunView.run()` rebuilds `VLMStage`. `plan-nonchat-aiui.md` OC2.
struct OCRRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let stage: any PipelineStage
    /// How this model's stage varies with a per-run prompt. `.none` ⇒ this view is
    /// byte-for-byte its pre-OCP-0 self and `stage` runs unchanged.
    var promptSupport: PromptSupport = .none
    /// Rebuilds the stage for a chosen mode string; supplied only when `promptSupport`
    /// is not `.none`. `nil`, or a `nil` return, falls back to `stage`.
    var rebuildStage: ((String) -> (any PipelineStage)?)? = nil

    @State private var model = OCRRunModel()
    @State private var image: CGImage?
    @State private var imageName: String?
    @State private var isTargeted = false
    @State private var showImporter = false
    /// Selected recognition mode for a `.modes` model; seeded from the default on appear.
    @State private var mode = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            imageWell
            modePicker
            Text(license ?? "License: see model page")
                .font(.caption).foregroundStyle(.secondary)
            if model.isRunning { runningRow }
            if let text = model.text { resultView(text) }
            if let error = model.errorText { errorBar(error) }
            Spacer(minLength: 0)
            footer
        }
        .padding(18)
        .frame(width: 560, height: 700)
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.image],
                      allowsMultipleSelection: false) { result in
            if case let .success(urls) = result, let url = urls.first { load(url) }
        }
        .onAppear {
            if case let .modes(_, defaultValue) = promptSupport, mode.isEmpty { mode = defaultValue }
        }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "doc.text.viewfinder")
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Image → extracted text").font(.caption).foregroundStyle(.secondary)
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
                        Text("Drop a document image here, or choose one").foregroundStyle(.secondary)
                    }
                }
            }
            .frame(height: 240)
            .onDrop(of: [.image], isTargeted: $isTargeted) { providers in loadDropped(providers) }
            HStack {
                Button("Choose Image…") { showImporter = true }
                Button("Use Sample Image") { loadSample() }
                Text(imageName ?? "No image selected")
                    .font(.callout).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            }
        }
    }

    /// Recognition-mode picker — shown only for a `.modes` model (OCP-0: PaddleOCR-VL).
    /// `.none` models render nothing here, keeping their surface unchanged.
    @ViewBuilder private var modePicker: some View {
        if case let .modes(values, _) = promptSupport {
            VStack(alignment: .leading, spacing: 6) {
                Text("Mode").font(.subheadline.bold())
                Picker("Mode", selection: $mode) {
                    ForEach(values, id: \.self) { Text(Self.modeLabel($0)).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .disabled(model.isRunning)
            }
        }
    }

    /// Friendly label for a PaddleOCR-VL mode id.
    private static func modeLabel(_ id: String) -> String {
        switch id {
        case "ocr": return "Text"
        case "table": return "Table"
        case "formula": return "Formula"
        case "chart": return "Chart"
        default: return id.capitalized
        }
    }

    private var runningRow: some View {
        HStack(spacing: 8) {
            ProgressView().scaleEffect(0.6).frame(width: 16, height: 16)
            Text("Extracting text…").font(.callout).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private func resultView(_ text: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Label("Extracted text", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green).font(.subheadline.bold())
                Spacer()
                Button("Copy") { copy(text) }
            }
            ScrollView {
                Text(text).textSelection(.enabled).font(.body.monospaced())
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 200)
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
            Button(model.isRunning ? "Extracting…" : "Extract Text") { run() }
                .keyboardShortcut(.defaultAction)
                .disabled(image == nil || model.isRunning)
        }
    }

    // MARK: - Actions

    private func run() {
        guard let image else { return }
        model.start(image: image, stage: effectiveStage)
    }

    /// The pre-built `stage` for `.none` models and for a `.modes` model still on its default
    /// mode (so that surface stays byte-for-byte its pre-OCP-0 self); otherwise a fresh stage
    /// for the selected mode, falling back to `stage` if the rebuild returns nil.
    private var effectiveStage: any PipelineStage {
        guard case let .modes(_, defaultValue) = promptSupport,
              let rebuildStage, !mode.isEmpty, mode != defaultValue else { return stage }
        return rebuildStage(mode) ?? stage
    }

    /// Load a picked file URL into a fully-decoded, memory-resident `CGImage` (sandbox-safe:
    /// the scope is only open here, so the image must not stay file-mapped — see `ImageLoader`).
    private func load(_ url: URL) {
        Task {
            let cg = await Task.detached(priority: .userInitiated) {
                ImageLoader.decodedCGImage(fromSecurityScoped: url)
            }.value
            if let cg {
                image = cg
                imageName = url.lastPathComponent
            } else {
                model.errorText = "Could not read image."
            }
        }
    }

    /// A document image bundled with the app (`Resources/Samples/PersonalBudget.png`) for
    /// repeated OCR testing — works in both editions since it ships inside the app bundle.
    private static let sampleResource = "PersonalBudget"

    /// Load the bundled sample image (bytes read + forced decode, like the drop path).
    private func loadSample() {
        guard let url = Bundle.main.url(forResource: Self.sampleResource, withExtension: "png") else {
            model.errorText = "Sample image \(Self.sampleResource).png is missing from the app bundle."
            return
        }
        Task {
            let cg = await Task.detached(priority: .userInitiated) { () -> CGImage? in
                guard let data = try? Data(contentsOf: url) else { return nil }
                return ImageLoader.decodedCGImage(from: data)
            }.value
            if let cg {
                image = cg
                imageName = url.lastPathComponent
            } else {
                model.errorText = "Could not read the bundled sample image."
            }
        }
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

/// View-model driving one OCR run: runs the stage off-actor and folds the text back onto
/// the main actor. Mirrors `ASRRunModel`.
@MainActor
@Observable
final class OCRRunModel {
    var isRunning = false
    var text: String?
    var errorText: String?

    func start(image: CGImage, stage: any PipelineStage) {
        isRunning = true
        text = nil
        errorText = nil

        Task {
            do {
                let result = try await stage.run(.image(ImageMedia(cgImage: image))) { _ in }
                if case let .text(extracted) = result { self.text = extracted }
                self.isRunning = false
            } catch {
                self.errorText = "\(error)"
                self.isRunning = false
            }
        }
    }
}
