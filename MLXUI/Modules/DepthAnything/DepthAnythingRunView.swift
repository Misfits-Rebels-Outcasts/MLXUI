import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// Depth Anything Run screen (owner decisions 5 and 6): pick a photo (Open… or drag and drop), Estimate
/// depth, see the input and the depth map side by side with a Grayscale / Colour preview toggle, Save PNG
/// (always the grayscale map). Near is bright; the depth is relative, not metres. No Stop button — one run
/// takes about a second.
struct DepthAnythingRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let modelID: String
    /// SC-5c: the stage the Run sheet handed (a `LeasingStage`); every run goes through it.
    let handedStage: any PipelineStage

    @State private var model: DepthRunModel
    @State private var showImporter = false
    @State private var isDropTargeted = false

    /// `model` is injectable so a test can host a finished run state without running one.
    init(modelDisplayName: String, license: String?, modelID: String, handedStage: any PipelineStage,
         model: DepthRunModel? = nil) {
        self.modelDisplayName = modelDisplayName
        self.license = license
        self.modelID = modelID
        self.handedStage = handedStage
        _model = State(initialValue: model ?? DepthRunModel())
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            picker
            if let limit = model.limitMessage { RunErrorBar(message: limit) }
            if model.isRunning { runningRow }
            if let error = model.errorText { RunErrorBar(message: error) }
            if model.hasResult { resultView }
            Spacer(minLength: 0)
            Text("Relative depth — brighter is nearer. Not a measurement in metres.")
                .font(.caption).foregroundStyle(.secondary)
            footer
        }
        .padding(18)
        .frame(width: 760, height: 680)
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.image], allowsMultipleSelection: false) { result in
            if case let .success(urls) = result, let url = urls.first { model.choose(url) }
        }
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first else { return false }
            model.choose(url)
            return true
        } isTargeted: { isDropTargeted = $0 }
        .overlay {
            if isDropTargeted {
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 2, dash: [6]))
                    .padding(6)
                    .allowsHitTesting(false)
            }
        }
        .onDisappear { model.teardown() }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: ModelType.depth.sfSymbol).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("How near or far every pixel is: one photo in, a grayscale depth map out")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var picker: some View {
        HStack(spacing: 8) {
            Button("Choose Photo…") { showImporter = true }.disabled(model.isRunning)
            Text(model.sourceName ?? "No photo selected (or drop one here)")
                .font(.callout).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            Spacer()
            if let summary = model.inputSummary {
                Text(summary).font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
        }
    }

    private var runningRow: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            Text("Estimating depth…").font(.callout).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private var resultView: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 12) {
                previewPanel("Photo", image: model.inputImage)
                previewPanel(model.previewMode == .grayscale ? "Depth (grayscale)" : "Depth (colour preview)",
                             image: model.previewImage)
            }
            HStack(spacing: 16) {
                Picker("Preview", selection: $model.previewMode) {
                    ForEach(DepthPreviewMode.allCases, id: \.self) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 220)
                Spacer()
                Button("Save PNG…") { model.save() }
            }
        }
    }

    private func previewPanel(_ title: String, image: CGImage?) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            ZStack {
                RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.35))
                if let image {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .interpolation(.medium)
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
            }
            .frame(maxWidth: .infinity, minHeight: 260, maxHeight: 330)
        }
        .frame(maxWidth: .infinity)
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            Button(model.isRunning ? "Working…" : "Estimate depth") { run() }
                .keyboardShortcut(.defaultAction)
                .disabled(!model.canRun)
        }
    }

    // MARK: - Actions

    private func run() {
        let modelID = self.modelID
        let stage = LeasingStage.perRun(handedStage, config: .default) { _ in
            DepthAnythingStage(modelID: modelID)
        }
        model.start(stage: stage)
    }
}

/// Display-only preview choice. `Save PNG` always writes the grayscale map (owner decision 6).
enum DepthPreviewMode: String, CaseIterable {
    case grayscale = "Grayscale"
    case colour = "Colour"
}

/// View-model for one depth run: decodes and checks the photo before any model work, runs the stage off the
/// main actor, keeps the grayscale map (and a display-only colour copy), saves the grayscale PNG. No Combine.
@MainActor
@Observable
final class DepthRunModel {
    var previewMode: DepthPreviewMode = .grayscale
    private(set) var sourceURL: URL?
    /// A label for a result hosted by the evidence harness (no file was chosen).
    private(set) var hostedSourceLabel: String?
    private(set) var sourceImage: CGImage?
    private(set) var limitMessage: String?
    private(set) var errorText: String?
    private(set) var grayImage: CGImage?
    private(set) var colourImage: CGImage?
    private(set) var pngData: Data?
    private(set) var isRunning = false

    private var task: Task<Void, Never>?
    private var inspecting: Task<Void, Never>?
    private var accessing = false

    var hasResult: Bool { grayImage != nil }
    var inputImage: CGImage? { sourceImage }
    var previewImage: CGImage? { previewMode == .grayscale ? grayImage : colourImage }
    var sourceName: String? { sourceURL?.lastPathComponent ?? hostedSourceLabel }
    var inputSummary: String? { sourceImage.map { "\($0.width) × \($0.height)" } }
    var canRun: Bool { sourceImage != nil && limitMessage == nil && !isRunning }

    func reset() {
        task?.cancel()
        inspecting?.cancel()
        task = nil
        inspecting = nil
        grayImage = nil
        colourImage = nil
        pngData = nil
        errorText = nil
        limitMessage = nil
        isRunning = false
    }

    func teardown() {
        reset()
        releaseSource()
    }

    private func releaseSource() {
        if accessing { sourceURL?.stopAccessingSecurityScopedResource() }
        accessing = false
    }

    /// Decode and check the limits before any model work; a refusal is one plain sentence.
    func choose(_ url: URL) {
        reset()
        releaseSource()
        sourceURL = url
        sourceImage = nil
        accessing = url.startAccessingSecurityScopedResource()
        inspecting = Task {
            do {
                let prepared = try await Self.prepareOffMain(url)
                guard !Task.isCancelled else { return }
                sourceImage = prepared.image
            } catch {
                guard !Task.isCancelled else { return }
                limitMessage = MLXErrorGuard.sentence(for: error)
            }
        }
    }

    func start(stage: any PipelineStage) {
        guard let sourceImage else { return }
        reset()
        isRunning = true
        let input = Media.image(ImageMedia(cgImage: sourceImage))
        task = Task {
            do {
                let produced = try await MLXErrorGuard.guardedRun {
                    try await Self.runOffMain(stage, input)
                }
                guard case let .image(media) = produced else {
                    throw StageError.kindMismatch(expected: .image, got: produced.kind)
                }
                let gray = media.cgImage
                let bytes = DepthAnythingPostprocessor.grayBytes(from: gray) ?? []
                grayImage = gray
                colourImage = try? DepthAnythingPostprocessor.colourImage(gray: bytes,
                                                                         width: gray.width, height: gray.height)
                pngData = PNGEncoder.pngData(from: gray)
            } catch {
                // A cancelled run is never an error (a memory wait throws its own error, not CancellationError).
                if error is CancellationError || Task.isCancelled {
                    errorText = nil
                } else {
                    errorText = MLXErrorGuard.sentence(for: error)
                }
            }
            isRunning = false
        }
    }

    /// Evidence seam: show a finished result without running the model (the screenshot harness).
    func hostResult(source: CGImage, gray: CGImage, colour: CGImage?, png: Data?, sourceLabel: String? = nil) {
        sourceImage = source
        hostedSourceLabel = sourceLabel
        grayImage = gray
        colourImage = colour
        pngData = png
        limitMessage = nil
        errorText = nil
        isRunning = false
    }

    func save() {
        guard let pngData else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = Self.savedFileName(for: sourceURL)
        guard panel.runModal() == .OK, let target = panel.url else { return }
        do {
            try pngData.write(to: target)
        } catch {
            errorText = "Save failed: \(error.localizedDescription)"
        }
    }

    nonisolated static func savedFileName(for url: URL?) -> String {
        let base = url?.deletingPathExtension().lastPathComponent ?? ""
        return (base.isEmpty ? "photo" : base) + "-depth.png"
    }

    /// Decoded off the main actor: a 12 MP photo's decode should not block the window.
    @concurrent nonisolated static func prepareOffMain(_ url: URL) async throws -> DepthAnythingInput.Prepared {
        try DepthAnythingInput.prepare(url: url)
    }

    @concurrent nonisolated static func runOffMain(_ stage: any PipelineStage, _ input: Media) async throws -> Media {
        try await stage.run(input) { _ in }
    }
}
