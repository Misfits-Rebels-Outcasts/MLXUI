import AppKit
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Run Model

/// View-model for one SAM3 segmentation session. Holds the source image, user-placed
/// point prompts, and the masked result. Encodes once per image; re-decodes on each
/// "Segment" press with the current point set.
@MainActor
@Observable
final class SegmentAnythingRunModel {
    var sourceImage: CGImage?
    var resultImage: CGImage?
    var imageSize: CGSize = .zero
    var points: [(CGPoint, Bool)] = []   // (normalized [0,1] coord, isForeground)
    var foregroundMode = true
    var isRunning = false
    var progress = 0.0
    var errorText: String?

    func setImage(_ cg: CGImage) {
        sourceImage = cg
        imageSize = CGSize(width: cg.width, height: cg.height)
        resultImage = nil
        points = []
        errorText = nil
    }

    func addPoint(_ pt: CGPoint) {
        points.append((pt, foregroundMode))
    }

    func clearPoints() {
        points = []
        resultImage = nil
    }

    func segment(modelID: String) {
        guard let source = sourceImage else { return }
        isRunning = true
        progress = 0.0
        errorText = nil
        resultImage = nil

        let pts = points.isEmpty ? [(CGPoint(x: 0.5, y: 0.5), true)] : points
        let floatPts = pts.map { [Float($0.0.x), Float($0.0.y)] }
        let labels = pts.map { $0.1 ? 1 : 0 }

        Task {
            do {
                progress = 0.1
                let embedding = try await SegmentAnythingEngine.encodeImage(source, modelID: modelID)
                progress = 0.7
                let (masks, iouScores) = try await SegmentAnythingEngine.decodeMasks(
                    embedding: embedding,
                    points: floatPts,
                    labels: labels,
                    modelID: modelID)
                progress = 0.9
                let composite = SegmentAnythingEngine.renderTopMask(masks: masks, iouScores: iouScores, sourceImage: source)
                resultImage = composite
                isRunning = false
                progress = 1.0
            } catch {
                errorText = "\(error)"
                isRunning = false
            }
        }
    }
}

// MARK: - Run View

/// Interactive Run surface for SAM3 image segmentation (SA-AM5).
/// Workflow: load image → click foreground/background points → Segment → colored mask overlay → Save PNG.
struct SegmentAnythingRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let modelID: String

    @State private var model = SegmentAnythingRunModel()
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
        .frame(width: 660, height: 800)
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
            Image(systemName: "lasso")
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Image → segmentation mask").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var imageCanvas: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Image").font(.subheadline.bold())
            ZStack {
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(
                        isTargeted ? Color.accentColor : .secondary.opacity(0.4),
                        style: StrokeStyle(lineWidth: 1.5, dash: [6])
                    )
                    .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 10))

                if let displayImage = model.resultImage ?? model.sourceImage {
                    GeometryReader { geo in
                        let imgSize = model.imageSize
                        let viewSize = geo.size
                        ZStack {
                            Image(nsImage: NSImage(cgImage: displayImage, size: .zero))
                                .resizable()
                                .scaledToFit()
                                .frame(maxWidth: .infinity, maxHeight: .infinity)

                            // Point-prompt overlay — dots on top of image or result
                            Canvas { ctx, size in
                                guard imgSize.width > 0, imgSize.height > 0 else { return }
                                let rect = fitRect(imageSize: imgSize, viewSize: size)
                                for (pt, isForeground) in model.points {
                                    let cx = rect.minX + pt.x * rect.width
                                    let cy = rect.minY + pt.y * rect.height
                                    let r: CGFloat = 7
                                    let bounds = CGRect(x: cx - r, y: cy - r, width: r * 2, height: r * 2)
                                    let circle = Path(ellipseIn: bounds)
                                    ctx.fill(circle, with: .color(isForeground ? .green : .red))
                                    ctx.stroke(circle, with: .color(.white), lineWidth: 1.5)
                                }
                            }
                        }
                        .contentShape(Rectangle())
                        .onTapGesture { location in
                            guard model.sourceImage != nil, imgSize.width > 0 else { return }
                            let rect = fitRect(imageSize: imgSize, viewSize: viewSize)
                            guard rect.contains(location) else { return }
                            let normalized = CGPoint(
                                x: (location.x - rect.minX) / rect.width,
                                y: (location.y - rect.minY) / rect.height
                            )
                            model.addPoint(normalized)
                            model.resultImage = nil   // clear stale result on new click
                        }
                    }
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "square.and.arrow.down").font(.title2)
                        Text("Drop an image here, or choose one")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .frame(height: 360)
            .onDrop(of: [.image], isTargeted: $isTargeted) { providers in loadDropped(providers) }
        }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                Button("Choose Image…") { showImporter = true }
                    .disabled(model.isRunning)
                Spacer()
                Text("Tap image to place point prompts")
                    .font(.caption).foregroundStyle(.secondary)
            }

            HStack(spacing: 10) {
                Picker("", selection: $model.foregroundMode) {
                    Text("Foreground").tag(true)
                    Text("Background").tag(false)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(maxWidth: 220)

                Circle()
                    .fill(model.foregroundMode ? Color.green : Color.red)
                    .frame(width: 10, height: 10)
                Text(model.foregroundMode ? "green dots" : "red dots")
                    .font(.caption).foregroundStyle(.secondary)

                Spacer()

                if !model.points.isEmpty {
                    Button("Clear Points") { model.clearPoints() }
                        .disabled(model.isRunning)
                }
            }

            HStack(spacing: 6) {
                if model.points.isEmpty {
                    Text("No points — center-point foreground prompt will be used.")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    let fg = model.points.filter(\.1).count
                    let bg = model.points.count - fg
                    Text("\(model.points.count) point\(model.points.count == 1 ? "" : "s"): \(fg) fg, \(bg) bg")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
            }

            if let result = model.resultImage {
                HStack(spacing: 12) {
                    Label("Mask ready", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green).font(.subheadline.bold())
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
            Text(model.progress < 0.6 ? "Loading weights + encoding image…" : "Decoding masks…")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            Button(model.isRunning ? "Segmenting…" : "Segment") {
                model.segment(modelID: modelID)
            }
            .keyboardShortcut(.defaultAction)
            .disabled(model.sourceImage == nil || model.isRunning)
        }
    }

    // MARK: - Helpers

    /// The rect within a view of `viewSize` where an image of `imageSize` is drawn by scaledToFit.
    private func fitRect(imageSize: CGSize, viewSize: CGSize) -> CGRect {
        let scale = min(viewSize.width / imageSize.width, viewSize.height / imageSize.height)
        let w = imageSize.width * scale
        let h = imageSize.height * scale
        return CGRect(x: (viewSize.width - w) / 2, y: (viewSize.height - h) / 2, width: w, height: h)
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
        panel.nameFieldStringValue = "sam3-mask.png"
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

// MARK: - ModelUI adapter

/// `ModelUI` adapter for the SAM3 segmentation module.
struct SegmentAnythingUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .segmentation else { return .no }
        let haystack = (model.id + " " + model.hfModelId).lowercased()
        guard haystack.contains("sam") else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(SegmentAnythingRunView(
            modelDisplayName: model.displayName,
            modelID: model.id))
    }
}
