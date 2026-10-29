import SwiftUI
import AppKit
import AVKit
import UniformTypeIdentifiers

/// The run surface: choose or drop a video, pick 2× or 4× the frame rate (and Scale 1.0 or 0.5 for 4K), Interpolate, preview the
/// result, save the MP4. The original audio is kept. Image-pair mode is not in this release (owner ruling 2026-10-28).
struct RIFERunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let modelID: String
    /// SC-5c: the stage the Run sheet handed (a `LeasingStage`); every run goes through it.
    let handedStage: any PipelineStage

    @State private var model = RIFERunModel()
    @State private var showImporter = false
    @State private var isDropTargeted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            inputs
            if model.isRunning { runningRow }
            if let error = model.errorText { RunErrorBar(message: error) }
            if model.hasResult { resultView }
            Spacer(minLength: 0)
            footer
        }
        .padding(18)
        .frame(width: 600, height: 640)
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.movie], allowsMultipleSelection: false) { result in
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
            Image(systemName: ModelType.interpolate.sfSymbol).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Smoother or slow-motion video: adds frames between the ones you have")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var inputs: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button("Choose Video…") { showImporter = true }.disabled(model.isRunning)
                Text(model.sourceName ?? "No video selected (or drop one here)")
                    .font(.callout).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            }
            if let info = model.info {
                Text(model.sourceSummary(info)).font(.caption).foregroundStyle(.secondary)
            }
            if let limit = model.limitMessage { RunErrorBar(message: limit) }
            HStack(spacing: 16) {
                Picker("Frame rate", selection: $model.times) {
                    Text("2×").tag(2)
                    Text("4×").tag(4)
                }
                .pickerStyle(.segmented).frame(maxWidth: 180)
                Picker("Scale", selection: $model.scale) {
                    Text("1.0").tag(Float(1.0))
                    Text("0.5").tag(Float(0.5))
                }
                .pickerStyle(.segmented).frame(maxWidth: 180)
            }
            .disabled(model.isRunning)
            if let info = model.info {
                Text("Output: \(Self.fps(info.outputFrameRate(times: model.times))) fps, about \(info.outputFrames(times: model.times)) frames, the same length as the original. The audio is copied unchanged.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text("Scale 0.5 estimates the motion at half size: use it for 4K video to save memory; 1.0 is more exact on smaller video.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Text("Handles clips up to \(RIFEVideoOptions.maxSourceFrames) frames (about 60 seconds at 30 fps) and 3840 × 2160. It adds frames; it doesn't make the picture sharper or larger.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Text(license ?? "License: see model page").font(.caption).foregroundStyle(.secondary)
        }
    }

    private var runningRow: some View {
        HStack(spacing: 8) {
            ProgressView(value: model.progress).frame(maxWidth: 260)
            Text("Interpolating… \(Int((model.progress * 100).rounded()))%").font(.callout).foregroundStyle(.secondary).monospacedDigit()
            Spacer()
            Button("Cancel") { model.cancel() }
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private var resultView: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let summary = model.resultSummary {
                Label(summary, systemImage: "checkmark.circle.fill").foregroundStyle(.green).font(.subheadline.bold())
            }
            if let player = model.player {
                VideoPlayer(player: player).frame(height: 240).clipShape(RoundedRectangle(cornerRadius: 8))
            }
            HStack {
                Spacer()
                Button("Save Video…") { model.save() }
            }
        }
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            Button(model.isRunning ? "Working…" : "Interpolate") { run() }
                .keyboardShortcut(.defaultAction)
                .disabled(!model.canRun)
        }
    }

    private static func fps(_ value: Double) -> String { String(format: abs(value.rounded() - value) < 0.005 ? "%.0f" : "%.2f", value) }

    // MARK: - Actions

    private func run() {
        let modelID = self.modelID
        let options = model.options
        let output = RIFEStage.temporaryOutputURL()
        let stage = LeasingStage.perRun(handedStage, config: .default) { _ in
            RIFEStage(modelID: modelID, options: options, outputURL: output)
        }
        model.start(stage: stage, output: output)
    }
}

/// View-model for one interpolation run: inspects the chosen video (readable refusals before any model loads), runs the stage off the
/// main actor, previews and saves the result. No Combine.
@MainActor
@Observable
final class RIFERunModel {
    var times = 2
    var scale: Float = 1.0
    var info: RIFEVideoInfo?
    var limitMessage: String?
    var isRunning = false
    var progress = 0.0
    var errorText: String?
    private(set) var sourceURL: URL?
    private(set) var outputURL: URL?
    private(set) var player: AVPlayer?
    private(set) var resultSummary: String?

    private var task: Task<Void, Never>?
    private var inspecting: Task<Void, Never>?
    private var accessing = false

    var options: RIFEVideoOptions { RIFEVideoOptions(times: times, scale: scale) }
    var hasResult: Bool { outputURL != nil }
    var sourceName: String? { sourceURL?.lastPathComponent }
    var canRun: Bool { sourceURL != nil && info != nil && limitMessage == nil && !isRunning }

    func sourceSummary(_ info: RIFEVideoInfo) -> String {
        let fps = abs(info.frameRate.rounded() - info.frameRate) < 0.005 ? String(format: "%.0f", info.frameRate) : String(format: "%.2f", info.frameRate)
        return "\(info.width) × \(info.height) · \(fps) fps · \(String(format: "%.1f", info.duration)) s" + (info.hasAudio ? " · audio" : "")
    }

    func reset() {
        task?.cancel()
        inspecting?.cancel()
        player?.pause()
        player = nil
        if let outputURL { try? FileManager.default.removeItem(at: outputURL) }
        outputURL = nil
        resultSummary = nil
        errorText = nil
        limitMessage = nil
        progress = 0
        isRunning = false
        info = nil
    }

    func teardown() {
        reset()
        releaseSource()
    }

    private func releaseSource() {
        if accessing { sourceURL?.stopAccessingSecurityScopedResource() }
        accessing = false
    }

    func choose(_ url: URL) {
        reset()
        releaseSource()
        sourceURL = url
        accessing = url.startAccessingSecurityScopedResource()
        inspecting = Task {
            do {
                let found = try await RIFEVideoPipeline.inspect(url)
                guard !Task.isCancelled else { return }
                info = found
                do { try RIFEVideoPipeline.validate(found) } catch { limitMessage = MLXErrorGuard.sentence(for: error) }
            } catch {
                guard !Task.isCancelled else { return }
                limitMessage = MLXErrorGuard.sentence(for: error)
            }
        }
    }

    func cancel() { task?.cancel() }

    func start(stage: any PipelineStage, output: URL) {
        guard let sourceURL, let info else { return }
        errorText = nil
        resultSummary = nil
        player?.pause()
        player = nil
        outputURL = nil
        progress = 0
        isRunning = true
        let times = self.times
        task = Task {
            do {
                // MLX-ERR-2 (see MLXErrorGuard.guardedRun): an MLX error becomes a thrown sentence, and a run in which the
                // process-wide fallback fired is never reported as successful.
                let produced = try await MLXErrorGuard.guardedRun {
                    try await Self.runOffMain(stage, .text(sourceURL.path)) { fraction in
                        Task { @MainActor in self.progress = fraction }
                    }
                }
                guard case let .text(path) = produced else { throw StageError.kindMismatch(expected: .text, got: produced.kind) }
                let url = URL(fileURLWithPath: path)
                outputURL = url
                player = AVPlayer(url: url)
                resultSummary = "Done: \(info.outputFrames(times: times)) frames at \(String(format: "%.0f", info.outputFrameRate(times: times))) fps"
                progress = 1
            } catch {
                // MP5-3g (owner ruling 2026-10-26): a cancelled run is never an error. The task's own flag decides, whatever was
                // thrown (a cancelled memory wait throws `MemoryBudget.WaitError`, not `CancellationError`). The pipeline has
                // already removed its partial file.
                if error is CancellationError || Task.isCancelled {
                    progress = 0
                } else {
                    errorText = MLXErrorGuard.sentence(for: error)
                }
                try? FileManager.default.removeItem(at: output)
            }
            isRunning = false
        }
    }

    /// Runs `stage` off the main actor (MP5-3g): `SWIFT_APPROACHABLE_CONCURRENCY` makes a plain `nonisolated async` call run on its
    /// caller's actor, and this model's `Task` is main-actor. `@concurrent` is a child hop, so the `MLXErrorGuard` task-local and
    /// cancellation carry across.
    @concurrent nonisolated private static func runOffMain(_ stage: any PipelineStage, _ input: Media,
                                                          progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try await stage.run(input, progress: progress)
    }

    func save() {
        guard let outputURL else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.mpeg4Movie]
        panel.nameFieldStringValue = Self.savedFileName(for: sourceURL, times: times)
        guard panel.runModal() == .OK, let target = panel.url else { return }
        do {
            if FileManager.default.fileExists(atPath: target.path) { try FileManager.default.removeItem(at: target) }
            try FileManager.default.copyItem(at: outputURL, to: target)
        } catch {
            errorText = "Save failed: \(error.localizedDescription)"
        }
    }

    nonisolated static func savedFileName(for url: URL?, times: Int) -> String {
        let base = url?.deletingPathExtension().lastPathComponent ?? ""
        return (base.isEmpty ? "video" : base) + "-\(times)x.mp4"
    }
}
