import SwiftUI
import AppKit
import AVFoundation
import UniformTypeIdentifiers

/// `ModelUI` for DeepFilterNet audio cleanup. Presents `DeepFilterNetRunView` (pick or drop an audio file, Clean, play the
/// original or the cleaned version, Save cleaned audio as a 48 kHz WAV). Same claim predicate as `DeepFilterNetSDK`.
struct DeepFilterNetUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .enhance,
              model.source == .mlx,
              model.family.lowercased().contains("deepfilternet") else {
            return .no
        }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(DeepFilterNetRunView(modelDisplayName: model.displayName,
                                     license: model.license,
                                     modelID: model.id,
                                     handedStage: stage))
    }
}

/// The run surface: choose or drop an audio file, Clean, compare before and after, save.
struct DeepFilterNetRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let modelID: String
    /// SC-5c: the stage the Run sheet handed (a `LeasingStage`); every run goes through it.
    let handedStage: any PipelineStage

    @State private var model = DeepFilterNetRunModel()
    @State private var audioURL: URL?
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
        .frame(width: 560, height: 520)
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.audio],
                      allowsMultipleSelection: false) { result in
            if case let .success(urls) = result { choose(urls.first) }
        }
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first else { return false }
            choose(url)
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
        .onDisappear { model.stopPlayback() }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: ModelType.enhance.sfSymbol).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Noise removal: speech recording → cleaned 48 kHz audio")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var inputs: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button("Choose Audio…") { showImporter = true }
                Text(audioURL?.lastPathComponent ?? "No file selected (or drop one here)")
                    .font(.callout).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            }
            Text("Built for recordings of speech. Any sample rate works: the audio is converted to 48 kHz first, and the cleaned file is always 48 kHz mono.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            if let limit = DeepFilterNetEngine.maxDurationSeconds {
                Text("Handles recordings up to \(DeepFilterNetEngine.spoken(limit)).")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text(license ?? "License: see model page").font(.caption).foregroundStyle(.secondary)
        }
    }

    private var runningRow: some View {
        HStack(spacing: 8) {
            ProgressView(value: model.progress).frame(maxWidth: 220)
            Text("Cleaning…").font(.callout).foregroundStyle(.secondary)
            Spacer()
            Button("Cancel") { model.cancel() }
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private var resultView: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Cleaned \(model.durationLabel)", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green).font(.subheadline.bold())
            HStack(spacing: 12) {
                Picker("Listen to", selection: Binding(get: { model.source }, set: { model.select($0) })) {
                    Text("Original").tag(DeepFilterNetRunModel.Source.original)
                    Text("Cleaned").tag(DeepFilterNetRunModel.Source.cleaned)
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 220)
                Button(model.isPlaying ? "Pause" : "Play") { model.togglePlayback() }
                    .keyboardShortcut(.space, modifiers: [])
                Spacer()
                Button("Save Cleaned Audio…") { model.save(suggestedName: audioURL) }
            }
            Text("Switch between Original and Cleaned while playing to compare; playback keeps its place.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            Button(model.isRunning ? "Working…" : "Clean") { run() }
                .keyboardShortcut(.defaultAction)
                .disabled(audioURL == nil || model.isRunning)
        }
    }

    // MARK: - Actions

    private func choose(_ url: URL?) {
        guard let url else { return }
        audioURL = url
        model.reset()
    }

    private func run() {
        guard let url = audioURL else { return }
        let modelID = self.modelID
        let stage = LeasingStage.perRun(handedStage, config: .default) { _ in
            DeepFilterNetStage(modelID: modelID)
        }
        model.start(audioURL: url, stage: stage)
    }
}

/// View-model driving one cleanup run: reads the chosen file, runs the stage, then plays the original or the cleaned audio
/// and saves the cleaned one. Reads the file via `DeepFilterNetInput` (readable errors). No Combine.
@MainActor
@Observable
final class DeepFilterNetRunModel: NSObject, AVAudioPlayerDelegate {
    enum Source: Hashable { case original, cleaned }

    var isRunning = false
    var progress = 0.0
    var errorText: String?
    var source: Source = .cleaned
    var isPlaying = false
    var durationLabel = ""

    private(set) var original: AudioBuffer?
    private(set) var cleaned: AudioBuffer?
    private var player: AVAudioPlayer?
    private var task: Task<Void, Never>?

    var hasResult: Bool { cleaned != nil }

    func reset() {
        task?.cancel()
        stopPlayback()
        original = nil
        cleaned = nil
        errorText = nil
        progress = 0
        isRunning = false
    }

    func cancel() { task?.cancel() }

    func start(audioURL: URL, stage: any PipelineStage) {
        reset()
        isRunning = true
        let accessed = audioURL.startAccessingSecurityScopedResource()
        task = Task {
            defer { if accessed { audioURL.stopAccessingSecurityScopedResource() } }
            do {
                // Decode off the main actor; a long recording is many megabytes of samples.
                let buffer = try await Task.detached { try DeepFilterNetInput.read(audioURL) }.value
                self.original = buffer
                let seconds = Double(buffer.samples.count) / Double(buffer.sampleRate)
                // Refuse an over-long recording before anything is loaded (the stage and the engine check again).
                try DeepFilterNetEngine.requireWithinLimit(seconds: seconds)
                // MLX-ERR-2 (see MLXErrorGuard.guardedRun): an MLX error becomes a thrown sentence, and a run in which the
                // process-wide fallback fired is never reported as successful.
                let output = try await MLXErrorGuard.guardedRun {
                    try await stage.run(.audio(buffer)) { fraction in
                        Task { @MainActor in self.progress = fraction }
                    }
                }
                guard case let .audio(audio) = output else { throw StageError.kindMismatch(expected: .audio, got: output.kind) }
                self.cleaned = audio
                self.source = .cleaned
                self.durationLabel = String(format: "%.1f s", Double(audio.samples.count) / Double(audio.sampleRate))
                self.progress = 1
            } catch is CancellationError {
                // The user cancelled, or declined the served-load question: not a failure.
            } catch {
                self.errorText = MLXErrorGuard.sentence(for: error)
                self.original = nil
            }
            self.isRunning = false
        }
    }

    // MARK: - Playback

    private func buffer(for source: Source) -> AudioBuffer? { source == .original ? original : cleaned }

    func togglePlayback() {
        if let player, player.isPlaying {
            player.pause()
            isPlaying = false
            return
        }
        if let player {   // paused mid-way: resume
            player.play()
            isPlaying = true
            return
        }
        startPlayer(at: 0)
    }

    /// Changes what is heard. While playing (or paused) the position carries over as a fraction of the length.
    func select(_ newSource: Source) {
        guard newSource != source else { return }
        let fraction = player.map { $0.duration > 0 ? $0.currentTime / $0.duration : 0 }
        let wasPlaying = isPlaying
        player?.stop()
        player = nil
        source = newSource
        isPlaying = false
        if let fraction, wasPlaying { startPlayer(at: fraction) }
        else if let fraction { startPlayer(at: fraction, play: false) }
    }

    private func startPlayer(at fraction: Double, play: Bool = true) {
        guard let buffer = buffer(for: source) else { return }
        do {
            let player = try AVAudioPlayer(data: AudioWriter.wavData(buffer))
            player.delegate = self
            player.currentTime = min(max(fraction, 0), 1) * player.duration
            self.player = player
            if play {
                player.play()
                isPlaying = true
            }
        } catch {
            errorText = "Playback failed: \(error.localizedDescription)"
        }
    }

    func stopPlayback() {
        player?.stop()
        player = nil
        isPlaying = false
    }

    /// Playback reached the end: flip the button back to "Play" and start from the top next time.
    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            self.isPlaying = false
            self.player = nil
        }
    }

    // MARK: - Save

    func save(suggestedName url: URL?) {
        guard let cleaned else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.wav]
        panel.nameFieldStringValue = Self.cleanedFileName(for: url)
        guard panel.runModal() == .OK, let target = panel.url else { return }
        do {
            try AudioWriter.writeWAV(cleaned, to: target)
        } catch {
            errorText = "Save failed: \(error.localizedDescription)"
        }
    }

    nonisolated static func cleanedFileName(for url: URL?) -> String {
        let base = url?.deletingPathExtension().lastPathComponent ?? ""
        return (base.isEmpty ? "audio" : base) + "-cleaned.wav"
    }
}
