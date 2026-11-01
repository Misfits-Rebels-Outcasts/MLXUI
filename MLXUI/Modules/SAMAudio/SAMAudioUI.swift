import SwiftUI
import AppKit
import AVFoundation
import UniformTypeIdentifiers

/// `ModelUI` for SAM-Audio sound isolation. Presents `SAMAudioRunView` (pick or drop a recording, say what to
/// isolate, Run/Stop, play the isolated track or everything else, save either as a WAV at the model's sample rate).
/// Same claim predicate as `SAMAudioSDK`.
struct SAMAudioUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .separate,
              model.source == .mlx,
              model.family.lowercased() == "sam-audio" else {
            return .no
        }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(SAMAudioRunView(modelDisplayName: model.displayName,
                                license: model.license,
                                licenseURL: model.licenseUrl,
                                modelID: model.id,
                                handedStage: stage,
                                model: SAMAudioRunModel()))
    }
}

/// The run surface: choose or drop a recording, say what to isolate, Run (with the length check and a time estimate
/// shown first), then compare the two tracks and save either one.
struct SAMAudioRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let licenseURL: String?
    let modelID: String
    /// SC-5c: the stage the Run sheet handed (a `LeasingStage`); every run goes through it with this run's description.
    let handedStage: any PipelineStage

    @State private var model = SAMAudioRunModel()
    @State private var audioURL: URL?
    @State private var durationSeconds: Double?
    @State private var description = ""
    @State private var showImporter = false
    @State private var isDropTargeted = false

    /// The app's path. The first three optional parameters are **render-test seams** (`ServerScreenRenderTests`'
    /// pattern): a snapshot test injects a run model and a chosen file, so the screen can be captured in its filled
    /// and result states without a GUI. They default to the empty state the sheet opens with.
    @MainActor init(modelDisplayName: String, license: String?, licenseURL: String?, modelID: String,
         handedStage: any PipelineStage,
         model: SAMAudioRunModel,
         initialAudioURL: URL? = nil, initialDescription: String = "") {
        self.modelDisplayName = modelDisplayName
        self.license = license
        self.licenseURL = licenseURL
        self.modelID = modelID
        self.handedStage = handedStage
        _model = State(initialValue: model)
        _audioURL = State(initialValue: initialAudioURL)
        _durationSeconds = State(initialValue: initialAudioURL.flatMap { SAMAudioInput.durationSeconds(of: $0) })
        _description = State(initialValue: initialDescription)
    }

    private static let suggestions = ["speech", "music", "singing voice", "background noise"]

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
        .frame(width: 600, height: 600)
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
            Image(systemName: ModelType.separate.sfSymbol).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Sound isolation: name a sound, keep it on one track and everything else on another")
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
            VStack(alignment: .leading, spacing: 6) {
                Text("What to isolate").font(.callout.weight(.medium))
                TextField("For example: speech", text: $description)
                    .textFieldStyle(.roundedBorder)
                    .disabled(model.isRunning)
                HStack(spacing: 6) {
                    ForEach(Self.suggestions, id: \.self) { suggestion in
                        Button(suggestion) { description = suggestion }
                            .buttonStyle(.link)
                            .disabled(model.isRunning)
                    }
                    Spacer()
                }
            }
            if let estimate = estimateText {
                Text(estimate).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            Text("Handles recordings up to \(SAMAudioEngine.spoken(SAMAudioEngine.maxDurationSeconds)); a longer one is refused, so split it into parts. Long recordings take minutes. Saved WAVs keep the model's own sample rate.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 4) {
                Text(licenseNotice).font(.caption).foregroundStyle(.secondary)
                if let licenseURL, let url = URL(string: licenseURL) {
                    Link("Read the license", destination: url).font(.caption)
                }
            }
        }
    }

    private var runningRow: some View {
        HStack(spacing: 8) {
            ProgressView(value: model.progress).frame(maxWidth: 220)
            Text("Isolating… \(Int((model.progress * 100).rounded()))%").font(.callout).foregroundStyle(.secondary).monospacedDigit()
            Spacer()
            Button("Stop") { model.cancel() }
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private var resultView: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Separated \(model.durationLabel)", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green).font(.subheadline.bold())
            trackRow(.isolated)
            trackRow(.other)
            Text("Play one track, then the other, to hear what was isolated and what was left behind.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func trackRow(_ track: SAMAudioRunModel.Track) -> some View {
        HStack(spacing: 12) {
            Text(track.title).font(.callout.weight(.medium)).frame(width: 130, alignment: .leading)
            Button(model.isPlaying(track) ? "Pause" : "Play") { model.toggle(track) }
                .disabled(model.buffer(track) == nil)
            Spacer()
            Button("Save WAV…") { model.save(track, suggestedName: audioURL) }
                .disabled(model.buffer(track) == nil)
        }
        .padding(.vertical, 2)
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            Button(model.isRunning ? "Working…" : "Run") { run() }
                .keyboardShortcut(.defaultAction)
                .disabled(audioURL == nil || model.isRunning || description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
    }

    // MARK: - Copy

    private var licenseNotice: String {
        guard let license else { return "License: see the model page." }
        return "License: \(DataNormalizer.normalizeLicense(license)) — the model page links the full terms."
    }

    /// The time estimate shown before a run (owner decision 5). Uses the file's header, so it appears as soon as a
    /// file is chosen, before anything is decoded or loaded.
    private var estimateText: String? {
        guard let durationSeconds else { return nil }
        let estimate = SAMAudioEngine.estimate(seconds: durationSeconds)
        return "This \(SAMAudioEngine.spoken(durationSeconds)) recording will take about \(SAMAudioEngine.spokenEstimate(estimate))."
    }

    // MARK: - Actions

    private func choose(_ url: URL?) {
        guard let url else { return }
        audioURL = url
        durationSeconds = SAMAudioInput.durationSeconds(of: url)
        model.reset()
    }

    private func run() {
        guard let url = audioURL else { return }
        let text = description.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        let modelID = self.modelID
        // One per run, so a second run never shows the first run's second track.
        let box = SAMAudioTrackBox()
        let stage = LeasingStage.perRun(handedStage, config: StageConfig(prompt: text)) { _ in
            SAMAudioStage(modelID: modelID, description: text, output: box)
        }
        model.start(audioURL: url, stage: stage, output: box)
    }
}

/// The view-model driving one isolation run: reads the chosen file, runs the stage (leasing the engine through the
/// handed `LeasingStage`), then plays either track and saves either one. No Combine.
@MainActor
@Observable
final class SAMAudioRunModel: NSObject, AVAudioPlayerDelegate {
    enum Track: Hashable, CaseIterable {
        case isolated, other
        var title: String { self == .isolated ? "Isolated" : "Everything else" }
        var fileSuffix: String { self == .isolated ? "isolated" : "everything-else" }
    }

    var isRunning = false
    var progress = 0.0
    var errorText: String?
    var durationLabel = ""

    private(set) var isolated: AudioBuffer?
    private(set) var other: AudioBuffer?
    private var isolatedPlayer: AVAudioPlayer?
    private var otherPlayer: AVAudioPlayer?
    private var playing: Track?
    private var task: Task<Void, Never>?

    var hasResult: Bool { isolated != nil }

    func buffer(_ track: Track) -> AudioBuffer? {
        switch track {
        case .isolated: return isolated
        case .other: return other
        }
    }

    func reset() {
        task?.cancel()
        stopPlayback()
        isolated = nil
        other = nil
        errorText = nil
        progress = 0
        isRunning = false
        durationLabel = ""
    }

    func cancel() { task?.cancel() }

    /// Runs one isolation. `output` is the box the stage hands the "everything else" track through (`Media` carries the
    /// isolated one). `maxDurationSeconds` is a parameter so a test can refuse without a 30-minute file.
    func start(audioURL: URL, stage: any PipelineStage, output: SAMAudioTrackBox,
               maxDurationSeconds: Double = SAMAudioEngine.maxDurationSeconds) {
        reset()
        output.clear()
        isRunning = true
        let accessed = audioURL.startAccessingSecurityScopedResource()
        task = Task {
            defer { if accessed { audioURL.stopAccessingSecurityScopedResource() } }
            do {
                // Refuse an over-long file from its header, before the decode allocates anything.
                if let header = SAMAudioInput.durationSeconds(of: audioURL) {
                    try SAMAudioEngine.requireWithinLimit(seconds: header, limit: maxDurationSeconds)
                }
                let buffer = try await Task.detached { try SAMAudioInput.read(audioURL) }.value
                let seconds = Double(buffer.samples.count) / Double(buffer.sampleRate)
                // Refuse an over-long recording before anything is loaded (the stage and the engine check again).
                try SAMAudioEngine.requireWithinLimit(seconds: seconds, limit: maxDurationSeconds)
                // MLX-ERR-2 (see MLXErrorGuard.guardedRun): an MLX error becomes a thrown sentence, and a run in which
                // the process-wide fallback fired is never reported as successful.
                let result = try await MLXErrorGuard.guardedRun {
                    try await Self.runOffMain(stage, .audio(buffer)) { fraction in
                        Task { @MainActor in self.progress = fraction }
                    }
                }
                guard case let .audio(audio) = result else {
                    throw StageError.kindMismatch(expected: .audio, got: result.kind)
                }
                self.isolated = audio
                self.other = output.other
                self.durationLabel = String(format: "%.1f s", Double(audio.samples.count) / Double(audio.sampleRate))
                self.progress = 1
            } catch {
                // MP5-3g's rule: a cancelled run is never an error. Drop the decoded audio and any partial output too.
                if error is CancellationError || Task.isCancelled {
                    self.isolated = nil
                    self.other = nil
                    self.progress = 0
                } else {
                    self.errorText = MLXErrorGuard.sentence(for: error)
                    self.isolated = nil
                    self.other = nil
                }
                output.clear()
            }
            self.isRunning = false
        }
    }

    /// Runs `stage` off the main actor. `SWIFT_APPROACHABLE_CONCURRENCY` makes a plain `nonisolated async` call run on
    /// its caller's actor, and this model's `Task` is main-actor, so without the hop a stage's synchronous window loop
    /// would run on the main thread: beachball, progress stuck, Stop unclickable. `@concurrent` is a child hop, so the
    /// `MLXErrorGuard` task-local and cancellation carry across, and a `LeasingStage` waiting for memory leaves the
    /// main actor too.
    @concurrent nonisolated private static func runOffMain(_ stage: any PipelineStage, _ input: Media,
                                                          progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try await stage.run(input, progress: progress)
    }

    // MARK: - Playback

    func isPlaying(_ track: Track) -> Bool { playing == track && (player(for: track)?.isPlaying ?? false) }

    func toggle(_ track: Track) {
        let player = player(for: track)
        if let player, player.isPlaying {
            player.pause()
            playing = nil
            return
        }
        // One track at a time: starting one stops the other, so the comparison is by ear, not by mix.
        stopPlayback(keeping: track)
        if let player {
            player.play()
            playing = track
            return
        }
        startPlayer(track, at: 0)
    }

    private func player(for track: Track) -> AVAudioPlayer? {
        track == .isolated ? isolatedPlayer : otherPlayer
    }

    private func startPlayer(_ track: Track, at fraction: Double, play: Bool = true) {
        guard let buffer = buffer(track) else { return }
        do {
            let player = try AVAudioPlayer(data: AudioWriter.wavData(buffer))
            player.delegate = self
            player.currentTime = min(max(fraction, 0), 1) * player.duration
            switch track {
            case .isolated: isolatedPlayer = player
            case .other: otherPlayer = player
            }
            if play {
                player.play()
                playing = track
            }
        } catch {
            errorText = "Playback failed: \(error.localizedDescription)"
        }
    }

    func stopPlayback() {
        stopPlayback(keeping: nil)
    }

    private func stopPlayback(keeping: Track?) {
        if keeping != .isolated, let isolatedPlayer, isolatedPlayer.isPlaying { isolatedPlayer.pause() }
        if keeping != .other, let otherPlayer, otherPlayer.isPlaying { otherPlayer.pause() }
        playing = keeping
    }

    /// Playback reached the end: flip the button back to "Play" and start from the top next time.
    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            self.isolatedPlayer = nil
            self.otherPlayer = nil
            self.playing = nil
        }
    }

    // MARK: - Save

    func save(_ track: Track, suggestedName url: URL?) {
        guard let buffer = buffer(track) else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.wav]
        panel.nameFieldStringValue = Self.fileName(for: url, track: track)
        guard panel.runModal() == .OK, let target = panel.url else { return }
        do {
            try AudioWriter.writeWAV(buffer, to: target)
        } catch {
            errorText = "Save failed: \(error.localizedDescription)"
        }
    }

    /// "interview-isolated.wav" and "interview-everything-else.wav"; "audio-…" when the chosen file has no stem.
    nonisolated static func fileName(for url: URL?, track: Track) -> String {
        let base = url?.deletingPathExtension().lastPathComponent ?? ""
        return (base.isEmpty ? "audio" : base) + "-\(track.fileSuffix).wav"
    }
}
