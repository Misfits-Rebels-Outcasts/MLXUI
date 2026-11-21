import SwiftUI
import AppKit
import AVFoundation
import UniformTypeIdentifiers

/// `ModelUI` for ACE-Step 1.5 (MC5-3j): `AceStepRunView`. Same claim as `AceStepSDK`.
struct AceStepUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore { AceStepSDK().claim(model) }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(AceStepRunView(modelDisplayName: model.displayName, license: model.license, modelID: model.id))
    }
}

/// One run window's state: the inputs, the phase of a run, the result (WAV bytes), playback. `@MainActor @Observable`, no Combine.
/// The Generate button calls `generate()`; a test calls the same method with a generator injected in place of the real engine.
@MainActor
@Observable
final class AceStepRunModel: NSObject, AVAudioPlayerDelegate {
    typealias Generator = @Sendable (AceStepRequest, @escaping @Sendable (AceStepPhase) -> Void) async throws -> AceStepAudio

    static let durationRange = 10.0 ... 240.0
    static let defaultDuration = 60.0

    // Inputs. The seed is text so a half-typed number is not rewritten under the cursor; empty = a random one each run.
    var caption = "warm lofi piano with soft drums, relaxed and dreamy"
    var lyrics = ""
    var language = "en"
    var seedText = ""
    var duration = AceStepRunModel.defaultDuration {
        didSet {
            let clamped = Self.clamp(duration)
            if clamped != duration { duration = clamped }
        }
    }

    // Run state.
    private(set) var isRunning = false
    /// MC5-T-fix: Cancel was pressed and the engine has not finished unwinding. The model stays "running" (Generate is refused) until the engine task has returned and released its
    /// memory, so two generations can never overlap (a second engine on a 19 GB Mac would not fit).
    private(set) var isCancelling = false
    private(set) var progress = 0.0
    private(set) var phaseLabel = ""
    private(set) var errorText: String?
    /// The seed the last run used (typed, or the random one it picked), so a song you like can be made again.
    private(set) var usedSeed: UInt64?
    private(set) var wavData: Data?
    private(set) var resultSeconds = 0.0
    private(set) var isPlaying = false
    var didGenerate: Bool { wavData != nil }
    private(set) var request: AceStepRequest?

    private let generator: Generator
    private var task: Task<Void, Never>?
    private var generation = 0
    private var player: AVAudioPlayer?

    /// Production: the installed model folder through the engine. Tests pass their own `generator`.
    init(modelID: String = "", directory: URL? = nil, generator: Generator? = nil) {
        self.generator = generator ?? AceStepEngine.generator(modelID: modelID, directory: directory)
        super.init()
    }

    nonisolated static func clamp(_ seconds: Double) -> Double { min(max(seconds, durationRange.lowerBound), durationRange.upperBound) }

    var canGenerate: Bool { !isRunning && !caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    var durationText: String { Self.format(duration) }

    nonisolated static func format(_ seconds: Double) -> String {
        let total = Int(seconds.rounded())
        return total < 60 ? "\(total) s" : (total % 60 == 0 ? "\(total / 60) min" : "\(total / 60) min \(total % 60) s")
    }

    /// The button's entry point.
    func generate() {
        guard canGenerate else { return }
        stopPlayback()
        player = nil
        isRunning = true
        progress = 0
        phaseLabel = Self.label(for: .text)
        errorText = nil
        wavData = nil
        let seed = UInt64(seedText.trimmingCharacters(in: .whitespaces)) ?? UInt64.random(in: 0 ..< (1 << 32))
        usedSeed = seed
        // Empty lyrics make an instrumental (the engine sends upstream's `[Instrumental]`); the language only matters when there are lyrics.
        let request = AceStepRequest(caption: caption.trimmingCharacters(in: .whitespacesAndNewlines),
                                     lyrics: lyrics.trimmingCharacters(in: .whitespacesAndNewlines), language: language,
                                     duration: Self.clamp(duration), seed: seed)
        self.request = request
        generation += 1
        let mine = generation
        let generator = self.generator
        task = Task { [weak self] in
            do {
                let audio = try await MLXErrorGuard.guardedRun {
                    try await generator(request) { phase in
                        Task { @MainActor in self?.report(phase, generation: mine) }
                    }
                }
                self?.finish(audio, generation: mine)
            } catch is CancellationError {
                self?.cancelled(generation: mine)
            } catch {
                self?.fail(error, generation: mine)
            }
        }
    }

    func cancel() {
        guard isRunning, !isCancelling else { return }
        isCancelling = true
        phaseLabel = "Stopping…"
        task?.cancel()
    }

    /// Waits for the current run to end (a test; the view never needs it).
    func finished() async { await task?.value }

    // MARK: Run transitions (stale generations are ignored)

    private func report(_ phase: AceStepPhase, generation g: Int) {
        guard g == generation, isRunning, !isCancelling else { return }
        progress = phase.fraction
        phaseLabel = Self.label(for: phase)
    }

    private func finish(_ audio: AceStepAudio, generation g: Int) {
        guard g == generation else { return }
        wavData = audio.wavData()
        resultSeconds = audio.seconds
        progress = 1
        isRunning = false
        isCancelling = false
        task = nil
    }

    private func cancelled(generation g: Int) {
        guard g == generation else { return }
        isRunning = false
        isCancelling = false
        progress = 0
        phaseLabel = ""
        task = nil
    }

    private func fail(_ error: Error, generation g: Int) {
        guard g == generation else { return }
        errorText = Self.message(for: error)
        isRunning = false
        isCancelling = false
        task = nil
    }

    nonisolated static func label(for phase: AceStepPhase) -> String {
        switch phase {
        case .text: return "Reading your description…"
        case .condition: return "Getting ready…"
        case .denoise(let step, let total): return "Making the music: step \(step) of \(total)"
        case .decode: return "Turning it into sound…"
        }
    }

    /// Plain-language error text. A damaged or incomplete install says so and points to reinstalling; nothing here downloads.
    nonisolated static func message(for error: Error) -> String {
        if let weights = error as? AceStepWeightsError {
            return "\(String(describing: weights)). Reinstall ACE-Step 1.5 from Browse to repair it."
        }
        return MLXErrorGuard.sentence(for: error)
    }

    // MARK: Playback and saving

    func togglePlayback() {
        guard let wavData else { return }
        if let player, player.isPlaying {
            player.pause()
            isPlaying = false
            return
        }
        do {
            let player = try AVAudioPlayer(data: wavData)
            player.delegate = self
            self.player = player
            player.play()
            isPlaying = true
        } catch {
            errorText = "Playback failed: \(error.localizedDescription)"
        }
    }

    func stopPlayback() {
        player?.stop()
        isPlaying = false
    }

    var hasPlayer: Bool { player != nil }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in self.isPlaying = false }
    }

    /// Writes the song as a .wav through the save panel (user-selected file access; no new entitlement).
    func save() {
        guard let wavData else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.wav]
        panel.nameFieldStringValue = "acestep.wav"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try wavData.write(to: url) } catch { errorText = "Save failed: \(error.localizedDescription)" }
    }
}

/// The run surface for ACE-Step: a style description, optional lyrics, length, seed, Generate / Cancel, live progress, then play and save.
/// Laid out like MusicGen's. The sheet that opens it has already checked the install and the Mac's memory.
struct AceStepRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    @State private var model: AceStepRunModel

    init(modelDisplayName: String, license: String?, modelID: String) {
        self.modelDisplayName = modelDisplayName
        self.license = license
        _model = State(initialValue: AceStepRunModel(modelID: modelID))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            descriptionField
            lyricsField
            options
            if model.isRunning { progressRow }
            if let error = model.errorText { RunErrorBar(message: error) }
            if model.didGenerate { resultRow }
            Spacer(minLength: 0)
            footer
        }
        .padding(18)
        .frame(width: 620, height: 720)
        .onDisappear { model.cancel(); model.stopPlayback() }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "music.quarternote.3").foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Describe a song and make it on your Mac").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var descriptionField: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("What kind of music?").font(.subheadline.bold())
            TextEditor(text: $model.caption)
                .font(.body)
                .frame(minHeight: 80)
                .padding(6)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
                .disabled(model.isRunning)
        }
    }

    private var lyricsField: some View {
        DisclosureGroup("Lyrics (optional)") {
            VStack(alignment: .leading, spacing: 6) {
                TextEditor(text: $model.lyrics)
                    .font(.body)
                    .frame(minHeight: 110)
                    .padding(6)
                    .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
                    .disabled(model.isRunning)
                HStack {
                    Text("Language of the lyrics").font(.caption)
                    Picker("", selection: $model.language) {
                        Text("English").tag("en")
                        Text("Chinese").tag("zh")
                        Text("Other").tag("unknown")
                    }
                    .labelsHidden().frame(maxWidth: 140).disabled(model.isRunning)
                    Spacer()
                }
                Text("Leave this empty to make music without singing.").font(.caption).foregroundStyle(.secondary)
            }
            .padding(.top, 6)
        }
    }

    private var options: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Length").frame(width: 60, alignment: .leading)
                Slider(value: $model.duration, in: AceStepRunModel.durationRange, step: 5).frame(maxWidth: 300).disabled(model.isRunning)
                Text(model.durationText).font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                Spacer()
            }
            HStack {
                Text("Seed").frame(width: 60, alignment: .leading)
                TextField("Random", text: $model.seedText).frame(width: 140).disabled(model.isRunning)
                if let used = model.usedSeed, !model.isRunning {
                    Text("Last song used seed \(used)").font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                }
                Spacer()
            }
            Text(license ?? "License: see model page").font(.caption).foregroundStyle(.secondary)
        }
    }

    private var progressRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                ProgressView(value: model.progress).progressViewStyle(.linear)
                Text("\(Int(model.progress * 100))%").font(.callout.monospacedDigit()).foregroundStyle(.secondary)
            }
            Text(model.phaseLabel).font(.caption).foregroundStyle(.secondary)
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private var resultRow: some View {
        HStack(spacing: 12) {
            Label("Ready", systemImage: "checkmark.circle.fill").foregroundStyle(.green).font(.subheadline.bold())
            Button { model.togglePlayback() } label: {
                Label(model.isPlaying ? "Pause" : "Play", systemImage: model.isPlaying ? "pause.fill" : "play.fill")
            }
            Button { model.save() } label: { Label("Save WAV…", systemImage: "square.and.arrow.down") }
            Text(AceStepRunModel.format(model.resultSeconds)).font(.callout.monospacedDigit()).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            if model.isRunning {
                Button(model.isCancelling ? "Stopping…" : "Cancel") { model.cancel() }.disabled(model.isCancelling)
            }
            Button(model.isRunning ? "Making music…" : "Generate") { model.generate() }
                .keyboardShortcut(.defaultAction)
                .disabled(!model.canGenerate)
        }
    }
}
