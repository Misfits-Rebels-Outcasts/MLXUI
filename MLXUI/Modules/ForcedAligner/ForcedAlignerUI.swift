import SwiftUI
import AppKit
import UniformTypeIdentifiers
import AVFoundation

/// `ModelUI` for Qwen3 forced alignment. Presents `ForcedAlignerRunView` (pick or drop an audio file, paste or open the
/// transcript, pick a language, Align → a `text,start,end` table; click a row to hear it; export CSV/SRT/VTT). The claim
/// predicate is `ForcedAlignerSDK`'s (trap 3).
struct ForcedAlignerUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .align,
              model.source == .mlx,
              model.family.lowercased() == "qwen3-forcedaligner" else {
            return .no
        }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(ForcedAlignerRunView(modelDisplayName: model.displayName,
                                     license: model.license,
                                     modelID: model.id,
                                     handedStage: stage))
    }
}

/// The Run surface (owner decision 7): an audio file, a transcript, one of the 11 languages, **Align** with a working
/// **Stop**, a table of `text`, `start`, `end`, click-to-play from a row, and **Export CSV/SRT/VTT**.
struct ForcedAlignerRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let modelID: String
    /// SC-5c: the stage the Run sheet handed (a `LeasingStage`); every run rebuilds it with this run's transcript and
    /// language, so the memory lease and the engine-cache key follow the run.
    let handedStage: any PipelineStage

    @State private var model: AlignRunModel
    @State private var selection: AlignRunModel.Row.ID?
    @State private var showAudioImporter = false
    @State private var showTranscriptImporter = false
    @State private var isDropTargeted = false

    init(modelDisplayName: String, license: String?, modelID: String, handedStage: any PipelineStage) {
        self.modelDisplayName = modelDisplayName
        self.license = license
        self.modelID = modelID
        self.handedStage = handedStage
        _model = State(initialValue: AlignRunModel(modelID: modelID))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            inputs
            caption
            controls
            if model.isRunning { runningRow }
            if let error = model.errorText { RunErrorBar(message: error) }
            if let result = model.result { resultView(result) }
            Spacer(minLength: 0)
            footer(result: model.result)
        }
        .padding(18)
        .frame(width: 760, height: 800)
        .fileImporter(isPresented: $showAudioImporter, allowedContentTypes: [.audio],
                      allowsMultipleSelection: false) { result in
            if case let .success(urls) = result, let url = urls.first {
                model.chooseAudio(url)
            }
        }
        .fileImporter(isPresented: $showTranscriptImporter, allowedContentTypes: [.plainText, .text],
                      allowsMultipleSelection: false) { result in
            if case let .success(urls) = result, let url = urls.first {
                model.openTranscript(url)
            }
        }
        .dropDestination(for: URL.self) { urls, _ in
            model.dropAudio(urls)
        } isTargeted: { isDropTargeted = $0 && model.canChangeInputs }
        .onDisappear { model.stop() }   // ruling 12 (see the Done button), however the sheet is closed
        .overlay {
            if isDropTargeted && model.canChangeInputs {
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 2, dash: [6]))
                    .padding(6)
                    .allowsHitTesting(false)
            }
        }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: ModelType.align.sfSymbol).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Word-level timestamps: audio + the transcript you already have → when each word is said")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var inputs: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button("Choose Audio…") { showAudioImporter = true }
                    .disabled(!model.canChangeInputs)
                Text(model.audioURL?.lastPathComponent ?? "No audio file selected (or drop one here)")
                    .font(.callout).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                Spacer()
                if let url = model.audioURL {
                    Text(Self.clock(model.audioDuration(of: url)))
                        .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                }
            }
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Transcript").font(.callout.bold())
                    // Owner ruling 16 (Boo, 2026-11-27): "Confirmed: the transcript box and the language picker stay editable
                    // during a run. The run keeps its own copies. The result table and exports belong to the run that produced
                    // them."
                    TextEditor(text: $model.transcript)
                        .font(.callout)
                        .frame(minHeight: 96, maxHeight: 140)
                        .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(.quaternary))
                }
                Button("Open…") { showTranscriptImporter = true }
                    .disabled(!model.canChangeInputs)
                    .help("Open a .txt file as the transcript")
            }
            HStack {
                Text("Language").frame(width: 70, alignment: .leading)
                // Editable during a run too (ruling 16, quoted at the transcript box).
                Picker("Language", selection: $model.languageCode) {
                    ForEach(ForcedAlignerEngine.languages) { language in
                        Text(language.name).tag(language.code)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: 220)
                Spacer()
            }
            Text(license ?? "License: see model page").font(.caption).foregroundStyle(.secondary)
        }
    }

    /// Decision 7's caption, verbatim.
    static let captionText = "Timestamps are in 80 ms steps. The transcript must match what is said. A leading intro or silence can pull the first word towards 0 s."

    private var caption: some View {
        Text(Self.captionText)
            .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
    }

    private var controls: some View {
        HStack(spacing: 8) {
            Button(model.runButtonTitle) { run() }
                .keyboardShortcut(.defaultAction)
                .disabled(!model.canChangeInputs || model.audioURL == nil)
            if model.isRunning {
                Button("Stop") { model.stop() }.keyboardShortcut(.cancelAction)
            }
            Spacer()
        }
    }

    private var runningRow: some View {
        HStack(spacing: 8) {
            ProgressView().scaleEffect(0.6).frame(width: 16, height: 16)
            ProgressView(value: model.progress).frame(maxWidth: 240)
            Text(model.progressText).font(.callout).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private func resultView(_ result: AlignRunModel.Result) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("\(result.units.count) unit\(result.units.count == 1 ? "" : "s"), \(AlignmentExport.seconds(result.duration)) s of audio",
                      systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green).font(.subheadline.bold())
                Spacer()
            }
            Table(result.rows, selection: $selection) {
                TableColumn("Text") { row in Text(row.unit.text) }
                TableColumn("Start") { row in Text(AlignmentExport.seconds(row.unit.start)).monospacedDigit() }
                TableColumn("End") { row in Text(AlignmentExport.seconds(row.unit.end)).monospacedDigit() }
            }
            .frame(minHeight: 220)
            .onChange(of: selection) { _, newValue in
                guard let row = result.rows.first(where: { $0.id == newValue }) else { return }
                model.play(from: row.unit.start)
            }
            Text("Click a row to play the audio from that word's start.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func footer(result: AlignRunModel.Result?) -> some View {
        HStack {
            // Owner ruling 12 (Boo, 2026-11-27): "Done cancels the run through the same path as Stop. The lease rule from
            // ruling 11 applies. Test that Done during a run discards the result and releases the lease only after the step
            // returns."
            // Owner ruling 13 (Boo, 2026-11-27): "Confirmed: Done has no keyboard shortcut while a run is going, so the
            // cancel key always means Stop."
            Button("Done") { model.stop(); dismiss() }.keyboardShortcut(model.isRunning ? nil : .cancelAction)
            Spacer()
            Button("Export CSV") { export(result, as: .csv) }.disabled(result == nil)
            Button("Export SRT") { export(result, as: .srt) }.disabled(result == nil)
            Button("Export VTT") { export(result, as: .vtt) }.disabled(result == nil)
        }
    }

    // MARK: - Actions

    private func run() {
        guard let url = model.audioURL else { return }
        selection = nil   // row ids are positions, so a new result must not inherit the last one's selection
        let modelID = self.modelID
        let text = model.transcript
        let language = model.languageCode
        // The per-run stage: the sheet's lease, this run's transcript and language (the cache key ignores the transcript).
        let stage = LeasingStage.perRun(handedStage,
                                        config: StageConfig(language: language, transcript: text)) { config in
            ForcedAlignerStage(modelID: modelID,
                               transcript: config.transcript ?? text,
                               language: config.language ?? language)
        }
        model.start(audioURL: url, transcript: text, language: language, stage: stage)
    }

    private func export(_ result: AlignRunModel.Result?, as format: ExportFormat) {
        guard let result else { return }
        let panel = NSSavePanel()
        let base = model.audioURL?.deletingPathExtension().lastPathComponent ?? "alignment"
        panel.nameFieldStringValue = "\(base).\(format.rawValue)"
        panel.allowedContentTypes = [UTType(filenameExtension: format.rawValue) ?? .plainText]
        guard panel.runModal() == .OK, let target = panel.url else { return }
        do {
            try format.render(result).write(to: target, atomically: true, encoding: .utf8)
        } catch {
            model.errorText = "The \(format.rawValue.uppercased()) file couldn't be saved — choose another folder and try again."
        }
    }

    enum ExportFormat: String {
        case csv, srt, vtt

        /// SRT and VTT end their cues at the sentences of the transcript that run aligned.
        func render(_ result: AlignRunModel.Result) -> String {
            switch self {
            case .csv: return AlignmentExport.csv(result.units)
            case .srt: return AlignmentExport.srt(result.units, transcript: result.transcript)
            case .vtt: return AlignmentExport.vtt(result.units, transcript: result.transcript)
            }
        }
    }

    /// `0:07.240` for the chosen file's length.
    static func clock(_ seconds: Double) -> String {
        let totalMs = Int((max(0, seconds) * 1000).rounded())
        let (ms, total) = (totalMs % 1000, totalMs / 1000)
        return String(format: "%d:%02d.%03d", total / 60, total % 60, ms)
    }
}

/// View-model driving one alignment run: checks the inputs, reads the file off the main actor, runs the stage and folds
/// the `text,start,end` CSV back into rows. Plays the recording from a clicked row. No Combine.
@MainActor
@Observable
final class AlignRunModel {
    struct Result: Equatable {
        let units: [AlignedUnit]
        let duration: Double
        /// What this run aligned (the transcript box may have been edited since); the SRT/VTT cues read it.
        let transcript: String

        var rows: [Row] { units.enumerated().map { Row(id: $0.offset, unit: $0.element) } }
    }

    /// A table row. Its identity is its position: two units can have the same text and times (a repeated word whose
    /// times the library filled from a neighbour).
    struct Row: Identifiable, Equatable {
        let id: Int
        let unit: AlignedUnit
    }

    var isRunning = false
    /// Stop was pressed and the step in flight hasn't returned yet (`isRunning` stays true until it does).
    private(set) var isStopping = false
    var runButtonTitle: String { isStopping ? "Stopping…" : isRunning ? "Aligning…" : "Align" }
    var progressText: String { isStopping ? "Stopping…" : "Timing the transcript…" }
    var progress = 0.0
    var errorText: String?
    var result: Result?
    /// The transcript box.
    var transcript = ""
    /// The language picker.
    var languageCode = ForcedAlignerEngine.defaultLanguageCode
    private let modelID: String
    private let tracker: AlignStepTracker

    /// `tracker` is the app-wide one unless a test passes its own.
    init(modelID: String, tracker: AlignStepTracker? = nil) {
        self.modelID = modelID
        self.tracker = tracker ?? .shared
    }

    private var player: AVAudioPlayer?
    private(set) var task: Task<Void, Never>?
    /// The one run that may write this model's state (nil after `reset()`). Since ruling 14 (`canChangeInputs`) nothing
    /// can replace a run in flight, so the checks against it are a backstop.
    private(set) var currentRun: UUID?
    /// The chosen recording (what Align runs on, and what a clicked row plays).
    private(set) var audioURL: URL?
    /// The audio file's length as read from its header, for the view's label (nil until read once; the run replaces it
    /// with the decoded duration).
    private var headerDuration: [URL: Double] = [:]

    func audioDuration(of url: URL) -> Double {
        if let known = headerDuration[url] { return known }
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        let seconds = (try? AVAudioFile(forReading: url)).map { file in
            file.processingFormat.sampleRate > 0 ? Double(file.length) / file.processingFormat.sampleRate : 0
        } ?? 0
        headerDuration[url] = seconds
        return seconds
    }

    // MARK: - Inputs

    /// Whether the audio and transcript file may change and a run may start: not while a step is in flight, which
    /// includes "Stopping…", whether this sheet started it or an earlier sheet of the same model did (ruling 15, see
    /// `AlignStepTracker`). Every path below, and `start`, asks this.
    ///
    /// Owner ruling 14 (Boo, 2026-11-27): "Disable them. While a run is going, including during 'Stopping…', choosing or
    /// dropping a new audio file and opening a transcript are disabled, along with Align. Add a test that no second run
    /// can start until the first step has returned."
    var canChangeInputs: Bool { !isRunning && !tracker.isBusy(modelID) }

    /// A file picked as the recording; false when refused.
    @discardableResult func chooseAudio(_ url: URL) -> Bool {
        guard canChangeInputs else { return false }
        reset()
        audioURL = url
        return true
    }

    /// Files dropped on the screen (the first is the recording); false when refused.
    @discardableResult func dropAudio(_ urls: [URL]) -> Bool {
        guard let url = urls.first else { return false }
        return chooseAudio(url)
    }

    /// A text file opened as the transcript; false when refused or unreadable.
    @discardableResult func openTranscript(_ url: URL) -> Bool {
        guard canChangeInputs else { return false }
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        do {
            transcript = try String(contentsOf: url, encoding: .utf8)
            reset()
            return true
        } catch {
            errorText = "That file isn't readable as text. Save it as UTF-8 .txt and try again."
            return false
        }
    }

    /// Clears the last run. Only reached when no step is in flight (the callers check `canChangeInputs`).
    private func reset() {
        task?.cancel()
        task = nil
        currentRun = nil
        isRunning = false
        isStopping = false
        progress = 0
        errorText = nil
        result = nil
        stopPlayback()
    }

    /// The one cancel path: the Stop button, Done, and the sheet disappearing. The run's memory lease is not touched
    /// here: `LeasingStage.run` returns it when the stage itself returns.
    ///
    /// Owner ruling 11 (Boo, 2026-11-27): "Accept that Stop can't interrupt the single alignment pass. But when Stop is
    /// pressed, the screen shows 'Stopping…' at once instead of 'running', and the MemoryBudget lease is released only
    /// when the step actually returns, never at the moment Stop is pressed. Test both."
    func stop() {
        guard isRunning else { return }
        isStopping = true
        task?.cancel()
    }

    func start(audioURL: URL, transcript: String, language: String, stage: any PipelineStage) {
        guard canChangeInputs else { return }   // one step at a time (ruling 14)
        reset()
        // Refuse with one plain sentence before any model work (owner decision 5): an empty transcript, an unsupported
        // language or an over-long recording never loads the model.
        let text = ForcedAlignerEngine.libraryTranscript(transcript)
        if text.isEmpty {
            errorText = ForcedAlignerEngine.EngineError.emptyTranscript.errorDescription
            return
        }
        guard ForcedAlignerEngine.libraryLanguageName(for: language) != nil else {
            errorText = ForcedAlignerEngine.EngineError.unsupportedLanguage(code: language).errorDescription
            return
        }
        isRunning = true
        self.audioURL = audioURL
        let run = UUID()
        currentRun = run
        tracker.begin(modelID)
        let accessed = audioURL.startAccessingSecurityScopedResource()
        task = Task {
            defer { self.tracker.end(self.modelID) }   // the step has returned, thrown or been cancelled
            defer { if accessed { audioURL.stopAccessingSecurityScopedResource() } }
            do {
                // Decode off the main actor; a long recording is many megabytes of samples.
                let buffer = try await Task.detached { try AudioFileReader.read(audioURL) }.value
                let duration = buffer.sampleRate > 0 ? Double(buffer.samples.count) / Double(buffer.sampleRate) : 0
                try ForcedAlignerEngine.requireWithinLimit(seconds: duration)
                // MLX-ERR-2 (see MLXErrorGuard.guardedRun): an MLX error becomes a thrown sentence, and a run in which the
                // process-wide fallback fired is never reported as successful.
                let output = try await MLXErrorGuard.guardedRun {
                    try await Self.runOffMain(stage, .audio(buffer)) { fraction in
                        Task { @MainActor in self.report(fraction, from: run) }
                    }
                }
                // Stop pressed during the forward pass, which can't be interrupted: the output is dropped.
                try Task.checkCancellation()
                guard case let .text(csv) = output else {
                    throw StageError.kindMismatch(expected: .text, got: output.kind)
                }
                self.result = Result(units: try AlignmentExport.parseCSV(csv), duration: duration, transcript: text)
                self.progress = 1
            } catch {
                // A replaced run writes nothing: `reset()` or the newer run owns the state now.
                guard self.currentRun == run else { return }
                // A cancelled run (Stop) is never an error, whatever the stage threw (the DeepFilterNet / Sortformer rule).
                if error is CancellationError || Task.isCancelled {
                    self.progress = 0
                } else {
                    self.errorText = MLXErrorGuard.sentence(for: error)
                }
            }
            self.isRunning = false
            self.isStopping = false
        }
    }

    /// A progress value from `run`'s stage; a stopped or replaced run's is dropped.
    private func report(_ fraction: Double, from run: UUID) {
        guard currentRun == run, task?.isCancelled == false else { return }
        progress = fraction
    }

    /// Runs the stage off the main actor (the DeepFilterNet runOffMain pattern): `SWIFT_APPROACHABLE_CONCURRENCY` would
    /// otherwise run the engine's synchronous `generate` on the main thread. `@concurrent` is a child hop, so the
    /// `MLXErrorGuard` task-local and cancellation carry across.
    @concurrent nonisolated private static func runOffMain(_ stage: any PipelineStage, _ input: Media,
                                                          progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try await stage.run(input, progress: progress)
    }

    // MARK: - Playback

    /// Play the recording from `seconds`. The file's bytes are read under its security scope and handed to
    /// `AVAudioPlayer(data:)`, so the player keeps them and the scope is not held while it plays.
    func play(from seconds: Double) {
        guard let audioURL else { return }
        stopPlayback()
        let accessed = audioURL.startAccessingSecurityScopedResource()
        defer { if accessed { audioURL.stopAccessingSecurityScopedResource() } }
        do {
            let data = try Data(contentsOf: audioURL)
            let player = try AVAudioPlayer(data: data)
            player.currentTime = min(max(seconds, 0), player.duration)
            self.player = player
            player.play()
        } catch {
            errorText = "The audio couldn't be played — the file may have moved or be in a format the player can't read."
        }
    }

    func stopPlayback() {
        player?.stop()
        player = nil
    }
}
