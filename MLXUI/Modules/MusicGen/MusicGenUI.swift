import SwiftUI
import AppKit
import AVFoundation
import UniformTypeIdentifiers

/// `ModelUI` for MusicGen text → music. Presents `MusicGenRunView` (prompt + duration + Run →
/// progress → playback + Save WAV). Same claim predicate as `MusicGenSDK`. MG-UI1.
struct MusicGenUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .music, model.source == .mlx else { return .no }
        let haystack = (model.id + " " + model.hfModelId).lowercased()
        guard haystack.contains("musicgen") else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(MusicGenRunView(modelDisplayName: model.displayName,
                                license: model.license,
                                modelID: model.id))
    }
}

/// The run surface for MusicGen text → music: a prompt, a duration picker, Run → live progress
/// → inline playback + Save WAV.
struct MusicGenRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let modelID: String

    @State private var model = MusicGenRunModel()
    @State private var prompt = "A cheerful acoustic guitar melody with light percussion."
    @State private var durationIndex = 0

    /// (maxSteps, label) — 250 ≈ 5 s, 500 ≈ 10 s, 1000 ≈ 20 s at 50 codec frames/s.
    private let durations: [(Int, String)] = [
        (250, "Short (~5 s)"),
        (500, "Medium (~10 s)"),
        (1000, "Long (~20 s)"),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            promptField
            options
            if model.isRunning { progressRow }
            if let error = model.errorText { errorBar(error) }
            if model.didGenerate { resultRow }
            Spacer(minLength: 0)
            footer
        }
        .padding(18)
        .frame(width: 600, height: 660)
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "music.note").foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Text → music (autoregressive)").font(.caption).foregroundStyle(.secondary)
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
                Text("Duration").frame(width: 60, alignment: .leading)
                Picker("", selection: $durationIndex) {
                    ForEach(0 ..< durations.count, id: \.self) { i in
                        Text(durations[i].1).tag(i)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: 240)
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
            Text(model.progressLabel)
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private var resultRow: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                Label("Ready", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green).font(.subheadline.bold())
                Button { model.togglePlayback() } label: {
                    Label(model.isPlaying ? "Pause" : "Play", systemImage: model.isPlaying ? "pause.fill" : "play.fill")
                }
                Button { save() } label: {
                    Label("Save WAV…", systemImage: "square.and.arrow.down")
                }
                Text(model.durationLabel)
                    .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                Spacer()
            }
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private func errorBar(_ message: String) -> some View {
        RunErrorBar(message: message)
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            Button(model.isRunning ? "Generating…" : "Generate") { run() }
                .keyboardShortcut(.defaultAction)
                .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || model.isRunning)
        }
    }

    // MARK: - Actions

    private func run() {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let maxSteps = durations[durationIndex].0
        let stage = MusicGenStage(modelID: modelID, maxSteps: maxSteps)
        model.generate(prompt: trimmed, stage: stage)
    }

    private func save() {
        model.save()
    }
}

/// View-model driving one MusicGen run: runs the stage off-actor, folds progress back, then
/// plays the result inline and offers a Save WAV. Mirrors `TTSRunModel`/`FluxRunModel`.
@MainActor
@Observable
final class MusicGenRunModel: NSObject, AVAudioPlayerDelegate {
    var isRunning = false
    var progress = 0.0
    var progressLabel = ""
    var errorText: String?
    var didGenerate = false
    var isPlaying = false
    var durationLabel = ""

    private var buffer: AudioBuffer?
    private var player: AVAudioPlayer?

    func generate(prompt: String, stage: any PipelineStage) {
        isRunning = true
        progress = 0.0
        didGenerate = false
        errorText = nil
        isPlaying = false
        player?.stop()
        player = nil

        Task {
            do {
                let result = try await stage.run(.text(prompt)) { fraction in
                    Task { @MainActor in
                        self.progress = fraction
                        self.progressLabel = Self.label(for: fraction)
                    }
                }
                guard case let .audio(audio) = result else {
                    throw StageError.kindMismatch(expected: .audio, got: result.kind)
                }
                self.buffer = audio
                self.didGenerate = true
                self.isRunning = false
                self.progress = 1.0
                self.durationLabel = String(format: "%.1f s", Double(audio.samples.count) / Double(audio.sampleRate))
            } catch {
                self.errorText = "\(error)"
                self.isRunning = false
            }
        }
    }

    func togglePlayback() {
        guard let buffer else { return }
        if let player, player.isPlaying {
            player.pause()
            isPlaying = false
            return
        }
        do {
            let player = try AVAudioPlayer(data: AudioWriter.wavData(buffer))
            player.delegate = self          // reset `isPlaying` when playback completes
            self.player = player            // retain so it isn't deallocated mid-play
            player.play()
            isPlaying = true
        } catch {
            errorText = "Playback failed: \(error.localizedDescription)"
        }
    }

    // MARK: - AVAudioPlayerDelegate

    /// Playback reached the end — flip the button back to "Play" (the toggle otherwise sticks
    /// on "Pause" because `AVAudioPlayer.isPlaying` only reflects an in-flight `play()`).
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        isPlaying = false
    }

    func save() {
        guard let buffer else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.wav]
        panel.nameFieldStringValue = "musicgen.wav"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try AudioWriter.writeWAV(buffer, to: url)
        } catch {
            errorText = "Save failed: \(error.localizedDescription)"
        }
    }

    private nonisolated static func label(for fraction: Double) -> String {
        switch fraction {
        case ..<0.15: return "Loading weights…"
        case ..<0.25: return "Encoding prompt…"
        case ..<0.85: return "Generating music…"
        case ..<0.95: return "Decoding audio…"
        default:      return "Finalizing…"
        }
    }
}
