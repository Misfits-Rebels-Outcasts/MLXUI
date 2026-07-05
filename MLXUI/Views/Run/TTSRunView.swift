import SwiftUI
import AppKit
import AVFoundation
import UniformTypeIdentifiers

/// Run surface for Kokoro text-to-speech: type text, pick a **voice** and **speed**,
/// Synthesize, hear it (auto-plays), Play again, or Save a `.wav`. Builds a fresh
/// `TTSStage` per run with the chosen options. Mirrors `ASRRunView`.
struct TTSRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let modelID: String
    /// Voices found in the installed model (may be empty until installed).
    let voices: [String]

    @State private var model = TTSRunModel()
    @State private var text = "Hello! This is Kokoro text-to-speech running locally with MLX."
    @State private var voice = KokoroEngine.defaultVoice
    @State private var speed = 1.0

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            inputs
            options
            if model.isRunning { runningRow }
            if model.didSynthesize { resultRow }
            if let error = model.errorText { errorBar(error) }
            Spacer(minLength: 0)
            footer
        }
        .padding(18)
        .frame(width: 560, height: 640)
        .onAppear {
            // Default to af_heart when present, else the first installed voice.
            if !voices.contains(voice) { voice = voices.first ?? KokoroEngine.defaultVoice }
        }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "mouth")
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Text → speech").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var inputs: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Text to speak").font(.subheadline.bold())
            TextEditor(text: $text)
                .font(.body)
                .frame(minHeight: 110)
                .padding(6)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private var options: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Voice").frame(width: 60, alignment: .leading)
                if voices.isEmpty {
                    Text("No voices found — reinstall the model.")
                        .font(.callout).foregroundStyle(.orange)
                } else {
                    Picker("", selection: $voice) {
                        ForEach(voices, id: \.self) { Text($0).tag($0) }
                    }
                    .labelsHidden()
                    .frame(maxWidth: 240)
                }
                Spacer()
            }
            HStack {
                Text("Speed").frame(width: 60, alignment: .leading)
                Slider(value: $speed, in: 0.5...2.0, step: 0.05)
                    .frame(maxWidth: 240)
                Text(String(format: "%.2f×", speed))
                    .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                Spacer()
            }
            Text(license ?? "License: see model page")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var runningRow: some View {
        HStack(spacing: 8) {
            ProgressView().scaleEffect(0.6).frame(width: 16, height: 16)
            Text("Synthesizing…").font(.callout).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private var resultRow: some View {
        HStack(spacing: 12) {
            Label("Ready", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green).font(.subheadline.bold())
            Button { model.play() } label: { Label("Play", systemImage: "play.fill") }
            Button { model.save() } label: { Label("Save WAV…", systemImage: "square.and.arrow.down") }
            Spacer()
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
            Button(model.isRunning ? "Synthesizing…" : "Synthesize") { run() }
                .keyboardShortcut(.defaultAction)
                .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || model.isRunning)
        }
    }

    private func run() {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let stage = TTSStage(modelID: modelID, voice: voice, speed: Float(speed))
        model.synthesize(text: trimmed, stage: stage)
    }
}

/// View-model driving one TTS run: runs the stage off-actor, then plays the resulting audio
/// and offers a Save. Mirrors `ASRRunModel`.
@MainActor
@Observable
final class TTSRunModel {
    var isRunning = false
    var didSynthesize = false
    var errorText: String?

    private var buffer: AudioBuffer?
    private var player: AVAudioPlayer?

    func synthesize(text: String, stage: any PipelineStage) {
        isRunning = true
        didSynthesize = false
        errorText = nil

        Task {
            do {
                let result = try await stage.run(.text(text)) { _ in }
                guard case let .audio(audio) = result else {
                    throw StageError.kindMismatch(expected: .audio, got: result.kind)
                }
                self.buffer = audio
                self.didSynthesize = true
                self.isRunning = false
                self.play()   // auto-play the result
            } catch {
                self.errorText = "\(error)"
                self.isRunning = false
            }
        }
    }

    func play() {
        guard let buffer else { return }
        do {
            let player = try AVAudioPlayer(data: AudioWriter.wavData(buffer))
            self.player = player          // retain so it isn't deallocated mid-play
            player.play()
        } catch {
            errorText = "Playback failed: \(error.localizedDescription)"
        }
    }

    func save() {
        guard let buffer else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.wav]
        panel.nameFieldStringValue = "speech.wav"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try AudioWriter.writeWAV(buffer, to: url)
        } catch {
            errorText = "Save failed: \(error.localizedDescription)"
        }
    }
}
