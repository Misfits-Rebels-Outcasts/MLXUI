import SwiftUI
import UniformTypeIdentifiers

/// Run surface for non-Kokoro MLX text-to-speech (Qwen3-TTS, chatterbox, orpheus). Type
/// text, choose a **voice** (preset/free-form) or a **reference clip** for cloning families,
/// Synthesize → auto-play, Play again, Save WAV. Builds a fresh `TTSStage` backed by
/// `MLXAudioTTSEngine` per run and reuses the model-agnostic `TTSRunModel` (defined in
/// `TTSRunView.swift`) for playback/save. `plan-nonchat-aiui.md` AT1/AT2.
struct MLXAudioTTSRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let modelID: String
    let family: String

    @State private var model = TTSRunModel()
    @State private var text = "Hello! This is text-to-speech running locally with MLX."
    /// Empty = the model's default speaker (`nil`); otherwise a preset or free-form voice id.
    @State private var voice = ""
    /// Reference clip for zero-shot cloning families (chatterbox).
    @State private var refAudioURL: URL?
    @State private var showAudioImporter = false

    private var presets: [String] { MLXAudioTTSVoices.presets(family: family) }
    private var usesReferenceAudio: Bool { MLXAudioTTSVoices.supportsReferenceAudio(family: family) }

    /// Cloning families require a reference clip; the underlying engine throws without one.
    private var canSynthesize: Bool {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return false }
        guard !model.isRunning else { return false }
        if usesReferenceAudio && refAudioURL == nil { return false }
        return true
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            inputs
            voiceControl
            Text(license ?? "License: see model page")
                .font(.caption).foregroundStyle(.secondary)
            if model.isRunning { runningRow }
            if model.didSynthesize { resultRow }
            if let error = model.errorText { errorBar(error) }
            Spacer(minLength: 0)
            footer
        }
        .padding(18)
        .frame(width: 560, height: 600)
        .fileImporter(isPresented: $showAudioImporter, allowedContentTypes: [.audio],
                      allowsMultipleSelection: false) { result in
            if case let .success(urls) = result { refAudioURL = urls.first }
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

    /// Reference-clip picker for cloning families (chatterbox); a preset picker for families
    /// with documented voices (orpheus); a free-form field otherwise (Qwen3-TTS).
    @ViewBuilder private var voiceControl: some View {
        if usesReferenceAudio {
            HStack {
                Text("Voice clip").frame(width: 70, alignment: .leading)
                Button("Choose Audio…") { showAudioImporter = true }
                Text(refAudioURL?.lastPathComponent ?? "Required for voice cloning")
                    .font(.callout)
                    .foregroundStyle(refAudioURL == nil ? .red : .secondary)
                    .lineLimit(1).truncationMode(.middle)
                Spacer()
            }
        } else if presets.isEmpty {
            HStack {
                Text("Voice").frame(width: 70, alignment: .leading)
                TextField("model default", text: $voice)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 240)
                Spacer()
            }
        } else {
            HStack {
                Text("Voice").frame(width: 70, alignment: .leading)
                Picker("", selection: $voice) {
                    Text("Default").tag("")
                    ForEach(presets, id: \.self) { Text($0).tag($0) }
                }
                .labelsHidden()
                .frame(maxWidth: 240)
                Spacer()
            }
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
                .disabled(!canSynthesize)
        }
    }

    private func run() {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let directory = MLXAudioTTSEngine.installedModelDirectory(id: modelID)

        if usesReferenceAudio {
            // Read the required reference clip (security-scoped). `canSynthesize` guarantees
            // a URL is set; cloning engines (chatterbox) throw without a clip.
            let refBuffer: AudioBuffer? = refAudioURL.flatMap { url in
                let accessed = url.startAccessingSecurityScopedResource()
                defer { if accessed { url.stopAccessingSecurityScopedResource() } }
                return try? AudioFileReader.read(url)
            }
            let stage = TTSStage(id: "mlx-audio-tts.\(modelID)", name: "TTS (\(family))") { text in
                try await MLXAudioTTSEngine.synthesize(
                    text, voice: nil, referenceAudio: refBuffer, modelDirectory: directory)
            }
            model.synthesize(text: trimmed, stage: stage)
        } else {
            let chosen = voice.trimmingCharacters(in: .whitespacesAndNewlines)
            let resolvedVoice = chosen.isEmpty ? nil : chosen
            let stage = TTSStage(id: "mlx-audio-tts.\(modelID)", name: "TTS (\(family))") { text in
                try await MLXAudioTTSEngine.synthesize(
                    text, voice: resolvedVoice, modelDirectory: directory)
            }
            model.synthesize(text: trimmed, stage: stage)
        }
    }
}
