import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// Reusable run surface for any audio→text (ASR) stage. The model module supplies the
/// concrete `stage`; this view is model-agnostic glue (`plan-whisper-aiui.md` W2): pick an
/// audio file, Run, watch an indeterminate spinner (ASR only reports 0.1→1.0, so a
/// determinate bar would be misleading), then read and copy the transcript.
struct ASRRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let stage: any PipelineStage
    /// Optional stage rebuilder: when non-nil, the view shows a Language picker and rebuilds
    /// the stage with the chosen Whisper language code per run (nil code = auto-detect). MLX
    /// Whisper conditions on the language token — without one a weak model can transcribe into
    /// the wrong language. Runners that don't support language selection omit this and use `stage`.
    var rebuildStage: ((String?) -> any PipelineStage)? = nil

    @State private var model = ASRRunModel()
    @State private var audioURL: URL?
    @State private var showImporter = false
    /// Selected language code; default English is the safe choice (the underlying library has
    /// no real auto-detection, so `nil`/auto can yield wrong-language output on weak models).
    @State private var languageCode: String? = "en"

    /// Curated Whisper languages for the picker. `nil` code = auto-detect (last).
    private static let languageOptions: [(code: String?, name: String)] = [
        ("en", "English"), ("zh", "Chinese"), ("es", "Spanish"), ("fr", "French"),
        ("de", "German"), ("it", "Italian"), ("pt", "Portuguese"), ("nl", "Dutch"),
        ("ru", "Russian"), ("ja", "Japanese"), ("ko", "Korean"), ("ar", "Arabic"),
        ("hi", "Hindi"), ("tr", "Turkish"), (nil, "Auto-detect"),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            inputs
            if model.isRunning { runningRow }
            if let transcript = model.transcript { resultView(transcript) }
            if let error = model.errorText { errorBar(error) }
            Spacer(minLength: 0)
            footer
        }
        .padding(18)
        .frame(width: 560, height: 640)
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.audio],
                      allowsMultipleSelection: false) { result in
            if case let .success(urls) = result { audioURL = urls.first }
        }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "waveform")
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Transcribe audio → text").font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var inputs: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button("Choose Audio…") { showImporter = true }
                Text(audioURL?.lastPathComponent ?? "No file selected")
                    .font(.callout).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            }
            if rebuildStage != nil { languageControl }
            // License shown before Run (plan §8).
            Text(license ?? "License: see model page")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    /// Language picker, shown only when the runner supports language selection.
    private var languageControl: some View {
        HStack {
            Text("Language").frame(width: 70, alignment: .leading)
            Picker("", selection: $languageCode) {
                ForEach(Self.languageOptions, id: \.code) { option in
                    Text(option.name).tag(option.code)
                }
            }
            .labelsHidden()
            .frame(maxWidth: 200)
            Spacer()
        }
    }

    private var runningRow: some View {
        HStack(spacing: 8) {
            ProgressView().scaleEffect(0.6).frame(width: 16, height: 16)
            Text("Transcribing…").font(.callout).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private func resultView(_ transcript: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Label("Transcript", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green).font(.subheadline.bold())
                Spacer()
                Button("Copy") { copy(transcript) }
            }
            ScrollView {
                Text(transcript).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 220)
            .padding(8)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private func errorBar(_ message: String) -> some View {
        RunErrorBar(message: message)
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            Button(model.isRunning ? "Transcribing…" : "Run") { run() }
                .keyboardShortcut(.defaultAction)
                .disabled(audioURL == nil || model.isRunning)
        }
    }

    // MARK: - Actions

    private func run() {
        guard let url = audioURL else { return }
        // Rebuild the stage with the chosen language when the runner supports it; otherwise
        // use the pre-built stage as-is.
        let activeStage = rebuildStage?(languageCode) ?? stage
        model.start(audioURL: url, stage: activeStage)
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

/// View-model driving one ASR run: reads the chosen file to a buffer, runs the stage
/// off-actor, and folds the transcript back onto the main actor. Reads the file directly
/// via `AudioFileReader`.
@MainActor
@Observable
final class ASRRunModel {
    var isRunning = false
    var transcript: String?
    var errorText: String?

    func start(audioURL: URL, stage: any PipelineStage) {
        isRunning = true
        transcript = nil
        errorText = nil

        let accessed = audioURL.startAccessingSecurityScopedResource()
        Task {
            defer { if accessed { audioURL.stopAccessingSecurityScopedResource() } }
            do {
                let buffer = try AudioFileReader.read(audioURL)
                let result = try await stage.run(.audio(buffer)) { _ in }
                if case let .text(text) = result { self.transcript = text }
                self.isRunning = false
            } catch {
                self.errorText = "\(error)"
                self.isRunning = false
            }
        }
    }
}
