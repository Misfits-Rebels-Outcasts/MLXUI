import SwiftUI
import UniformTypeIdentifiers

/// Minimal run surface for the audio→summary MVP pipeline (M12): pick an audio file,
/// choose an installed LLM, Run, watch per-stage progress, read the summary.
struct PipelineRunView: View {
    @Environment(AppState.self) private var appState
    @Environment(\.dismiss) private var dismiss

    @State private var model = PipelineRunModel()
    @State private var audioURL: URL?
    @State private var selectedLLM: ModelEntry?
    @State private var showImporter = false

    private var llms: [ModelEntry] { appState.installedLLMEntries }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            inputs
            if !model.stageLabels.isEmpty { progressList }
            if let summary = model.resultText { resultView(summary) }
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
        .onAppear { selectedLLM = selectedLLM ?? llms.first }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "waveform.badge.magnifyingglass")
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text("Summarize Audio").font(.headline)
                Text("Read audio → transcribe → summarize").font(.caption)
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
            if llms.isEmpty {
                Label("Install an LLM first (Browse → install a chat model).", systemImage: "exclamationmark.triangle")
                    .font(.callout).foregroundStyle(.orange)
            } else {
                Picker("Summarize with", selection: $selectedLLM) {
                    ForEach(llms) { Text($0.displayName).tag(Optional($0)) }
                }
            }
            Text("Transcription uses WhisperKit “base” (downloads on first run).")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var progressList: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(model.stageLabels.enumerated()), id: \.offset) { index, label in
                HStack(spacing: 8) {
                    stageIcon(for: index)
                    Text(label).font(.callout)
                        .foregroundStyle(index == model.currentStageIndex ? .primary : .secondary)
                    Spacer()
                }
            }
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private func stageIcon(for index: Int) -> some View {
        if model.completed || index < (model.currentStageIndex ?? 0) {
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        } else if index == model.currentStageIndex && model.isRunning {
            ProgressView().scaleEffect(0.6).frame(width: 16, height: 16)
        } else {
            Image(systemName: "circle").foregroundStyle(.tertiary)
        }
    }

    private func resultView(_ summary: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Summary").font(.subheadline.bold())
            ScrollView {
                Text(summary).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 160)
            .padding(8)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private func errorBar(_ message: String) -> some View {
        Label(message, systemImage: "exclamationmark.triangle.fill")
            .font(.caption).foregroundStyle(.red)
            .padding(8).frame(maxWidth: .infinity, alignment: .leading)
            .background(.red.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            Button(model.isRunning ? "Running…" : "Run") { run() }
                .keyboardShortcut(.defaultAction)
                .disabled(audioURL == nil || selectedLLM == nil || model.isRunning)
        }
    }

    // MARK: - Run

    private func run() {
        guard let url = audioURL, let llm = selectedLLM else { return }
        model.start(audioURL: url, model: llm)
    }
}

/// View-model driving one pipeline run: builds the pipeline on the main actor, runs it
/// off-actor, and folds `PipelineEvent`s back onto the main actor for the UI.
@MainActor
@Observable
final class PipelineRunModel {
    var stageLabels: [String] = []
    var currentStageIndex: Int?
    var isRunning = false
    var completed = false
    var resultText: String?
    var errorText: String?

    func start(audioURL: URL, model: ModelEntry) {
        let outURL = AudioSummaryPipeline.defaultOutputURL()
        try? FileManager.default.createDirectory(
            at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        let pipeline = AudioSummaryPipeline.pipeline(for: model, saveTo: outURL)
        stageLabels = pipeline.stages.map(\.name)
        currentStageIndex = nil
        isRunning = true
        completed = false
        resultText = nil
        errorText = nil

        let accessed = audioURL.startAccessingSecurityScopedResource()
        Task {
            defer { if accessed { audioURL.stopAccessingSecurityScopedResource() } }
            do {
                let result = try await pipeline.run(.text(audioURL.path)) { event in
                    Task { @MainActor [weak self] in self?.apply(event) }
                }
                if case let .text(summary) = result { self.resultText = summary }
                self.completed = true
                self.isRunning = false
            } catch {
                self.errorText = "\(error)"
                self.isRunning = false
            }
        }
    }

    private func apply(_ event: PipelineEvent) {
        switch event {
        case let .stageStarted(index, _):
            currentStageIndex = index
        case let .failed(_, error):
            errorText = "\(error)"
        case .stageProgress, .stageFinished:
            break
        }
    }
}
