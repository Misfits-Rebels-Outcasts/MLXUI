import SwiftUI
import AppKit

/// `ModelUI` for TranslateGemma. Presents the **Translate** screen (owner decision 4): From (Auto + every language),
/// To (every language), a swap button, a text box, a read-only result with Copy, Translate and Stop, and the live
/// token count against the model's 2K input limit. No chat history, no system prompt, no attachments.
struct TranslateGemmaUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        TranslateGemmaSDK().claim(model)
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        let footprint = Int64(model.ramGB * 1_073_741_824)
        return AnyView(TranslateGemmaRunView(
            modelDisplayName: model.displayName,
            license: model.license,
            licenseURL: model.licenseUrl,
            modelID: model.id,
            handedStage: stage,
            model: TranslateGemmaRunModel(
                modelID: model.id,
                displayName: model.displayName,
                modelDirectory: TranslateGemmaEngine.modelDirectory(for: model.id),
                footprintBytes: footprint)))
    }
}

/// The Translate screen. The view binds to ``TranslateGemmaRunModel`` and does nothing else; every decision the model
/// can make (which codes go out, what is refused, what is shown as detected) lives there and is unit-tested.
struct TranslateGemmaRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let licenseURL: String?
    let modelID: String
    /// SC-5c: the stage the Run sheet handed (a `LeasingStage`); each run gets its own stage with its two codes.
    let handedStage: any PipelineStage

    @State private var model: TranslateGemmaRunModel

    /// The app's path. The model is a render-test seam (`SAMAudioRunView`'s pattern) so a snapshot test can capture the
    /// filled and result states without a GUI.
    @MainActor init(modelDisplayName: String, license: String?, licenseURL: String?, modelID: String,
                    handedStage: any PipelineStage, model: TranslateGemmaRunModel) {
        self.modelDisplayName = modelDisplayName
        self.license = license
        self.licenseURL = licenseURL
        self.modelID = modelID
        self.handedStage = handedStage
        _model = State(initialValue: model)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            languages
            sourceBox
            if model.isRunning { runningRow } else { countRow }
            if let error = model.errorText { RunErrorBar(message: error) }
            resultBox
            Spacer(minLength: 0)
            footer
        }
        .padding(18)
        .frame(width: 660, height: 640)
        .onChange(of: model.sourceText) { _, _ in model.sourceTextChanged() }
        .onChange(of: model.targetSelection) { _, _ in model.targetSelectionChanged() }
        .task { await model.refreshTokenCount() }
        .onDisappear { model.cancel() }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "translate").foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Translate text between languages: pick the two languages, paste the text, and read the translation.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var languages: some View {
        HStack(alignment: .bottom, spacing: 10) {
            VStack(alignment: .leading, spacing: 4) {
                Text("From").font(.caption.weight(.medium)).foregroundStyle(.secondary)
                Picker("From", selection: $model.sourceSelection) {
                    Text("Auto (detect)").tag(TranslateGemmaRunModel.auto)
                    favouritesSection
                    allLanguagesSection
                }
                .labelsHidden()
                .frame(width: 200)
            }
            Button {
                model.swap()
            } label: {
                Image(systemName: "arrow.left.arrow.right")
            }
            .help("Swap the languages")
            .disabled(!model.canSwap)
            VStack(alignment: .leading, spacing: 4) {
                Text("To").font(.caption.weight(.medium)).foregroundStyle(.secondary)
                Picker("To", selection: $model.targetSelection) {
                    favouritesSection
                    allLanguagesSection
                }
                .labelsHidden()
                .frame(width: 200)
            }
            Spacer()
            if let detected = model.detected, model.sourceSelection == TranslateGemmaRunModel.auto {
                Label("Detected: \(detected.name)", systemImage: "character.bubble")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private var favouritesSection: some View {
        Section("Favourites") {
            ForEach(TranslateGemmaLanguages.favourites) { language in
                Text(language.name).tag(language.code)
            }
        }
    }

    @ViewBuilder private var allLanguagesSection: some View {
        Section("All languages") {
            ForEach(model.otherLanguages) { language in
                Text(language.name).tag(language.code)
            }
        }
    }

    private var sourceBox: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Text").font(.callout.weight(.medium))
            TextEditor(text: $model.sourceText)
                .font(.body)
                .frame(minHeight: 140)
                .overlay {
                    RoundedRectangle(cornerRadius: 6)
                        .strokeBorder(.secondary.opacity(0.35))
                        .allowsHitTesting(false)
                }
                .disabled(model.isRunning)
            Text("Auto reads the language from the text on this Mac; if it can't tell, pick the source language yourself.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var countRow: some View {
        HStack {
            Spacer()
            Text(model.tokenCountLabel)
                .font(.caption.monospacedDigit())
                .foregroundStyle(model.isOverLimit ? .red : .secondary)
        }
    }

    private var runningRow: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            Text("Translating…").font(.callout).foregroundStyle(.secondary)
            Spacer()
            Button("Stop") { model.cancel() }
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private var resultBox: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Translation").font(.callout.weight(.medium))
                Spacer()
                Button("Copy") { model.copyResult() }
                    .disabled(!model.canCopy)
            }
            ScrollView {
                Text(model.resultText ?? (model.didStop ? "Stopped." : "The translation appears here."))
                    .font(.body)
                    .foregroundStyle(model.resultText == nil ? .secondary : .primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(6)
            }
            .frame(minHeight: 100)
            .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 6))
            HStack(spacing: 4) {
                Text(licenseNotice).font(.caption).foregroundStyle(.secondary)
                if let licenseURL, let url = URL(string: licenseURL) {
                    Link("Read the license", destination: url).font(.caption)
                }
            }
        }
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            Button(model.isRunning ? "Translating…" : "Translate") { translate() }
                .keyboardShortcut(.defaultAction)
                .disabled(!model.canTranslate)
        }
    }

    // MARK: - Copy

    private var licenseNotice: String {
        guard let license else { return "License: see the model page." }
        return "License: \(DataNormalizer.normalizeLicense(license)) — the model page links the full terms."
    }

    // MARK: - Actions

    private func translate() {
        guard let source = model.resolvedSourceCode,
              let target = model.targetLanguage?.code else { return }
        // One stage per run, carrying this run's two codes; the handed stage keeps the engine-cache identity.
        let stage = LeasingStage.perRun(handedStage,
                                        config: StageConfig(sourceLanguage: source, targetLanguage: target)) { _ in
            model.makeStage(source, target)
        }
        model.start(stage: stage)
    }
}

/// The Translate screen's view model: the two language selections, Auto detection, the live rendered-token count, and
/// one translation run. `@MainActor @Observable` — the view only binds to it. The stage seam is `makeStage`; tests
/// inject a stub stage and a stub counter, so none of this needs weights.
@MainActor @Observable
final class TranslateGemmaRunModel {

    static let auto = "auto"
    /// Implementer's call, pending owner confirmation: a fresh screen translates *into* English until the person
    /// changes it — the app's own interface language.
    static let defaultTarget = "en"

    let modelID: String
    let displayName: String
    let modelDirectory: URL
    let footprintBytes: Int64

    /// A language code from `TranslateGemmaLanguages`, or ``auto``.
    var sourceSelection = auto
    /// A language code from `TranslateGemmaLanguages` (never ``auto``).
    var targetSelection = defaultTarget
    var sourceText = ""
    var resultText: String?
    var detected: TranslateGemmaLanguageDetector.Detection?
    var isRunning = false
    var didStop = false
    var errorText: String?
    /// The rendered request's token count, or `nil` while it cannot be rendered (no source yet, or an empty text).
    var promptTokenCount: Int?

    /// The app's defaults read the real on-device detector and tokenizer; tests inject stubs.
    @ObservationIgnored var detect: @Sendable (String) -> TranslateGemmaLanguageDetector.Detection? =
        { TranslateGemmaLanguageDetector.detect($0) }
    @ObservationIgnored var countTokens: @Sendable (TranslateGemmaEngine.Request) async -> Int? = { _ in nil }
    @ObservationIgnored var makeStage: @Sendable (String, String) -> any PipelineStage = { source, target in
        TranslateGemmaStage(id: "unbound", name: "TranslateGemma", sourceCode: source, targetCode: target) { _ in
            throw TranslateGemmaError.missingLanguages
        }
    }
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var countTask: Task<Void, Never>?

    init(modelID: String, displayName: String, modelDirectory: URL, footprintBytes: Int64? = nil) {
        self.modelID = modelID
        self.displayName = displayName
        self.modelDirectory = modelDirectory
        self.footprintBytes = footprintBytes ?? 0
        let directory = modelDirectory
        let footprint = footprintBytes ?? 0
        countTokens = { request in
            try? await TranslateGemmaTokenCounter.shared.tokenCount(for: request, modelDirectory: directory)
        }
        makeStage = { source, target in
            TranslateGemmaStage(modelID: modelID, displayName: displayName, sourceCode: source, targetCode: target,
                                footprintBytes: footprint)
        }
    }

    // MARK: - Derived state

    static var tokenLimit: Int { TranslateGemmaLanguages.maxPromptTokens }

    /// All languages except the favourites block (the pickers show the favourites above them, and a duplicated tag
    /// would make the selection ambiguous).
    var otherLanguages: [TranslateGemmaLanguages.Language] {
        let favouriteCodes = Set(TranslateGemmaLanguages.favourites.map(\.code))
        return TranslateGemmaLanguages.all.filter { !favouriteCodes.contains($0.code) }
    }

    var targetLanguage: TranslateGemmaLanguages.Language? {
        TranslateGemmaLanguages.language(matching: targetSelection)
    }

    /// The code the request will send: the chosen one, or the detected one under Auto.
    var resolvedSourceCode: String? {
        if sourceSelection != Self.auto { return TranslateGemmaLanguages.requestCode(for: sourceSelection) }
        return detected?.code
    }

    var sourceLabel: String {
        if sourceSelection == Self.auto {
            return detected.map { "Auto (\($0.name))" } ?? "Auto (detect)"
        }
        return TranslateGemmaLanguages.language(matching: sourceSelection)?.name ?? sourceSelection
    }

    var targetLabel: String { targetLanguage?.name ?? targetSelection }

    var canSwap: Bool {
        sourceSelection == Self.auto ? detected != nil : true
    }

    var isOverLimit: Bool { (promptTokenCount ?? 0) > Self.tokenLimit }

    var tokenCountLabel: String {
        guard let promptTokenCount else { return "— / \(Self.tokenLimit) tokens" }
        return "\(promptTokenCount) / \(Self.tokenLimit) tokens"
    }

    var canCopy: Bool { !isRunning && resultText != nil }

    var canTranslate: Bool {
        guard !isRunning else { return false }
        guard !sourceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return false }
        guard resolvedSourceCode != nil, targetLanguage != nil else { return false }
        return !isOverLimit
    }

    // MARK: - Selections

    /// Swap the two languages. Under Auto the source is what was detected, so a swap needs a detection.
    func swap() {
        if sourceSelection == Self.auto {
            guard let detected else { return }
            let oldTarget = targetSelection
            targetSelection = detected.code
            sourceSelection = oldTarget
            self.detected = nil
        } else {
            let oldSource = sourceSelection
            sourceSelection = targetSelection
            targetSelection = oldSource
        }
        sourceTextChanged()
    }

    func sourceTextChanged() {
        refreshDetection()
        scheduleTokenCount()
    }

    func targetSelectionChanged() {
        scheduleTokenCount()
    }

    /// Auto detection is a suggestion: `nil` here means the run asks the person to pick rather than guessing.
    func refreshDetection() {
        guard sourceSelection == Self.auto else {
            detected = nil
            return
        }
        let text = sourceText.trimmingCharacters(in: .whitespacesAndNewlines)
        detected = text.isEmpty ? nil : detect(text)
    }

    // MARK: - Token count (decision 8)

    private func scheduleTokenCount() {
        countTask?.cancel()
        countTask = Task { await refreshTokenCount() }
    }

    /// Count the rendered request with the model's own tokenizer. `nil` while it can't be rendered yet.
    func refreshTokenCount() async {
        let trimmed = sourceText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let source = resolvedSourceCode,
              let target = targetLanguage?.code,
              !trimmed.isEmpty else {
            promptTokenCount = nil
            return
        }
        let request = TranslateGemmaEngine.Request(text: sourceText, sourceCode: source, targetCode: target)
        let count = await countTokens(request)
        guard !Task.isCancelled else { return }
        promptTokenCount = count
    }

    // MARK: - One run

    func start(stage: any PipelineStage) {
        errorText = nil
        didStop = false
        resultText = nil
        guard !sourceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            errorText = TranslateGemmaError.emptyText.description
            return
        }
        guard let source = resolvedSourceCode, let target = targetLanguage?.code else {
            errorText = "I couldn't tell which language this is. Pick the source language and try again."
            return
        }
        if isOverLimit {
            errorText = TranslateGemmaError.promptTooLong(tokens: promptTokenCount ?? 0).description
            return
        }
        // The engine validates too (and refuses before rendering); this is only so a code-picker mistake is a sentence.
        guard TranslateGemmaLanguages.canonicalTemplateCode(source) != nil,
              TranslateGemmaLanguages.canonicalTemplateCode(target) != nil else {
            errorText = TranslateGemmaError.unknownLanguage(source).description
            return
        }
        isRunning = true
        let text = sourceText
        task = Task {
            do {
                let media = try await MLXErrorGuard.guardedRun {
                    try await Self.runStageOffMain(stage, .text(text))
                }
                guard case .text(let translated) = media else { throw TranslateGemmaError.noTranslation }
                guard !Task.isCancelled else {
                    self.finishCancelled()
                    return
                }
                self.resultText = translated
                self.isRunning = false
            } catch {
                if error is CancellationError || Task.isCancelled {
                    self.finishCancelled()
                } else {
                    self.errorText = MLXErrorGuard.sentence(for: error)
                    self.isRunning = false
                }
            }
        }
    }

    func cancel() {
        task?.cancel()
    }

    private func finishCancelled() {
        isRunning = false
        didStop = true
        resultText = nil
        errorText = nil
    }

    /// Runs the stage off the main actor (a synchronous engine loop on the main thread would beachball, and Stop would
    /// be unclickable): `@concurrent` is a child hop, so MLX-ERR-2's task-local guard and cancellation carry across.
    @concurrent nonisolated private static func runStageOffMain(_ stage: any PipelineStage,
                                                                _ input: Media) async throws -> Media {
        try await stage.run(input) { _ in }
    }

    // MARK: - Result

    func copyResult() {
        guard let resultText, canCopy else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(resultText, forType: .string)
    }
}
