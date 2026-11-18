import Foundation
import AppKit
import UniformTypeIdentifiers

/// PF-7 — the Redact Run screen's view model. `@MainActor @Observable`; the view only binds to it.
///
/// Owner decision 6, in one place:
///  * paste text, or open a `.txt`/`.md` file (or drop one on the window) — the text is read as UTF-8;
///  * **Redact** with a working **Stop** (cancellation is observed between windows, so the app never freezes);
///  * side by side: the original with every detected span, and the redacted text;
///  * eight category checkboxes, all on by default. They change **only which spans are replaced**: toggling one
///    re-renders the redacted text from the cached spans and never calls the model again;
///  * **Copy redacted**, **Save .txt**, **Export spans .json** (UTF-8 byte offsets);
///  * errors are one plain sentence and never quote the input.
///
/// The stage/box seams are how the tests drive it with no weights: `makeStage` builds the per-run stage, and the stage
/// hands its ``PrivacyFilterEngine/Outcome`` back through a ``RedactionBox`` (the pipeline boundary is text-only, so the
/// spans have no other way back to the view).
@MainActor @Observable
final class RedactRunModel {

    /// The result one run's stage produces. One box per run; written on the stage's thread, read on the main actor after
    /// the run returns.
    final class RedactionBox: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: PrivacyFilterEngine.Outcome?

        func store(_ outcome: PrivacyFilterEngine.Outcome) {
            lock.lock(); stored = outcome; lock.unlock()
        }

        var outcome: PrivacyFilterEngine.Outcome? {
            lock.lock(); defer { lock.unlock() }
            return stored
        }
    }

    let modelID: String
    let displayName: String
    let modelDirectory: URL
    let footprintBytes: Int64

    /// The text in the editor. Editable except while a run is in flight.
    var inputText = ""
    /// Which categories are replaced in the redacted text. All eight by default (decision 6).
    var selectedCategories: Set<PrivacyFilterCategory> = Set(PrivacyFilterCategory.allCases)

    private(set) var spans: [PrivacyFilterSpan] = []
    private(set) var redactedText: String?
    private(set) var tokenCount: Int?
    /// The text the cached `spans` were detected in — the checkbox re-render works on this, never on a newer edit.
    private(set) var resultSourceText = ""
    private(set) var isRunning = false
    private(set) var didStop = false
    private(set) var errorText: String?
    /// The last opened file's display name, for the header (never its contents).
    private(set) var sourceName: String?

    /// The engine's stage builder for a per-run config; the default binds the installed folder, and tests inject a stub.
    @ObservationIgnored var makeStage: @Sendable (StageConfig, RedactionBox) -> any PipelineStage
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var accessingScopedResource = false
    @ObservationIgnored private var sourceURL: URL?

    init(modelID: String, displayName: String, modelDirectory: URL, footprintBytes: Int64? = nil) {
        self.modelID = modelID
        self.displayName = displayName
        self.modelDirectory = modelDirectory
        self.footprintBytes = footprintBytes ?? 0
        let directory = modelDirectory
        let footprint = footprintBytes ?? 0
        makeStage = { config, box in
            PrivacyFilterStage(id: modelID, name: displayName, only: config.redactOnly) { text in
                let outcome = try await PrivacyFilterEngine.redact(
                    text, categories: config.redactOnly, modelDirectory: directory,
                    footprintBytes: footprint, holder: .runWindow)
                box.store(outcome)
                return outcome
            }
        }
    }

    // MARK: - Derived state

    static let tokenLimit = PrivacyFilterInference.maxTokens

    /// The verbatim caption from decision 6.
    static let caption = "Finds common personal data, mainly in English. It can miss things, so review before you share. "
        + "Not an anonymisation or compliance guarantee."

    var hasResult: Bool { redactedText != nil }

    var canRedact: Bool {
        !isRunning && !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var canCopy: Bool { !isRunning && redactedText != nil }

    var canExportSpans: Bool { !isRunning && !spans.isEmpty }

    /// All eight selected == "replace everything"; passed to the engine/redactor as `nil` (decision 4's default).
    var effectiveCategories: Set<PrivacyFilterCategory>? {
        selectedCategories.count == PrivacyFilterCategory.allCases.count ? nil : selectedCategories
    }

    var selectedCategoryCount: Int { selectedCategories.count }

    // MARK: - Input

    /// Open a `.txt`/`.md` file as UTF-8 text. A read failure is one plain sentence (never the file's contents).
    func open(_ url: URL) {
        reset()
        releaseSource()
        sourceURL = url
        accessingScopedResource = url.startAccessingSecurityScopedResource()
        do {
            let text = try String(contentsOf: url, encoding: .utf8)
            inputText = text
            sourceName = url.lastPathComponent
        } catch {
            errorText = "That file couldn't be read as UTF-8 text. Save it as UTF-8 and try again."
        }
    }

    /// Drop a single `.txt`/`.md` file. Other types are refused with a sentence rather than silently ignored.
    func drop(_ urls: [URL]) {
        guard let url = urls.first else { return }
        let ext = url.pathExtension.lowercased()
        guard Self.readableExtensions.contains(ext) else {
            errorText = "Drop a .txt or .md file — other file types aren't read here."
            return
        }
        open(url)
    }

    static let readableExtensions: Set<String> = ["txt", "md"]

    func clear() {
        reset()
        releaseSource()
        inputText = ""
        sourceName = nil
    }

    // MARK: - One run

    /// Redact. `handedStage` is the Run sheet's stage (a `LeasingStage`, which books the shared-budget load and asks
    /// before pushing out a served model); the per-run stage this run actually executes carries the category selection.
    func start(handedStage: any PipelineStage) {
        reset()
        let text = inputText
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            errorText = PrivacyFilterInferenceError.emptyText.errorDescription
                ?? "There is no text to redact."
            return
        }
        isRunning = true
        let box = RedactionBox()
        let config = StageConfig(redactOnly: selectedCategories)
        let stage = LeasingStage.perRun(handedStage, config: config) { config in
            self.makeStage(config, box)
        }
        task = Task {
            do {
                let produced = try await MLXErrorGuard.guardedRun {
                    try await Self.runStageOffMain(stage, .text(text))
                }
                guard case .text(let redacted) = produced else {
                    throw StageError.kindMismatch(expected: .text, got: produced.kind)
                }
                guard !Task.isCancelled else {
                    self.finishCancelled()
                    return
                }
                let outcome = box.outcome
                self.spans = outcome?.spans ?? []
                self.tokenCount = outcome?.tokenCount
                self.resultSourceText = text
                self.redactedText = redacted
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

    /// Stop. The run is cancelled; the stage's `redaction` observes it between windows and throws `CancellationError`.
    func stop() {
        task?.cancel()
    }

    func teardown() {
        task?.cancel()
        task = nil
        releaseSource()
    }

    private func finishCancelled() {
        isRunning = false
        didStop = true
        redactedText = nil
        tokenCount = nil
        errorText = nil
    }

    private func reset() {
        task?.cancel()
        task = nil
        isRunning = false
        didStop = false
        errorText = nil
        redactedText = nil
        spans = []
        tokenCount = nil
        resultSourceText = ""
    }

    private func releaseSource() {
        if accessingScopedResource { sourceURL?.stopAccessingSecurityScopedResource() }
        accessingScopedResource = false
        sourceURL = nil
    }

    /// Runs the stage off the main actor (the engine is a synchronous MLX loop: on the main thread the window
    /// would beachball and Stop would be unclickable). `@concurrent` is a child hop, so task-local guards and
    /// cancellation carry across.
    @concurrent nonisolated private static func runStageOffMain(_ stage: any PipelineStage,
                                                                _ input: Media) async throws -> Media {
        try await stage.run(input) { _ in }
    }

    // MARK: - Category checkboxes (decision 6: re-render, never re-run)

    /// A box was toggled. Rebuild the redacted text from the cached spans — no model call, no tokenization.
    func categoriesChanged() {
        guard hasResult else { return }
        redactedText = PrivacyFilterRedactor.redactedText(
            resultSourceText, spans: spans, categories: effectiveCategories)
    }

    // MARK: - Copy, save, export

    func copyRedacted() {
        guard let redactedText, canCopy else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(redactedText, forType: .string)
    }

    func saveRedacted() {
        guard let redactedText, canCopy else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText]
        panel.nameFieldStringValue = Self.suggestedFileName(sourceName: sourceName, suffix: "redacted", ext: "txt")
        guard panel.runModal() == .OK, let target = panel.url else { return }
        do {
            try redactedText.write(to: target, atomically: true, encoding: .utf8)
        } catch {
            errorText = "The redacted text couldn't be saved there. Pick another folder and try again."
        }
    }

    func exportSpans() {
        guard canExportSpans else { return }
        let data = Self.spansJSON(spans)
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = Self.suggestedFileName(sourceName: sourceName, suffix: "spans", ext: "json")
        guard panel.runModal() == .OK, let target = panel.url else { return }
        do {
            try data.write(to: target)
        } catch {
            errorText = "The span list couldn't be saved there. Pick another folder and try again."
        }
    }

    /// One record per detected span: its category names and the **UTF-8 byte offsets** into the original input
    /// (decision 4 — the UI counts Swift `Character`s, the exported JSON counts bytes). Sorted by start offset, so a
    /// consumer can walk the file independently of detection order.
    nonisolated static func spansJSON(_ spans: [PrivacyFilterSpan]) -> Data {
        struct Record: Encodable {
            let category: String
            let label: String
            let start: Int
            let end: Int
            let text: String
        }
        struct Document: Encodable {
            let version: Int
            let encoding: String
            let spans: [Record]
        }
        let ordered = spans.sorted { $0.utf8Start == $1.utf8Start ? $0.utf8End < $1.utf8End : $0.utf8Start < $1.utf8Start }
        let document = Document(
            version: 1,
            encoding: "utf-8-bytes",
            spans: ordered.map {
                Record(category: $0.category.rawValue, label: $0.category.modelLabel,
                       start: $0.utf8Start, end: $0.utf8End, text: $0.text)
            })
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        // Encoding a value this shape cannot fail; the fallback keeps the signature total for the view.
        return (try? encoder.encode(document)) ?? Data("{\"spans\":[]}".utf8)
    }

    nonisolated static func suggestedFileName(sourceName: String?, suffix: String, ext: String) -> String {
        guard let sourceName, !sourceName.isEmpty else { return "\(suffix).\(ext)" }
        let stem = (sourceName as NSString).deletingPathExtension
        return "\(stem.isEmpty ? suffix : stem)-\(suffix).\(ext)"
    }
}
