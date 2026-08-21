import Foundation
import Observation

/// The visible half of a run (CFM-R2-8): an `@Observable` view-model that consumes the
/// `FlowEvent` stream and drives each row's dot `○ → ● → ✓`, shows the row's error sentence
/// on `✗`, exposes a cancel, and tracks the install sheet state. Held by `FlowListView`.
@Observable
final class FlowRunSession {
    /// Per-row run state, keyed by row id.
    struct RowState: Equatable {
        var status: FlowStatus = .notRun
        var errorSentence: String?
    }

    private(set) var rowStates: [UUID: RowState] = [:]
    /// The last finished output per row id (the inspector's content source).
    private(set) var outputs: [UUID: Asset] = [:]
    /// A `CatalogBridge` substitution note per row id (CFM-R2-2 rule 3), or absent.
    private(set) var substitutionNotes: [UUID: String] = [:]
    /// The row the inspector pane is bound to (nil = nothing selected).
    var selectedRowID: UUID?
    var isRunning = false
    var errorSentence: String?
    /// The preflight result that gates Run (models to install / blocked reason).
    var preflight: FlowPreflight.Result?
    var installPrompted = false
    var showInstallSheet = false

    private var runTask: Task<Void, Never>?

    // MARK: - Gating

    /// Whether Run is available: runnable, not already running, and no blocked models.
    var canRun: Bool {
        !isRunning && (preflight?.isBlocked ?? false) == false
    }

    /// The sentence explaining why Run is disabled, or nil when runnable.
    var runDisabledReason: String? {
        if isRunning { return nil }
        return preflight.flatMap { FlowPreflight.blockedReason($0, totalRAMGB: SystemInfo.detect().totalRAMGB) }
    }

    /// Dot state for a row id.
    func status(for rowID: UUID) -> FlowStatus {
        rowStates[rowID]?.status ?? .notRun
    }

    /// Whether any row has earned a result (any succeeded dot) — gates "re-run from here".
    var hasRunResults: Bool {
        rowStates.values.contains { $0.status == .succeeded }
    }

    /// Error sentence for a failed row.
    func errorSentence(for rowID: UUID) -> String? {
        rowStates[rowID]?.errorSentence
    }

    // MARK: - Start

    /// Start a run. When `resume` is true, the run starts at the first gray row (answer
    /// `f3`): rows already `✓` keep their dots and outputs, and Run re-executes only the
    /// gray rows. Otherwise the whole flow runs (dots reset).
    func start(doc: FlowDocument, runner: FlowRunner, context: FlowRunner.RunContext,
               resume: Bool = false) {
        guard !isRunning else { return }
        isRunning = true
        errorSentence = nil

        let startIndex = resume ? firstGrayIndex(in: doc) : 0
        let resumeOutputs = resume ? outputs : [:]
        if startIndex == 0 {
            rowStates = Dictionary(uniqueKeysWithValues: doc.rows.map { ($0.id, RowState()) })
        } else {
            // Keep the ✓ rows' states; gray out everything from startIndex down.
            for (id, var state) in rowStates {
                if isDownstream(id, of: startIndex, in: doc) {
                    state.status = .notRun
                    state.errorSentence = nil
                    rowStates[id] = state
                    outputs[id] = nil
                }
            }
        }

        let stream = runner.run(doc, context: context, startIndex: startIndex,
                                resumeOutputs: resumeOutputs)
        runTask = Task {
            for await event in stream {
                if Task.isCancelled { break }
                apply(event)
            }
            isRunning = false
        }
    }

    /// The index of the first non-`✓` row (the re-run-from-here boundary). A row is gray if
    /// it isn't `.succeeded` (covers `.notRun`, `.failed`, `.needsAttention`).
    private func firstGrayIndex(in doc: FlowDocument) -> Int {
        for (i, row) in doc.rows.enumerated() {
            if status(for: row.id) != .succeeded { return i }
        }
        return doc.rows.count   // everything done — a fresh run from the top
    }

    /// Whether `id` is at or after `fromIndex` in the flow's top-level rows.
    private func isDownstream(_ id: UUID, of fromIndex: Int, in doc: FlowDocument) -> Bool {
        guard let idx = doc.rows.firstIndex(where: { $0.id == id }) else { return false }
        return idx >= fromIndex
    }

    // MARK: - Clear

    /// Clear Run: reset every row's dot to gray and drop the inspector outputs. The cache
    /// store is untouched (Clear Cache is separate).
    func clearRun(doc: FlowDocument) {
        cancel()
        rowStates = Dictionary(uniqueKeysWithValues: doc.rows.map { ($0.id, RowState()) })
        outputs = [:]
        selectedRowID = nil
    }

    /// Clear Cache: drop every stored asset (entries + blobs) from the flow cache store.
    /// Always safe — clearing only costs recomputation. Returns the number of entries
    /// cleared, or nil when the store was already empty.
    func clearCache() -> Int? {
        let store = FlowCacheStore.shared
        let count = store.entryCount
        try? store.clear()
        return count > 0 ? count : nil
    }

    private func apply(_ event: FlowEvent) {
        switch event {
        case .queued(let id):
            setStatus(.notRun, for: id)
        case .started(let id):
            setStatus(.running, for: id)
        case .progress(let id, _):
            if rowStates[id]?.status != .failed { setStatus(.running, for: id) }
        case .finished(let id, let asset):
            outputs[id] = asset
            setStatus(.succeeded, for: id)
        case .failed(let id, let error):
            rowStates[id]?.status = .failed
            rowStates[id]?.errorSentence = FlowErrorDisplay.sentence(for: error)
            errorSentence = FlowErrorDisplay.sentence(for: error)
        case .cacheHit(let id):
            setStatus(.succeeded, for: id)
        case .flagRaised(let id, let message):
            rowStates[id]?.errorSentence = message
        }
    }

    private func setStatus(_ status: FlowStatus, for id: UUID) {
        if rowStates[id] != nil { rowStates[id]?.status = status }
    }

    // MARK: - Cancel

    func cancel() {
        runTask?.cancel()
        runTask = nil
        isRunning = false
        // The partially-run flow keeps the dots it earned (dots aren't reset on cancel).
    }

    // MARK: - Install

    /// Register the preflight result, compute substitution notes, and decide whether an
    /// install sheet must be shown.
    func prepareInstall(_ result: FlowPreflight.Result, doc: FlowDocument? = nil) {
        preflight = result
        substitutionNotes = Self.substitutionNotes(result, doc: doc)
        if !result.toDownload.isEmpty && !installPrompted {
            installPrompted = true
            showInstallSheet = true
        }
    }

    /// The `CatalogBridge` substitution note per row id (CFM-R2-2 rule 3): `.sameFamily` /
    /// `.substitute` rows get "running <display> as <candidate>" so the substitution is
    /// surfaced, never hidden. `.same`/`.requantized` run silently.
    private static func substitutionNotes(_ result: FlowPreflight.Result,
                                          doc: FlowDocument?) -> [UUID: String] {
        var notes: [UUID: String] = [:]
        guard let doc else { return notes }
        let needsByDisplay = Dictionary(result.needs.map { ($0.display, $0) }, uniquingKeysWith: { a, _ in a })
        for row in allRows(doc.rows) {
            guard let display = row.model,
                  let need = needsByDisplay[display],
                  let equivalence = need.equivalence,
                  let model = need.model,
                  let note = equivalence.note(display: display, substitutedID: model.hfModelId) else {
                continue
            }
            notes[row.id] = note
        }
        return notes
    }

    private static func allRows(_ rows: [Row]) -> [Row] {
        rows.flatMap { [$0] + allRows($0.children) }
    }
}

/// A small async wait for a single install to land on disk (the `.installed` marker),
/// letting the flow drive `InstallManager.install` sequentially (one prompt, models one at
/// a time). `InstallManager` isn't `@Observable`, so we poll its `isInstalled` marker rather
/// than observing `modelStates`.
enum InstallPoller {
    /// Await the `.installed` marker for `modelID`, polling every 0.5 s. Returns `false` on
    /// a ~10-minute timeout or cancellation.
    static func awaitInstalled(modelID: String, installManager: InstallManager) async -> Bool {
        var attempts = 0
        while !Task.isCancelled {
            if installManager.isInstalled(modelID) { return true }
            attempts += 1
            if attempts > 1200 { return false }   // ~10 min
            try? await Task.sleep(for: .milliseconds(500))
        }
        return false
    }
}
