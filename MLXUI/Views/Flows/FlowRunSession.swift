import Foundation
import Observation
import Darwin
import MLX

/// The visible half of a run (CFM-R2-8): an `@Observable` view-model that consumes the
/// `FlowEvent` stream and drives each row's dot `○ → ● → ✓`, shows the row's error sentence
/// on `✗`, exposes a cancel, and tracks the install sheet state. Held by `FlowListView`.
@Observable
final class FlowRunSession {
    /// Per-row run state, keyed by row id.
    struct RowState: Equatable {
        var status: FlowStatus = .notRun
        var errorSentence: String?
        /// DA-10: this row was dropped by an enclosing `<each on_error=skip>` — `errorSentence`
        /// then holds the skip reason, not a failure. Renders as △ (not ✗), and the run log
        /// line is styled as a skip, so "the flow chose to continue" reads distinctly.
        var wasSkipped = false
    }

    private(set) var rowStates: [UUID: RowState] = [:]
    /// CFM-R11-1: the run's peak-memory record — sampled at each row boundary, so a real
    /// cold run can tell whether the preflight's largest-single-row RAM rule is sound.
    private(set) var metrics = RunMetrics()
    /// Rows served from the flow's output cache this run (their dots would have computed
    /// fresh) — the per-row "cached" marker's source.
    private(set) var cacheHitRows: Set<UUID> = []
    /// The last finished output per row id (the inspector's content source).
    private(set) var outputs: [UUID: Asset] = [:]
    /// A `CatalogBridge` substitution note per row id (CFM-R2-2 rule 3), or absent.
    private(set) var substitutionNotes: [UUID: String] = [:]
    /// The row the inspector pane is bound to (nil = nothing selected).
    var selectedRowID: UUID?
    var isRunning = false
    var errorSentence: String?
    /// CFM-R10-Human: the parked human row awaiting an answer, or nil.
    private(set) var parked: ParkedInfo?
    /// The preflight result that gates Run (models to install / blocked reason).
    var preflight: FlowPreflight.Result?
    /// The parsed flow document — lets `canRun` consult `FlowRunner.canRun(doc)` (B3), so the
    /// Run button refuses out-of-scope language even if the gallery metadata mis-tags a flow.
    private(set) var document: FlowDocument?
    /// CFM-R17-3: the run scope for a workspace flow, so `runnability` resolves its `uses:`
    /// graph instead of refusing a used-flow call as an unknown task. `nil` for a plain flow.
    private(set) var scope: FlowScope?
    /// True while the flow's to-download models are being installed (spinner state).
    var isInstalling = false
    var showInstallSheet = false

    private var runTask: Task<Void, Never>?

    /// CFM-R10-Human: a parked `wait=forever` human row awaiting an answer.
    struct ParkedInfo: Equatable {
        let rowID: UUID
        let prompt: String
        let policy: String
        /// The interpreter's activation key the answer must be keyed by (`"3@1"`).
        let execPath: String
        /// When the run falls back to the row's default if unanswered — nil = waits forever.
        let deadline: Date?
    }

    // MARK: - Gating

    /// Whether the flow's models still need installing before it can run.
    var needsInstall: Bool {
        (preflight?.downloadSet.isEmpty ?? true) == false
    }

    /// The `FlowRunner.canRun` verdict for the loaded document, or nil when no document.
    /// CFM-R17-3: a workspace flow is checked against its scope so its `uses:` graph resolves.
    var runnability: Runnability? {
        guard let document else { return nil }
        if let scope { return FlowRunner.canRun(document, scope: scope) }
        return FlowRunner.canRun(document)
    }

    /// Whether Run is available: not running/installing, no blocked models, nothing left to
    /// download, and the flow's language is in the linear subset (`FlowRunner.canRun`).
    var canRun: Bool {
        guard !isRunning, !isInstalling else { return false }
        if preflight?.isBlocked ?? false { return false }
        if !(preflight?.toDownload.isEmpty ?? true) { return false }
        if case .notRunnable = runnability { return false }
        return true
    }

    /// The only thing between this flow and a run is downloading models — no hard block
    /// (a door, an out-of-subset language, a RAM ceiling), just an install. CFM-R17-FIX-7:
    /// an auto-run opens the install sheet in this case instead of doing nothing.
    var blockedOnlyOnDownloads: Bool {
        guard !isRunning, !isInstalling else { return false }
        guard let preflight else { return false }
        if preflight.isBlocked { return false }
        if case .notRunnable = runnability { return false }
        return !preflight.toDownload.isEmpty
    }

    /// The sentence explaining why Run is disabled, or nil when runnable.
    var runDisabledReason: String? {
        if isInstalling { return "Installing the required models…" }
        if isRunning { return nil }
        if case .notRunnable(let reason)? = runnability { return reason }
        guard let preflight else { return nil }
        if !preflight.toDownload.isEmpty {
            return "Install the required models first."
        }
        return FlowPreflight.blockedReason(preflight, totalRAMGB: SystemInfo.detect().totalRAMGB)
    }

    /// Dot state for a row id.
    func status(for rowID: UUID) -> FlowStatus {
        rowStates[rowID]?.status ?? .notRun
    }

    /// CFM-R12-FIX-3: every row a run can touch — top-level and block children alike — so
    /// `rowStates` seeding and Clear Run cover child ids (a child's dot must advance).
    nonisolated static func flattenedRows(_ rows: [Row]) -> [Row] {
        rows.flatMap { [$0] + flattenedRows($0.children) }
    }

    /// Whether any row has earned a result (any succeeded dot) — gates "re-run from here".
    var hasRunResults: Bool {
        rowStates.values.contains { $0.status == .succeeded }
    }

    /// Error sentence for a failed row.
    func errorSentence(for rowID: UUID) -> String? {
        rowStates[rowID]?.errorSentence
    }

    /// DA-10: whether `errorSentence(for:)` on this row is a skip reason (an enclosing
    /// `<each on_error=skip>` dropped it), not a failure — the run log line renders it
    /// distinctly.
    func wasSkipped(_ rowID: UUID) -> Bool {
        rowStates[rowID]?.wasSkipped ?? false
    }

    /// DA-10-FIX-1: a one-line run summary of every row an `<each on_error=skip>` dropped, so
    /// the reason is visible without hunting for the inline caption under the (indented, block-
    /// child) row. Deduplicated — a per-item `<each>` where every item failed the same way
    /// reads as one reason, not N. `nil` when nothing was skipped.
    var skipSummary: String? {
        let skipped = rowStates.values.filter { $0.wasSkipped }
        guard !skipped.isEmpty else { return nil }
        let reasons = skipped.compactMap { $0.errorSentence }.filter { !$0.isEmpty }
        let unique = Set(reasons).sorted()
        let noun = skipped.count == 1 ? "step was skipped" : "\(skipped.count) steps were skipped"
        guard !unique.isEmpty else { return "\(noun) (on_error=skip)." }
        return "\(noun) (on_error=skip): " + unique.joined(separator: " · ")
    }

    /// DA-10-FIX-1: the sentence to show in the inspector for a selected row — its skip reason
    /// (△) or its failure sentence (✗), or nil when the row neither failed nor was skipped.
    func statusNote(for rowID: UUID) -> (text: String, isSkip: Bool)? {
        guard let state = rowStates[rowID], let sentence = state.errorSentence, !sentence.isEmpty else {
            return nil
        }
        switch state.status {
        case .needsAttention: return (sentence, true)
        case .failed:         return (sentence, false)
        default:              return nil
        }
    }

    // MARK: - Start

    /// Start a run. When `resume` is true, the run starts at the first gray row (answer
    /// `f3`): rows already `✓` keep their dots and outputs, and Run re-executes only the
    /// gray rows. Otherwise the whole flow runs (dots reset).
    func start(doc: FlowDocument, runner: FlowRunner, context: FlowRunner.RunContext,
               resume: Bool = false,
               answers: [String: FlowInterpreter.HumanAnswer] = [:],
               occurrence: FlowInterpreter.Occurrence? = nil) {
        guard !isRunning else { return }
        isRunning = true
        errorSentence = nil
        parked = nil

        let startIndex = resume ? firstGrayIndex(in: doc) : 0
        RunMetrics.resetPeak()      // R11-1: measure this run, not the process lifetime
        if startIndex == 0 { cacheHitRows = [] }
        let resumeOutputs = resume ? outputs : [:]
        if startIndex == 0 {
            rowStates = Dictionary(Self.flattenedRows(doc.rows).map { ($0.id, RowState()) },
                             uniquingKeysWith: { a, _ in a })
        } else {
            // Keep the ✓ rows' states; gray out everything from startIndex down.
            for (id, var state) in rowStates {
                if isDownstream(id, of: startIndex, in: doc) {
                    state.status = .notRun
                    state.errorSentence = nil
                    state.wasSkipped = false
                    rowStates[id] = state
                    outputs[id] = nil
                }
            }
        }

        let stream = runner.run(doc, context: context, startIndex: startIndex,
                                resumeOutputs: resumeOutputs, answers: answers,
                                occurrence: occurrence)
        runTask = Task {
            for await event in stream {
                if Task.isCancelled { break }
                apply(event)
            }
            isRunning = false
            // CFM-R11-1: also land the record in the console so a run's peak survives
            // scrollback / logs, not just the on-screen footer.
            if !metrics.rowSamples.isEmpty {
                print("CFM-R11-1 run peak: \(metrics.summary())")
            }
        }
    }

    /// CFM-R10-Human: answer a parked `Ask Human` row with a tag (the row's decide clause
    /// continues on it) and re-run from the top — cached rows replay fast, the human row
    /// resolves, and the run continues past it.
    func answer(tag: String, for parked: ParkedInfo, doc: FlowDocument,
                runner: FlowRunner, context: FlowRunner.RunContext) {
        start(doc: doc, runner: runner, context: context,
              answers: [parked.execPath: FlowInterpreter.HumanAnswer(tag: tag, text: nil)])
    }

    /// CFM-R10-Human: answer a parked `Human Input` row with typed text.
    func answer(text: String, for parked: ParkedInfo, doc: FlowDocument,
                runner: FlowRunner, context: FlowRunner.RunContext) {
        start(doc: doc, runner: runner, context: context,
              answers: [parked.execPath: FlowInterpreter.HumanAnswer(tag: nil, text: text)])
    }

    /// CFM-R10-Human: no live answer by a `timeout=` row's deadline — re-run with the row's
    /// declared fallback (the `default=` tag for Ask Human, "unchanged" for Human Input),
    /// which the interpreter resolves and discloses as F002. A no-op if the run is no longer
    /// parked on this row (the user answered or stopped in the same instant).
    func fallbackTimeout(for parked: ParkedInfo, doc: FlowDocument,
                         runner: FlowRunner, context: FlowRunner.RunContext) {
        guard parked.deadline != nil, self.parked == parked else { return }
        start(doc: doc, runner: runner, context: context,
              answers: [parked.execPath: FlowInterpreter.HumanAnswer(tag: nil, text: nil,
                                                                     isTimeoutDefault: true)])
    }

    /// Parse a parked row's `timeout=` policy into an absolute deadline — nil when it waits
    /// forever or the duration doesn't parse.
    private static func deadline(fromPolicy policy: String) -> Date? {
        guard policy.hasPrefix("timeout=") else { return nil }
        let raw = String(policy.dropFirst("timeout=".count))
        guard let interval = FlowSettingsEditor.parseDuration(raw) else { return nil }
        return Date().addingTimeInterval(interval)
    }

    /// The index of the first non-`✓` row (the re-run-from-here boundary). A row is gray if
    /// it isn't `.succeeded` (covers `.notRun`, `.failed`, `.needsAttention`). When every
    /// row is already `✓`, returns 0 so a re-run is a fresh run from the top (H7) — the
    /// caller's `startIndex` must select rows, and `doc.rows.count` would select none.
    private func firstGrayIndex(in doc: FlowDocument) -> Int {
        for (i, row) in doc.rows.enumerated() {
            if status(for: row.id) != .succeeded { return i }
        }
        return 0
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
        rowStates = Dictionary(Self.flattenedRows(doc.rows).map { ($0.id, RowState()) },
                             uniquingKeysWith: { a, _ in a })
        outputs = [:]
        selectedRowID = nil
        parked = nil
        metrics = RunMetrics()
        cacheHitRows = []
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
            // DA-10-FIX-3: an `<each on_error=skip>` body row reuses the same row id across
            // items (fact already documented at `.finished` below). Once one item skipped,
            // a *later* item's own `.started` must not repaint the dot blue — that silently
            // undoes "the dot stays △ for the rest of the run" the moment the next item
            // begins, and since a skipped row's `.finished` never promotes back to ✓ either
            // (below), the dot was left stuck on ● with no event left to move it anywhere:
            // exactly the bug this fixes (row 3.1 of `31-ResearchBrief.cat` stuck blue while
            // rows 4-6 already finished green).
            if rowStates[id]?.wasSkipped != true { setStatus(.running, for: id) }
        case .progress(let id, _):
            if rowStates[id]?.status != .failed, rowStates[id]?.wasSkipped != true {
                setStatus(.running, for: id)
            }
        case .finished(let id, let asset):
            outputs[id] = asset
            // DA-10-FIX-2: an `<each on_error=skip>` body row reuses the same row id across
            // items — a later item's own `.finished` must not paint over an earlier item's
            // skip on this run. Once any item skipped, the dot stays △ for the rest of the
            // run (the caption's reason is real evidence *this row* dropped something, even
            // though a later item of the same row went on to succeed).
            if rowStates[id]?.wasSkipped != true { setStatus(.succeeded, for: id) }
            metrics.record(rowID: id)
        case .failed(let id, let error):
            rowStates[id]?.status = .failed
            rowStates[id]?.errorSentence = FlowErrorDisplay.sentence(for: error)
            errorSentence = FlowErrorDisplay.sentence(for: error)
            metrics.record(rowID: id)
        case .cacheHit(let id):
            if rowStates[id]?.wasSkipped != true { setStatus(.succeeded, for: id) }
            cacheHitRows.insert(id)
            metrics.record(rowID: id)
        case .flagRaised(let id, let message):
            rowStates[id]?.errorSentence = message
        case .skipped(let id, let reason):
            // DA-10 (SPEC-Q216): the enclosing `<each on_error=skip>` dropped this row and
            // the run continues. Leave `.running` behind — it is neither running nor failed —
            // and surface the reason. Not promoted to `session.errorSentence`: the run is not
            // failing.
            rowStates[id]?.status = .needsAttention
            rowStates[id]?.errorSentence = reason
            rowStates[id]?.wasSkipped = true
            metrics.record(rowID: id)
        case .parked(let id, let prompt, let policy, let execPath):
            isRunning = false
            parked = ParkedInfo(rowID: id, prompt: prompt, policy: policy, execPath: execPath,
                                deadline: Self.deadline(fromPolicy: policy))
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

    /// Dismiss the parked prompt without answering (the run stays parked until re-run).
    func clearParked() {
        parked = nil
    }

    // MARK: - CFM-R10-FIX-6: Improvise undo, reachable from the app

    private static func allRows(_ rows: [Row]) -> [Row] {
        rows.flatMap { [$0] + allRows($0.children) }
    }

    /// Whether the flow has an `Improvise` row (whose workdir can be undone).
    func hasImprovise(_ doc: FlowDocument) -> Bool {
        Self.allRows(doc.rows).contains { $0.task == "Improvise" }
    }

    /// Restore an Improvise row's declared workdir from its pre-run snapshot. Direct build
    /// only (the App Store tier refuses Improvise outright). Returns a plain sentence.
    func undoImprovise(doc: FlowDocument, workspace: FlowWorkspace, flowID: String) throws -> String {
        guard let improvised = Self.allRows(doc.rows).first(where: { $0.task == "Improvise" }),
              let workdirRaw = FlowSettings(improvised.settings).value(for: "workdir") else {
            throw FlowError.stageFailure(row: "Improvise", message: "no declared `workdir=` to undo")
        }
        #if DIRECT_BUILD
        let workdir = try workspace.resolve(workdirRaw, flowID: flowID)
        try FencedRunner.undo(workdir)
        return "Restored the improvise workdir to its pre-run state."
        #else
        throw FlowError.stageFailure(row: "Improvise",
                                     message: "the App Store build refuses Improvise — distribute it directly instead.")
        #endif
    }

    // MARK: - Install

    /// Register the preflight result and compute substitution notes. Does **not** prompt —
    /// the install sheet is shown only on explicit user action ("Install Required Models").
    /// The `doc` is kept so `canRun`/`runnability` consult `FlowRunner.canRun(doc)` (B3).
    func prepareInstall(_ result: FlowPreflight.Result, doc: FlowDocument? = nil,
                        scope: FlowScope? = nil) {
        document = doc
        self.scope = scope
        preflight = result
        substitutionNotes = Self.substitutionNotes(result, doc: doc)
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

}

/// A small async wait for a single install to land on disk (the `.installed` marker),
/// letting the flow drive `InstallManager.install` sequentially (one prompt, models one at
/// a time). `InstallManager` isn't `@Observable`, so we poll its `isInstalled` marker rather
/// than observing `modelStates`.
enum InstallPoller {
    /// Await the `.installed` marker for `model`, polling every 0.5 s. Returns `false` on
    /// a ~10-minute timeout or cancellation. MoC-5-FIX-1: takes the whole `ModelEntry` so
    /// `isInstalled` can resolve the marker by repo.
    static func awaitInstalled(model: ModelEntry, installManager: InstallManager) async -> Bool {
        var attempts = 0
        while !Task.isCancelled {
            if installManager.isInstalled(model) { return true }
            attempts += 1
            if attempts > 1200 { return false }   // ~10 min
            try? await Task.sleep(for: .milliseconds(500))
        }
        return false
    }
}

/// CFM-R11-1 — the per-run memory recorder. Peak MLX GPU memory is reset at run start (the
/// C-side peak counts the whole process lifetime otherwise), then sampled at every row
/// boundary alongside the process's resident set. A real cold run records `summary()` into
/// the journal, which decides whether the preflight's largest-single-row RAM rule is sound.
nonisolated struct RunMetrics: Equatable {
    /// A sample at one row boundary, in bytes.
    struct Sample: Equatable {
        var gpuActive: Int64
        var gpuCache: Int64
        var gpuPeak: Int64
        var rss: Int64
    }

    private(set) var rowSamples: [UUID: Sample] = [:]
    private(set) var peakGPU: Int64 = 0
    private(set) var peakRSS: Int64 = 0

    /// Reset the C-side MLX peak so a run measures only itself. `Memory.peakMemory`'s setter
    /// is the Swift spelling of `mlx_reset_peak_memory()`.
    static func resetPeak() {
        Memory.peakMemory = 0
    }

    /// Sample now and fold it into the run's peaks.
    mutating func record(rowID: UUID) {
        let stats = Memory.snapshot()
        let sample = Sample(gpuActive: Int64(stats.activeMemory),
                            gpuCache: Int64(stats.cacheMemory),
                            gpuPeak: Int64(stats.peakMemory),
                            rss: currentRSSBytes())
        rowSamples[rowID] = sample
        peakGPU = max(peakGPU, sample.gpuPeak)
        peakRSS = max(peakRSS, sample.rss)
    }

    /// A one-line journal record (R11-1's done-when): the run's peak GPU + peak RSS, plus
    /// the engine-cache state (R11-2) so a run reveals at a glance whether warm engines are
    /// being held.
    func summary() -> String {
        let engines = EngineCache.shared.count
        let warm = String(format: "%.1f GB", Double(EngineCache.shared.totalCachedBytes) / 1_073_741_824)
        return "peak GPU \(mb(peakGPU)) (active \(mb(rowSamples.values.map(\.gpuActive).max() ?? 0)) / cache \(mb(rowSamples.values.map(\.gpuCache).max() ?? 0))) · peak RSS \(mb(peakRSS)) · warm engines \(engines) (\(warm))"
    }

    private func mb(_ bytes: Int64) -> String {
        String(format: "%.1f MB", Double(bytes) / 1_048_576)
    }
}

/// The process's resident set size (bytes) via Mach `task_info` — the closest cheap measure
/// of "how much RAM the app is holding" at a row boundary.
nonisolated func currentRSSBytes() -> Int64 {
    var info = task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<task_basic_info>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_BASIC_INFO), $0, &count)
        }
    }
    guard result == KERN_SUCCESS else { return 0 }
    return Int64(info.resident_size)
}
