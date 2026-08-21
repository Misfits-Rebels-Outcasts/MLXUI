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

    /// Error sentence for a failed row.
    func errorSentence(for rowID: UUID) -> String? {
        rowStates[rowID]?.errorSentence
    }

    // MARK: - Start

    /// Start a run: reset dots, consume the stream, drive states. `installFirst` (when
    /// non-empty) is confirmed by the caller before this is called.
    func start(doc: FlowDocument, runner: FlowRunner, context: FlowRunner.RunContext) {
        guard !isRunning else { return }
        isRunning = true
        errorSentence = nil
        rowStates = Dictionary(uniqueKeysWithValues: doc.rows.map { ($0.id, RowState()) })

        let stream = runner.run(doc, context: context)
        runTask = Task {
            for await event in stream {
                if Task.isCancelled { break }
                apply(event)
            }
            isRunning = false
        }
    }

    private func apply(_ event: FlowEvent) {
        switch event {
        case .queued(let id):
            setStatus(.notRun, for: id)
        case .started(let id):
            setStatus(.running, for: id)
        case .progress(let id, _):
            if rowStates[id]?.status != .failed { setStatus(.running, for: id) }
        case .finished(let id, _):
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

    /// Register the preflight result and decide whether an install sheet must be shown.
    func prepareInstall(_ result: FlowPreflight.Result) {
        preflight = result
        if !result.toDownload.isEmpty && !installPrompted {
            installPrompted = true
            showInstallSheet = true
        }
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
