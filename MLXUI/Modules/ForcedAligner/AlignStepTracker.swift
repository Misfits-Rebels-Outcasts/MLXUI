import Observation

/// Which aligner models have an alignment step in flight, across every Run sheet. `AlignRunModel` marks its model when
/// a run starts and clears the mark when that run's step returns, however it ends; Stop and Done clear nothing.
/// Run sheets only: an `Align` row in an AI Workflow is not tracked here.
///
/// Owner ruling 15 (Boo, 2026-11-27): "Close it. The aligner module tracks the running step per model, across sheets,
/// and keeps Align disabled in every sheet until that step returns. Test: Done during a run, reopen the sheet, Align is
/// disabled until the first step returns."
@MainActor
@Observable
final class AlignStepTracker {
    static let shared = AlignStepTracker()

    private var busy: Set<String> = []

    func isBusy(_ modelID: String) -> Bool { busy.contains(modelID) }

    /// The caller has checked `isBusy` and calls `end` when its step returns.
    func begin(_ modelID: String) { busy.insert(modelID) }

    func end(_ modelID: String) { busy.remove(modelID) }
}
