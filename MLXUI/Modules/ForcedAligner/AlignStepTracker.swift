import Observation
import os

/// Which aligner models have an alignment step in flight, across every Run sheet. `AlignRunModel` marks its model when
/// a run starts and clears the mark when that run's step returns, however it ends; Stop and Done clear nothing.
/// Run sheets only: an `Align` row in an AI Workflow is not tracked here (a sheet's run queues behind one at
/// `ForcedAlignerGate`). A sheet this disables says why (ruling 21, quoted at `AlignRunModel.blockedText`).
///
/// Each step's start and end is logged (the model id and nothing of the run's content), so a person can tell whether a
/// step was still in flight when a sheet was reopened:
/// `log show --last 10m --predicate 'subsystem == "net.connectcode.mlxui" && category == "align"'`.
/// *Implementer's call, pending owner confirmation* (it answers ruling 40's "Check with a log whether the step was still
/// running", Boo, 2026-11-28, for the next smoke run).
///
/// Owner ruling 15 (Boo, 2026-11-27): "Close it. The aligner module tracks the running step per model, across sheets,
/// and keeps Align disabled in every sheet until that step returns. Test: Done during a run, reopen the sheet, Align is
/// disabled until the first step returns."
@MainActor
@Observable
final class AlignStepTracker {
    static let shared = AlignStepTracker()
    private static let log = Logger(subsystem: "net.connectcode.mlxui", category: "align")

    private var busy: Set<String> = []

    func isBusy(_ modelID: String) -> Bool { busy.contains(modelID) }

    /// The caller has checked `isBusy` and calls `end` when its step returns.
    func begin(_ modelID: String) {
        busy.insert(modelID)
        Self.log.notice("alignment step began: \(modelID, privacy: .public)")
    }

    func end(_ modelID: String) {
        busy.remove(modelID)
        Self.log.notice("alignment step ended: \(modelID, privacy: .public)")
    }
}
