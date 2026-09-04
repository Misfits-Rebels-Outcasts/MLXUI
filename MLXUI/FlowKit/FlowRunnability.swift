import Foundation

/// CFM-R12-FIX-1 — the single live refusal authority, replacing the stale hand-written
/// `notRunnableReason` strings in `_metadata.json` (which silently cancelled R12-8/R12-9:
/// six flows still read as "net/staged row" and never reached the runnable state).
///
/// The refusal = `FlowRunner.canRun` (language, doors, tools, channels) **or** the
/// `FlowPreflight` verdict (a model row with no runnable model, or a RAM overrun) — the two
/// gates the UI actually depends on, derived from the tree at load time, never from a JSON
/// string. B3's original ask ("derive the refusal from the language gates, not the JSON
/// string"), now applied to both gate layers.
nonisolated enum FlowRunnability {
    /// The refusal sentence for a flow, or nil when it can run. `scope` (CFM-R17-3) resolves
    /// a workspace flow's `uses:` graph so a used-flow call isn't refused as an unknown task;
    /// `nil` keeps the pre-R17 plain-flow check.
    static func refusalReason(for doc: FlowDocument, catalog: [ModelEntry],
                              installed: Set<String>, totalRAMGB: Double,
                              scope: FlowScope? = nil) -> String? {
        let runnability = scope.map { FlowRunner.canRun(doc, scope: $0) } ?? FlowRunner.canRun(doc)
        if case .notRunnable(let reason) = runnability {
            return reason
        }
        let preflight = FlowPreflight.run(doc, catalog: catalog,
                                          installedModelIDs: installed, totalRAMGB: totalRAMGB)
        return FlowPreflight.blockedReason(preflight, totalRAMGB: totalRAMGB)
    }
}
