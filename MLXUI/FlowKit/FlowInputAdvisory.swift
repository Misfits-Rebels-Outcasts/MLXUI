import Foundation

/// FIP-2 — surfaces `FlowInputFile` (FIP-1) as the same `FlowPreflight.RowAdvisory` shape the
/// UI already renders as a non-blocking banner, so the places that offer a button and then
/// fail on row 1 (a flow's Run, a workspace card's Build/Ask) can say so first.
///
/// Owner ruling, 2026-09-16 (`RSI/DelegateFlowInputPreflightBacklog.md`, Q1): *"Warn only
/// (Recommended)"* — this never blocks. A row fed by a matching upstream (row 3 writes what
/// row 5 reads) is legitimate and must never be warned about; `FlowInputFile.isMissing`
/// already carries that precedence, so this is purely a rendering shim on top of it.
nonisolated enum FlowInputAdvisory {
    /// The first row (document order, depth-first through blocks) whose input is missing, or
    /// nil when every `Read *` row's file is present or supplied by a matching upstream.
    /// Mirrors `FlowPreflight.setupAdvisory`'s "first, not all" contract — one banner, one row.
    static func advisory(for document: FlowDocument, scope: FlowScope) -> FlowPreflight.RowAdvisory? {
        for row in allRows(document.rows) {
            guard let task = row.task, SampleSeed.readTaskNames.contains(task) else { continue }
            guard FlowInputFile.isMissing(for: row, document: document, scope: scope),
                  let url = FlowInputFile.target(for: row, document: document, scope: scope)
            else { continue }
            let reason = task == "Read Index"
                ? "\(task) needs its index built first — run the builder flow, then try again."
                : "\(task) needs '\(url.lastPathComponent)', which isn't in the flow's folder yet."
            return FlowPreflight.RowAdvisory(task: task, reason: reason, action: nil)
        }
        return nil
    }

    private static func allRows(_ rows: [Row]) -> [Row] {
        var out: [Row] = []
        for row in rows {
            out.append(row)
            out.append(contentsOf: allRows(row.children))
        }
        return out
    }
}
