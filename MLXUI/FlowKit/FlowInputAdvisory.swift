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
        advisory(in: document.rows, document: document, scope: scope, prefix: "")
    }

    /// Document order, depth-first. RP-2: besides a missing `Read *` file, the first
    /// prompt-driven generator with no prompt and nothing upstream — same sentence as the
    /// editor's warning (`RowSources.missingValueSentence`), so Run says it before row 1 fails.
    private static func advisory(in rows: [Row], document: FlowDocument, scope: FlowScope,
                                 prefix: String) -> FlowPreflight.RowAdvisory? {
        for (i, row) in rows.enumerated() {
            let path = prefix.isEmpty ? "\(i + 1)" : "\(prefix).\(i + 1)"
            if let task = row.task {
                if SampleSeed.readTaskNames.contains(task),
                   FlowInputFile.isMissing(for: row, document: document, scope: scope),
                   let url = FlowInputFile.target(for: row, document: document, scope: scope) {
                    let reason = task == "Read Index"
                        ? "\(task) needs its index built first — run the builder flow, then try again."
                        : "\(task) needs '\(url.lastPathComponent)', which isn't in the flow's folder yet."
                    return FlowPreflight.RowAdvisory(task: task, reason: reason, action: nil)
                }
                if RowSources.entry(for: task)?.value == .prompt,
                   let sentence = RowSources.missingValueSentence(for: row, path: path),
                   !RowSources.isFed(row, previous: i > 0 ? rows[i - 1] : nil) {
                    return FlowPreflight.RowAdvisory(task: task, reason: sentence, action: nil)
                }
            }
            if let found = advisory(in: row.children, document: document, scope: scope, prefix: path) {
                return found
            }
        }
        return nil
    }
}
