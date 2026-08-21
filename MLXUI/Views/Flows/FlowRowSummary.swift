import Foundation

/// Pure, testable row-summary logic for the read-only flow list (CFM-R1-5). Extracted so
/// the UI stays trivial and the display rules (model-name subtitle, `(2)`/`(1,2)` reference
/// labels computed from row ids at display time, chain-break detection) are pinned by tests.
nonisolated enum FlowRowSummary {

    /// The friendly task name for a row. Block rows (`task == nil`) render their block
    /// kind: `<each>`, `<parallel>`, or `<list>`.
    static func taskName(for row: Row) -> String {
        if let task = row.task { return task }
        switch row.blockKind {
        case .each:     return "<each>"
        case .parallel: return "<parallel>"
        case .list:     return "<list>"
        case nil:       return "Block"
        }
    }

    /// A one-line description of a row's task, used as the subtitle when the row has no
    /// model. Covers the tasks the R1–R4 gallery flows use; anything unknown falls back to
    /// a generic line. **Not** the raw settings text (design doc §B.1, answer `b1`).
    static func taskDescription(for task: String?) -> String {
        guard let task else {
            return "Runs its rows in order"
        }
        switch task {
        case "Read Audio":    return "Reads an audio file"
        case "Read Text":     return "Reads a text file"
        case "Read Images":   return "Reads a folder of images"
        case "Transcribe":    return "Turns speech into text"
        case "Summarize":     return "Summarizes the text"
        case "Speak":         return "Reads text aloud"
        case "Save Audio":    return "Saves audio into the flow folder"
        case "Save Text":     return "Saves text into the flow folder"
        case "Save Images":   return "Saves images into the flow folder"
        case "Diff":          return "Compares two documents"
        case "Resize":        return "Resizes an image"
        case "Watermark":     return "Adds a watermark"
        default:              return "Runs this step"
        }
    }

    /// The row's subtitle: the model display name if the row has one, else a one-line
    /// description of its task.
    static func subtitle(for row: Row) -> String {
        if let model = row.model { return model }
        return taskDescription(for: row.task)
    }

    /// The inline reference label — the literal `(2)` / `(1,2)` the `.cat` uses — computed
    /// from row ids at display time (answer `d1`). `nil` when the row has no row refs.
    /// `inputRef`/`paramRef` are not row references and don't produce a `(N)` label.
    static func referenceLabel(for row: Row, in rows: [Row]) -> String? {
        let numbers = row.refs.compactMap { ref -> Int? in
            guard case .rowRef(let id) = ref else { return nil }
            return displayNumber(forID: id, in: rows)
        }
        guard !numbers.isEmpty else { return nil }
        return "(" + numbers.map(String.init).joined(separator: ",") + ")"
    }

    /// The 1-based display number of the row with `id` within `rows`, or `nil`.
    static func displayNumber(forID id: UUID, in rows: [Row]) -> Int? {
        guard let index = rows.firstIndex(where: { $0.id == id }) else { return nil }
        return index + 1
    }

    /// Whether the row starts a new chain — the blank line in the source. Renders as a
    /// `Divider` in the list (answer `e3`).
    static func hasChainBreakBefore(_ row: Row) -> Bool {
        row.chainBreak
    }

    /// Whether a row is a block (`<each>`, `<parallel>`, `<list>`) with children.
    static func isBlock(_ row: Row) -> Bool {
        row.blockKind != nil
    }
}
