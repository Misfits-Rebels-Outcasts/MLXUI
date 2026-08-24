import Foundation

/// Pure row-summary logic for the flow list (CFM-R6-2): after the list renders the
/// serializer's canonical lines, only the task name (inspector header), chain-break
/// detection, block detection, and display numbers are left. The English one-line
/// descriptions (`taskDescription`/`subtitle`) and the hand-computed `referenceLabel` were
/// removed — the ref column now comes from `CatSerializer`.
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
