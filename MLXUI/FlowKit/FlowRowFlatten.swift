import Foundation

/// CFM-R12-3 — flatten a flow's rows into one entry per actual row (block headers and their
/// children), each with its indent depth, so both the editor and the runner list render each
/// child as a selectable row with its own status dot instead of one blob per block. Children
/// of a collapsed block are skipped — collapse hides them from the list.
nonisolated enum FlowRowFlatten {
    struct Entry: Identifiable, Equatable {
        let row: Row
        let depth: Int
        var id: UUID { row.id }
    }

    static func flatten(_ rows: [Row], collapsed: Set<UUID>, depth: Int = 0) -> [Entry] {
        var out: [Entry] = []
        for row in rows {
            out.append(Entry(row: row, depth: depth))
            if row.blockKind != nil, !collapsed.contains(row.id) {
                out.append(contentsOf: flatten(row.children, collapsed: collapsed, depth: depth + 1))
            }
        }
        return out
    }
}
