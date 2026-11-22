import CoreGraphics

/// ER-FIX-1 — the outbox disclosure's list height. The disclosure sits under a row list and inspector that both ask for all the
/// height they can get, so an unbounded list was squeezed until a print entry's "Print…" button was a couple of points tall and a
/// mouse could not reach it. The list now has a *definite* height (an estimate of its content, capped), inside a `ScrollView`, and the
/// disclosure has layout priority over the rows above it: a long outbox scrolls, a short one shows whole.
///
/// The per-entry heights are estimates of the rendered entry, not measurements; an underestimate only means the list scrolls a little.
nonisolated enum OutboxLayout {
    /// The list never grows past this; beyond it the list scrolls.
    static let maxListHeight: CGFloat = 220
    /// A send / post entry: the kind line, up to three lines of text, the staged line, and padding.
    static let plainEntryHeight: CGFloat = 64
    /// A print entry: the same, plus the "Print…" button row.
    static let printEntryHeight: CGFloat = 96
    /// A print entry whose last attempt failed also shows the error line.
    static let printErrorLineHeight: CGFloat = 34

    static func entryHeight(kind: String, hasError: Bool) -> CGFloat {
        guard kind == "print" else { return plainEntryHeight }
        return printEntryHeight + (hasError ? printErrorLineHeight : 0)
    }

    /// The height to give the entry list: the entries' estimated total, capped at `maxListHeight`; `0` for no entries.
    static func listHeight(for entries: [(kind: String, hasError: Bool)]) -> CGFloat {
        min(maxListHeight, entries.reduce(0) { $0 + entryHeight(kind: $1.kind, hasError: $1.hasError) })
    }

    /// Whether the estimated content is taller than the list, i.e. the list scrolls.
    static func scrolls(for entries: [(kind: String, hasError: Bool)]) -> Bool {
        entries.reduce(0) { $0 + entryHeight(kind: $1.kind, hasError: $1.hasError) } > maxListHeight
    }
}
