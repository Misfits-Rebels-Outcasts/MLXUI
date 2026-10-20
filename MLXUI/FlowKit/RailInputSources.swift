import Foundation

/// KR-2 (`RSI/DelegateKindRailBacklog.md`) — where a row's input comes from, for the kind rail.
/// Display only: built from `RowSources` (`takesNoUpstreamInput`, `ownValue`, `isFed`) and the
/// row's own refs, with no compatibility logic of its own — whether the input *fits* is the
/// editor's `warning(for:)`, which the rail quotes verbatim.
nonisolated enum RailInputSource: Equatable, Sendable {
    /// Fed by the row directly above (or, for the first row of a block chain, by the block's input).
    case above
    /// A numbered ref: `(N)`, a top-level `(input:N)`, or a block-scoped ref resolved to the
    /// number of the row it actually comes from (KR-2-FIX-1).
    case row(Int)
    /// The same, for a source that lives inside a nested block ("3.1").
    case path(String)
    /// A block input that doesn't come from a single row.
    case blockInput(Int)
    /// A starting row using its own value; the word names it ("prompt", "file", "question", …).
    case own(String)
    /// Unfed, and no value of its own.
    case missing

    /// The chip's wording.
    var label: String {
        switch self {
        case .above:        return ""
        case .row(let n):   return "from \(n)"
        case .path(let p):  return "from \(p)"
        case .blockInput(let n): return "from block input \(n)"
        case .own(let w):   return "own \(w)"
        case .missing:      return "no input"
        }
    }
}

nonisolated enum RailInputSources {

    /// The sources of `row`'s input, in ref order. `previous` is the row above in the same scope
    /// with the blank line honoured (`FlowEditorModel.previousRow(before:)`); `inBlock` says the
    /// row is a block child, whose chain start is fed by the block's input rather than by a row.
    /// `number` maps a ref's row id to the number the file writes; an unresolvable ref is `.missing`.
    /// `blockInput` says what `(input:p)` reads as — by default `.row(p)` (top level); inside a block
    /// the editor passes the row that input resolves to.
    static func sources(of row: Row, previous: Row?, inBlock: Bool = false,
                        number: (UUID) -> Int?,
                        blockInput: (Int) -> RailInputSource = { .row($0) }) -> [RailInputSource] {
        let task = row.task
        // A row that reads only its own settings ignores whatever is upstream.
        let ignoresUpstream = task.map(RowSources.takesNoUpstreamInput) ?? false
        if !ignoresUpstream {
            var fromRefs: [RailInputSource] = []
            for ref in row.refs {
                switch ref {
                case .rowRef(let id):    fromRefs.append(number(id).map(RailInputSource.row) ?? .missing)
                case .inputRef(let p):   fromRefs.append(blockInput(p))
                case .paramRef:          break
                }
            }
            if !fromRefs.isEmpty { return fromRefs }
            if row.blockKind != nil, row.task == nil {
                // A block header has no task for `isFed` to judge; only say what is certain.
                return previous != nil || inBlock ? [.above] : [.missing]
            }
            if RowSources.isFed(row, previous: previous) { return [.above] }
            if inBlock, previous == nil, !isOwnValueRow(row) { return [.above] }
        }
        if let task, let entry = RowSources.entry(for: task),
           RowSources.ownValue(of: row, entry: entry) != nil {
            return [.own(word(for: entry.value))]
        }
        return [.missing]
    }

    private static func isOwnValueRow(_ row: Row) -> Bool {
        guard let task = row.task, let entry = RowSources.entry(for: task) else { return false }
        return RowSources.ownValue(of: row, entry: entry) != nil
    }

    private static func word(for value: RowSources.Value) -> String {
        switch value {
        case .file: return "file"
        case .prompt: return "prompt"
        case .question: return "question"
        case .pattern: return "text"
        case .range: return "range"
        case .url: return "URL"
        case .query: return "search"
        }
    }
}
