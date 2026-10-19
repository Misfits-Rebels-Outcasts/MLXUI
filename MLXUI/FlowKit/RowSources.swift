import Foundation

/// RP-1 (`RSI/DelegateRowPortsBacklog.md`, plan `RSI/plan-row-ports-2026-10.md` §3) — the one
/// declared table of **rows that can start a flow**: tasks that produce their output from their
/// own settings with no input.
///
/// Before this, `FlowEditorModel.startingNodes()` did two jobs from one list — what an empty
/// flow's picker offers, and which row-1 tasks escape the "nothing feeds it" warning — and got
/// both wrong through the `.anyKind` arm (Save/frame/latent rows were offered though they throw
/// with no input; web rows were warned though they source themselves). Both readers now ask
/// here. `CatFlowRowSourcesTests` runs every entry through `RealExecutor` with no input, so
/// the table cannot claim what the runtime does not do.
///
/// Owner ruling G1 (2026-10-19, "G0 yes, G1 yes."). Plain-text `Generate` stays out (G1b,
/// pending; implementer's call).
nonisolated enum RowSources {

    /// What the row's own value is — the thing it falls back on when nothing feeds it.
    enum Value: Sendable, Equatable {
        case file, prompt, question, pattern, range, url, query
    }

    struct Entry: Sendable, Equatable {
        let task: String
        let value: Value
        /// Whether an empty flow's + Add step offers it. Web rows need the `network` flag
        /// (CFM-R8-FIX-7); `Read Video`/`Read Index`/`Read Context` have no seeded sample.
        let offeredOnEmptyFlow: Bool
    }

    /// The table, in no particular order — `startingNodes()` keeps catalog order.
    static let entries: [Entry] = {
        // "Runnable ⟺ seeded" (R11-0c): the Read rows come straight from `SampleSeed.samples`.
        let reads = SampleSeed.samples.keys.map { Entry(task: $0, value: .file, offeredOnEmptyFlow: true) }
        let generators = ["Generate Image", "Generate Sound", "Generate Video"]
            .map { Entry(task: $0, value: .prompt, offeredOnEmptyFlow: true) }
        let own: [Entry] = [
            Entry(task: "Human Input", value: .question, offeredOnEmptyFlow: true),
            Entry(task: "Template", value: .pattern, offeredOnEmptyFlow: true),
            Entry(task: "Range", value: .range, offeredOnEmptyFlow: true),
        ]
        // Not offered (no seeded sample), but they source themselves from their own settings, so row 1 must not
        // warn about them (they were exempt before RP-1, and `Read Index kb.index` is a
        // legitimate row 1).
        // Rule (owner, 2026-10-19): a row that reads a file named in its own settings is
        // exempt — `Store Query orders.db; table=orders` is the same shape as `Read Index`.
        let unseededReads = ["Read Video", "Read Index", "Read Context", "Store Query", "Store Read"]
            .map { Entry(task: $0, value: .file, offeredOnEmptyFlow: false) }
        let web = ["Web Fetch", "HTTP Get", "Fetch Feed", "Download File"]
            .map { Entry(task: $0, value: .url, offeredOnEmptyFlow: false) }
            + [Entry(task: "Web Search", value: .query, offeredOnEmptyFlow: false)]
        return reads + generators + own + unseededReads + web
    }()

    static func entry(for task: String) -> Entry? {
        entries.first { $0.task == task }
    }

    /// Whether `task` can sit at row 1 on its own value — the warning's exemption.
    static func contains(_ task: String) -> Bool { entry(for: task) != nil }

    /// Whether an empty flow's picker offers `task`.
    static func isOfferedOnEmptyFlow(_ task: String) -> Bool { entry(for: task)?.offeredOnEmptyFlow ?? false }

    // MARK: - RP-2: the row's own value, and what to say when it is missing

    /// The row's own value (a file, prompt, pattern, range, url or query), trimmed; nil when
    /// empty or absent. `.question` is `Human Input`'s, which has its own HR-1 handling.
    static func ownValue(of row: Row, entry: Entry) -> String? {
        let s = FlowSettings(row.settings)
        let raw: String?
        switch entry.value {
        case .file: raw = s.pathValue()
        case .prompt, .pattern, .range, .question: raw = s.firstBare()
        case .url: raw = s.value(for: "url") ?? s.firstBare()
        case .query: raw = s.value(for: "query") ?? s.firstBare()
        }
        guard let trimmed = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty else { return nil }
        return trimmed
    }

    /// The non-blocking sentence for a table row whose own value is missing, or nil when the
    /// row has its value (or is `Human Input`, whose missing question is not a warning).
    /// `path` is the row's display number.
    static func missingValueSentence(for row: Row, path: String) -> String? {
        guard let task = row.task, let entry = entry(for: task),
              ownValue(of: row, entry: entry) == nil else { return nil }
        switch entry.value {
        case .prompt: return "Row \(path) needs a prompt — type one in the Step tab, or connect a row that makes text."
        case .file: return "Row \(path) needs a file — choose one in the Step tab, or connect a row that makes one."
        case .pattern: return "Row \(path) needs text — type it in the Step tab."
        case .range: return "Row \(path) needs a range — e.g. 1..3."
        case .url: return "Row \(path) needs a URL — add url= in the Step tab."
        case .query: return "Row \(path) needs a search — add query= in the Step tab."
        case .question: return nil
        }
    }

    /// Whether something upstream supplies the row's input: an explicit ref, or the row above
    /// auto-chaining a compatible shape. `Range` is never fed — `RangeTool` reads only its
    /// own settings, so an upstream row does not give it a range.
    static func isFed(_ row: Row, previous: Row?) -> Bool {
        if row.task == "Range" { return false }
        if !row.refs.isEmpty { return true }
        guard let previous, !previous.chainBreak,
              let to = row.task.flatMap({ TaskCatalog.get($0) }),
              let from = previous.task.flatMap({ TaskCatalog.get($0) }) else { return false }
        let given: Shape = from.gives == .sameAsInput ? .anyKind : from.gives
        return Shape.bundleCompatible(accepts: to.accepts, given: [given], rk: to.refKind)
    }
}
