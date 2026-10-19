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
        // Not offered, but they source themselves from their own settings, so row 1 must not
        // warn about them (they were exempt before RP-1, and `Read Index kb.index` is a
        // legitimate row 1).
        let unseededReads = ["Read Video", "Read Index", "Read Context"]
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
}
