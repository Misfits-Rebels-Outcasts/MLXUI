import Foundation

/// FIP-1 — the input counterpart to `FlowSavedFile`: given a `Read *` row and enough context
/// to know what feeds it, the file (or index directory) it will actually read when the flow
/// runs, or nil when nothing here needs checking. Pure, no UI, directly tested.
///
/// **Why this exists.** `FlowRunner.canRun` is a static gate — task exists, model served,
/// capability flags, `uses:` resolution — and never touches the filesystem. `FlowPreflight.run`
/// cannot help either: its signature carries no scope and no workspace, so it cannot see a
/// file even in principle. Nothing between pressing a button and row 1 executing asks whether
/// the files a flow reads are actually there — the gap `KW-4-FIX-1` hit for real (a brand-new
/// workspace's starter named `notes.txt`/`question.txt`, neither of which existed).
///
/// **The eleven `SampleSeed.readTaskNames` are not uniform about "upstream wins."** The rule
/// documented at `ReadTools.swift:52-54` — *"an upstream item of the matching kind wins, else
/// the settings' `path=`/first bare token"* — is what `ReadPath.resolve` implements for
/// `Read Image`/`Read Images`/`Read Files`/`Read PDF`; `ContextTools.swift`'s `Read Context`
/// implements the same rule inline (its own `FIX-11` comment); `RealExecutor.resolveFile`
/// implements it for `Read CSV`/`Read JSON`. But `Read Audio`, `Read Text`, `Read Video`, and
/// `Read Index` (`SpokenSummaryTools.swift`, `VideoTools.swift`, `IndexStoreTools.swift`)
/// always read their own settings, regardless of what feeds them — verified by reading each
/// tool's `run(_:progress:)` directly, not assumed. `checksUpstream` mirrors that split
/// exactly: getting it wrong in either direction is precisely the false-positive/false-negative
/// risk this type exists to avoid ("a check that is wrong is worse than no check").
nonisolated enum FlowInputFile {
    private static let upstreamAwareTasks: Set<String> = [
        "Read Image", "Read Images", "Read Files", "Read PDF",
        "Read Context", "Read CSV", "Read JSON",
    ]

    /// Whether `task`'s own tool checks an upstream `.file`-kind item before falling back to
    /// its settings path. `Read Audio`/`Read Text`/`Read Video`/`Read Index` do not — they
    /// always read their own settings.
    static func checksUpstream(_ task: String) -> Bool { upstreamAwareTasks.contains(task) }

    /// The file (or index directory, for `Read Index`) `row` will actually read — nil when
    /// `row` isn't a `Read *` task, when an upstream row of the matching kind supplies it (no
    /// read from this row's own path at all, even when that path is stale), or when it names
    /// no path. Resolves through `FlowWorkspace.resolve` — the CFM-R1-4 security boundary —
    /// never `appendingPathComponent` by hand, so a path that escapes the flow directory is
    /// refused rather than probed.
    ///
    /// Auto-chain detection walks only the document's top level. A row inside a block that
    /// takes its input from the block's own iteration (`(input:N)`) already carries an
    /// explicit ref, which the check below treats conservatively as "upstream supplies it"
    /// without needing to infer the block's per-iteration kind — the same conservative
    /// direction as the false-positive rule above: it can only under-warn, never wrongly warn
    /// on a correct flow.
    static func target(for row: Row, document: FlowDocument, scope: FlowScope) -> URL? {
        guard let task = row.task, SampleSeed.readTaskNames.contains(task) else { return nil }

        if !row.refs.isEmpty { return nil }

        if checksUpstream(task),
           let previous = previousTopLevelRow(before: row, in: document),
           let previousTask = previous.task,
           let gives = TaskCatalog.get(previousTask)?.gives,
           Shape.baseKind(gives) == .file {
            return nil
        }

        guard let raw = FlowSettings(row.settings).pathValue() else { return nil }
        return try? scope.workspace.resolve(raw, flowID: scope.locationID)
    }

    /// Whether `row`'s target is missing on disk — the file itself for every read task, or
    /// `manifest.json` inside the directory for `Read Index` (an index is a directory
    /// carrying one; `WorkspaceListView.manifest(for:)` already reads it the same way).
    static func isMissing(for row: Row, document: FlowDocument, scope: FlowScope) -> Bool {
        guard let url = target(for: row, document: document, scope: scope) else { return false }
        let checkURL = row.task == "Read Index" ? url.appendingPathComponent("manifest.json") : url
        return !FileManager.default.fileExists(atPath: checkURL.path)
    }

    /// The immediately preceding top-level row, honoring `chainBreak` (a blank line stops the
    /// chain — the same rule `FlowEditorModel`'s own auto-chain uses).
    private static func previousTopLevelRow(before row: Row, in document: FlowDocument) -> Row? {
        guard let index = document.rows.firstIndex(where: { $0.id == row.id }), index > 0 else { return nil }
        let previous = document.rows[index - 1]
        return previous.chainBreak ? nil : previous
    }
}
