import Foundation

/// CFM-R17-4 — the knowledge-base surface, **derived** from a workspace's flows. No
/// `workspace.json`, no naming convention, no new syntax: classify each parsed flow by what
/// it does with an index, and when a builder and a querier name the same index, the workspace
/// view offers it as one card with two verbs.
nonisolated enum WorkspaceKnowledge {

    /// How a flow relates to an index. Row lists are walked pre-order (`flatten`) — **source
    /// order**, recursing into blocks; branch clauses (`-> { tag: N }`, `-> N`, `resume`) are
    /// not followed, so "row order" means source order, not execution order (CFM-R17-FIX-9(d)).
    ///
    /// CFM-R17-FIX-11(a) (owner-ruled 2026-09-04): **non-exclusive and position-free.** A flow
    /// that stores an index builds it, full stop — every `Store Index` row anywhere counts, not
    /// only the flow's last row. A flow that reads an index and retrieves from it queries it. A
    /// flow may do both, to the same or different names (a self-contained "ingest then ask"
    /// flow is a real shape). The two arms used to be exclusive (`builds` was tested first and
    /// shadowed `queries`) and the builder arm alone was gated on `rows.last?.task ==
    /// "Store Index"` — which made whether an *earlier* `Store Index` counted depend on an
    /// unrelated later row, and collected names from `Gate`/`<each>` branches that might never
    /// execute. Both are gone: `builds` and `queries` are each computed independently over the
    /// whole flattened row list.
    struct FlowIndexUse: Equatable, Sendable {
        /// Every `Store Index <name>` row in the flow (any depth), source order, de-duplicated,
        /// normalized.
        var builds: [String] = []
        /// Every `Read Index <name>` row (any depth), when the flow has a `Retrieve` /
        /// `Keyword Search` somewhere — source order, de-duplicated, normalized. Empty when the
        /// flow reads an index but never retrieves from it (CFM-R17-3: `readIndexAloneIsPlain`).
        var queries: [String] = []

        var isPlain: Bool { builds.isEmpty && queries.isEmpty }
    }

    /// One index name at least one flow mentions — a candidate for a Knowledge Base card. A
    /// `Side` is `.one` when exactly one flow does that job (its button runs that flow),
    /// `.ambiguous` when more than one does (no button for it — the note names them), `.none`
    /// when no flow does.
    ///
    /// CFM-R17-FIX-9(c): an ambiguous side no longer suppresses the whole card. `1 builder +
    /// 2 queriers` keeps its Build button and notes the Ask collision; the mirror keeps Ask.
    ///
    /// CFM-R17-FIX-11(c) (owner-ruled 2026-09-04): `.none` is now reachable — `classify` emits
    /// a card for **every** name any flow mentions, including one-sided names; it stays pure
    /// (no filesystem) and leaves *whether to render* to the caller, since only the caller can
    /// see disk. `WorkspaceListView` renders a card when it has a builder, or when the index
    /// exists on disk (a prebuilt, queryable index — `uses_example`'s `kb.index` — is a real,
    /// fully-functional state with no builder in the workspace) — and suppresses a
    /// querier-only card for an index that doesn't exist yet (nothing actionable, and a card
    /// reading "not built yet" with no Build button is a dead end).
    struct IndexCard: Equatable, Sendable {
        let indexName: String
        let builder: Side
        let querier: Side

        enum Side: Equatable, Sendable {
            case none
            case one(String)
            /// More than one flow, in source order.
            case ambiguous([String])
        }

        /// The flow the **Build** button runs, or nil when the builder side is empty or
        /// ambiguous.
        var buildFile: String? {
            if case .one(let f) = builder { return f }
            return nil
        }

        /// The flow the **Ask** button runs, or nil when the querier side is empty or
        /// ambiguous.
        var askFile: String? {
            if case .one(let f) = querier { return f }
            return nil
        }

        /// A plain note covering only the ambiguous side(s), or nil when neither is ambiguous.
        var ambiguityNote: String? {
            var clauses: [String] = []
            if case .ambiguous(let files) = builder {
                clauses.append("\(WorkspaceKnowledge.list(files)) all build ‘\(indexName)’ — no single Build button")
            }
            if case .ambiguous(let files) = querier {
                clauses.append("\(WorkspaceKnowledge.list(files)) all query ‘\(indexName)’ — no single Ask button")
            }
            guard !clauses.isEmpty else { return nil }
            return clauses.joined(separator: "; ") + ". Rename one so this index has a single flow on each side."
        }

        /// "`Ingest.cat` builds" / "`DocChat.cat` queries" / both joined — the flow each button
        /// runs, for whichever side is `.one`. `nil` when neither side is (both `.none`, both
        /// `.ambiguous`, or one of each) — `ambiguityNote` names an ambiguous side instead.
        ///
        /// CFM-R17-FIX-11(b): `WorkspaceListView` used to name a flow only in the "not built
        /// yet" placeholder, which vanished the moment a manifest existed — after the first
        /// Build the card read `[Rebuild] [Ask]` with no filenames anywhere, and `runFlow`'s
        /// `autoRun: true` meant a click executed a flow the user was never shown the name of.
        /// This is unconditional on whether the index is built.
        var namesCaption: String? {
            var parts: [String] = []
            if let builder = buildFile { parts.append("\(builder) builds") }
            if let querier = askFile { parts.append("\(querier) queries") }
            guard !parts.isEmpty else { return nil }
            return parts.joined(separator: " · ")
        }
    }

    /// Classify one parsed flow. `usesGraph` is `doc`'s own resolved `uses:` graph (the caller
    /// resolves it — `UsesResolver` needs a workspace + id, so this stays pure); a row calling
    /// an entry in it inherits that used flow's index use, recursively through its own further
    /// `uses:` (CFM-R17-FIX-11(e), owner-ruled 2026-09-04). A caller whose retrieval lives
    /// entirely behind a `uses:` composite — `AskYourDocs.cat` calling `RagQuery.cat` — used to
    /// classify `.plain`; it now inherits `RagQuery`'s `Read Index`/`Retrieve` and classifies as
    /// the querier. `usesGraph` defaults empty, so a flow with no `uses:` (or a caller nobody
    /// passed a graph for) classifies exactly as before.
    static func role(of doc: FlowDocument, usesGraph: [String: FlowInterpreter.UsedFlow] = [:]) -> FlowIndexUse {
        var use = indexUse(rows: doc.rows)
        for row in flatten(doc.rows) {
            guard let task = row.task, let used = usesGraph[task] else { continue }
            let callee = indexUseOfUsedFlow(used)
            use.builds += callee.builds
            use.queries += callee.queries
        }
        return FlowIndexUse(builds: dedupe(use.builds), queries: dedupe(use.queries))
    }

    /// The Knowledge Base card candidates across a workspace's flows (filename → parsed doc),
    /// sorted by index name — one card per name **any** flow mentions on either side, absent
    /// side `.none` (CFM-R17-FIX-11(c)). Pure: no filesystem access, so the caller decides what
    /// to render (and, per `usesGraphs` below, what counts as a candidate at all).
    ///
    /// `usesGraphs` is `[file: doc's own resolved uses: graph]` (CFM-R17-FIX-11(e)) — sparse,
    /// only flows with a `uses:` section need an entry, and a missing one classifies that flow
    /// with no inheritance (same as omitting the parameter). **The caller must not include a
    /// flow that is itself the *target* of some sibling's `uses:` line in `flows`** — a `uses:`
    /// composite (`RagQuery.cat`) is a library component, not a runnable entry point (its first
    /// row typically takes input the caller supplies, per `AppFlowExecutorFactory`'s
    /// caller-only `transforms` map — the same "the caller supplies what the callee needs"
    /// shape `-10` found for `transforms:`); leaving it in would make it compete with, and
    /// often out-vote, the caller its `uses:` line exists to reach — the exact defect this item
    /// fixes, arrived at a different way.
    static func classify(flows: [(file: String, doc: FlowDocument)],
                         usesGraphs: [String: [String: FlowInterpreter.UsedFlow]] = [:]) -> [IndexCard] {
        var builders: [String: [String]] = [:]
        var queriers: [String: [String]] = [:]
        for entry in flows {
            let use = role(of: entry.doc, usesGraph: usesGraphs[entry.file] ?? [:])
            for name in use.builds { builders[name, default: []].append(entry.file) }
            for name in use.queries { queriers[name, default: []].append(entry.file) }
        }
        func side(_ files: [String]) -> IndexCard.Side {
            switch files.count {
            case 0: return .none
            case 1: return .one(files[0])
            default: return .ambiguous(files)
            }
        }
        return Set(builders.keys).union(queriers.keys).sorted().map { name in
            IndexCard(indexName: name, builder: side(builders[name] ?? []), querier: side(queriers[name] ?? []))
        }
    }

    /// Whether `card` is worth rendering as a Knowledge Base card (CFM-R17-FIX-11(c),
    /// owner-ruled 2026-09-04): a builder makes it always actionable (Build creates the
    /// index); with no builder, only an index that already exists earns a card — a querier
    /// with nothing to read yet is a dead end, not a card. Pure: the caller (the one place
    /// that can see disk, `WorkspaceListView.manifest(for:)`) supplies `indexExists`.
    static func isCardWorthRendering(_ card: IndexCard, indexExists: Bool) -> Bool {
        card.builder != .none || indexExists
    }

    // MARK: - Helpers

    /// "a", "a and b", "a, b, and c".
    static func list(_ files: [String]) -> String {
        switch files.count {
        case 0, 1: return files.joined()
        case 2: return "\(files[0]) and \(files[1])"
        default: return files.dropLast().joined(separator: ", ") + ", and " + files[files.count - 1]
        }
    }

    /// The name a `Store Index` row writes: `name=`, else the first bare token — exactly what
    /// `StoreIndexTool` resolves (`value(for: "name") ?? firstBare()`, and `index_store.py`'s
    /// `s.get("name") or s.first_bare()`). CFM-R17-FIX-9(a): **no `row.model` fallback.** The
    /// parser's model/settings split does park a bare `/`-containing name (`Store Index
    /// dir/kb.index`, `Read Index ./kb.index`) in `row.model` — but the tools don't read it
    /// there and neither does the Python, so such a row cannot run in either runtime. Reading
    /// it here would pair a card whose buttons always throw; leave it unclassified instead.
    static func storeIndexName(_ row: Row) -> String? {
        let s = FlowSettings(row.settings)
        return s.value(for: "name") ?? s.firstBare()
    }

    /// The name a `Read Index` row reads — `path=`, else the first bare token — matching
    /// `ReadIndexTool` / `_resolve_path`. No `row.model` fallback (CFM-R17-FIX-9(a); see
    /// `storeIndexName`).
    static func readIndexName(_ row: Row) -> String? {
        FlowSettings(row.settings).pathValue()
    }

    /// Normalise an index name for pairing. `Read Index path=./library.index` and `Store Index
    /// name=library.index` name the same directory (`FlowWorkspace.resolve` collapses both);
    /// compared raw they never pair (CFM-R17-FIX-5). Strips surrounding whitespace, a leading
    /// `./`, and a trailing slash.
    static func normalizedIndexName(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespaces)
        while s.hasPrefix("./") { s.removeFirst(2) }
        while s.hasSuffix("/") { s.removeLast() }
        return s
    }

    private static func dedupe(_ names: [String]) -> [String] {
        var seen = Set<String>()
        return names.filter { seen.insert($0).inserted }
    }

    private static func flatten(_ rows: [Row]) -> [Row] {
        rows.flatMap { [$0] + flatten($0.children) }
    }

    /// `builds`/`queries` over one row list, un-deduplicated (the caller dedupes after
    /// merging in whatever a `uses:` expansion contributes) — the row-scanning half of `role`,
    /// shared with `indexUseOfUsedFlow` since a used flow's `rows` is the same `[Row]` shape.
    private static func indexUse(rows: [Row]) -> FlowIndexUse {
        let flat = flatten(rows)
        let builds = flat
            .compactMap { $0.task == "Store Index" ? storeIndexName($0) : nil }
            .map(normalizedIndexName)
        var queries: [String] = []
        if flat.contains(where: { $0.task == "Retrieve" || $0.task == "Keyword Search" }) {
            queries = flat
                .compactMap { $0.task == "Read Index" ? readIndexName($0) : nil }
                .map(normalizedIndexName)
        }
        return FlowIndexUse(builds: builds, queries: queries)
    }

    /// CFM-R17-FIX-11(e): a used flow's own index use, plus whatever it inherits through its
    /// further `uses:` (`used.nested` — `UsesResolver` already resolved these recursively and
    /// ruled out cycles, E115). Mirrors `role(of:usesGraph:)`'s row-walk, over
    /// `FlowInterpreter.UsedFlow.rows` instead of a `FlowDocument`'s.
    private static func indexUseOfUsedFlow(_ used: FlowInterpreter.UsedFlow) -> FlowIndexUse {
        var use = indexUse(rows: used.rows)
        for row in flatten(used.rows) {
            guard let task = row.task, let nested = used.nested[task] else { continue }
            let callee = indexUseOfUsedFlow(nested)
            use.builds += callee.builds
            use.queries += callee.queries
        }
        return use
    }
}
