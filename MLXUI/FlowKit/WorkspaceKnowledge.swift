import Foundation

/// CFM-R17-4 — the knowledge-base surface, **derived** from a workspace's flows. No
/// `workspace.json`, no naming convention, no new syntax: classify each parsed flow by what
/// it does with an index, and when a builder and a querier name the same index, the workspace
/// view offers it as one card with two verbs.
nonisolated enum WorkspaceKnowledge {

    /// How a flow relates to an index.
    enum FlowRole: Equatable, Sendable {
        /// The flow's **terminal** row — the last row in execution order, recursing into a
        /// trailing block — is `Store Index <name>`. (CFM-R17-FIX-4: was the last *top-level*
        /// row, so a `Store Index` ending a `<list>` was invisible; the querier side has
        /// always scanned every depth, and now both sides do.)
        case builds(index: String)
        /// The flow reads one or more indexes (`Read Index <name>` at any depth) **and** has a
        /// `Retrieve` / `Keyword Search`. It queries *every* index it reads — names in source
        /// order, de-duplicated, normalized (CFM-R17-FIX-4: a flow reading N indexes yields N
        /// names, not just the first).
        case queries(indexes: [String])
        /// Anything else.
        case plain
    }

    /// One index a workspace's flows work with — a card in the Knowledge Base. A `Side` is
    /// `.one` when exactly one flow does that job (its button runs that flow), `.ambiguous`
    /// when more than one does (no button for it — the note names them), `.none` when none do.
    /// A card exists only when **both** sides have at least one flow (CFM-R17-3: a one-sided
    /// name still just lists plainly).
    ///
    /// CFM-R17-FIX-9(c): an ambiguous side no longer suppresses the whole card. `1 builder +
    /// 2 queriers` keeps its Build button and notes the Ask collision; the mirror keeps Ask.
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

        /// A plain note covering only the ambiguous side(s), or nil when both are unambiguous.
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
    }

    /// Classify one parsed flow.
    static func role(of doc: FlowDocument) -> FlowRole {
        let rows = flatten(doc.rows)
        // builds — the terminal row (last in execution order) is Store Index <name>.
        if let terminal = rows.last, terminal.task == "Store Index",
           let name = storeIndexName(terminal) {
            return .builds(index: normalizedIndexName(name))
        }
        // queries — every Read Index <name>, plus a Retrieve / Keyword Search somewhere.
        let names = rows
            .compactMap { $0.task == "Read Index" ? readIndexName($0) : nil }
            .map(normalizedIndexName)
        if !names.isEmpty,
           rows.contains(where: { $0.task == "Retrieve" || $0.task == "Keyword Search" }) {
            return .queries(indexes: dedupe(names))
        }
        return .plain
    }

    /// The Knowledge Base cards across a workspace's flows (filename → parsed doc), sorted by
    /// index name. One card per normalized index name that has **both** a builder and a
    /// querier; each side is `.one` / `.ambiguous` / `.none` per how many flows do that job.
    /// A name with only one side present produces no card (CFM-R17-3).
    static func classify(flows: [(file: String, doc: FlowDocument)]) -> [IndexCard] {
        var builders: [String: [String]] = [:]
        var queriers: [String: [String]] = [:]
        for entry in flows {
            switch role(of: entry.doc) {
            case .builds(let name):
                builders[name, default: []].append(entry.file)
            case .queries(let names):
                for name in names { queriers[name, default: []].append(entry.file) }
            case .plain:
                continue
            }
        }
        func side(_ files: [String]) -> IndexCard.Side {
            switch files.count {
            case 0: return .none
            case 1: return .one(files[0])
            default: return .ambiguous(files)
            }
        }
        var cards: [IndexCard] = []
        for name in Set(builders.keys).union(queriers.keys).sorted() {
            let b = builders[name] ?? []
            let q = queriers[name] ?? []
            guard !b.isEmpty, !q.isEmpty else { continue }   // one-sided → no card (CFM-R17-3)
            cards.append(IndexCard(indexName: name, builder: side(b), querier: side(q)))
        }
        return cards
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
}
