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

    /// One index a workspace's flows pair up on.
    struct IndexPairing: Equatable, Sendable {
        let indexName: String
        /// The `.cat` file whose terminal row builds it.
        let builderFile: String
        /// The `.cat` file that reads and retrieves from it.
        let querierFile: String
    }

    /// An index name more than one flow builds, or more than one flow queries — so a single
    /// "Build"/"Ask" button would be a silent coin-flip (CFM-R17-FIX-5). Surfaced instead of
    /// paired.
    struct IndexCollision: Equatable, Sendable {
        let indexName: String
        /// Every flow whose terminal row builds this name (source order).
        let builderFiles: [String]
        /// Every flow that queries this name (source order).
        let querierFiles: [String]

        /// A plain sentence naming which flows collide and how to fix it.
        var message: String {
            var clauses: [String] = []
            if builderFiles.count > 1 {
                clauses.append("\(list(builderFiles)) all build it")
            }
            if querierFiles.count > 1 {
                clauses.append("\(list(querierFiles)) all query it")
            }
            let detail = clauses.joined(separator: "; ")
            return "'\(indexName)' isn't paired into a card: \(detail). Rename an index so each one has a single builder and a single querier."
        }

        private func list(_ files: [String]) -> String {
            switch files.count {
            case 0, 1: return files.joined()
            case 2: return "\(files[0]) and \(files[1])"
            default: return files.dropLast().joined(separator: ", ") + ", and " + files[files.count - 1]
            }
        }
    }

    /// The knowledge base derived from a workspace's flows: the unambiguous pairs, and the
    /// name collisions that were held back from pairing.
    struct Knowledge: Equatable, Sendable {
        /// Builder + querier pairs where exactly one flow does each, sorted by index name.
        let pairings: [IndexPairing]
        /// Names where more than one flow builds or more than one queries (both sides present
        /// — a one-sided name still just lists plainly, CFM-R17-3), sorted by index name.
        let collisions: [IndexCollision]

        var isEmpty: Bool { pairings.isEmpty && collisions.isEmpty }
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

    /// The knowledge base across a workspace's flows (filename → parsed doc). A builder and a
    /// querier of the same normalized name pair; a name with builders **and** queriers but
    /// more than one on a side is a collision, not a coin-flip (CFM-R17-FIX-5); a name with
    /// only one side does not pair and its flows still list plainly (CFM-R17-3).
    static func classify(flows: [(file: String, doc: FlowDocument)]) -> Knowledge {
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
        var pairings: [IndexPairing] = []
        var collisions: [IndexCollision] = []
        for name in Set(builders.keys).union(queriers.keys).sorted() {
            let b = builders[name] ?? []
            let q = queriers[name] ?? []
            guard !b.isEmpty, !q.isEmpty else { continue }   // one-sided → no card, no collision
            if b.count == 1, q.count == 1 {
                pairings.append(IndexPairing(indexName: name, builderFile: b[0], querierFile: q[0]))
            } else {
                collisions.append(IndexCollision(indexName: name, builderFiles: b, querierFiles: q))
            }
        }
        return Knowledge(pairings: pairings, collisions: collisions)
    }

    /// The unambiguous index pairings only — the pre-CFM-R17-FIX-5 shape, kept for callers
    /// that render just the cards.
    static func pairings(flows: [(file: String, doc: FlowDocument)]) -> [IndexPairing] {
        classify(flows: flows).pairings
    }

    // MARK: - Helpers

    /// The name a `Store Index` row writes: `name=`, else the first bare token, else the row's
    /// `model` slot — the parser's model/settings heuristic parks a `./`-prefixed path there
    /// (`CatParser.splitModelSettings`, Python-parity), so a bare `./library.index` shows up
    /// as `row.model`, not in `settings`.
    static func storeIndexName(_ row: Row) -> String? {
        let s = FlowSettings(row.settings)
        return s.value(for: "name") ?? s.firstBare() ?? row.model
    }

    /// The name a `Read Index` row reads — `path=`, else the first bare token, else `row.model`
    /// (same parser quirk as `storeIndexName`).
    static func readIndexName(_ row: Row) -> String? {
        FlowSettings(row.settings).pathValue() ?? row.model
    }

    /// Normalise an index name for pairing. `Read Index ./library.index` and `Store Index
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
