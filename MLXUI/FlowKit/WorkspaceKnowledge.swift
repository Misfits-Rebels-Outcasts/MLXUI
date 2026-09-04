import Foundation

/// CFM-R17-4 — the knowledge-base surface, **derived** from a workspace's flows. No
/// `workspace.json`, no naming convention, no new syntax: classify each parsed flow by what
/// it does with an index, and when a builder and a querier name the same index, the workspace
/// view offers it as one card with two verbs.
nonisolated enum WorkspaceKnowledge {

    /// How a flow relates to an index.
    enum FlowRole: Equatable, Sendable {
        /// The flow's **last** row is `Store Index <name>`.
        case builds(index: String)
        /// The flow has a `Read Index <name>` row **plus** a `Retrieve` or `Keyword Search`.
        case queries(index: String)
        /// Anything else.
        case plain
    }

    /// One index a workspace's flows pair up on.
    struct IndexPairing: Equatable, Sendable {
        let indexName: String
        /// The `.cat` file whose last row builds it.
        let builderFile: String
        /// The `.cat` file that reads and retrieves from it.
        let querierFile: String
    }

    /// Classify one parsed flow.
    static func role(of doc: FlowDocument) -> FlowRole {
        // builds — the last top-level row is Store Index <name>.
        if let last = doc.rows.last, last.task == "Store Index",
           let name = storeIndexName(last) {
            return .builds(index: name)
        }
        // queries — a Read Index <name> anywhere, plus a Retrieve / Keyword Search anywhere.
        let rows = flatten(doc.rows)
        if let read = rows.first(where: { $0.task == "Read Index" }),
           let name = readIndexName(read),
           rows.contains(where: { $0.task == "Retrieve" || $0.task == "Keyword Search" }) {
            return .queries(index: name)
        }
        return .plain
    }

    /// The index pairings across a workspace's flows (filename → parsed doc), sorted by name.
    /// The first builder and the first querier of a given index name pair; a name with only
    /// one side does not pair (and its flows still list plainly — CFM-R17-3).
    static func pairings(flows: [(file: String, doc: FlowDocument)]) -> [IndexPairing] {
        var builders: [String: String] = [:]
        var queriers: [String: String] = [:]
        for entry in flows {
            switch role(of: entry.doc) {
            case .builds(let name):  if builders[name] == nil { builders[name] = entry.file }
            case .queries(let name): if queriers[name] == nil { queriers[name] = entry.file }
            case .plain: continue
            }
        }
        return builders.compactMap { name, builder in
            queriers[name].map { IndexPairing(indexName: name, builderFile: builder, querierFile: $0) }
        }
        .sorted { $0.indexName < $1.indexName }
    }

    // MARK: - Helpers

    static func storeIndexName(_ row: Row) -> String? {
        let s = FlowSettings(row.settings)
        return s.value(for: "name") ?? s.firstBare()
    }

    static func readIndexName(_ row: Row) -> String? {
        FlowSettings(row.settings).pathValue()
    }

    private static func flatten(_ rows: [Row]) -> [Row] {
        rows.flatMap { [$0] + flatten($0.children) }
    }
}
