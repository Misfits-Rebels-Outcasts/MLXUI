//
//  SemanticSearchTool.swift
//  MLXUI — agentic chat tools, slice AG4b.
//
//  `semantic_search` — search text documents by meaning (embedding similarity). Provide
//  `documents` (one per line) to build a small on-disk index; omit them on later calls to
//  search the persisted index. Backed by a locally-installed embedding model.
//
//  The index lives in the app container (`AI Browser/agent-index/index.json`), which is always
//  writable under the sandbox — no entitlement needed. Ranking + parsing + persistence are pure
//  helpers exercised by synchronous tests; only the embedding call itself isn't unit-tested.
//

import Foundation
import MLXLMCommon

/// A tiny persisted vector index: documents plus their embeddings, all produced by one embedding
/// model (so the vectors are comparable). `nonisolated` + `Sendable` + `Codable` for off-MainActor
/// use and JSON persistence.
nonisolated struct SemanticIndex: Codable, Sendable, Equatable {
    struct Entry: Codable, Sendable, Equatable {
        let text: String
        let vector: [Float]
    }

    /// The embedding model the vectors were produced with; the query is embedded with the same one.
    let modelID: String
    var entries: [Entry]

    /// Top-k documents by cosine similarity to the query vector, highest first.
    func search(queryVector: [Float], topK: Int) -> [(text: String, score: Float)] {
        let ranked = entries
            .map { (text: $0.text, score: EmbedTextTool.cosine(queryVector, $0.vector)) }
            .sorted { $0.score > $1.score }
        return Array(ranked.prefix(max(0, topK)))
    }

    /// Split a newline-separated blob into trimmed, non-empty documents.
    static func documents(from raw: String) -> [String] {
        raw.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    // MARK: Persistence (app container — no entitlement needed)

    private static var storeURL: URL {
        ModelStore.shared.agentIndexDirectory.appendingPathComponent("index.json")
    }

    static func load() -> SemanticIndex? {
        guard let data = try? Data(contentsOf: storeURL) else { return nil }
        return try? JSONDecoder().decode(SemanticIndex.self, from: data)
    }

    func save() throws {
        let url = Self.storeURL
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(self).write(to: url, options: .atomic)
    }
}

/// Searches text documents by meaning using a locally-installed embedding model, building or
/// reusing a small on-disk index.
nonisolated struct SemanticSearchTool: AgentTool {
    let name = "semantic_search"
    let toolDescription = """
        Search text documents by meaning (semantic similarity), returning the most relevant ones. \
        Provide 'documents' (one per line) to build the searchable index — this replaces any \
        existing index. On later calls, omit 'documents' to search the index you already built. \
        Uses a locally-installed embedding model.
        """
    let parameters: [ToolParameter] = [
        .required("query", type: .string, description: "What to search for."),
        .optional("documents", type: .string,
                  description: "Optional newline-separated documents to index (one per line). Replaces any existing index."),
        .optional("top_k", type: .int, description: "How many results to return (default 3)."),
    ]

    func execute(arguments: [String: JSONValue]) async throws -> String {
        let query = (arguments.string("query") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return "Error: missing required 'query' argument." }
        let topK = min(20, max(1, arguments.int("top_k") ?? 3))

        let index: SemanticIndex
        let documentsRaw = arguments.string("documents")?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let documentsRaw, !documentsRaw.isEmpty {
            let docs = SemanticIndex.documents(from: documentsRaw)
            guard !docs.isEmpty else { return "Error: 'documents' contained no non-empty lines." }
            guard let modelID = InstalledModelIndex.loadInstalled().best(kind: .embedding) else {
                return "Error: no embedding model is installed. Install one from the Browse tab first."
            }
            let vectors = try await EmbeddingEngine.embed(
                docs, modelDirectory: EmbeddingEngine.installedModelDirectory(id: modelID))
            guard vectors.count == docs.count else {
                return "Error: embedding did not index all documents."
            }
            index = SemanticIndex(modelID: modelID,
                                  entries: zip(docs, vectors).map { .init(text: $0, vector: $1) })
            try? index.save()
        } else {
            guard let existing = SemanticIndex.load(), !existing.entries.isEmpty else {
                return "Error: no search index yet. Provide 'documents' (one per line) to build one first."
            }
            index = existing
        }

        // Embed the query with the SAME model the index was built with (comparable vectors).
        guard let queryVector = try await EmbeddingEngine.embed(
            [query], modelDirectory: EmbeddingEngine.installedModelDirectory(id: index.modelID)).first
        else { return "Error: failed to embed the query." }

        return Self.formatResults(index.search(queryVector: queryVector, topK: topK))
    }

    /// Numbered "[score] text" lines, best match first. Pure, so it is unit-testable.
    static func formatResults(_ results: [(text: String, score: Float)]) -> String {
        guard !results.isEmpty else { return "No matching documents." }
        return results.enumerated().map { i, r in
            "\(i + 1). [\(String(format: "%.3f", r.score))] \(r.text)"
        }.joined(separator: "\n")
    }
}
