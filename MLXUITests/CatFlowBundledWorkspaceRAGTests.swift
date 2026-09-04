import Testing
import Foundation
@testable import MLXUI

/// CFM-R17-6 — `ask_your_docs` ships `16-IngestFolder` + `18-DocChat` as one workspace
/// sharing `library.index`. It carries **no prebuilt index** — `Ingest.cat` builds it from
/// the workspace's own `docs/` PDFs, `DocChat.cat` reads and chats against it. This is the
/// first bundled workspace that runs a RAG loop end to end (real models — a smoke row);
/// here the structure, the pairing, and the cache-key freshness are pinned.
struct CatFlowBundledWorkspaceRAGTests {

    private func prepared() throws -> (URL, FlowWorkspace, WorkspaceStore.Workspace) {
        let meta = try #require(BundledWorkspaces.meta(id: "ask_your_docs"))
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-ayd-\(UUID().uuidString)")
        let root = base.appendingPathComponent("workspaces")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let ws = FlowWorkspace(root: root)
        try BundledWorkspaces.prepare(meta, workspace: ws)
        let listed = try #require(WorkspaceStore.scan(workspace: ws).first)
        return (base, ws, listed)
    }

    @Test func materialisesTheTwoFlowsAndTheCorpusButNoIndex() throws {
        let (base, ws, _) = try prepared()
        defer { try? FileManager.default.removeItem(at: base) }
        let dir = ws.directory(for: "ask_your_docs")
        let fm = FileManager.default
        #expect(fm.fileExists(atPath: dir.appendingPathComponent("Ingest.cat").path))
        #expect(fm.fileExists(atPath: dir.appendingPathComponent("DocChat.cat").path))
        for pdf in ["doc-a.pdf", "doc-b.pdf", "doc-c.pdf"] {
            #expect(fm.fileExists(atPath: dir.appendingPathComponent("docs/\(pdf)").path), "\(pdf) missing")
        }
        // No prebuilt fixture — Build makes it.
        #expect(!fm.fileExists(atPath: dir.appendingPathComponent("library.index").path))
    }

    @Test func listsAsATwoFlowWorkspaceThatPairsIntoOneIndexCard() throws {
        let (base, _, listed) = try prepared()
        defer { try? FileManager.default.removeItem(at: base) }
        #expect(listed.workspaceID == "ask_your_docs")
        #expect(listed.flows.map(\.title).sorted() == ["DocChat", "Ingest"])

        let parsed = listed.flows.compactMap { f -> (file: String, doc: FlowDocument)? in
            (try? WorkspaceStore.loadDocument(flow: f)).map { (f.url.lastPathComponent, $0) }
        }
        let c = WorkspaceKnowledge.classify(flows: parsed)
        #expect(c.count == 1)
        let card = try #require(c.first)
        #expect(card.indexName == "library.index")
        #expect(card.builder == .one("Ingest.cat"))
        #expect(card.querier == .one("DocChat.cat"))
        #expect(card.ambiguityNote == nil)
    }

    @Test func theBuilderFlowCanRunUnderTheWorkspaceScope() throws {
        let (base, ws, _) = try prepared()
        defer { try? FileManager.default.removeItem(at: base) }
        let text = try String(contentsOf: ws.directory(for: "ask_your_docs").appendingPathComponent("Ingest.cat"),
                              encoding: .utf8)
        let doc = try CatParser.parse(text)
        let scope = FlowScope(identity: "Ingest", workspace: ws, locationID: "ask_your_docs",
                              flowText: text,
                              selfFile: ws.directory(for: "ask_your_docs").appendingPathComponent("Ingest.cat"))
        #expect(FlowRunner.canRun(doc, scope: scope) == .runnable)
    }

    /// The done-when's freshness clause, against this workspace's own `library.index`: a
    /// Rebuild that changed the corpus changes `Retrieve`'s cache key, so a second Ask
    /// re-runs rather than serving the first Ask's chunks.
    @Test func rebuildingChangesTheQueryCacheKey() async throws {
        let (base, ws, _) = try prepared()
        defer { try? FileManager.default.removeItem(at: base) }
        let dir = ws.directory(for: "ask_your_docs")

        func build(_ chunks: [String]) async throws {
            let vecs = try chunks.indices.map { i -> Item in
                var v = [Float](repeating: 0, count: 8); v[i % 8] = 1
                let u = dir.appendingPathComponent("v\(i).npy")
                try NpyCodec.save(v, to: u)
                return Item(kind: .vector, value: nil, path: u, sourceText: nil)
            }
            _ = try await StoreIndexTool(workspace: ws, flowID: "ask_your_docs",
                                         settings: "library.index; embedder=BGE-M3")
                .run(inputs: [
                    Asset(items: chunks.map { Item(kind: .text, value: $0, path: nil, sourceText: nil) }),
                    Asset(items: vecs),
                ])
        }
        let q = dir.appendingPathComponent("q.npy")
        try NpyCodec.save([0, 1, 0, 0, 0, 0, 0, 0], to: q)
        let indexDir = dir.appendingPathComponent("library.index")
        func key() throws -> String {
            try CacheKey.cacheKey(task: "Retrieve", model: nil, settings: "top_k=5",
                inputs: [Asset(items: [Item(kind: .index, value: nil, path: indexDir, sourceText: nil)]),
                         Asset(items: [Item(kind: .vector, value: nil, path: q, sourceText: nil)])],
                realism: "real")
        }
        try await build(["the pto policy changed in 2026", "expenses over 50 need approval"])
        let before = try key()
        try await build(["the pto policy is unchanged", "expenses over 50 need approval", "new: remote stipend"])
        #expect(before != (try key()))
    }
}
