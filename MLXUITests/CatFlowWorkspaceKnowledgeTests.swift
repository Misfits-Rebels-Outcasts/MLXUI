import Testing
import Foundation
@testable import MLXUI

/// CFM-R17-4 — the knowledge-base surface is **derived** from the rows: a flow whose last row
/// is `Store Index <name>` builds it; a flow with `Read Index <name>` plus `Retrieve` /
/// `Keyword Search` queries it; a builder + a querier of the same name pair into one card.
/// Freshness needs no new machinery — a rebuilt index changes `Retrieve`'s cache key through
/// the directory hash.
struct CatFlowWorkspaceKnowledgeTests {

    private func doc(_ text: String) throws -> FlowDocument { try CatParser.parse(text) }

    // MARK: - Classification

    @Test func lastRowStoreIndexBuilds() throws {
        let d = try doc("catflow 0.8\n1. Read Files   docs/\n2. Embed   BGE-M3\n6. Store Index   (1,2)   library.index\n")
        #expect(WorkspaceKnowledge.role(of: d) == .builds(index: "library.index"))
    }

    @Test func storeIndexNotLastIsNotABuilder() throws {
        // Store Index in the middle, a Save Text after it — not "the last row".
        let d = try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   library.index\n3. Save Text   done.md\n")
        #expect(WorkspaceKnowledge.role(of: d) == .plain)
    }

    @Test func readIndexPlusRetrieveQueries() throws {
        let d = try doc("catflow 0.8\n1. Read Index   library.index\n2. Embed   BGE-M3\n3. Retrieve   (1,2)\n4. Answer   Qwen3 8B\n")
        #expect(WorkspaceKnowledge.role(of: d) == .queries(indexes: ["library.index"]))
    }

    @Test func readIndexPlusKeywordSearchQueries() throws {
        let d = try doc("catflow 0.8\n1. Read Index   kb.index\n2. Keyword Search   (1,1)\n")
        #expect(WorkspaceKnowledge.role(of: d) == .queries(indexes: ["kb.index"]))
    }

    // MARK: - CFM-R17-FIX-4: N indexes → N names; builder/querier over the same row set

    @Test func aFlowReadingTwoIndexesQueriesBoth() throws {
        // `20-TwoIndexAnalyst`-shaped: both Read Index rows live inside a `<list>` block.
        let two = try String(contentsOf: galleryURL("20-TwoIndexAnalyst"), encoding: .utf8)
        #expect(WorkspaceKnowledge.role(of: try doc(two)) == .queries(indexes: ["hr.index", "eng.index"]))
    }

    @Test func routerDeskQueriesBothBranchIndexes() throws {
        let router = try String(contentsOf: galleryURL("45-RouterDesk"), encoding: .utf8)
        #expect(WorkspaceKnowledge.role(of: try doc(router)) == .queries(indexes: ["billing-kb.index", "docs-kb.index"]))
    }

    @Test func repeatedReadIndexNameIsListedOnce() throws {
        let d = try doc("""
        catflow 0.8
        1. Read Index   kb.index
        2. Read Index   kb.index
        3. Retrieve   (1,1)
        """)
        #expect(WorkspaceKnowledge.role(of: d) == .queries(indexes: ["kb.index"]))
    }

    @Test func storeIndexEndingATrailingListStillBuilds() throws {
        // The asymmetry FIX-4 names: the terminal row is `Store Index`, one level down.
        let d = try doc("""
        catflow 0.8
        1. Read Files   docs/
        2. <list finish>
             1. Embed         BGE-M3
             2. Store Index   library.index
        """)
        #expect(WorkspaceKnowledge.role(of: d) == .builds(index: "library.index"))
    }

    @Test func storeIndexInATrailingListFollowedByMoreIsNotABuilder() throws {
        let d = try doc("""
        catflow 0.8
        1. <list build>
             1. Embed         BGE-M3
             2. Store Index   library.index
        2. Save Text   done.md
        """)
        #expect(WorkspaceKnowledge.role(of: d) == .plain)
    }

    // MARK: - CFM-R17-FIX-5: normalization + ambiguous pairings

    @Test func aLeadingDotSlashAndATrailingSlashNormaliseToTheSameName() throws {
        // Both names sit where the tools read them (`name=` / `path=`), so both rows run —
        // and `./library.index` normalises to `library.index` for the pairing (CFM-R17-FIX-5;
        // FIX-9(a) took the `row.model` fallback back out, so a *bare* `./x` no longer pairs).
        let builder = try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   (1,1)   name=library.index\n")
        let querier = try doc("catflow 0.8\n1. Read Index   path=./library.index\n2. Retrieve   (1,1)\n")
        let pairs = WorkspaceKnowledge.pairings(flows: [
            (file: "Build.cat", doc: builder), (file: "Ask.cat", doc: querier),
        ])
        #expect(pairs.count == 1)
        #expect(pairs.first?.indexName == "library.index")
    }

    @Test func aBareSlashNameLandsInModelAndIsNotClassified() throws {
        // CFM-R17-FIX-9(a): `Store Index dir/kb.index` / `Read Index ./kb.index` — the parser
        // parks the name in `row.model`, where neither the tools nor the Python read it. The
        // classifier must not pair a card whose Build/Ask would always throw; the rows stay
        // unclassified (`.plain`), exactly as before CFM-R17-FIX-5.
        let builder = try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   indexes/kb.index\n")
        let querier = try doc("catflow 0.8\n1. Read Index   ./kb.index\n2. Retrieve   (1,1)\n")
        #expect(WorkspaceKnowledge.role(of: builder) == .plain)
        #expect(WorkspaceKnowledge.role(of: querier) == .plain)
        #expect(WorkspaceKnowledge.classify(flows: [
            (file: "Build.cat", doc: builder), (file: "Ask.cat", doc: querier),
        ]).isEmpty)
    }

    @Test func twoBuildersOfOneNameCollideInsteadOfCoinFlipping() throws {
        let k = WorkspaceKnowledge.classify(flows: [
            (file: "Ingest.cat", doc: try doc("catflow 0.8\n1. Read Files   docs/\n2. Embed   BGE-M3\n3. Store Index   (1,2)   library.index\n")),
            (file: "InboxIngest.cat", doc: try doc("catflow 0.8\n1. On File   inbox/\n2. Embed   BGE-M3\n3. Store Index   (1,2)   library.index\n")),
            (file: "Ask.cat", doc: try doc("catflow 0.8\n1. Read Index   library.index\n2. Retrieve   (1,1)\n")),
        ])
        #expect(k.pairings.isEmpty)
        let c = try #require(k.collisions.first)
        #expect(c.indexName == "library.index")
        #expect(Set(c.builderFiles) == ["Ingest.cat", "InboxIngest.cat"])
        #expect(c.message.contains("Ingest.cat"))
        #expect(c.message.contains("InboxIngest.cat"))
        #expect(c.message.contains("all build it"))
    }

    @Test func twoQueriersOfOneNameCollide() throws {
        let k = WorkspaceKnowledge.classify(flows: [
            (file: "Build.cat", doc: try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   (1,1)   kb.index\n")),
            (file: "AskA.cat", doc: try doc("catflow 0.8\n1. Read Index   kb.index\n2. Retrieve   (1,1)\n")),
            (file: "AskB.cat", doc: try doc("catflow 0.8\n1. Read Index   kb.index\n2. Keyword Search   (1,1)\n")),
        ])
        #expect(k.pairings.isEmpty)
        #expect(k.collisions.first?.querierFiles.sorted() == ["AskA.cat", "AskB.cat"])
        #expect(k.collisions.first?.message.contains("all query it") == true)
    }

    @Test func twoBuildersButNoQuerierIsNotACollision() throws {
        // One-sided stays CFM-R17-3 behaviour: no card, and nothing to disambiguate.
        let k = WorkspaceKnowledge.classify(flows: [
            (file: "A.cat", doc: try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   x.index\n")),
            (file: "B.cat", doc: try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   x.index\n")),
        ])
        #expect(k.isEmpty)
    }

    @Test func twoIndexAnalystPlusTwoBuildersProducesTwoCards() throws {
        let two = try String(contentsOf: galleryURL("20-TwoIndexAnalyst"), encoding: .utf8)
        let k = WorkspaceKnowledge.classify(flows: [
            (file: "Analyst.cat", doc: try doc(two)),
            (file: "BuildHR.cat", doc: try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   (1,1)   hr.index\n")),
            (file: "BuildEng.cat", doc: try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   (1,1)   eng.index\n")),
        ])
        #expect(k.collisions.isEmpty)
        #expect(k.pairings.map(\.indexName) == ["eng.index", "hr.index"])
        #expect(k.pairings.allSatisfy { $0.querierFile == "Analyst.cat" })
        #expect(Set(k.pairings.map(\.builderFile)) == ["BuildHR.cat", "BuildEng.cat"])
    }

    @Test func readIndexAloneIsPlain() throws {
        let d = try doc("catflow 0.8\n1. Read Index   kb.index\n2. Save Text   x.md\n")
        #expect(WorkspaceKnowledge.role(of: d) == .plain)
    }

    @Test func explicitNameSettingWins() throws {
        let d = try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   (1,1)   name=hr.index\n")
        #expect(WorkspaceKnowledge.role(of: d) == .builds(index: "hr.index"))
    }

    // MARK: - Pairing (the 16 + 18 done-when)

    @Test func ingestAndDocChatRowsPairIntoOneIndexCard() throws {
        let ingest = try String(contentsOf: galleryURL("16-IngestFolder"), encoding: .utf8)
        let docChat = try String(contentsOf: galleryURL("18-DocChat"), encoding: .utf8)
        let pairs = WorkspaceKnowledge.pairings(flows: [
            (file: "Ingest.cat", doc: try doc(ingest)),
            (file: "Ask.cat", doc: try doc(docChat)),
        ])
        #expect(pairs.count == 1)
        let p = try #require(pairs.first)
        #expect(p.indexName == "library.index")
        #expect(p.builderFile == "Ingest.cat")
        #expect(p.querierFile == "Ask.cat")
    }

    @Test func flowsThatDoNotPairUpYieldNoCard() throws {
        let pairs = WorkspaceKnowledge.pairings(flows: [
            (file: "A.cat", doc: try doc("catflow 0.8\n1. Read Text   a.txt\n2. Save Text   b.md\n")),
            (file: "B.cat", doc: try doc("catflow 0.8\n1. Read Index   only.index\n2. Retrieve   (1,1)\n")),
        ])
        #expect(pairs.isEmpty)   // a querier with no matching builder — not an error, no card
    }

    @Test func differentIndexNamesDoNotPair() throws {
        let pairs = WorkspaceKnowledge.pairings(flows: [
            (file: "Build.cat", doc: try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   (1,1)   hr.index\n")),
            (file: "Ask.cat", doc: try doc("catflow 0.8\n1. Read Index   eng.index\n2. Retrieve   (1,1)\n")),
        ])
        #expect(pairs.isEmpty)
    }

    // MARK: - Freshness at the cache-key boundary

    @Test func rebuildingTheIndexChangesRetrievesCacheKey() async throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-kb-fresh-\(UUID().uuidString)")
        let ws = FlowWorkspace(root: base.appendingPathComponent("workspaces"))
        let dir = ws.directory(for: "kb")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }

        func buildIndex(chunks: [String]) async throws {
            let vectorItems = try chunks.indices.map { i -> Item in
                let url = dir.appendingPathComponent("v\(i).npy")
                var v = [Float](repeating: 0, count: 8); v[i % 8] = 1
                try NpyCodec.save(v, to: url)
                return Item(kind: .vector, value: nil, path: url, sourceText: nil)
            }
            let store = StoreIndexTool(workspace: ws, flowID: "kb", settings: "library.index; embedder=BGE-M3")
            _ = try await store.run(inputs: [
                Asset(items: chunks.map { Item(kind: .text, value: $0, path: nil, sourceText: nil) }),
                Asset(items: vectorItems),
            ])
        }

        let queryURL = dir.appendingPathComponent("q.npy")
        try NpyCodec.save([0, 1, 0, 0, 0, 0, 0, 0], to: queryURL)
        let indexDir = dir.appendingPathComponent("library.index")

        func retrieveKey() throws -> String {
            try CacheKey.cacheKey(
                task: "Retrieve", model: nil, settings: "top_k=5",
                inputs: [
                    Asset(items: [Item(kind: .index, value: nil, path: indexDir, sourceText: nil)]),
                    Asset(items: [Item(kind: .vector, value: nil, path: queryURL, sourceText: nil)]),
                ],
                realism: "real")
        }

        try await buildIndex(chunks: ["alpha", "beta", "gamma"])
        let keyA = try retrieveKey()

        // A Rebuild that changed the corpus.
        try await buildIndex(chunks: ["delta", "epsilon", "zeta", "eta"])
        let keyB = try retrieveKey()

        #expect(keyA != keyB, "a rebuilt index must change Retrieve's key — else Ask serves stale chunks")
    }

    // MARK: - helpers

    private func galleryURL(_ stem: String) -> URL {
        var dir = URL(fileURLWithPath: #filePath)
        while dir.lastPathComponent != "MLXUITests" { dir = dir.deletingLastPathComponent() }
        return dir.deletingLastPathComponent()
            .appendingPathComponent("MLXUI/Resources/Gallery/\(stem).cat")
    }
}
