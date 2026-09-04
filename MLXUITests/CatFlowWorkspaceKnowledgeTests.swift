import Testing
import Foundation
@testable import MLXUI

/// CFM-R17-4 — the knowledge-base surface is **derived** from the rows: a flow builds every
/// index its `Store Index` rows name (anywhere, CFM-R17-FIX-11(a)); a flow with `Read Index`
/// plus `Retrieve`/`Keyword Search` queries every index it reads. The two are independent — a
/// flow may build, query, or do both to the same or different names. `classify` pairs a
/// builder and a querier of the same name into one card; a name only one side mentions still
/// gets a card, one-sided (CFM-R17-FIX-11(c)) — whether that's worth *rendering* is the
/// caller's call (`isCardWorthRendering`), since only the caller can see whether the index
/// exists on disk. Freshness needs no new machinery — a rebuilt index changes `Retrieve`'s
/// cache key through the directory hash.
struct CatFlowWorkspaceKnowledgeTests {

    private func doc(_ text: String) throws -> FlowDocument { try CatParser.parse(text) }

    /// Classify a set of `(file, .cat text)` pairs into Knowledge Base card candidates.
    private func cards(_ flows: [(String, String)]) throws -> [WorkspaceKnowledge.IndexCard] {
        WorkspaceKnowledge.classify(flows: try flows.map { ($0.0, try doc($0.1)) })
    }

    // MARK: - Classification

    @Test func lastRowStoreIndexBuilds() throws {
        let d = try doc("catflow 0.8\n1. Read Files   docs/\n2. Embed   BGE-M3\n6. Store Index   (1,2)   library.index\n")
        #expect(WorkspaceKnowledge.role(of: d) == WorkspaceKnowledge.FlowIndexUse(builds: ["library.index"]))
    }

    @Test func storeIndexAnywhereBuildsRegardlessOfLaterRows() throws {
        // CFM-R17-FIX-11(a) (owner-ruled 2026-09-04): position-free. Before, this was `.plain`
        // because the `Store Index` wasn't the flow's *last* row — whether it counted depended
        // on the unrelated `Save Text` after it.
        let d = try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   library.index\n3. Save Text   done.md\n")
        #expect(WorkspaceKnowledge.role(of: d) == WorkspaceKnowledge.FlowIndexUse(builds: ["library.index"]))
    }

    @Test func aFlowEndingInTwoStoreIndexRowsBuildsBoth() throws {
        // CFM-R17-FIX-9(b): the builder side collects every `Store Index` name, not just one —
        // symmetric with the querier side.
        let d = try doc("catflow 0.8\n1. Read Files   docs/\n2. Embed   BGE-M3\n3. Store Index   (1,2)   name=hr.index\n4. Store Index   (1,2)   name=eng.index\n")
        #expect(WorkspaceKnowledge.role(of: d) == WorkspaceKnowledge.FlowIndexUse(builds: ["hr.index", "eng.index"]))
    }

    @Test func aFlowThatBothBuildsAndQueriesOneIndexRegistersOnBothSides() throws {
        // CFM-R17-FIX-11(a): the two arms used to be exclusive — `role` tested `builds` first
        // and returned immediately, so a flow whose last row is `Store Index` never had its
        // earlier `Read Index`/`Retrieve` rows checked. Now independent: a self-contained
        // "ingest, then ask" flow (single-file RAG) is a real, expressible shape.
        let d = try doc("""
        catflow 0.8
        1. Read Index   library.index
        2. Retrieve     (1,1)
        3. Embed        BGE-M3
        4. Store Index  library.index
        """)
        #expect(WorkspaceKnowledge.role(of: d) == WorkspaceKnowledge.FlowIndexUse(
            builds: ["library.index"], queries: ["library.index"]))
    }

    @Test func aBuilderOfTwoIndexesPlusTwoQueriersProducesTwoCards() throws {
        let c = try cards([
            ("BuildBoth.cat", "catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   (1,1)   name=hr.index\n3. Store Index   (1,1)   name=eng.index\n"),
            ("AskHR.cat", "catflow 0.8\n1. Read Index   hr.index\n2. Retrieve   (1,1)\n"),
            ("AskEng.cat", "catflow 0.8\n1. Read Index   eng.index\n2. Retrieve   (1,1)\n"),
        ])
        #expect(c.map(\.indexName) == ["eng.index", "hr.index"])
        #expect(c.allSatisfy { $0.builder == .one("BuildBoth.cat") })
        #expect(Set(c.compactMap(\.askFile)) == ["AskHR.cat", "AskEng.cat"])
    }

    @Test func readIndexPlusRetrieveQueries() throws {
        let d = try doc("catflow 0.8\n1. Read Index   library.index\n2. Embed   BGE-M3\n3. Retrieve   (1,2)\n4. Answer   Qwen3 8B\n")
        #expect(WorkspaceKnowledge.role(of: d) == WorkspaceKnowledge.FlowIndexUse(queries: ["library.index"]))
    }

    @Test func readIndexPlusKeywordSearchQueries() throws {
        let d = try doc("catflow 0.8\n1. Read Index   kb.index\n2. Keyword Search   (1,1)\n")
        #expect(WorkspaceKnowledge.role(of: d) == WorkspaceKnowledge.FlowIndexUse(queries: ["kb.index"]))
    }

    // MARK: - CFM-R17-FIX-4: N indexes → N names; builder/querier over the same row set

    @Test func aFlowReadingTwoIndexesQueriesBoth() throws {
        // `20-TwoIndexAnalyst`-shaped: both Read Index rows live inside a `<list>` block.
        let two = try String(contentsOf: galleryURL("20-TwoIndexAnalyst"), encoding: .utf8)
        #expect(WorkspaceKnowledge.role(of: try doc(two)) == WorkspaceKnowledge.FlowIndexUse(
            queries: ["hr.index", "eng.index"]))
    }

    @Test func routerDeskQueriesBothBranchIndexes() throws {
        let router = try String(contentsOf: galleryURL("45-RouterDesk"), encoding: .utf8)
        #expect(WorkspaceKnowledge.role(of: try doc(router)) == WorkspaceKnowledge.FlowIndexUse(
            queries: ["billing-kb.index", "docs-kb.index"]))
    }

    @Test func repeatedReadIndexNameIsListedOnce() throws {
        let d = try doc("""
        catflow 0.8
        1. Read Index   kb.index
        2. Read Index   kb.index
        3. Retrieve   (1,1)
        """)
        #expect(WorkspaceKnowledge.role(of: d) == WorkspaceKnowledge.FlowIndexUse(queries: ["kb.index"]))
    }

    @Test func storeIndexEndingATrailingListStillBuilds() throws {
        // The asymmetry FIX-4 names: the last row is `Store Index`, one level down.
        let d = try doc("""
        catflow 0.8
        1. Read Files   docs/
        2. <list finish>
             1. Embed         BGE-M3
             2. Store Index   library.index
        """)
        #expect(WorkspaceKnowledge.role(of: d) == WorkspaceKnowledge.FlowIndexUse(builds: ["library.index"]))
    }

    @Test func storeIndexInATrailingListStillBuildsEvenFollowedByMore() throws {
        // Position-free (CFM-R17-FIX-11(a)) — the same reclassification as
        // `storeIndexAnywhereBuildsRegardlessOfLaterRows`, one level down inside a block.
        let d = try doc("""
        catflow 0.8
        1. <list build>
             1. Embed         BGE-M3
             2. Store Index   library.index
        2. Save Text   done.md
        """)
        #expect(WorkspaceKnowledge.role(of: d) == WorkspaceKnowledge.FlowIndexUse(builds: ["library.index"]))
    }

    // MARK: - CFM-R17-FIX-5 / -9(c): normalization, and ambiguity that keeps the good side

    @Test func aLeadingDotSlashAndATrailingSlashNormaliseToTheSameName() throws {
        // Both names sit where the tools read them (`name=` / `path=`), so both rows run —
        // and `./library.index` normalises to `library.index` for the pairing (CFM-R17-FIX-5;
        // FIX-9(a) took the `row.model` fallback back out, so a *bare* `./x` no longer pairs).
        let c = try cards([
            ("Build.cat", "catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   (1,1)   name=library.index\n"),
            ("Ask.cat", "catflow 0.8\n1. Read Index   path=./library.index\n2. Retrieve   (1,1)\n"),
        ])
        #expect(c.count == 1)
        #expect(c.first?.indexName == "library.index")
        #expect(c.first?.builder == .one("Build.cat"))
        #expect(c.first?.querier == .one("Ask.cat"))
    }

    @Test func aBareSlashNameLandsInModelAndIsNotClassified() throws {
        // CFM-R17-FIX-9(a): `Store Index dir/kb.index` / `Read Index ./kb.index` — the parser
        // parks the name in `row.model`, where neither the tools nor the Python read it. The
        // classifier must not offer a card whose Build/Ask would always throw; the rows stay
        // unclassified.
        let builder = try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   indexes/kb.index\n")
        let querier = try doc("catflow 0.8\n1. Read Index   ./kb.index\n2. Retrieve   (1,1)\n")
        #expect(WorkspaceKnowledge.role(of: builder).isPlain)
        #expect(WorkspaceKnowledge.role(of: querier).isPlain)
        #expect(try cards([
            ("Build.cat", "catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   indexes/kb.index\n"),
            ("Ask.cat", "catflow 0.8\n1. Read Index   ./kb.index\n2. Retrieve   (1,1)\n"),
        ]).isEmpty)
    }

    @Test func twoBuildersKeepTheAskButtonAndNoteTheBuildCollision() throws {
        // CFM-R17-FIX-9(c): the Ask side is unambiguous, so the card keeps its Ask button —
        // the old collision notice threw it away.
        // CFM-R17-FIX-11(d): "Nightly.cat"/"InboxIngest.cat", not "Ingest.cat"/"InboxIngest.cat"
        // — the latter pair made the `note.contains("Ingest.cat")` conjunct vacuous, since
        // "InboxIngest.cat" contains "Ingest.cat" as a substring.
        let c = try #require(try cards([
            ("Nightly.cat", "catflow 0.8\n1. Read Files   docs/\n2. Embed   BGE-M3\n3. Store Index   (1,2)   library.index\n"),
            ("InboxIngest.cat", "catflow 0.8\n1. On File   inbox/\n2. Embed   BGE-M3\n3. Store Index   (1,2)   library.index\n"),
            ("Ask.cat", "catflow 0.8\n1. Read Index   library.index\n2. Retrieve   (1,1)\n"),
        ]).first)
        #expect(c.indexName == "library.index")
        #expect(c.buildFile == nil)              // ambiguous — no Build button
        #expect(c.askFile == "Ask.cat")          // ...but Ask still works
        if case .ambiguous(let b) = c.builder { #expect(Set(b) == ["Nightly.cat", "InboxIngest.cat"]) }
        else { Issue.record("builder side should be .ambiguous") }
        let note = try #require(c.ambiguityNote)
        #expect(note.contains("Nightly.cat") && note.contains("InboxIngest.cat"))
        #expect(note.contains("Build") && !note.contains("Ask"))   // only the ambiguous side
        // CFM-R17-FIX-11(b): the button that survives is still named on the card.
        #expect(c.namesCaption == "Ask.cat queries")
    }

    @Test func twoQueriersKeepTheBuildButtonAndNoteTheAskCollision() throws {
        // CFM-R17-FIX-11(d): "QueryA.cat"/"QueryB.cat", not "AskA.cat"/"AskB.cat" — the latter
        // pair made `note.contains("Ask")` vacuous (satisfied by the filenames regardless of
        // the sentence); only `!note.contains("Build")` carried information.
        let c = try #require(try cards([
            ("Build.cat", "catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   (1,1)   kb.index\n"),
            ("QueryA.cat", "catflow 0.8\n1. Read Index   kb.index\n2. Retrieve   (1,1)\n"),
            ("QueryB.cat", "catflow 0.8\n1. Read Index   kb.index\n2. Keyword Search   (1,1)\n"),
        ]).first)
        #expect(c.buildFile == "Build.cat")      // unambiguous — Build works
        #expect(c.askFile == nil)                // ambiguous — no Ask button
        let note = try #require(c.ambiguityNote)
        #expect(note.contains("QueryA.cat") && note.contains("QueryB.cat"))
        #expect(note.contains("Ask") && !note.contains("Build"))
        #expect(c.namesCaption == "Build.cat builds")
    }

    @Test func bothSidesAmbiguousIsACardWithNoButtonsAndTwoNotes() throws {
        let c = try #require(try cards([
            ("B1.cat", "catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   kb.index\n"),
            ("B2.cat", "catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   kb.index\n"),
            ("Q1.cat", "catflow 0.8\n1. Read Index   kb.index\n2. Retrieve   (1,1)\n"),
            ("Q2.cat", "catflow 0.8\n1. Read Index   kb.index\n2. Retrieve   (1,1)\n"),
        ]).first)
        #expect(c.buildFile == nil)
        #expect(c.askFile == nil)
        let note = try #require(c.ambiguityNote)
        #expect(note.contains("Build") && note.contains("Ask"))
        // CFM-R17-FIX-11(b): nothing unambiguous to name — the note already names both sides.
        #expect(c.namesCaption == nil)
    }

    @Test func twoIndexAnalystPlusTwoBuildersProducesTwoCards() throws {
        let two = try String(contentsOf: galleryURL("20-TwoIndexAnalyst"), encoding: .utf8)
        let c = try cards([
            ("Analyst.cat", two),
            ("BuildHR.cat", "catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   (1,1)   hr.index\n"),
            ("BuildEng.cat", "catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   (1,1)   eng.index\n"),
        ])
        #expect(c.map(\.indexName) == ["eng.index", "hr.index"])
        #expect(c.allSatisfy { $0.querier == .one("Analyst.cat") })
        #expect(c.allSatisfy { $0.ambiguityNote == nil })
        #expect(Set(c.map(\.buildFile)) == ["BuildHR.cat", "BuildEng.cat"])
    }

    @Test func readIndexAloneIsPlain() throws {
        let d = try doc("catflow 0.8\n1. Read Index   kb.index\n2. Save Text   x.md\n")
        #expect(WorkspaceKnowledge.role(of: d).isPlain)
    }

    @Test func explicitNameSettingWins() throws {
        let d = try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   (1,1)   name=hr.index\n")
        #expect(WorkspaceKnowledge.role(of: d) == WorkspaceKnowledge.FlowIndexUse(builds: ["hr.index"]))
    }

    // MARK: - Pairing (the 16 + 18 done-when)

    @Test func ingestAndDocChatRowsPairIntoOneIndexCard() throws {
        let ingest = try String(contentsOf: galleryURL("16-IngestFolder"), encoding: .utf8)
        let docChat = try String(contentsOf: galleryURL("18-DocChat"), encoding: .utf8)
        let c = try cards([("Ingest.cat", ingest), ("Ask.cat", docChat)])
        #expect(c.count == 1)
        let card = try #require(c.first)
        #expect(card.indexName == "library.index")
        #expect(card.builder == .one("Ingest.cat"))
        #expect(card.querier == .one("Ask.cat"))
        #expect(card.ambiguityNote == nil)
        // CFM-R17-FIX-11(b): both sides unambiguous — the card names both, always (not only
        // in the pre-build placeholder, which this Ingest.cat/DocChat.cat pair also covers).
        #expect(card.namesCaption == "Ingest.cat builds · Ask.cat queries")
    }

    // MARK: - CFM-R17-FIX-11(c): one-sided names still classify, as their own card

    @Test func aQuerierWithNoMatchingBuilderStillProducesAOneSidedCard() throws {
        // `classify` is pure and no longer requires both sides — a querier with no builder in
        // this workspace is `uses_example`'s exact shape (a prebuilt `kb.index`, nothing here
        // builds it). Whether the *view* renders it is `isCardWorthRendering`'s call, below.
        let c = try #require(try cards([
            ("A.cat", "catflow 0.8\n1. Read Text   a.txt\n2. Save Text   b.md\n"),
            ("B.cat", "catflow 0.8\n1. Read Index   only.index\n2. Retrieve   (1,1)\n"),
        ]).first)
        #expect(c.indexName == "only.index")
        #expect(c.builder == .none)
        #expect(c.buildFile == nil)
        #expect(c.querier == .one("B.cat"))
    }

    @Test func twoBuildersButNoQuerierProducesAOneSidedCard() throws {
        let c = try #require(try cards([
            ("A.cat", "catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   x.index\n"),
            ("B.cat", "catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   x.index\n"),
        ]).first)
        #expect(c.indexName == "x.index")
        #expect(c.querier == .none)
        #expect(c.askFile == nil)
        if case .ambiguous(let b) = c.builder { #expect(Set(b) == ["A.cat", "B.cat"]) }
        else { Issue.record("builder side should be .ambiguous") }
    }

    @Test func differentIndexNamesProduceTwoOneSidedCards() throws {
        let c = try cards([
            ("Build.cat", "catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   (1,1)   hr.index\n"),
            ("Ask.cat", "catflow 0.8\n1. Read Index   eng.index\n2. Retrieve   (1,1)\n"),
        ])
        #expect(c.map(\.indexName) == ["eng.index", "hr.index"])
        let engCard = try #require(c.first { $0.indexName == "eng.index" })
        #expect(engCard.builder == .none)
        #expect(engCard.querier == .one("Ask.cat"))
        let hrCard = try #require(c.first { $0.indexName == "hr.index" })
        #expect(hrCard.builder == .one("Build.cat"))
        #expect(hrCard.querier == .none)
    }

    @Test func aCardWithABuilderIsWorthRenderingRegardlessOfDisk() throws {
        let c = try #require(try cards([
            ("Ingest.cat", "catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   kb.index\n"),
        ]).first)
        #expect(WorkspaceKnowledge.isCardWorthRendering(c, indexExists: false))
        #expect(WorkspaceKnowledge.isCardWorthRendering(c, indexExists: true))
    }

    @Test func aQuerierOnlyCardNeedsTheIndexOnDiskToBeWorthRendering() throws {
        // uses_example-shaped: a flow queries a prebuilt kb.index nothing in the workspace
        // builds. Not worth a card until the index actually exists — otherwise it's a dead
        // end (a card reading "not built yet" with no Build button).
        let c = try #require(try cards([
            ("RagQuery.cat", "catflow 0.8\n1. Read Index   kb.index\n2. Retrieve   (1,1)\n"),
        ]).first)
        #expect(c.builder == .none)
        #expect(!WorkspaceKnowledge.isCardWorthRendering(c, indexExists: false),
                "nothing actionable yet — a dead end, not a card")
        #expect(WorkspaceKnowledge.isCardWorthRendering(c, indexExists: true),
                "a prebuilt, queryable index is a real state — uses_example ships exactly this")
    }

    // MARK: - CFM-R17-FIX-11(e): a caller inherits its callees' index use through uses:

    /// A resolved `uses:` entry built directly from source text — the shape `UsesResolver`
    /// would hand `role(of:usesGraph:)`, without needing a workspace on disk.
    private func usedFlow(_ text: String, nested: [String: FlowInterpreter.UsedFlow] = [:]) throws -> FlowInterpreter.UsedFlow {
        let d = try doc(text)
        return FlowInterpreter.UsedFlow(params: d.params, rows: d.rows,
                                        definitions: d.definitions, presets: d.presets, nested: nested)
    }

    @Test func aCallerInheritsItsCalleesIndexUse() throws {
        // uses_example-shaped: AskYourDocs.cat calls RagQuery.cat, and RagQuery.cat's own
        // Read Index + Retrieve is what makes AskYourDocs a querier — it has neither itself.
        let caller = try doc("catflow 0.8\n1. RagQuery\n\nuses:\n  RagQuery = ./RagQuery.cat\n")
        let ragQuery = try usedFlow("""
        catflow 0.8
        1. Embed       BGE-M3
        2. Read Index  kb.index
        3. Retrieve    (2,1)
        4. Answer      Qwen3 8B
        """)
        // Without the graph, still `.plain` — inheritance is opt-in per call site.
        #expect(WorkspaceKnowledge.role(of: caller).isPlain)
        #expect(WorkspaceKnowledge.role(of: caller, usesGraph: ["RagQuery": ragQuery])
            == WorkspaceKnowledge.FlowIndexUse(queries: ["kb.index"]))
    }

    @Test func inheritanceRecursesThroughNestedUsesChains() throws {
        // A uses B uses C; C is where the Read Index + Retrieve actually live.
        let c = try usedFlow("catflow 0.8\n1. Read Index   deep.index\n2. Retrieve   (1,1)\n")
        let b = try usedFlow("catflow 0.8\n1. C\n\nuses:\n  C = ./C.cat\n", nested: ["C": c])
        let a = try doc("catflow 0.8\n1. B\n\nuses:\n  B = ./B.cat\n")
        #expect(WorkspaceKnowledge.role(of: a, usesGraph: ["B": b])
            == WorkspaceKnowledge.FlowIndexUse(queries: ["deep.index"]))
    }

    @Test func classifyWithAUsesGraphMakesTheCallerTheSoleQuerier() throws {
        // Mirrors what `WorkspaceListView` does: the callee is excluded from `flows` entirely
        // (it's a library component, not a candidate) — only its resolved graph is passed,
        // attributed to the caller. One card, one unambiguous Ask, no coin-flip with the callee.
        let ragQuery = try usedFlow("""
        catflow 0.8
        1. Embed       BGE-M3
        2. Read Index  kb.index
        3. Retrieve    (2,1)
        4. Answer      Qwen3 8B
        """)
        let c = WorkspaceKnowledge.classify(
            flows: [("AskYourDocs.cat", try doc("catflow 0.8\n1. RagQuery\n\nuses:\n  RagQuery = ./RagQuery.cat\n"))],
            usesGraphs: ["AskYourDocs.cat": ["RagQuery": ragQuery]])
        #expect(c.count == 1)
        #expect(c.first?.indexName == "kb.index")
        #expect(c.first?.querier == .one("AskYourDocs.cat"))
        #expect(c.first?.askFile == "AskYourDocs.cat")
    }

    @Test func bundledUsesExampleKnowledgeCardAsksTheCallerNotTheCallee() throws {
        // The real fixture, through the real extracted pipeline (CFM-R17-FIX-12(c)): both
        // `AskYourDocs.cat` *and* `RagQuery.cat` go into `flows` — `classifyWorkspace` itself
        // must do the exclusion (`RagQuery.cat` cannot run standalone — its row 1 `Embed` has
        // no inline value and no upstream), not a test that hand-picks which file to pass.
        let meta = try #require(BundledWorkspaces.meta(id: "uses_example"))
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-kb-uses-\(UUID().uuidString)")
        let root = base.appendingPathComponent("workspaces")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let ws = FlowWorkspace(root: root)
        try BundledWorkspaces.prepare(meta, workspace: ws)
        let dir = ws.directory(for: "uses_example")
        let flows = try meta.files.filter { $0.ext == "cat" }.map { file -> (file: String, doc: FlowDocument, url: URL) in
            let url = dir.appendingPathComponent(file.destination)
            return (file.destination, try CatParser.parse(try String(contentsOf: url, encoding: .utf8)), url)
        }
        #expect(Set(flows.map(\.file)) == ["AskYourDocs.cat", "RagQuery.cat"])

        let c = WorkspaceKnowledge.classifyWorkspace(
            flows: flows,
            resolveUses: { doc, selfFile in
                UsesResolver.resolve(doc, workspace: ws, flowID: "uses_example", selfFile: selfFile)
            },
            resolvePath: { rawPath in try? ws.resolve(rawPath, flowID: "uses_example") })
        #expect(c.count == 1)
        let card = try #require(c.first)
        #expect(card.indexName == "kb.index")
        #expect(card.querier == .one("AskYourDocs.cat"))
        #expect(card.askFile == "AskYourDocs.cat")
        // The real-executor proof that this target actually clears row 1 lives in
        // `CatFlowUsesEndToEndTests.bundledUsesExampleClearsRowOneUnderARealExecutor` — this
        // test is the other half: the card now points there, not at `RagQuery.cat`.
    }

    @Test func classifyWorkspaceLeavesAnUnrelatedThirdFlowAsACandidate() throws {
        // Only the actual uses: target is excluded — a third flow with no relation to the
        // caller/callee pair stays a normal candidate, on either side.
        let caller = try doc("catflow 0.8\n1. RagQuery\n\nuses:\n  RagQuery = ./RagQuery.cat\n")
        let calleeText = "catflow 0.8\n1. Embed   BGE-M3\n2. Read Index   kb.index\n3. Retrieve   (2,1)\n"
        let callee = try doc(calleeText)
        let calleeUsed = try usedFlow(calleeText)
        let other = try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   other.index\n")
        let callerURL = URL(fileURLWithPath: "/ws/AskYourDocs.cat")
        let calleeURL = URL(fileURLWithPath: "/ws/RagQuery.cat")
        let otherURL = URL(fileURLWithPath: "/ws/Other.cat")

        let cards = WorkspaceKnowledge.classifyWorkspace(
            flows: [
                ("AskYourDocs.cat", caller, callerURL),
                ("RagQuery.cat", callee, calleeURL),
                ("Other.cat", other, otherURL),
            ],
            resolveUses: { _, selfFile in selfFile == callerURL ? ["RagQuery": calleeUsed] : [:] },
            resolvePath: { raw in raw == "./RagQuery.cat" ? calleeURL : nil })

        #expect(cards.map(\.indexName).sorted() == ["kb.index", "other.index"])
        let kbCard = try #require(cards.first { $0.indexName == "kb.index" })
        #expect(kbCard.querier == .one("AskYourDocs.cat"))
        #expect(kbCard.builder == .none)
        let otherCard = try #require(cards.first { $0.indexName == "other.index" })
        #expect(otherCard.builder == .one("Other.cat"))
    }

    @Test func classifyWorkspaceWithNoUsesAnywhereNeverCallsTheResolvers() throws {
        let build = try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   (1,1)   kb.index\n")
        let ask = try doc("catflow 0.8\n1. Read Index   kb.index\n2. Retrieve   (1,1)\n")
        var resolverCalls = 0
        let cards = WorkspaceKnowledge.classifyWorkspace(
            flows: [
                ("Build.cat", build, URL(fileURLWithPath: "/ws/Build.cat")),
                ("Ask.cat", ask, URL(fileURLWithPath: "/ws/Ask.cat")),
            ],
            resolveUses: { _, _ in resolverCalls += 1; return [:] },
            resolvePath: { _ in resolverCalls += 1; return nil })
        #expect(resolverCalls == 0, "no flow declares uses: — the resolvers must never be invoked")
        #expect(cards.count == 1)
        #expect(cards.first?.builder == .one("Build.cat"))
        #expect(cards.first?.querier == .one("Ask.cat"))
    }

    @Test func classifyWorkspaceLeavesADeclaredButNeverCalledEntryUntouched() throws {
        // CFM-R17-FIX-12(a): `Nightly.cat` declares `uses: Ingest = ./Ingest.cat` but no row
        // calls it (mid-edit, commented out). `Ingest.cat` still builds `library.index` on its
        // own rows and must keep its card — a declaration alone no longer costs it candidacy.
        let nightly = try doc("catflow 0.8\n1. Save Text   noop.md\n\nuses:\n  Ingest = ./Ingest.cat\n")
        let ingest = try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   library.index\n")
        let ask = try doc("catflow 0.8\n1. Read Index   library.index\n2. Retrieve   (1,1)\n")
        let ingestURL = URL(fileURLWithPath: "/ws/Ingest.cat")

        let cards = WorkspaceKnowledge.classifyWorkspace(
            flows: [
                ("Nightly.cat", nightly, URL(fileURLWithPath: "/ws/Nightly.cat")),
                ("Ingest.cat", ingest, ingestURL),
                ("Ask.cat", ask, URL(fileURLWithPath: "/ws/Ask.cat")),
            ],
            resolveUses: { _, _ in [:] },   // never consulted — "Ingest" is never *called*
            resolvePath: { raw in raw == "./Ingest.cat" ? ingestURL : nil })

        let libraryCard = try #require(cards.first { $0.indexName == "library.index" })
        #expect(libraryCard.builder == .one("Ingest.cat"))
        #expect(libraryCard.querier == .one("Ask.cat"))
    }

    @Test func classifyWorkspaceKeepsAnEntryShadowedByASameNamedDefinitionsComposite() throws {
        // CFM-R17-FIX-12(a): `Caller.cat` declares both `definitions: Ingest` (a plain inline
        // composite) *and* `uses: Ingest = ./Ingest.cat` — same name. The interpreter dispatches
        // `1. Ingest` to the `definitions:` composite first and never reaches the `uses:` entry
        // (`FlowInterpreter`'s scope-expansion switch). `Ingest.cat` must not lose its card.
        let caller = try doc("""
        catflow 0.8
        1. Ingest

        definitions:
          composite Ingest   text -> text
             1. Save Text   noop.md

        uses:
          Ingest = ./Ingest.cat
        """)
        let ingest = try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   library.index\n")
        let ingestURL = URL(fileURLWithPath: "/ws/Ingest.cat")

        let cards = WorkspaceKnowledge.classifyWorkspace(
            flows: [
                ("Caller.cat", caller, URL(fileURLWithPath: "/ws/Caller.cat")),
                ("Ingest.cat", ingest, ingestURL),
            ],
            resolveUses: { _, _ in [:] },
            resolvePath: { raw in raw == "./Ingest.cat" ? ingestURL : nil })

        let libraryCard = try #require(cards.first { $0.indexName == "library.index" })
        #expect(libraryCard.builder == .one("Ingest.cat"))
    }

    @Test func roleDoesNotInheritThroughAnAliasShadowedByADefinitionsComposite() throws {
        // The inheritance half of the same precedence: `Caller.cat` would otherwise inherit
        // `RagQuery.cat`'s Read Index/Retrieve, but `definitions: RagQuery` intercepts the row
        // first, so nothing is ever actually reached — `role` must not inherit from it.
        let caller = try doc("""
        catflow 0.8
        1. RagQuery

        definitions:
          composite RagQuery   text -> text
             1. Save Text   noop.md

        uses:
          RagQuery = ./RagQuery.cat
        """)
        let ragQuery = try usedFlow("catflow 0.8\n1. Read Index   kb.index\n2. Retrieve   (1,1)\n")
        #expect(WorkspaceKnowledge.role(of: caller, usesGraph: ["RagQuery": ragQuery]).isPlain)
    }

    @Test func classifyWorkspaceHandlesASelfReferencingUsesEntry() throws {
        // CFM-R17-FIX-12(a): `A.cat` declares `uses: A = ./A.cat` (self-use) and calls it.
        // `UsesResolver` drops a self-referencing entry outright (E115, the candidate is
        // already on the stack), so it's never in the resolved graph — `graph["A"] == nil`,
        // which the "called and resolved" rule reads as "never survived," not excluded.
        let a = try doc("catflow 0.8\n1. A\n\nuses:\n  A = ./A.cat\n")
        let aURL = URL(fileURLWithPath: "/ws/A.cat")
        let cards = WorkspaceKnowledge.classifyWorkspace(
            flows: [("A.cat", a, aURL)],
            resolveUses: { _, _ in [:] },   // mirrors UsesResolver: a self-reference never resolves
            resolvePath: { raw in raw == "./A.cat" ? aURL : nil })
        #expect(cards.isEmpty)   // A.cat has no index use of its own — nothing to exclude either
    }

    @Test func classifyWorkspaceResolvesAMutualUsesCycleWhenOneSideCanStandAlone() throws {
        // CFM-R17-FIX-12(a)+(b) against the REAL `UsesResolver`/`FlowWorkspace`: A calls B, B
        // calls A, both genuinely resolve (only the *nested*, one-level-deeper back-reference
        // is dropped, E115 — "called and resolved" alone can't tell these two exclusions apart,
        // journal `2026-189`'s residual). `-12(b)` breaks the tie: A's row 1 (`Embed BGE-M3`,
        // no inline value) can't stand alone, so A stays excluded; B's row 1 (`Read Index
        // library.index`) can, so B keeps its candidacy — and, inheriting A's `Store Index`
        // through the very `uses: A` that would have excluded it, ends up the single clean
        // card for the index both files touch.
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-kb-cycle-\(UUID().uuidString)")
        let root = base.appendingPathComponent("workspaces")
        let dir = root.appendingPathComponent("cycle", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let aText = "catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   library.index\n3. B\n\nuses:\n  B = ./B.cat\n"
        let bText = "catflow 0.8\n1. Read Index   library.index\n2. Retrieve   (1,1)\n3. A\n\nuses:\n  A = ./A.cat\n"
        let aURL = dir.appendingPathComponent("A.cat")
        let bURL = dir.appendingPathComponent("B.cat")
        try aText.write(to: aURL, atomically: true, encoding: .utf8)
        try bText.write(to: bURL, atomically: true, encoding: .utf8)
        let ws = FlowWorkspace(root: root)

        let cards = WorkspaceKnowledge.classifyWorkspace(
            flows: [
                ("A.cat", try CatParser.parse(aText), aURL),
                ("B.cat", try CatParser.parse(bText), bURL),
            ],
            resolveUses: { doc, selfFile in
                UsesResolver.resolve(doc, workspace: ws, flowID: "cycle", selfFile: selfFile)
            },
            resolvePath: { raw in try? ws.resolve(raw, flowID: "cycle") })

        let card = try #require(cards.first { $0.indexName == "library.index" })
        #expect(card.builder == .one("B.cat"))
        #expect(card.querier == .one("B.cat"))
    }

    @Test func classifyWorkspaceLeavesAMutualCycleExcludedWhenNeitherSideCanStandAlone() throws {
        // The narrower residual `-12(b)` doesn't reach: both A and B genuinely need external
        // input (empty settings on row 1) and call each other. Neither can verify the other as
        // a real entry point, so both stay excluded — arguably correct, not just unresolved:
        // nothing in this pair can actually be kicked off without a caller from outside it.
        let a = try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   library.index\n3. B\n\nuses:\n  B = ./B.cat\n")
        let b = try doc("catflow 0.8\n1. Embed   BGE-M3\n2. Read Index   library.index\n3. Retrieve   (1,2)\n4. A\n\nuses:\n  A = ./A.cat\n")
        let aURL = URL(fileURLWithPath: "/ws/A.cat")
        let bURL = URL(fileURLWithPath: "/ws/B.cat")
        let aUsed = try usedFlow("catflow 0.8\n1. Embed   BGE-M3\n2. Store Index   library.index\n")
        let bUsed = try usedFlow("catflow 0.8\n1. Embed   BGE-M3\n2. Read Index   library.index\n3. Retrieve   (1,2)\n")

        let cards = WorkspaceKnowledge.classifyWorkspace(
            flows: [("A.cat", a, aURL), ("B.cat", b, bURL)],
            resolveUses: { _, selfFile in selfFile == aURL ? ["B": bUsed] : ["A": aUsed] },
            resolvePath: { raw in raw == "./B.cat" ? bURL : (raw == "./A.cat" ? aURL : nil) })

        #expect(cards.isEmpty)
    }

    // MARK: - CFM-R17-FIX-12(b): only a callee that cannot stand alone is excluded

    @Test func canStandAloneChecksRowOneForItsOwnValue() throws {
        #expect(!WorkspaceKnowledge.canStandAlone(try doc("catflow 0.8\n1. Embed   BGE-M3\n")))
        #expect(WorkspaceKnowledge.canStandAlone(
            try doc("catflow 0.8\n1. Embed   BGE-M3; \"literal text\"\n")))
        #expect(WorkspaceKnowledge.canStandAlone(try doc("catflow 0.8\n1. Read Index   kb.index\n")))
        #expect(WorkspaceKnowledge.canStandAlone(try doc("catflow 0.8; events\n1. On File   inbox/\n")))
        #expect(!WorkspaceKnowledge.canStandAlone(FlowDocument(version: "0.8", rows: [])))
    }

    @Test func canStandAloneDescendsIntoALeadingBlocksFirstChild() throws {
        // `<each>`/`<list>`/`<parallel>` all execute their first child before anything after
        // the block, so a leading block's own "row 1" is really its first child.
        let withValue = FlowDocument(version: "0.8", rows: [
            Row(blockKind: .each, blockName: "e", children: [Row(task: "Read Index", settings: "kb.index")]),
        ])
        #expect(WorkspaceKnowledge.canStandAlone(withValue))
        let withoutValue = FlowDocument(version: "0.8", rows: [
            Row(blockKind: .each, blockName: "e", children: [Row(task: "Embed", model: "BGE-M3")]),
        ])
        #expect(!WorkspaceKnowledge.canStandAlone(withoutValue))
    }

    @Test func classifyWorkspaceKeepsACalleeThatCanStandAloneAsItsOwnCandidate() throws {
        // CFM-R17-FIX-12(b) (owner-ruled, recommended option A): `RagFlow.cat` is a
        // self-contained ingest-then-ask flow — being `BatchReport.cat`'s `uses:` target
        // doesn't cost it its own card. Both end up registering `kb.index` (`RagFlow.cat`
        // directly, `BatchReport.cat` by inheritance through the very call that would have
        // excluded it) — an honest ambiguity naming two real entry points, not
        // `BatchReport.cat` silently winning Ask with no `Human Input` row.
        let ragFlowText = "catflow 0.8\n1. Read Index   kb.index\n2. Retrieve   (1,1)\n3. Embed   BGE-M3\n4. Store Index   kb.index\n"
        let ragFlow = try doc(ragFlowText)
        let ragFlowUsed = try usedFlow(ragFlowText)
        let batchReport = try doc("catflow 0.8\n1. RagFlow\n\nuses:\n  RagFlow = ./RagFlow.cat\n")
        let ragFlowURL = URL(fileURLWithPath: "/ws/RagFlow.cat")
        let batchURL = URL(fileURLWithPath: "/ws/BatchReport.cat")
        #expect(WorkspaceKnowledge.canStandAlone(ragFlow))

        let cards = WorkspaceKnowledge.classifyWorkspace(
            flows: [
                ("BatchReport.cat", batchReport, batchURL),
                ("RagFlow.cat", ragFlow, ragFlowURL),
            ],
            resolveUses: { _, selfFile in selfFile == batchURL ? ["RagFlow": ragFlowUsed] : [:] },
            resolvePath: { raw in raw == "./RagFlow.cat" ? ragFlowURL : nil })

        let card = try #require(cards.first { $0.indexName == "kb.index" })
        if case .ambiguous(let builders) = card.builder {
            #expect(Set(builders) == ["BatchReport.cat", "RagFlow.cat"])
        } else {
            Issue.record("builder side should be ambiguous — both are real entry points")
        }
        if case .ambiguous(let queriers) = card.querier {
            #expect(Set(queriers) == ["BatchReport.cat", "RagFlow.cat"])
        } else {
            Issue.record("querier side should be ambiguous — both are real entry points")
        }
    }

    @Test func classifyWorkspaceStillExcludesACalleeThatCannotStandAlone() throws {
        // Regression guard: RagQuery.cat's shape (row 1 `Embed BGE-M3`, no inline value) still
        // gets excluded — -12(b) narrows the rule further, it doesn't remove it.
        let ragQueryText = "catflow 0.8\n1. Embed   BGE-M3\n2. Read Index   kb.index\n3. Retrieve   (1,2)\n"
        let ragQuery = try doc(ragQueryText)
        #expect(!WorkspaceKnowledge.canStandAlone(ragQuery))
        let ragQueryUsed = try usedFlow(ragQueryText)
        let caller = try doc("catflow 0.8\n1. RagQuery\n\nuses:\n  RagQuery = ./RagQuery.cat\n")
        let callerURL = URL(fileURLWithPath: "/ws/AskYourDocs.cat")
        let ragQueryURL = URL(fileURLWithPath: "/ws/RagQuery.cat")

        let cards = WorkspaceKnowledge.classifyWorkspace(
            flows: [
                ("AskYourDocs.cat", caller, callerURL),
                ("RagQuery.cat", ragQuery, ragQueryURL),
            ],
            resolveUses: { _, selfFile in selfFile == callerURL ? ["RagQuery": ragQueryUsed] : [:] },
            resolvePath: { raw in raw == "./RagQuery.cat" ? ragQueryURL : nil })

        #expect(cards.count == 1)
        let card = try #require(cards.first)
        #expect(card.querier == .one("AskYourDocs.cat"))
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
