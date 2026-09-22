import Testing
import Foundation
@testable import MLXUI

/// CFM-R17-3 — the workspace view is a container around the existing flow surfaces. The
/// SwiftUI views themselves have no automated coverage yet, but the plumbing they stand on
/// does: `WorkspaceRef` → a workspace-rooted `FlowScope`, two flows in one workspace
/// resolving the same relative path to the same file, and a workspace flow running through
/// the same `FlowRunner`/`FlowRunSession` the gallery uses.
struct CatFlowWorkspaceViewTests {

    private func makeWorkspace(id: String = "docs",
                               files: [(name: String, text: String)]) throws -> (URL, FlowWorkspace) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-wsview-\(UUID().uuidString)")
        let root = base.appendingPathComponent("workspaces")
        let dir = root.appendingPathComponent(id, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for f in files {
            try f.text.write(to: dir.appendingPathComponent(f.name), atomically: true, encoding: .utf8)
        }
        return (base, FlowWorkspace(root: root))
    }

    private let validCat = "mlxflow 0.8\n1. Read Text   memo.txt\n2. Save Text   out.md\n"

    // MARK: - WR-1: FlowDisplay.resolve precedence across all three load paths

    /// The bug (`RSI/DelegateWorkspaceRunBacklog.md`, WR-1): `FlowListView.display` used to be
    /// computed from only `metadata`/`userEntry`, so a workspace flow's `display` was `nil` for
    /// the life of the view and the page never left "Loading flow…". `FlowDisplay.resolve` is
    /// the extracted decision, reachable here directly — delete the workspace branch inside it
    /// and this test goes red (confirmed locally).
    @Test func flowDisplayResolvesMetadataFirst() {
        let metadata = GalleryFlowMetadata(number: 1, title: "Gallery Title", filename: "01-F.cat",
                                           category: "Test", description: "gallery desc", tier: nil)
        let userEntry = UserFlowStore.Entry(flowID: "f", url: URL(fileURLWithPath: "/tmp/f.cat"),
                                            title: "User Title", modifiedAt: Date(), parseIssue: nil)
        let workspace = WorkspaceRef(workspaceID: "ask_your_docs", flowFile: "Ingest.cat")
        let display = FlowDisplay.resolve(metadata: metadata, userEntry: userEntry, workspace: workspace)
        #expect(display?.title == "Gallery Title")
        #expect(display?.description == "gallery desc")
    }

    @Test func flowDisplayResolvesUserEntrySecond() {
        let userEntry = UserFlowStore.Entry(flowID: "f", url: URL(fileURLWithPath: "/tmp/f.cat"),
                                            title: "User Title", modifiedAt: Date(), parseIssue: nil)
        let workspace = WorkspaceRef(workspaceID: "ask_your_docs", flowFile: "Ingest.cat")
        let display = FlowDisplay.resolve(metadata: nil, userEntry: userEntry, workspace: workspace)
        #expect(display?.title == "User Title")
        #expect(display?.description == nil)
    }

    /// The regression itself: a workspace flow with no gallery metadata and no user-shelf
    /// entry — every workspace flow, today — must still resolve to a non-nil display so
    /// `FlowListView.body`'s content branch is reachable.
    @Test func flowDisplayResolvesWorkspaceThird() {
        let workspace = WorkspaceRef(workspaceID: "ask_your_docs", flowFile: "Ingest.cat")
        let display = FlowDisplay.resolve(metadata: nil, userEntry: nil, workspace: workspace)
        #expect(display?.title == "Ingest")
        #expect(display?.description == "in ask_your_docs")
    }

    @Test func flowDisplayResolvesNilWhenAllThreeSourcesAreNil() {
        let display = FlowDisplay.resolve(metadata: nil, userEntry: nil, workspace: nil)
        #expect(display == nil)
    }

    // MARK: - WR-3: FlowReassessment recomputes the refusal AND the preflight

    /// Root cause 3: `refreshPreflight()` used to recompute only `session.prepareInstall(...)`
    /// — never `notRunnableReason`, which was derived once at load time and never revisited.
    /// Read `CatalogBridge.resolve`, `ModelSlot.readiness` and `TaskAvailability.state`: none
    /// of the three consult `installedModelIDs` to decide whether a model row *resolves* — only
    /// whether an already-resolved one is `.ready` vs `.needsDownload`. So the refusal half is
    /// genuinely a function of `catalog`/`claimableModelIDs` (an unresolvable display name),
    /// not of `installed`; the preflight half (`toDownload` vs `installed`) is the one that
    /// genuinely depends on `installed`. `FlowReassessment.compute` must get both right, on the
    /// same document, and this test drives each half with the input that actually decides it.
    /// Deleting the workspace-refusal-clearing behavior locally (reverting to the old
    /// preflight-only recompute) turns the first two `#expect`s red — confirmed before
    /// restoring.
    ///
    /// `preflight` is asserted non-nil even on the refused path — owner review, 2026-09-22: an
    /// earlier version returned `nil` there, which let `FlowListView.apply` skip
    /// `session.prepareInstall` and leave a stale prior preflight in place. `compute` now runs
    /// `FlowPreflight.run` exactly once and always returns it.
    @Test func reassessmentRecomputesBothTheRefusalAndThePreflightFromTheSameDocument() throws {
        let doc = try CatParser.parse("mlxflow 0.8\n1. Embed   Test Model\n")
        let model = makeEntry(id: "test-embed", displayName: "Test Model")
        let scope = FlowScope.plain("t")

        // Half 1 (the refusal): an empty catalog can't resolve "Test Model" at all —
        // `CatalogBridge.resolve`'s terminal `.notRunnable` fallback — independent of
        // `installed`, so this is genuinely "unresolvable," not "not downloaded yet."
        let unresolved = FlowReassessment.compute(doc: doc, catalog: [], installed: [],
                                                  totalRAMGB: 128, claimableModelIDs: [],
                                                  refusalScope: nil, inputScope: scope)
        #expect(unresolved.notRunnableReason != nil)
        #expect(unresolved.preflight.isBlocked == true)

        // Same document. The catalog now carries the named model, so it resolves — no longer
        // refused, whether or not it's installed yet (that's half 2, below).
        let resolvedNotInstalled = FlowReassessment.compute(
            doc: doc, catalog: [model], installed: [],
            totalRAMGB: 128, claimableModelIDs: [model.id], refusalScope: nil, inputScope: scope)
        #expect(resolvedNotInstalled.notRunnableReason == nil)
        #expect(resolvedNotInstalled.preflight.toDownload.contains { $0.model?.id == model.id })

        // Half 2 (the preflight): same document, same catalog — only `installed` changes —
        // moves the model out of `toDownload` and into `installed`. This half already worked
        // before WR-3 (`refreshPreflight()`'s whole job); `reassess()` must not regress it.
        let installed = FlowReassessment.compute(
            doc: doc, catalog: [model], installed: [model.id],
            totalRAMGB: 128, claimableModelIDs: [model.id], refusalScope: nil, inputScope: scope)
        #expect(installed.notRunnableReason == nil)
        #expect(installed.preflight.toDownload.isEmpty)
        #expect(installed.preflight.installed.contains { $0.model?.id == model.id })
    }

    // MARK: - WR-4: AutoRunResume — the re-arm decision as a pure function

    /// Root cause 4: `didAutoRun` burns once, by design (CFM-R17-FIX-7), so an auto-run that
    /// opens the install sheet never opens it twice — but nothing resumed the run once the
    /// install it asked for actually finished ("Build → Install → nothing runs"). The obvious
    /// fix, resetting `didAutoRun`, is wrong: it would re-arm the Cancel path and every later,
    /// unrelated install too. `AutoRunResume.shouldRun` is the alternative, decided purely from
    /// (was this flow's own auto-run waiting on an install, what closed the sheet) — never
    /// touching `didAutoRun`. Deleting the `dismissal == .installSucceeded` half of the
    /// condition (always `true` when pending) turns the `.cancelled`/`.installFailed` cases red;
    /// deleting `autoRunPending &&` turns the `autoRunPending: false` cases red — confirmed
    /// both locally before restoring.
    @Test func autoRunResumesOnlyOnThePathThatOpenedTheInstallSheetItself() {
        // The path WR-4 exists for: this flow's own auto-run opened the sheet, the user
        // installed, it succeeded.
        #expect(AutoRunResume.shouldRun(autoRunPending: true, dismissal: .installSucceeded))

        // Cancel — the sheet the auto-run raised closes without an install. Never runs.
        #expect(!AutoRunResume.shouldRun(autoRunPending: true, dismissal: .cancelled))

        // A failed download — surfaces `session.errorSentence` elsewhere; never runs, and
        // (per the view's own handling) never leaves `autoRunPending` armed for a retry.
        #expect(!AutoRunResume.shouldRun(autoRunPending: true, dismissal: .installFailed))

        // Not this flow's auto-run waiting — a manual "Install Required Models" press, or a
        // plain manual visit with no auto-run at all. Never runs, regardless of outcome.
        #expect(!AutoRunResume.shouldRun(autoRunPending: false, dismissal: .installSucceeded))
        #expect(!AutoRunResume.shouldRun(autoRunPending: false, dismissal: .cancelled))
        #expect(!AutoRunResume.shouldRun(autoRunPending: false, dismissal: .installFailed))
    }

    // MARK: - WorkspaceRef → FlowScope

    @Test func workspaceRefBuildsAWorkspaceRootedScope() {
        let ref = WorkspaceRef(workspaceID: "docs", flowFile: "AskYourDocs.cat")
        #expect(ref.flowStem == "AskYourDocs")
        let scope = ref.scope(text: "mlxflow 0.8\n")
        #expect(scope.identity == "AskYourDocs")          // run seed / On Flow / records
        #expect(scope.locationID == "docs")               // where paths resolve
        #expect(scope.directory == ModelStore.shared.workspacesDirectory
            .appendingPathComponent("docs", isDirectory: true))
        #expect(scope.selfFile == scope.directory.appendingPathComponent("AskYourDocs.cat"))
    }

    // MARK: - Two flows, one folder, one file

    @Test func twoWorkspaceFlowsResolveTheSameRelativePathToTheSameFile() throws {
        let (base, _) = try makeWorkspace(id: "kb", files: [
            ("Build.cat", "mlxflow 0.8\n1. Read Files   docs/\n6. Store Index   library.index\n"),
            ("Ask.cat", "mlxflow 0.8\n1. Read Index   library.index\n2. Embed   BGE-M3\n"),
        ])
        defer { try? FileManager.default.removeItem(at: base) }

        let buildRef = WorkspaceRef(workspaceID: "kb", flowFile: "Build.cat")
        let askRef = WorkspaceRef(workspaceID: "kb", flowFile: "Ask.cat")
        // Each flow keeps its own identity …
        #expect(buildRef.scope(text: nil).identity != askRef.scope(text: nil).identity)
        // … but `library.index` resolves to the one shared file for both.
        let a = try buildRef.workspace.resolve("library.index", flowID: buildRef.workspaceID)
        let b = try askRef.workspace.resolve("library.index", flowID: askRef.workspaceID)
        #expect(a == b)
        #expect(a.deletingLastPathComponent().lastPathComponent == "kb")
    }

    // MARK: - A workspace flow runs through the same session the gallery uses

    @Test func aWorkspaceFlowRunsThroughFlowRunSession() async throws {
        let askText = "mlxflow 0.8\n1. Read Text   question.txt\n2. RagQuery\n3. Save Text   answer.md\n\nuses:\n  RagQuery = ./RagQuery.cat\n"
        let ragText = "mlxflow 0.8\n1. Embed   BGE-M3\n2. Read Index   kb.index\n3. Retrieve   (2,1)\n4. Answer   Qwen3 8B\n"
        let (base, ws) = try makeWorkspace(id: "rag", files: [
            ("AskYourDocs.cat", askText), ("RagQuery.cat", ragText),
        ])
        defer { try? FileManager.default.removeItem(at: base) }

        let ref = WorkspaceRef(workspaceID: "rag", flowFile: "AskYourDocs.cat")
        // The scope's workspace is rooted at ModelStore's workspaces dir; point it at the temp
        // root for the test by resolving through `ws` directly.
        let scope = FlowScope(identity: "AskYourDocs", workspace: ws, locationID: "rag",
                              flowText: askText,
                              selfFile: ws.directory(for: "rag").appendingPathComponent("AskYourDocs.cat"))
        _ = ref  // (the production path builds `scope` from `ref`; here we inject the temp root)

        // The session's own Run gate resolves the `uses:` graph and does not refuse.
        let session = FlowRunSession()
        let doc = try CatParser.parse(askText)
        session.prepareInstall(FlowPreflight.Result(), doc: doc, scope: scope)
        #expect(session.runnability == .runnable)

        // And it runs: row 2 (the RagQuery call) finishes.
        let blob = base.appendingPathComponent("blobs")
        let context = FlowRunner.RunContext(scope: scope, blobDirectory: blob,
                                            executor: MockExecutor(blobDirectory: blob))
        var events: [FlowEvent] = []
        for await event in FlowRunner().run(doc, context: context) { events.append(event) }
        #expect(events.contains { if case .finished = $0 { return true } else { return false } })
        #expect(!events.contains { if case .failed = $0 { return true } else { return false } })
    }

    // MARK: - KW-4-1: the New Workspace starter pair (Q4 — "a working pair of flows")

    /// `NewWorkspaceStarter`'s literal content — the same text `FlowGalleryView.createWorkspace`
    /// writes — parses, is genuinely runnable (not just syntactically valid), and produces a
    /// real Knowledge Base card with a working Build **and** a working Ask on the one index
    /// name `newWorkspaceBadge`'s copy promises. Before `KW-4-1` the single-flow starter
    /// touched no index at all; this is the proof the two-file replacement actually delivers
    /// what the badge says.
    @Test func newWorkspaceStarterProducesAWorkingKnowledgeBaseCard() throws {
        let builderDoc = try CatParser.parse(NewWorkspaceStarter.builderText)
        let querierDoc = try CatParser.parse(NewWorkspaceStarter.querierText)
        #expect(FlowRunner.canRun(builderDoc) == .runnable)
        #expect(FlowRunner.canRun(querierDoc) == .runnable)

        // KW-4-FIX-1: everything createWorkspace actually writes -- the two .cat files *and*
        // the two sample inputs their row-1s name. Without notes.txt/question.txt the pair is
        // well-formed and canRun == .runnable, but Build still fails on row 1: canRun is a
        // static gate (task exists, model served, uses: resolves) that never touches disk.
        let (base, ws) = try makeWorkspace(id: "starter", files: [
            (NewWorkspaceStarter.builderFilename, NewWorkspaceStarter.builderText),
            (NewWorkspaceStarter.querierFilename, NewWorkspaceStarter.querierText),
            (NewWorkspaceStarter.notesFilename, NewWorkspaceStarter.notesText),
            (NewWorkspaceStarter.questionFilename, NewWorkspaceStarter.questionText),
        ])
        defer { try? FileManager.default.removeItem(at: base) }
        let dir = ws.directory(for: "starter")

        // Resolve each Read Text row's path the way ReadPath.resolve actually does (settings'
        // bare token, through the workspace's own security boundary) and require the file be
        // there already -- not a hardcoded filename list, which would pass even if
        // createWorkspace wrote the sample under a name no row actually reads.
        for (doc, docFilename) in [(builderDoc, NewWorkspaceStarter.builderFilename),
                                   (querierDoc, NewWorkspaceStarter.querierFilename)] {
            for row in doc.rows where row.task == "Read Text" {
                let raw = try #require(FlowSettings(row.settings).pathValue(),
                                       "\(docFilename)'s Read Text row has no path")
                let url = try ws.resolve(raw, flowID: "starter")
                #expect(FileManager.default.fileExists(atPath: url.path),
                       "\(docFilename)'s Read Text row names '\(raw)', which createWorkspace never wrote")
            }
        }

        // Store Index and Read Index must resolve to the *same* file (proving the builder and
        // querier genuinely share one index, not just a coincidentally-matching raw string) --
        // and that file must NOT exist yet: building it is Store Index's job when the user
        // presses Build, exactly as the bundled ask_your_docs workspace ships no prebuilt
        // library.index either.
        let storeIndexRow = try #require(builderDoc.rows.first { $0.task == "Store Index" })
        let readIndexRow = try #require(querierDoc.rows.first { $0.task == "Read Index" })
        let storeIndexRaw = try #require(FlowSettings(storeIndexRow.settings).pathValue())
        let readIndexRaw = try #require(FlowSettings(readIndexRow.settings).pathValue())
        let storeIndexURL = try ws.resolve(storeIndexRaw, flowID: "starter")
        let readIndexURL = try ws.resolve(readIndexRaw, flowID: "starter")
        #expect(storeIndexURL == readIndexURL)
        #expect(!FileManager.default.fileExists(atPath: storeIndexURL.path))

        let parsed = [
            (file: NewWorkspaceStarter.builderFilename, doc: builderDoc,
             url: dir.appendingPathComponent(NewWorkspaceStarter.builderFilename)),
            (file: NewWorkspaceStarter.querierFilename, doc: querierDoc,
             url: dir.appendingPathComponent(NewWorkspaceStarter.querierFilename)),
        ]
        let cards = WorkspaceKnowledge.classifyWorkspace(
            flows: parsed,
            resolveUses: { doc, selfFile in
                UsesResolver.resolve(doc, workspace: ws, flowID: "starter", selfFile: selfFile)
            },
            resolvePath: { rawPath in try? ws.resolve(rawPath, flowID: "starter") })
        let card = try #require(cards.first { $0.indexName == "library.index" })
        #expect(card.buildFile == NewWorkspaceStarter.builderFilename)
        #expect(card.askFile == NewWorkspaceStarter.querierFilename)
    }

    // MARK: - KW-2-1: New Flow in this workspace

    /// Mirrors `WorkspaceListView.addFlow()`'s exact steps (write the starter under a
    /// collision-free name, rescan) against a workspace that already holds two flows, then
    /// confirms a `uses:` reference in an existing flow resolves to the one just added —
    /// `UsesResolver` needs real siblings, which is the whole point of adding one this way.
    @Test func newFlowNeverOverwritesAndIsResolvableViaUses() throws {
        let (base, ws) = try makeWorkspace(id: "docs", files: [
            ("Ingest.cat", validCat),
            ("Flow.cat", "mlxflow 0.8\n1. Read Text   old.txt\n"),   // a pre-existing collision
        ])
        defer { try? FileManager.default.removeItem(at: base) }
        let dir = ws.directory(for: "docs")

        let starter = "mlxflow 0.8\n1. Read Text   notes.txt\n2. Save Text   out.md\n"
        let filename = WorkspaceStore.firstFreeFlowName(stem: "Flow", in: dir)
        #expect(filename == "Flow-2.cat")               // never the colliding "Flow.cat"
        try starter.write(to: dir.appendingPathComponent(filename), atomically: true, encoding: .utf8)

        // The pre-existing "Flow.cat" survives untouched.
        #expect(try String(contentsOf: dir.appendingPathComponent("Flow.cat"), encoding: .utf8)
            .contains("old.txt"))

        let listed = WorkspaceStore.scan(workspace: ws)
        let workspace = try #require(listed.first { $0.workspaceID == "docs" })
        #expect(workspace.flows.count == 3)
        #expect(workspace.flows.map(\.title).contains("Flow-2"))

        // A sibling's `uses:` line resolves to the flow just added.
        let callerText = "mlxflow 0.8\n1. Read Text   memo.txt\n2. NewStep\n\nuses:\n  NewStep = ./Flow-2.cat\n"
        let callerDoc = try CatParser.parse(callerText)
        let selfFile = dir.appendingPathComponent("Caller.cat")
        let resolved = UsesResolver.resolve(callerDoc, workspace: ws, flowID: "docs", selfFile: selfFile)
        #expect(resolved["NewStep"] != nil)
    }

    // MARK: - KW-2-2: deleting a flow resolves a Knowledge Base ambiguity

    /// Two builders on the same index name make the card's builder side ambiguous
    /// (`buildFile == nil`). Deleting one — `WorkspaceStore.removeFlow`, then a fresh
    /// `classifyWorkspace` over the rescanned flows — should turn it back into a single,
    /// actionable builder. `WorkspaceListView` gets this for free at the SwiftUI layer because
    /// `workspace` is a live lookup (`KW-2-1-FIX`); this proves the plumbing underneath it.
    @Test func deletingOneOfTwoBuildersResolvesTheAmbiguity() throws {
        let buildA = "mlxflow 0.8\n1. Read Files   docs/\n6. Store Index   library.index\n"
        let buildB = "mlxflow 0.8\n1. Read Files   more/\n6. Store Index   library.index\n"
        let askText = "mlxflow 0.8\n1. Read Index   library.index\n2. Embed   BGE-M3\n"
        let (base, ws) = try makeWorkspace(id: "kb", files: [
            ("BuildA.cat", buildA), ("BuildB.cat", buildB), ("Ask.cat", askText),
        ])
        defer { try? FileManager.default.removeItem(at: base) }
        let dir = ws.directory(for: "kb")

        func cards() throws -> [WorkspaceKnowledge.IndexCard] {
            let flows = WorkspaceStore.scan(workspace: ws).first!.flows
            let parsed = flows.compactMap { flow -> (file: String, doc: FlowDocument, url: URL)? in
                guard let doc = try? WorkspaceStore.loadDocument(flow: flow) else { return nil }
                return (flow.url.lastPathComponent, doc, flow.url)
            }
            return WorkspaceKnowledge.classifyWorkspace(
                flows: parsed,
                resolveUses: { doc, selfFile in
                    UsesResolver.resolve(doc, workspace: ws, flowID: "kb", selfFile: selfFile)
                },
                resolvePath: { rawPath in try? ws.resolve(rawPath, flowID: "kb") })
        }

        let before = try #require(try cards().first { $0.indexName == "library.index" })
        #expect(before.buildFile == nil)          // ambiguous: two builders

        try WorkspaceStore.removeFlow(file: dir.appendingPathComponent("BuildB.cat"), from: dir)

        let after = try #require(try cards().first { $0.indexName == "library.index" })
        #expect(after.buildFile == "BuildA.cat")   // resolved: one builder, Build button back
    }

    // MARK: - The bundled uses_example is on the shelf

    @Test func bundledUsesExampleIsListedByWorkspaceStoreAfterPrepare() throws {
        let meta = try #require(BundledWorkspaces.meta(id: "uses_example"))
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-wsview-bw-\(UUID().uuidString)")
        let root = base.appendingPathComponent("workspaces")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let ws = FlowWorkspace(root: root)

        try BundledWorkspaces.prepare(meta, workspace: ws)
        let listed = WorkspaceStore.scan(workspace: ws)
        #expect(listed.count == 1)
        #expect(listed.first?.workspaceID == "uses_example")
        #expect(listed.first?.flows.count == 2)
        // The delete sentence names what it removes.
        let sentence = WorkspaceStore.deletionSummary(try #require(listed.first))
        #expect(sentence.contains("2 flows"))
        #expect(sentence.contains("workspaces folder"))
    }
}
