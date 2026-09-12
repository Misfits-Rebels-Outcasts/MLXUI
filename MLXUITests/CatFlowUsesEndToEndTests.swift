import Testing
import Foundation
@testable import MLXUI

/// CFM-R17-5 — `uses:` becomes reachable end to end. A workspace directory (siblings in one
/// folder) is the first thing that can produce a `uses:` graph the app actually runs, so this
/// exercises code that has never executed in the app: `UsesResolver`, `FlowRunner`'s
/// `usesGraph` wiring, `FlowRunner.canRun`'s `uses:`-row recognition, and the App Store E117
/// channel guarantee (smoke row 39). The pair is `catflow-mlx/library/uses_example/`, ported
/// as the bundled workspace `uses_example`.
struct CatFlowUsesEndToEndTests {

    /// A workspaces root + a workspace directory `id`, with the given `.cat` files written
    /// side by side. Returns the ws-rooted `FlowWorkspace`, the workspace id, and the base.
    private func makeWorkspace(id: String = "ws",
                               files: [(name: String, text: String)]) throws -> (FlowWorkspace, String, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-uses-e2e-\(UUID().uuidString)")
        let root = base.appendingPathComponent("workspaces")
        let dir = root.appendingPathComponent(id, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for f in files {
            try f.text.write(to: dir.appendingPathComponent(f.name), atomically: true, encoding: .utf8)
        }
        return (FlowWorkspace(root: root), id, base)
    }

    private let askYourDocs = """
    mlxflow 0.8
    1. Read Text   question.txt
    2. RagQuery    top_k=5
    3. Save Text   answer.md

    uses:
      RagQuery = ./RagQuery.cat
    """

    private let ragQuery = """
    mlxflow 0.8
    params: top_k = 5
    1. Embed                BGE-M3
    2. Read Index           kb.index
    3. Retrieve     (2,1)   top_k={top_k}
    4. Template             "Context:\\n{input}\\n\\nAnswer the question."
    5. Answer               Qwen3 8B
    """

    // MARK: - The used flow's rows execute

    @Test func usesResolverBuildsTheGraphFromSiblings() throws {
        let (ws, id, base) = try makeWorkspace(files: [
            ("AskYourDocs.cat", askYourDocs), ("RagQuery.cat", ragQuery),
        ])
        defer { try? FileManager.default.removeItem(at: base) }
        let doc = try CatParser.parse(askYourDocs)
        let graph = UsesResolver.resolve(doc, workspace: ws, flowID: id,
                                         selfFile: ws.directory(for: id).appendingPathComponent("AskYourDocs.cat"))
        let used = try #require(graph["RagQuery"])
        #expect(used.rows.count == 5)
        #expect(used.params.map(\.name) == ["top_k"])
        #expect(used.rows.first?.task == "Embed")
        #expect(used.nested.isEmpty)
    }

    @Test func theUsedFlowsRowsExecute() async throws {
        let (ws, id, base) = try makeWorkspace(files: [
            ("AskYourDocs.cat", askYourDocs), ("RagQuery.cat", ragQuery),
        ])
        defer { try? FileManager.default.removeItem(at: base) }
        let doc = try CatParser.parse(askYourDocs)
        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))
        let graph = UsesResolver.resolve(doc, workspace: ws, flowID: id,
                                         selfFile: ws.directory(for: id).appendingPathComponent("AskYourDocs.cat"))

        let events = try await FlowInterpreter.run(doc, executor: mock,
                                                   definitions: doc.definitions, presets: doc.presets,
                                                   usesGraph: graph)
        #expect(events.last?.kind == .runCompleted)
        // `2. RagQuery` expanded; its own rows ran as 2.1 … 2.5.
        for sub in ["2.1", "2.2", "2.3", "2.4", "2.5"] {
            #expect(events.contains { $0.kind == .rowCompleted && $0.path == sub },
                    "RagQuery row \(sub) did not execute")
        }
        // No row failed.
        #expect(!events.contains { $0.kind == .rowFailed })
    }

    @Test func fullRunnerPathExecutesTheUsedFlow() async throws {
        let (ws, id, base) = try makeWorkspace(files: [
            ("AskYourDocs.cat", askYourDocs), ("RagQuery.cat", ragQuery),
        ])
        defer { try? FileManager.default.removeItem(at: base) }
        let doc = try CatParser.parse(askYourDocs)
        let blob = base.appendingPathComponent("blobs")
        let scope = FlowScope(identity: "AskYourDocs", workspace: ws, locationID: id,
                              flowText: askYourDocs,
                              selfFile: ws.directory(for: id).appendingPathComponent("AskYourDocs.cat"))
        let context = FlowRunner.RunContext(scope: scope, blobDirectory: blob,
                                            executor: MockExecutor(blobDirectory: blob))
        var events: [FlowEvent] = []
        for await event in FlowRunner().run(doc, context: context) { events.append(event) }

        // Row 2 (the `uses:` call) finished — it wouldn't if RagQuery's rows had thrown.
        #expect(events.contains { if case .finished = $0 { return true } else { return false } })
        #expect(!events.contains { if case .failed = $0 { return true } else { return false } })
    }

    // MARK: - canRun recognises a `uses:` row

    @Test func canRunAcceptsAResolvedUsesCall() throws {
        let (ws, id, base) = try makeWorkspace(files: [
            ("AskYourDocs.cat", askYourDocs), ("RagQuery.cat", ragQuery),
        ])
        defer { try? FileManager.default.removeItem(at: base) }
        let doc = try CatParser.parse(askYourDocs)
        let scope = FlowScope(identity: "AskYourDocs", workspace: ws, locationID: id, flowText: askYourDocs,
                              selfFile: ws.directory(for: id).appendingPathComponent("AskYourDocs.cat"))
        #expect(FlowRunner.canRun(doc, scope: scope) == .runnable)
    }

    @Test func canRunRefusesAnUnresolvableUsesCall() throws {
        // `RagQuery.cat` is simply not there.
        let (ws, id, base) = try makeWorkspace(files: [("AskYourDocs.cat", askYourDocs)])
        defer { try? FileManager.default.removeItem(at: base) }
        let doc = try CatParser.parse(askYourDocs)
        let scope = FlowScope(identity: "AskYourDocs", workspace: ws, locationID: id, flowText: askYourDocs,
                              selfFile: ws.directory(for: id).appendingPathComponent("AskYourDocs.cat"))
        guard case .notRunnable(let reason) = FlowRunner.canRun(doc, scope: scope) else {
            Issue.record("expected .notRunnable"); return
        }
        #expect(reason.contains("RagQuery"))
    }

    @Test func canRunRefusesAUsedFlowWithAnUnknownRow() throws {
        let badUsed = "mlxflow 0.8\n1. Frobnicate   x.txt\n"
        let (ws, id, base) = try makeWorkspace(files: [
            ("AskYourDocs.cat", askYourDocs), ("RagQuery.cat", badUsed),
        ])
        defer { try? FileManager.default.removeItem(at: base) }
        let doc = try CatParser.parse(askYourDocs)
        let scope = FlowScope(identity: "AskYourDocs", workspace: ws, locationID: id, flowText: askYourDocs,
                              selfFile: ws.directory(for: id).appendingPathComponent("AskYourDocs.cat"))
        guard case .notRunnable(let reason) = FlowRunner.canRun(doc, scope: scope) else {
            Issue.record("expected .notRunnable"); return
        }
        #expect(reason.contains("Frobnicate"))
    }

    // MARK: - E113 / E114 / E115 fire for the workspace layout

    @Test func e113FiresForAnUnresolvedName() throws {
        // The flow has a `uses:` section, but row 3 names something not in it — E113's own
        // wording tells the user to add `Retriever = ./Retriever.cat` to `uses:`.
        let strayName = """
        mlxflow 0.8
        1. Read Text   q.txt
        2. RagQuery
        3. Retriever
        uses:
          RagQuery = ./RagQuery.cat
        """
        let parsed = try CatParser.parseForValidation(strayName)
        let issues = FlowValidator.checkFlow(parsed)
        let e113 = try #require(issues.first { $0.code == "E113" })
        #expect(e113.row == "3")
        #expect(e113.message.contains("Retriever"))
    }

    @Test func e114FiresForAPathOutsideTheWorkspace() throws {
        let escaping = """
        mlxflow 0.8
        1. RagQuery
        uses:
          RagQuery = ../evil.cat
        """
        let (ws, id, base) = try makeWorkspace(files: [("AskYourDocs.cat", escaping)])
        defer { try? FileManager.default.removeItem(at: base) }
        let parsed = try CatParser.parseForValidation(escaping)
        let issues = FlowValidator.checkFlow(parsed, workspace: ws, flowID: id,
                                             rootFile: ws.directory(for: id).appendingPathComponent("AskYourDocs.cat"))
        #expect(issues.contains { $0.code == "E114" })
    }

    @Test func e115FiresForACycle() throws {
        let aCat = "mlxflow 0.8\n1. B\nuses:\n  B = ./B.cat\n"
        let bCat = "mlxflow 0.8\n1. A\nuses:\n  A = ./A.cat\n"
        let (ws, id, base) = try makeWorkspace(files: [("A.cat", aCat), ("B.cat", bCat)])
        defer { try? FileManager.default.removeItem(at: base) }
        let parsed = try CatParser.parseForValidation(aCat)
        let issues = FlowValidator.checkFlow(parsed, workspace: ws, flowID: id,
                                             rootFile: ws.directory(for: id).appendingPathComponent("A.cat"))
        #expect(issues.contains { $0.code == "E115" })
    }

    // MARK: - E117 — the App Store channel guarantee (smoke row 39)

    @Test func e117RefusesInAppStoreWhenTheUsedFlowImprovises() throws {
        // The *used* flow declares `improvise`; the caller's header doesn't.
        let usedImprovises = "mlxflow 0.8; improvise\n1. Read Text   lib.txt\n"
        let (ws, id, base) = try makeWorkspace(files: [
            ("AskYourDocs.cat", askYourDocs), ("RagQuery.cat", usedImprovises),
        ])
        defer { try? FileManager.default.removeItem(at: base) }
        let doc = try CatParser.parse(askYourDocs)
        let scope = FlowScope(identity: "AskYourDocs", workspace: ws, locationID: id, flowText: askYourDocs,
                              selfFile: ws.directory(for: id).appendingPathComponent("AskYourDocs.cat"))

        #expect(CapabilityGate.isAppStoreBuild, "this proof only holds in the APPSTORE_BUILD test host")
        guard case .notRunnable(let reason) = FlowRunner.canRun(doc, scope: scope) else {
            Issue.record("App Store build must refuse an inherited `improvise`"); return
        }
        #expect(reason.contains("improvise"))
    }

    @Test func e117RefusalHaltsTheRunnerBeforeAnyRow() async throws {
        let usedImprovises = "mlxflow 0.8; improvise\n1. Read Text   lib.txt\n"
        let (ws, id, base) = try makeWorkspace(files: [
            ("AskYourDocs.cat", askYourDocs), ("RagQuery.cat", usedImprovises),
        ])
        defer { try? FileManager.default.removeItem(at: base) }
        let doc = try CatParser.parse(askYourDocs)
        let blob = base.appendingPathComponent("blobs")
        let scope = FlowScope(identity: "AskYourDocs", workspace: ws, locationID: id, flowText: askYourDocs,
                              selfFile: ws.directory(for: id).appendingPathComponent("AskYourDocs.cat"))
        let context = FlowRunner.RunContext(scope: scope, blobDirectory: blob,
                                            executor: MockExecutor(blobDirectory: blob))
        var events: [FlowEvent] = []
        for await event in FlowRunner().run(doc, context: context) { events.append(event) }
        // Exactly one event: the refusal on row 1. No row started.
        #expect(events.count == 1)
        guard case .failed(_, let error) = events.first else { Issue.record("expected .failed"); return }
        #expect("\(error)".contains("improvise") || "\(error)".contains("App Store"))
    }

    // MARK: - CFM-R17-FIX-2 — canRun's per-row class gates recurse into `uses:`

    // The caller (`AskYourDocs.cat`) declares nothing; the *used* flow carries an
    // **undeclared** `Improvise` row (no `; improvise` header). `CapabilityGate.effectiveFlags`
    // sees no flag; before FIX-2(b) `usedFlowRefusal` only asked `TaskCatalog.get(task) == nil`
    // (and `Improvise` is in the catalog), so `canRun` returned `.runnable` and the `.agent`
    // refusal only landed mid-run through `RealExecutor`'s channel gate — after preceding rows
    // had written real files. FIX-2(b) applies the same `rowClassRefusal` verdict recursively.

    private var usedImproviseUndeclared: String {
        // No `; improvise` — the row is undeclared.
        "mlxflow 0.8\n1. Improvise   \"rewrite the file\"\n"
    }

    /// The caller flow: a `Save Text` **before** the `uses:` call, so a mid-run refusal would
    /// already have written a real file by the time it fires.
    private var callerWithSaveBeforeUses: String {
        """
        mlxflow 0.8
        1. Read Text   q.txt
        2. Save Text   out.md
        3. Helper

        uses:
          Helper = ./Helper.cat
        """
    }

    @Test func appStoreCanRunRefusesAnUndeclaredImproviseReachedThroughUses() throws {
        #expect(CapabilityGate.isAppStoreBuild, "this proof only holds in the APPSTORE_BUILD test host")
        let (ws, id, base) = try makeWorkspace(files: [
            ("AskYourDocs.cat", callerWithSaveBeforeUses),
            ("Helper.cat", usedImproviseUndeclared),
        ])
        defer { try? FileManager.default.removeItem(at: base) }
        let doc = try CatParser.parse(callerWithSaveBeforeUses)
        let scope = FlowScope(identity: "AskYourDocs", workspace: ws, locationID: id,
                              flowText: callerWithSaveBeforeUses,
                              selfFile: ws.directory(for: id).appendingPathComponent("AskYourDocs.cat"))

        guard case .notRunnable(let reason) = FlowRunner.canRun(doc, scope: scope) else {
            Issue.record("App Store build must refuse an undeclared `improvise` row reached through `uses:`")
            return
        }
        #expect(reason.contains("Helper"))
        #expect(reason.contains("App Store"))
    }

    @Test func appStoreRefusalForAUsedImproviseHaltsBeforeAnyRowAndWritesNothing() async throws {
        #expect(CapabilityGate.isAppStoreBuild, "this proof only holds in the APPSTORE_BUILD test host")
        let (ws, id, base) = try makeWorkspace(files: [
            ("AskYourDocs.cat", callerWithSaveBeforeUses),
            ("Helper.cat", usedImproviseUndeclared),
        ])
        defer { try? FileManager.default.removeItem(at: base) }
        let dir = ws.directory(for: id)
        try "the question".write(to: dir.appendingPathComponent("q.txt"), atomically: true, encoding: .utf8)
        let outURL = dir.appendingPathComponent("out.md")

        let doc = try CatParser.parse(callerWithSaveBeforeUses)
        let blob = base.appendingPathComponent("blobs")
        let scope = FlowScope(identity: "AskYourDocs", workspace: ws, locationID: id,
                              flowText: callerWithSaveBeforeUses,
                              selfFile: dir.appendingPathComponent("AskYourDocs.cat"))
        // A real executor so `Save Text` genuinely writes — the damage FIX-2 is about is files
        // on disk from rows that ran before a refusal that should have fired before row 1.
        let executor = RealExecutor(
            workspace: ws, flowID: id, blobDirectory: blob,
            makeModelStage: { _, _ in throw FlowError.unknownTask(row: "no model in this flow") },
            installedModelIDs: [], catalog: [])
        let context = FlowRunner.RunContext(scope: scope, blobDirectory: blob, executor: executor)

        var events: [FlowEvent] = []
        for await event in FlowRunner().run(doc, context: context) { events.append(event) }

        // Refused before row 1: exactly one `.failed`, nothing finished, out.md never written.
        #expect(!events.contains { if case .finished = $0 { return true } else { return false } },
                "no row should finish — the flow must be refused before row 1")
        #expect(!FileManager.default.fileExists(atPath: outURL.path), "out.md must not be written")
        guard case .failed(_, let error)? = events.first else {
            Issue.record("expected a single .failed refusal, got \(events)")
            return
        }
        #expect("\(error)".contains("App Store") || "\(error)".contains("Helper"))
    }

    /// Phase WS ported `Web Search` (this test's own former example of an unported
    /// `.net` task) — `TaskAvailability.supportedNetTools` now covers every `.net`-class
    /// catalog task, so there is no real task left to demonstrate the `.unported`
    /// recursion-through-`uses:` refusal with (same gap `canRunAllowsWebSearchNowThat
    /// ItIsPorted` in `CatFlowTaskAvailabilityTests` notes for the non-`uses:` case).
    /// This now pins the positive fact instead: a used flow calling `Web Search`
    /// recurses to `.runnable`, not a refusal — needing no key to clear this gate either
    /// (`.needsSetup` isn't a `canRun`/`uses:` recursion refusal).
    @Test func appStoreCanRunAcceptsWebSearchReachedThroughUsesNowThatItIsPorted() throws {
        #expect(CapabilityGate.isAppStoreBuild)
        let caller = "mlxflow 0.8\n1. Helper\n\nuses:\n  Helper = ./Helper.cat\n"
        let (ws, id, base) = try makeWorkspace(files: [
            ("Caller.cat", caller),
            ("Helper.cat", "mlxflow 0.8\n1. Web Search   \"latest news\"\n"),
        ])
        defer { try? FileManager.default.removeItem(at: base) }
        let doc = try CatParser.parse(caller)
        let scope = FlowScope(identity: "Caller", workspace: ws, locationID: id, flowText: caller,
                              selfFile: ws.directory(for: id).appendingPathComponent("Caller.cat"))
        #expect(FlowRunner.canRun(doc, scope: scope) == .runnable)
    }
    // (a well-formed `uses:` call still resolving to `.runnable` is covered by
    // `canRunAcceptsAResolvedUsesCall` above — the FIX-2 recursion runs on that path too.)

    // MARK: - CFM-R17-FIX-10 — a used flow's own `transforms:` is unsupported

    private var callerOfHelper: String {
        "mlxflow 0.8\n1. Helper\n\nuses:\n  Helper = ./Helper.cat\n"
    }

    /// `RealExecutor.transforms` is built once from the *caller's* `doc.transforms`
    /// (`AppFlowExecutorFactory`), so a transform declared inside a used flow could never run.
    /// FIX-2(b) caught it only incidentally, as `.unknownTask`. FIX-10: refuse it up front with
    /// the real reason — and **not** behind an `isAppStoreBuild` branch, so it holds in the
    /// Direct build too (where a caller's own `transforms:` *would* run, fenced).
    @Test func aUsedFlowCallingItsOwnTransformIsRefusedWithTheRealReason() throws {
        let (ws, id, base) = try makeWorkspace(files: [
            ("Caller.cat", callerOfHelper),
            ("Helper.cat", "mlxflow 0.8\n1. Tidy\n\ntransforms:\n  Tidy  text -> text\n    run: script.sh\n"),
        ])
        defer { try? FileManager.default.removeItem(at: base) }
        let doc = try CatParser.parse(callerOfHelper)
        let scope = FlowScope(identity: "Caller", workspace: ws, locationID: id, flowText: callerOfHelper,
                              selfFile: ws.directory(for: id).appendingPathComponent("Caller.cat"))

        guard case .notRunnable(let reason) = FlowRunner.canRun(doc, scope: scope) else {
            Issue.record("a used flow that declares and calls its own `transforms:` must be refused up front")
            return
        }
        #expect(reason.contains("Tidy"))
        #expect(reason.contains("transforms:"))
        #expect(reason.contains("can't run its own transforms"))
        // Not "unknown task" (the old incidental path), and edition-independent.
        #expect(!reason.contains("isn't a task this version of Flows knows"))
        #expect(!reason.contains("App Store"))
    }

    /// The limit is "declares **and calls**": a used flow that declares a transform it never
    /// invokes still runs (the `transformNames` check keys on the row, not the section).
    @Test func aUsedFlowThatDeclaresButNeverCallsATransformStillRuns() throws {
        let (ws, id, base) = try makeWorkspace(files: [
            ("Caller.cat", callerOfHelper),
            ("Helper.cat", "mlxflow 0.8\n1. Read Text   a.txt\n2. Save Text   b.md\n\ntransforms:\n  Tidy  text -> text\n    run: script.sh\n"),
        ])
        defer { try? FileManager.default.removeItem(at: base) }
        let doc = try CatParser.parse(callerOfHelper)
        let scope = FlowScope(identity: "Caller", workspace: ws, locationID: id, flowText: callerOfHelper,
                              selfFile: ws.directory(for: id).appendingPathComponent("Caller.cat"))
        #expect(FlowRunner.canRun(doc, scope: scope) == .runnable)
    }

    // MARK: - The bundled workspace

    @Test func bundledUsesExampleMaterializesAndListsAsAWorkspace() async throws {
        let meta = try #require(BundledWorkspaces.meta(id: "uses_example"))
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-bw-\(UUID().uuidString)")
        let root = base.appendingPathComponent("workspaces")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let ws = FlowWorkspace(root: root)

        try BundledWorkspaces.prepare(meta, workspace: ws)
        let dir = ws.directory(for: "uses_example")
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("AskYourDocs.cat").path))
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("RagQuery.cat").path))
        // CFM-R17-FIX-6: the assets the rows read ship too — no failure on row 1.
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("question.txt").path))
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("kb.index/manifest.json").path))

        // It lists as a two-flow workspace (a bundled workspace shows in the shelf like any
        // other — CFM-R17-FIX-8).
        let listed = try #require(WorkspaceStore.scan(workspace: ws).first)
        #expect(listed.workspaceID == "uses_example")
        #expect(listed.flows.map(\.title).sorted() == ["AskYourDocs", "RagQuery"])

        // The `uses:` graph resolves and the interpreter walks both flows. (This uses
        // `MockExecutor`, which synthesizes every output from the task shape and never touches
        // the filesystem — it proves the composition, not that a real RAG run succeeds. The
        // real-executor floor is `bundledUsesExampleClearsRowOneUnderARealExecutor`.)
        let text = try String(contentsOf: dir.appendingPathComponent(meta.entryFlow), encoding: .utf8)
        let doc = try CatParser.parse(text)
        let graph = UsesResolver.resolve(doc, workspace: ws, flowID: "uses_example",
                                         selfFile: dir.appendingPathComponent(meta.entryFlow))
        #expect(graph["RagQuery"] != nil)
        let events = try await FlowInterpreter.run(doc, executor: MockExecutor(blobDirectory: base.appendingPathComponent("blobs")),
                                                   definitions: doc.definitions, presets: doc.presets, usesGraph: graph)
        #expect(events.last?.kind == .runCompleted)
    }

    /// CFM-R17-FIX-6 — with the shipped `question.txt` and `kb.index/`, materialising the
    /// bundled workspace and running it under a **real** executor clears row 1 (`Read Text
    /// question.txt`). The run still can't finish in a unit host with no model weights — the
    /// wall is the `Embed` row inside `RagQuery`, one row into the used flow, which is exactly
    /// where journal `2026-169` expects it, not on row 1.
    @Test func bundledUsesExampleClearsRowOneUnderARealExecutor() async throws {
        let meta = try #require(BundledWorkspaces.meta(id: "uses_example"))
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-bw-real-\(UUID().uuidString)")
        let root = base.appendingPathComponent("workspaces")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let ws = FlowWorkspace(root: root)

        try BundledWorkspaces.prepare(meta, workspace: ws)
        let dir = ws.directory(for: "uses_example")
        let text = try String(contentsOf: dir.appendingPathComponent(meta.entryFlow), encoding: .utf8)
        let doc = try CatParser.parse(text)
        let blob = base.appendingPathComponent("blobs")
        let graph = UsesResolver.resolve(doc, workspace: ws, flowID: "uses_example",
                                         selfFile: dir.appendingPathComponent(meta.entryFlow))
        let executor = RealExecutor(
            workspace: ws, flowID: "uses_example", blobDirectory: blob,
            makeModelStage: { _, _ in throw FlowError.unknownTask(row: "no model weights in a unit host") },
            installedModelIDs: [], catalog: [])
        let events = try await FlowInterpreter.run(doc, executor: executor,
                                                   definitions: doc.definitions, presets: doc.presets,
                                                   usesGraph: graph)

        // Row 1 read its file.
        #expect(events.contains { $0.kind == .rowCompleted && $0.path == "1" },
                "Read Text question.txt should complete — the asset ships now")
        // The used flow was entered and it's the model row that stops it, not row 1.
        #expect(events.contains { $0.kind == .rowFailed && $0.path == "2.1" },
                "the wall is `Embed` (RagQuery row 1 → 2.1), one row into the used flow")
        #expect(!events.contains { $0.kind == .rowCompleted && $0.path == "3" },
                "Save Text must not run — the flow never reaches it")
    }

    @Test func prepareIsIdempotentAndLeavesEditsAlone() throws {
        let meta = try #require(BundledWorkspaces.meta(id: "uses_example"))
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-bw2-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base.appendingPathComponent("workspaces"),
                                                withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let ws = FlowWorkspace(root: base.appendingPathComponent("workspaces"))
        try BundledWorkspaces.prepare(meta, workspace: ws)
        let ask = ws.directory(for: "uses_example").appendingPathComponent("AskYourDocs.cat")
        try "mlxflow 0.8\n1. Read Text edited.txt\n".write(to: ask, atomically: true, encoding: .utf8)
        try BundledWorkspaces.prepare(meta, workspace: ws)   // second call
        #expect(try String(contentsOf: ask, encoding: .utf8).contains("edited.txt"))
    }
}
