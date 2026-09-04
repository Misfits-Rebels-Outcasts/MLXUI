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
    catflow 0.8
    1. Read Text   question.txt
    2. RagQuery    top_k=5
    3. Save Text   answer.md

    uses:
      RagQuery = ./RagQuery.cat
    """

    private let ragQuery = """
    catflow 0.8
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
        let badUsed = "catflow 0.8\n1. Frobnicate   x.txt\n"
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
        catflow 0.8
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
        catflow 0.8
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
        let aCat = "catflow 0.8\n1. B\nuses:\n  B = ./B.cat\n"
        let bCat = "catflow 0.8\n1. A\nuses:\n  A = ./A.cat\n"
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
        let usedImprovises = "catflow 0.8; improvise\n1. Read Text   lib.txt\n"
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
        let usedImprovises = "catflow 0.8; improvise\n1. Read Text   lib.txt\n"
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

        // It lists as a two-flow workspace, and is excluded from a user's own list.
        let listed = try #require(WorkspaceStore.scan(workspace: ws).first)
        #expect(listed.workspaceID == "uses_example")
        #expect(listed.flows.map(\.title).sorted() == ["AskYourDocs", "RagQuery"])
        #expect(WorkspaceStore.scan(workspace: ws, bundledWorkspaceIDs: BundledWorkspaces.ids).isEmpty)

        // And the bundled pair runs end to end.
        let text = try String(contentsOf: dir.appendingPathComponent(meta.entryFlow), encoding: .utf8)
        let doc = try CatParser.parse(text)
        let graph = UsesResolver.resolve(doc, workspace: ws, flowID: "uses_example",
                                         selfFile: dir.appendingPathComponent(meta.entryFlow))
        #expect(graph["RagQuery"] != nil)
        let events = try await FlowInterpreter.run(doc, executor: MockExecutor(blobDirectory: base.appendingPathComponent("blobs")),
                                                   definitions: doc.definitions, presets: doc.presets, usesGraph: graph)
        #expect(events.last?.kind == .runCompleted)
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
        try "catflow 0.8\n1. Read Text edited.txt\n".write(to: ask, atomically: true, encoding: .utf8)
        try BundledWorkspaces.prepare(meta, workspace: ws)   // second call
        #expect(try String(contentsOf: ask, encoding: .utf8).contains("edited.txt"))
    }
}
