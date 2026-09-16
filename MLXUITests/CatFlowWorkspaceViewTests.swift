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
