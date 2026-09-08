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
