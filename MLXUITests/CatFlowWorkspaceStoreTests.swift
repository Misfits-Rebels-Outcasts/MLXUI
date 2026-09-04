import Testing
import Foundation
@testable import MLXUI

/// CFM-R17-2 — `WorkspaceStore`: the workspaces root beside `flows/`. Mirrors
/// `UserFlowStore`'s shape with the **opposite** cardinality (one or more `.cat` per
/// folder), lists a broken file with its sentence instead of dropping it, and round-trips a
/// whole directory through export → import. `UserFlowStore` is not touched — its own tests
/// (`CatFlowUserFlowStoreTests`) still pass unchanged.
struct CatFlowWorkspaceStoreTests {

    private func makeRoot() throws -> (FlowWorkspace, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-ws-\(UUID().uuidString)")
        let root = base.appendingPathComponent("workspaces")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return (FlowWorkspace(root: root), base)
    }

    private func makeWorkspace(root: URL, id: String, flows: [(name: String, text: String)]) throws {
        let dir = root.appendingPathComponent(id, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for f in flows {
            try f.text.write(to: dir.appendingPathComponent("\(f.name).cat"), atomically: true, encoding: .utf8)
        }
    }

    private func teardown(_ base: URL) { try? FileManager.default.removeItem(at: base) }

    private let validCat = """
    catflow 0.8
    1. Read Text   memo.txt
    2. Save Text   out.md
    """

    // MARK: - scan

    @Test func emptyRootYieldsNoWorkspaces() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        #expect(WorkspaceStore.scan(workspace: ws).isEmpty)
    }

    @Test func threeCatFilesListAsThreeFlowsInOneWorkspace() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [
            ("Index", validCat), ("Ask", validCat), ("Rebuild", validCat),
        ])
        let all = WorkspaceStore.scan(workspace: ws)
        #expect(all.count == 1)
        let w = try #require(all.first)
        #expect(w.workspaceID == "docs")
        #expect(w.title == "docs")
        #expect(w.flows.count == 3)
        // Sorted by name, each parses.
        #expect(w.flows.map(\.title) == ["Ask", "Index", "Rebuild"])
        #expect(w.flows.allSatisfy { $0.parseIssue == nil })
    }

    @Test func oneUnparseableFileListsWithItsSentenceOthersSurvive() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [
            ("Good1", validCat),
            ("Broken", "this is not a cat file at all \u{1}"),
            ("Good2", validCat),
        ])
        let w = try #require(WorkspaceStore.scan(workspace: ws).first)
        #expect(w.flows.count == 3)                       // the broken one did not vanish
        let broken = try #require(w.flows.first { $0.title == "Broken" })
        #expect(broken.parseIssue != nil)
        #expect(w.flows.filter { $0.title.hasPrefix("Good") }.allSatisfy { $0.parseIssue == nil })
    }

    @Test func folderWithNoFlowFileIsNotAWorkspace() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        let bare = ws.root.appendingPathComponent("just-files", isDirectory: true)
        try FileManager.default.createDirectory(at: bare, withIntermediateDirectories: true)
        try Data([1, 2, 3]).write(to: bare.appendingPathComponent("notes.txt"))
        #expect(WorkspaceStore.scan(workspace: ws).isEmpty)
    }

    @Test func newestWorkspaceFirst() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "older", flows: [("A", validCat)])
        try? FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSinceNow: -3600)],
            ofItemAtPath: ws.root.appendingPathComponent("older").path)
        try makeWorkspace(root: ws.root, id: "newer", flows: [("A", validCat)])
        #expect(WorkspaceStore.scan(workspace: ws).map(\.workspaceID) == ["newer", "older"])
    }

    @Test func bundledWorkspaceIsExcluded() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "mine", flows: [("A", validCat)])
        try makeWorkspace(root: ws.root, id: "AskYourDocs", flows: [("Ask", validCat)])
        #expect(WorkspaceStore.scan(workspace: ws, bundledWorkspaceIDs: ["AskYourDocs"])
            .map(\.workspaceID) == ["mine"])
    }

    // MARK: - loadDocument

    @Test func loadDocumentParsesAWorkspaceFlow() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("Ask", validCat)])
        let flow = try #require(WorkspaceStore.scan(workspace: ws).first?.flows.first)
        let doc = try WorkspaceStore.loadDocument(flow: flow)
        #expect(doc.rows.count == 2)
        #expect(doc.rows.first?.task == "Read Text")
    }

    // MARK: - deletionSummary

    @Test func deletionSummaryNamesFlowsAndIndexes() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("Index", validCat), ("Ask", validCat)])
        // Add one real index directory (a subfolder with a manifest.json) and a decoy folder.
        let dir = ws.root.appendingPathComponent("docs")
        let idx = dir.appendingPathComponent("library.index", isDirectory: true)
        try FileManager.default.createDirectory(at: idx, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: idx.appendingPathComponent("manifest.json"))
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("docs", isDirectory: true),
                                                withIntermediateDirectories: true)

        let w = try #require(WorkspaceStore.scan(workspace: ws).first)
        let sentence = WorkspaceStore.deletionSummary(w)
        #expect(sentence.contains("2 flows"))
        #expect(sentence.contains("1 index"))
        #expect(sentence.contains("workspaces folder"))
        #expect(sentence.contains("can't be undone"))
    }

    @Test func deletionSummarySingularAndNoIndex() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "solo", flows: [("Only", validCat)])
        let w = try #require(WorkspaceStore.scan(workspace: ws).first)
        let sentence = WorkspaceStore.deletionSummary(w)
        #expect(sentence.contains("1 flow"))
        #expect(!sentence.contains("index"))
    }

    // MARK: - remove

    @Test func removeDeletesTheWholeDirectory() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("Ask", validCat)])
        try WorkspaceStore.remove(workspaceID: "docs", workspace: ws)
        #expect(!FileManager.default.fileExists(atPath: ws.root.appendingPathComponent("docs").path))
        #expect(throws: WorkspaceStoreError.notFound("docs")) {
            try WorkspaceStore.remove(workspaceID: "docs", workspace: ws)
        }
    }

    // MARK: - import / export

    @Test func importCopiesEveryFlowAndSharedFile() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        let source = base.appendingPathComponent("Ask Your Docs", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try validCat.write(to: source.appendingPathComponent("Index.cat"), atomically: true, encoding: .utf8)
        try validCat.write(to: source.appendingPathComponent("Ask.cat"), atomically: true, encoding: .utf8)
        let docs = source.appendingPathComponent("docs", isDirectory: true)
        try FileManager.default.createDirectory(at: docs, withIntermediateDirectories: true)
        try "hello".write(to: docs.appendingPathComponent("note.txt"), atomically: true, encoding: .utf8)

        let w = try WorkspaceStore.importWorkspace(from: source, workspace: ws)
        #expect(w.workspaceID == "Ask Your Docs")         // name preserved, not a uuid
        #expect(w.flows.map(\.title) == ["Ask", "Index"])
        let dir = ws.directory(for: w.workspaceID)
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("Index.cat").path))
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("Ask.cat").path))
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("docs/note.txt").path))
        #expect(WorkspaceStore.scan(workspace: ws).count == 1)
    }

    @Test func importRefusesAFolderWithNoFlowFile() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        let empty = base.appendingPathComponent("empty", isDirectory: true)
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        try "x".write(to: empty.appendingPathComponent("readme.txt"), atomically: true, encoding: .utf8)
        #expect(throws: WorkspaceStoreError.importNeedsAFlowFile) {
            try WorkspaceStore.importWorkspace(from: empty, workspace: ws)
        }
    }

    @Test func importRefusesAnExistingWorkspaceID() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("A", validCat)])
        let source = base.appendingPathComponent("docs", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try validCat.write(to: source.appendingPathComponent("B.cat"), atomically: true, encoding: .utf8)
        #expect(throws: WorkspaceStoreError.importDestinationExists("docs")) {
            try WorkspaceStore.importWorkspace(from: source, workspace: ws)
        }
    }

    @Test func exportThenImportRoundTripsTheDirectoryVerbatim() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("Index", validCat), ("Ask", validCat)])
        let dir = ws.directory(for: "docs")
        try Data([7, 7, 7]).write(to: dir.appendingPathComponent("corpus.bin"))
        let idx = dir.appendingPathComponent("library.index", isDirectory: true)
        try FileManager.default.createDirectory(at: idx, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: idx.appendingPathComponent("manifest.json"))

        let destination = base.appendingPathComponent("exports", isDirectory: true)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let exported = try WorkspaceStore.export(workspaceID: "docs", name: "Docs Workspace",
                                                 to: destination, workspace: ws)
        #expect(exported.lastPathComponent == "Docs Workspace")

        // Re-import the exported folder; every file comes back at the same relative path.
        let reimported = try WorkspaceStore.importWorkspace(from: exported, workspace: ws)
        let back = ws.directory(for: reimported.workspaceID)
        for rel in ["Index.cat", "Ask.cat", "corpus.bin", "library.index/manifest.json"] {
            #expect(FileManager.default.contentsEqual(
                atPath: dir.appendingPathComponent(rel).path,
                andPath: back.appendingPathComponent(rel).path), "\(rel) did not round-trip")
        }
        #expect(reimported.flows.map(\.title) == ["Ask", "Index"])
    }

    @Test func exportRefusesAnExistingDestination() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("A", validCat)])
        let destination = base.appendingPathComponent("exports", isDirectory: true)
        try FileManager.default.createDirectory(
            at: destination.appendingPathComponent("Docs", isDirectory: true),
            withIntermediateDirectories: true)
        #expect(throws: WorkspaceStoreError.exportDestinationExists("Docs")) {
            try WorkspaceStore.export(workspaceID: "docs", name: "Docs",
                                      to: destination, workspace: ws)
        }
    }

    // MARK: - ModelStore

    @Test func modelStoreExposesAWorkspacesDirectoryBesideFlows() {
        let store = ModelStore(baseDirectory: URL(fileURLWithPath: "/tmp/aibrowser-test"))
        #expect(store.workspacesDirectory.lastPathComponent == "workspaces")
        #expect(store.workspacesDirectory.deletingLastPathComponent().path == store.baseDirectory.path)
        #expect(store.workspacesDirectory.deletingLastPathComponent().path
            == store.flowsDirectory.deletingLastPathComponent().path)
    }
}
