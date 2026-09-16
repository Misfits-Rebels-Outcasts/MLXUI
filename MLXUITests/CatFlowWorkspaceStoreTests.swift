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
    mlxflow 0.8
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

    @Test func trulyEmptyFolderIsNotAWorkspace() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        let bare = ws.root.appendingPathComponent("nothing-here", isDirectory: true)
        try FileManager.default.createDirectory(at: bare, withIntermediateDirectories: true)
        #expect(WorkspaceStore.scan(workspace: ws).isEmpty)
    }

    /// KW-1-2 (Q2, owner ruling 2026-09-16): a directory with no `.cat` but *something* else
    /// in it (an index, leftover user files) lists as a workspace with no flows, rather than
    /// vanishing with no UI that can reach it for Remove.
    @Test func folderWithNoFlowFileButOtherContentListsWithNoFlows() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        let bare = ws.root.appendingPathComponent("just-files", isDirectory: true)
        try FileManager.default.createDirectory(at: bare, withIntermediateDirectories: true)
        try Data([1, 2, 3]).write(to: bare.appendingPathComponent("notes.txt"))
        let all = WorkspaceStore.scan(workspace: ws)
        #expect(all.count == 1)
        let w = try #require(all.first)
        #expect(w.workspaceID == "just-files")
        #expect(w.flows.isEmpty)
        // Reachable and removable — the whole point of the ruling.
        try WorkspaceStore.remove(workspaceID: "just-files", workspace: ws)
        #expect(!FileManager.default.fileExists(atPath: bare.path))
    }

    /// The same state, but the leftover is a built index — the exact "stranded, potentially
    /// gigabytes" scenario the finding describes.
    @Test func folderWithOnlyAnIndexListsWithNoFlowsAndDeletionSummaryNamesTheIndex() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        let bare = ws.root.appendingPathComponent("orphaned", isDirectory: true)
        let idx = bare.appendingPathComponent("library.index", isDirectory: true)
        try FileManager.default.createDirectory(at: idx, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: idx.appendingPathComponent("manifest.json"))
        let w = try #require(WorkspaceStore.scan(workspace: ws).first)
        #expect(w.flows.isEmpty)
        let sentence = WorkspaceStore.deletionSummary(w)
        #expect(sentence.contains("1 index"))
        #expect(!sentence.contains("0 flow"))
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

    @Test func aBundledWorkspaceIDListsLikeAnyOther() throws {
        // CFM-R17-FIX-8: `scan` has no exclusion set — a bundled workspace's materialised copy
        // is a real, editable, Remove/Restore-able workspace (CFM-R17-FIX-1), so it appears in
        // the shelf next to the user's own. (Contrast `UserFlowStore.scan`'s `bundledFlowIDs`,
        // which is live: a gallery flow's working directory *is* hidden.)
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "mine", flows: [("A", validCat)])
        try makeWorkspace(root: ws.root, id: "ask_your_docs", flows: [("Ask", validCat)])
        #expect(Set(WorkspaceStore.scan(workspace: ws).map(\.workspaceID)) == ["mine", "ask_your_docs"])
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

    @Test func deletionSummaryCountsTheHiddenTrash() throws {
        // CFM-R17-FIX-8: `.trash` accumulates a full copy of the previous index on every
        // Rebuild and is invisible in Shared Files — the delete sentence must name its size.
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("Ask", validCat)])
        let trash = ws.root.appendingPathComponent("docs/.trash", isDirectory: true)
        try FileManager.default.createDirectory(at: trash, withIntermediateDirectories: true)
        try Data(repeating: 0, count: 40_000).write(to: trash.appendingPathComponent("library.index.1700000000000"))

        let w = try #require(WorkspaceStore.scan(workspace: ws).first)
        let sentence = WorkspaceStore.deletionSummary(w)
        #expect(sentence.contains(".trash"))
        #expect(sentence.contains("earlier versions"))
    }

    @Test func deletionSummarySkipsAnAbsentTrash() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("Ask", validCat)])
        let w = try #require(WorkspaceStore.scan(workspace: ws).first)
        #expect(!WorkspaceStore.deletionSummary(w).contains(".trash"))
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

    // MARK: - KW-2-1: firstFreeFlowName

    @Test func firstFreeFlowNamePicksFlowDotCatWhenNothingCollides() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("Ingest", validCat)])
        let dir = ws.directory(for: "docs")
        #expect(WorkspaceStore.firstFreeFlowName(stem: "Flow", in: dir) == "Flow.cat")
    }

    @Test func firstFreeFlowNameSkipsToFlow2WhenFlowDotCatExists() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("Flow", validCat)])
        let dir = ws.directory(for: "docs")
        #expect(WorkspaceStore.firstFreeFlowName(stem: "Flow", in: dir) == "Flow-2.cat")
    }

    @Test func firstFreeFlowNameKeepsIncrementingPastMultipleCollisions() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs",
                         flows: [("Flow", validCat), ("Flow-2", validCat), ("Flow-3", validCat)])
        let dir = ws.directory(for: "docs")
        #expect(WorkspaceStore.firstFreeFlowName(stem: "Flow", in: dir) == "Flow-4.cat")
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

    // MARK: - KW-2-2: removeFlow

    @Test func removeFlowDeletesOnlyThatFileSiblingsUntouched() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs",
                         flows: [("Ingest", validCat), ("Ask", validCat)])
        let dir = ws.directory(for: "docs")
        try WorkspaceStore.removeFlow(file: dir.appendingPathComponent("Ask.cat"), from: dir)
        #expect(!FileManager.default.fileExists(atPath: dir.appendingPathComponent("Ask.cat").path))
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("Ingest.cat").path))
        let w = try #require(WorkspaceStore.scan(workspace: ws).first)
        #expect(w.flows.map(\.title) == ["Ingest"])
    }

    @Test func removeFlowRefusesAPathOutsideTheGivenDirectory() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("Ingest", validCat)])
        try makeWorkspace(root: ws.root, id: "other", flows: [("Ask", validCat)])
        let docsDir = ws.directory(for: "docs")
        let otherFile = ws.directory(for: "other").appendingPathComponent("Ask.cat")
        #expect(throws: WorkspaceStoreError.flowNotFound("Ask.cat")) {
            try WorkspaceStore.removeFlow(file: otherFile, from: docsDir)
        }
        #expect(FileManager.default.fileExists(atPath: otherFile.path))
    }

    @Test func removeFlowRefusesAMissingFile() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("Ingest", validCat)])
        let dir = ws.directory(for: "docs")
        #expect(throws: WorkspaceStoreError.flowNotFound("Ghost.cat")) {
            try WorkspaceStore.removeFlow(file: dir.appendingPathComponent("Ghost.cat"), from: dir)
        }
    }

    // MARK: - KW-2-2: flowDeletionSummary (Q3, owner ruling 2026-09-16)

    @Test func flowDeletionSummaryNamesTheFileWhenOthersRemain() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs",
                         flows: [("Ingest", validCat), ("Ask", validCat)])
        let w = try #require(WorkspaceStore.scan(workspace: ws).first)
        let sentence = WorkspaceStore.flowDeletionSummary(fileName: "Ask.cat", isLastFlow: false, workspace: w)
        #expect(sentence.contains("Ask.cat"))
        #expect(!sentence.contains("no flows"))
    }

    @Test func flowDeletionSummaryWarnsWhatStaysBehindOnTheLastFlow() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("Ingest", validCat)])
        let dir = ws.directory(for: "docs")
        let idx = dir.appendingPathComponent("library.index", isDirectory: true)
        try FileManager.default.createDirectory(at: idx, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: idx.appendingPathComponent("manifest.json"))
        let w = try #require(WorkspaceStore.scan(workspace: ws).first)
        let sentence = WorkspaceStore.flowDeletionSummary(fileName: "Ingest.cat", isLastFlow: true, workspace: w)
        #expect(sentence.contains("Ingest.cat"))
        #expect(sentence.contains("no flows"))
        #expect(sentence.contains("its index"))
    }

    // MARK: - KW-2-2: renameFlow

    @Test func renameFlowRenamesInPlace() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("Ingest", validCat)])
        let dir = ws.directory(for: "docs")
        let newURL = try WorkspaceStore.renameFlow(file: dir.appendingPathComponent("Ingest.cat"),
                                                   toStem: "Build", in: dir)
        #expect(newURL.lastPathComponent == "Build.cat")
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("Build.cat").path))
        #expect(!FileManager.default.fileExists(atPath: dir.appendingPathComponent("Ingest.cat").path))
        #expect(try String(contentsOf: newURL, encoding: .utf8) == validCat)
    }

    @Test func renameFlowRefusesOntoADifferentSiblingsName() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs",
                         flows: [("Ingest", validCat), ("Ask", validCat)])
        let dir = ws.directory(for: "docs")
        #expect(throws: WorkspaceStoreError.flowNameTaken("Ask.cat")) {
            _ = try WorkspaceStore.renameFlow(file: dir.appendingPathComponent("Ingest.cat"),
                                              toStem: "Ask", in: dir)
        }
        // Neither file moved.
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("Ingest.cat").path))
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("Ask.cat").path))
    }

    @Test func renameFlowAllowsACaseOnlyRename() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("Ingest", validCat)])
        let dir = ws.directory(for: "docs")
        let newURL = try WorkspaceStore.renameFlow(file: dir.appendingPathComponent("Ingest.cat"),
                                                   toStem: "INGEST", in: dir)
        #expect(newURL.lastPathComponent == "INGEST.cat")
        let files = try FileManager.default.contentsOfDirectory(atPath: dir.path)
            .filter { $0.hasSuffix(".cat") }
        #expect(files == ["INGEST.cat"])
    }

    // MARK: - KW-2-FIX-3: renameFlow's containment and stem guards

    @Test func renameFlowRefusesASourceOutsideTheDirectory() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("Ingest", validCat)])
        try makeWorkspace(root: ws.root, id: "other", flows: [("Ask", validCat)])
        let docsDir = ws.directory(for: "docs")
        let otherFile = ws.directory(for: "other").appendingPathComponent("Ask.cat")
        #expect(throws: WorkspaceStoreError.flowNotFound("Ask.cat")) {
            _ = try WorkspaceStore.renameFlow(file: otherFile, toStem: "Stolen", in: docsDir)
        }
        #expect(FileManager.default.fileExists(atPath: otherFile.path))
    }

    @Test func renameFlowRefusesAnEmptyStem() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("Ingest", validCat)])
        let dir = ws.directory(for: "docs")
        #expect(throws: WorkspaceStoreError.invalidFlowName("")) {
            _ = try WorkspaceStore.renameFlow(file: dir.appendingPathComponent("Ingest.cat"), toStem: "", in: dir)
        }
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("Ingest.cat").path))
    }

    /// The exact `KW-1-2` orphan-state trap named in the finding: a leading-dot stem would
    /// write a hidden file that `scan`'s `.skipsHiddenFiles` drops from the shelf entirely.
    @Test func renameFlowRefusesALeadingDotStem() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("Ingest", validCat)])
        let dir = ws.directory(for: "docs")
        #expect(throws: WorkspaceStoreError.invalidFlowName(".old")) {
            _ = try WorkspaceStore.renameFlow(file: dir.appendingPathComponent("Ingest.cat"), toStem: ".old", in: dir)
        }
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("Ingest.cat").path))
    }

    @Test func renameFlowRefusesAStemContainingASlash() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        try makeWorkspace(root: ws.root, id: "docs", flows: [("Ingest", validCat)])
        let dir = ws.directory(for: "docs")
        #expect(throws: WorkspaceStoreError.invalidFlowName("../escape")) {
            _ = try WorkspaceStore.renameFlow(file: dir.appendingPathComponent("Ingest.cat"),
                                              toStem: "../escape", in: dir)
        }
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("Ingest.cat").path))
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
