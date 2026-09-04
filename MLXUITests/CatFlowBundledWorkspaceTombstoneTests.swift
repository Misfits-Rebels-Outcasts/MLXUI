import Testing
import Foundation
@testable import MLXUI

/// CFM-R17-FIX-1 — removing a *bundled* workspace used to delete its directory and then
/// `reloadWorkspaces()` re-materialised it one screen later, silently destroying the index
/// the user built and any documents they added while the dialog claimed the delete "can't be
/// undone". The fix: a persisted per-id tombstone so a removed bundled workspace stays
/// removed, a "Restore bundled workspaces" affordance that clears it, and dialog wording that
/// matches. A *user* workspace is unaffected — it has no tombstone and Remove is final.
struct CatFlowBundledWorkspaceTombstoneTests {

    private func makeRoot() throws -> (FlowWorkspace, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-ws-tomb-\(UUID().uuidString)")
        let root = base.appendingPathComponent("workspaces")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return (FlowWorkspace(root: root), base)
    }

    // MARK: - prepareAll skips tombstoned ids

    @Test func removedBundledWorkspaceStaysRemovedAcrossTwoReloads() throws {
        let (ws, base) = try makeRoot()
        defer { try? FileManager.default.removeItem(at: base) }
        let fm = FileManager.default

        // First materialisation brings both bundled workspaces in.
        BundledWorkspaces.prepareAll(into: ws)
        #expect(fm.fileExists(atPath: ws.directory(for: "ask_your_docs").path))
        #expect(fm.fileExists(atPath: ws.directory(for: "uses_example").path))

        // The user removes one, and it is tombstoned.
        try WorkspaceStore.remove(workspaceID: "ask_your_docs", workspace: ws)
        let removed: Set<String> = ["ask_your_docs"]

        // Two more reloads (the shelf reappears, the workspace page appears) — it stays gone.
        BundledWorkspaces.prepareAll(into: ws, removed: removed)
        BundledWorkspaces.prepareAll(into: ws, removed: removed)
        #expect(!fm.fileExists(atPath: ws.directory(for: "ask_your_docs").path))
        // The other bundled workspace is untouched.
        #expect(fm.fileExists(atPath: ws.directory(for: "uses_example").path))

        // Clearing the tombstone (Restore) brings it back.
        BundledWorkspaces.prepareAll(into: ws, removed: [])
        #expect(fm.fileExists(atPath: ws.directory(for: "ask_your_docs").path))
    }

    // MARK: - tombstone persistence

    @Test func tombstonePersistenceRoundTripsAndDropsStaleIDs() throws {
        let suite = "tombstone-test-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }

        BundledWorkspaceTombstones.save(["ask_your_docs"], to: defaults)
        #expect(BundledWorkspaceTombstones.load(from: defaults) == ["ask_your_docs"])

        // An id that no longer ships is discarded, not carried forever.
        BundledWorkspaceTombstones.save(["ask_your_docs", "workspace-that-was-deleted"], to: defaults)
        #expect(BundledWorkspaceTombstones.load(from: defaults) == ["ask_your_docs"])

        // Clearing removes the key entirely.
        BundledWorkspaceTombstones.save([], to: defaults)
        #expect(BundledWorkspaceTombstones.load(from: defaults).isEmpty)
        #expect(defaults.array(forKey: BundledWorkspaceTombstones.defaultsKey) == nil)
    }

    @Test func isBundledOnlyRecognisesShippedIDs() {
        #expect(BundledWorkspaces.isBundled("ask_your_docs"))
        #expect(BundledWorkspaces.isBundled("uses_example"))
        #expect(!BundledWorkspaces.isBundled("Workspace-1a2b3c4d"))
        #expect(!BundledWorkspaces.isBundled("my-notes"))
    }

    // MARK: - dialog wording

    /// The bundled-removal path genuinely destroys a built index — so its dialog must say so,
    /// and must not claim the false "can't be undone".
    @Test func bundledRemovalDialogNamesTheLostIndexAndOffersRestore() throws {
        let (ws, base) = try makeRoot()
        defer { try? FileManager.default.removeItem(at: base) }
        BundledWorkspaces.prepareAll(into: ws)

        // Simulate the user having built the index.
        let idx = ws.directory(for: "ask_your_docs").appendingPathComponent("library.index", isDirectory: true)
        try FileManager.default.createDirectory(at: idx, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: idx.appendingPathComponent("manifest.json"))

        let w = try #require(WorkspaceStore.scan(workspace: ws).first { $0.workspaceID == "ask_your_docs" })
        let sentence = WorkspaceStore.deletionSummary(w, bundled: true)

        #expect(sentence.contains("1 index"))
        #expect(sentence.contains("index you built"))
        #expect(sentence.contains("Restore bundled workspaces"))
        #expect(!sentence.contains("can't be undone"))
    }

    @Test func userWorkspaceRemovalWordingIsUnchanged() throws {
        let (ws, base) = try makeRoot()
        defer { try? FileManager.default.removeItem(at: base) }
        let dir = ws.root.appendingPathComponent("my-notes", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try "catflow 0.8\n1. Read Text memo.txt\n2. Save Text out.md\n"
            .write(to: dir.appendingPathComponent("Notes.cat"), atomically: true, encoding: .utf8)

        let w = try #require(WorkspaceStore.scan(workspace: ws).first)
        let sentence = WorkspaceStore.deletionSummary(w)   // bundled: false (default)

        #expect(sentence.contains("1 flow"))
        #expect(sentence.contains("This can't be undone."))
        #expect(!sentence.contains("Restore bundled workspaces"))
    }
}
