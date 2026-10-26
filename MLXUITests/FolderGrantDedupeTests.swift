import Foundation
import Testing
@testable import MLXUI

/// CW-FIX-3 — one button per real folder. Temp dirs, a symlink, an injected defaults suite.
@MainActor
struct FolderGrantDedupeTests {

    /// <root>/a/workfolder, <root>/b/workfolder and <root>/link -> <root>/a.
    private func makeTree() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("grant-dedupe-\(UUID().uuidString)", isDirectory: true)
        for name in ["a", "b"] {
            try FileManager.default.createDirectory(
                at: root.appendingPathComponent("\(name)/workfolder"), withIntermediateDirectories: true)
        }
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("link"),
            withDestinationURL: root.appendingPathComponent("a"))
        return root
    }

    private func makeDefaults() -> (UserDefaults, () -> Void) {
        let suite = "FolderGrantDedupeTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        return (defaults, { defaults.removePersistentDomain(forName: suite) })
    }

    @Test func grantingTheSameFolderUnderASymlinkSpellingKeepsOneGrant() throws {
        let root = try makeTree()
        let (defaults, cleanup) = makeDefaults()
        defer { cleanup(); try? FileManager.default.removeItem(at: root) }

        try FolderGrants.addGrant(for: root.appendingPathComponent("a/workfolder"), to: defaults)
        try FolderGrants.addGrant(for: root.appendingPathComponent("link/workfolder"), to: defaults)
        let grants = FolderGrants.load(from: defaults)
        #expect(grants.count == 1)
        // Newest wins, and the path is stored as given.
        #expect(grants.first?.path == root.appendingPathComponent("link/workfolder").path)
    }

    @Test func loadCollapsesStoredDuplicatesKeepingTheLastAndPersistsTheCleanList() throws {
        let root = try makeTree()
        let (defaults, cleanup) = makeDefaults()
        defer { cleanup(); try? FileManager.default.removeItem(at: root) }

        let real = root.appendingPathComponent("a/workfolder").path
        let other = root.appendingPathComponent("b/workfolder").path
        let stored = [
            FolderGrant(path: real, bookmark: Data([1])),
            FolderGrant(path: other, bookmark: Data([9])),
            FolderGrant(path: real + "/", bookmark: Data([2])),
            FolderGrant(path: root.appendingPathComponent("link/workfolder").path, bookmark: Data([3])),
        ]
        FolderGrants.save(stored, to: defaults)

        let loaded = FolderGrants.load(from: defaults)
        #expect(loaded.map(\.bookmark) == [Data([9]), Data([3])])
        // The cleaned list was written back: the raw stored value now holds two entries.
        let raw = try JSONDecoder().decode(
            [FolderGrant].self, from: defaults.data(forKey: "agentFolderGrants") ?? Data())
        #expect(raw.count == 2)
    }

    @Test func twoRealFoldersWithOneNameStayTwoCandidatesWithParentLabels() throws {
        let root = try makeTree()
        defer { try? FileManager.default.removeItem(at: root) }
        let a = root.appendingPathComponent("a/workfolder").path
        let b = root.appendingPathComponent("b/workfolder").path

        let candidates = RunChatView.grantCandidates(attached: [], granted: [a, b])
        #expect(candidates == [a, b])
        #expect(RunChatView.folderLabel(a, among: candidates) == "workfolder — a")
        #expect(RunChatView.folderLabel(b, among: candidates) == "workfolder — b")
    }

    @Test func uniqueNamesKeepTheBareLabelAndDeepClashesAddMoreParents() {
        let paths = ["/x/p/docs", "/x/q/docs", "/y/p/docs", "/x/notes"]
        let labels = FolderLabels.labels(for: paths)
        #expect(labels["/x/notes"] == "notes")
        #expect(Set(labels.values).count == 4)
        #expect(labels["/x/p/docs"] == "docs — x/p")
    }

    @Test func oneFolderUnderTwoSpellingsIsOneCandidate() throws {
        let root = try makeTree()
        defer { try? FileManager.default.removeItem(at: root) }
        let real = root.appendingPathComponent("a/workfolder").path
        let linked = root.appendingPathComponent("link/workfolder").path
        #expect(RunChatView.grantCandidates(attached: [], granted: [real, linked]) == [real])
        #expect(RunChatView.grantCandidates(attached: [linked], granted: [real]) == [linked])
    }

    @Test func attachingAnAlreadyAttachedFolderByItsOtherSpellingKeepsOneChip() throws {
        let root = try makeTree()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("grant-dedupe-store-\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: directory)
        }
        let real = root.appendingPathComponent("a/workfolder").path
        let linked = root.appendingPathComponent("link/workfolder").path

        let runner = ModelRunner()
        runner.prepare(for: makeEntry(id: "model-a", displayName: "Test Model"), in: directory)
        runner.attachFolder(path: real)
        runner.attachFolder(path: linked)
        #expect(runner.activeAttachedFolders == [real])
        runner.detachFolder(path: linked)
        #expect(runner.activeAttachedFolders.isEmpty)
        #expect(runner.activeWorkingFolder == nil)
    }
}
