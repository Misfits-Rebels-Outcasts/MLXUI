import Foundation
import Testing
@testable import MLXUI

/// CW-4 — `FolderListing`: one level of a folder for the inspector's disclosure tree, plus the
/// grant-aware wrapper the view calls. No model, no runner: a temp directory and the injected seam
/// stand in for a granted folder.
struct FolderListingTests {

    private func makeTree() throws -> URL {
        let fm = FileManager.default
        let base = fm.temporaryDirectory
            .appendingPathComponent("folder-listing-\(UUID().uuidString)", isDirectory: true)
        for name in ["beta", "Alpha", "docs"] {
            try fm.createDirectory(
                at: base.appendingPathComponent(name), withIntermediateDirectories: true)
        }
        for name in ["note.txt", ".hidden", "Zed.md", "alpha.md"] {
            try "x".write(
                to: base.appendingPathComponent(name), atomically: true, encoding: .utf8)
        }
        return base
    }

    // MARK: - one level

    /// Folders first, then files, each case-insensitively alphabetical (localizedStandardCompare).
    @Test func childrenSortFoldersFirstThenNames() throws {
        let base = try makeTree()
        defer { try? FileManager.default.removeItem(at: base) }

        let result = FolderListing.children(of: base)
        #expect(result.error == nil)
        #expect(result.entries.map(\.name) == ["Alpha", "beta", "docs", "alpha.md", "note.txt", "Zed.md"])
        #expect(result.entries.filter(\.isDirectory).map(\.name) == ["Alpha", "beta", "docs"])
        #expect(result.total == 6)
        #expect(result.hidden == 0)
    }

    @Test func hiddenFilesAreSkippedByDefaultAndShownOnRequest() throws {
        let base = try makeTree()
        defer { try? FileManager.default.removeItem(at: base) }

        #expect(!FolderListing.children(of: base).entries.contains { $0.name == ".hidden" })
        #expect(FolderListing.children(of: base, includeHidden: true)
            .entries.contains { $0.name == ".hidden" })
    }

    @Test func theCapTrimsAndReportsHowManyAreHidden() throws {
        let fm = FileManager.default
        let base = fm.temporaryDirectory
            .appendingPathComponent("folder-cap-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        for index in 0..<12 {
            try "x".write(
                to: base.appendingPathComponent(String(format: "f%02d.txt", index)),
                atomically: true, encoding: .utf8)
        }

        let result = FolderListing.children(of: base, limit: 5)
        #expect(result.entries.count == 5)
        #expect(result.total == 12)
        #expect(result.hidden == 7)
        #expect(result.entries.map(\.name) == ["f00.txt", "f01.txt", "f02.txt", "f03.txt", "f04.txt"])
    }

    /// A symlink is listed but never expandable — that is what makes a link loop unwalkable, since
    /// the tree only ever descends into `isExpandable` rows.
    @Test func aSymlinkIsListedButNotExpandable() throws {
        let base = try makeTree()
        defer { try? FileManager.default.removeItem(at: base) }
        let link = base.appendingPathComponent("loop")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: base)

        let result = FolderListing.children(of: base)
        let row = try #require(result.entries.first { $0.name == "loop" })
        #expect(row.isDirectory == false)
        #expect(row.isExpandable == false)
    }

    @Test func anUnreadableDirectoryReportsAnErrorRatherThanAnEmptyList() {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("nope-\(UUID().uuidString)", isDirectory: true)
        let result = FolderListing.children(of: missing)
        #expect(result.entries.isEmpty)
        #expect(result.error != nil)
    }

    // MARK: - the wrapper the view calls

    /// `listing(for:)` is what the tree actually calls; a test injects a closure so no grant or
    /// real bookmark is involved (the same seam idea as the scoped file tools).
    @Test func listingUsesTheInjectedReaderWhenGiven() throws {
        let base = try makeTree()
        defer { try? FileManager.default.removeItem(at: base) }
        var asked: [String] = []
        let result = FolderListing.listing(for: base) { directory in
            asked.append(directory.path)
            return FolderListing.children(of: directory)
        }
        #expect(asked == [base.path])
        #expect(result.total == 6)
    }

    @Test func listingReturnsTheInjectedRefusalUnchanged() {
        let refused = FolderListing.Result(
            entries: [], total: 0, error: "This folder is not attached to this chat.")
        let result = FolderListing.listing(
            for: URL(fileURLWithPath: "/tmp/whatever")) { _ in refused }
        #expect(result == refused)
    }

    /// Without an injection the real wrapper runs. A temp directory is not the app container and
    /// is not granted, so the wrapper must refuse it — the tree never shows a folder the tools
    /// could not read.
    @Test func listingRefusesAFolderOutsideTheContainerAndGrants() throws {
        let base = try makeTree()
        defer { try? FileManager.default.removeItem(at: base) }
        let result = FolderListing.listing(for: base)
        #expect(result.error != nil)
        #expect(result.entries.isEmpty)
    }
}
