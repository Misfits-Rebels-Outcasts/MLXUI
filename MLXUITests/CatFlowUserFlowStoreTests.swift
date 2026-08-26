import Testing
import Foundation
@testable import MLXUI

/// CFM-R12-1 — the "My Workflows" shelf store: enumerates the user's saved flows, never
/// shows a bundled flow's working directory, and lists a broken file with an error instead
/// of dropping it. All over a temp root.
struct CatFlowUserFlowStoreTests {

    private func makeRoot() throws -> (FlowWorkspace, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-shelf-\(UUID().uuidString)")
        let root = base.appendingPathComponent("flows")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return (FlowWorkspace(root: root), base)
    }

    private func makeFlow(root: URL, name: String, text: String, ext: String = "cat") throws -> String {
        let dir = root.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try text.write(to: dir.appendingPathComponent("\(name).\(ext)"), atomically: true, encoding: .utf8)
        return dir.lastPathComponent
    }

    private func teardown(_ base: URL) {
        try? FileManager.default.removeItem(at: base)
    }

    @Test func emptyRootYieldsNoFlows() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        #expect(UserFlowStore.scan(workspace: ws, bundledFlowIDs: []).isEmpty)
    }

    @Test func oneFlowListsWithTitleFromTheStem() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        let id = try makeFlow(root: ws.root, name: "My Notes", text: validCat)
        let entries = UserFlowStore.scan(workspace: ws, bundledFlowIDs: [])
        #expect(entries.count == 1)
        #expect(entries[0].flowID == id)
        #expect(entries[0].title == "My Notes")
        #expect(entries[0].parseIssue == nil)
    }

    @Test func catpipelineCountsToo() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        _ = try makeFlow(root: ws.root, name: "Look", text: validPipeline, ext: "catpipeline")
        let entries = UserFlowStore.scan(workspace: ws, bundledFlowIDs: [])
        #expect(entries.count == 1)
        #expect(entries[0].title == "Look")
    }

    @Test func bundledWorkingDirectoryNeverAppears() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        let id = try makeFlow(root: ws.root, name: "Mine", text: validCat)
        // A bundled flow's working dir — id matches a gallery id, and even holds a .cat.
        let bundled = ws.root.appendingPathComponent("01-SpokenSummary", isDirectory: true)
        try FileManager.default.createDirectory(at: bundled, withIntermediateDirectories: true)
        try validCat.write(to: bundled.appendingPathComponent("01-SpokenSummary.cat"),
                           atomically: true, encoding: .utf8)
        let entries = UserFlowStore.scan(workspace: ws, bundledFlowIDs: ["01-SpokenSummary"])
        #expect(entries.map(\.flowID) == [id])
    }

    @Test func unparseableFileListsWithAnError() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        let id = try makeFlow(root: ws.root, name: "Broken", text: "this is not a cat file at all \u{1}")
        let entries = UserFlowStore.scan(workspace: ws, bundledFlowIDs: [])
        #expect(entries.count == 1)
        #expect(entries[0].flowID == id)
        #expect(entries[0].parseIssue != nil)
    }

    @Test func loadDocumentParsesAValidFlow() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        _ = try makeFlow(root: ws.root, name: "Mine", text: validCat)
        let entry = try #require(UserFlowStore.scan(workspace: ws, bundledFlowIDs: []).first)
        let doc = try UserFlowStore.loadDocument(entry: entry)
        #expect(doc.rows.count == 2)
        #expect(doc.rows.first?.task == "Read Text")
    }

    @Test func newestFirstSorting() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        let older = ws.root.appendingPathComponent("old", isDirectory: true)
        try FileManager.default.createDirectory(at: older, withIntermediateDirectories: true)
        let olderFile = older.appendingPathComponent("Older.cat")
        try validCat.write(to: olderFile, atomically: true, encoding: .utf8)
        // Touch the folder to a clearly older date.
        try? FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSinceNow: -3600)],
                                               ofItemAtPath: older.path)

        let newer = ws.root.appendingPathComponent("new", isDirectory: true)
        try FileManager.default.createDirectory(at: newer, withIntermediateDirectories: true)
        try validCat.write(to: newer.appendingPathComponent("Newer.cat"), atomically: true, encoding: .utf8)

        let entries = UserFlowStore.scan(workspace: ws, bundledFlowIDs: [])
        #expect(entries.map(\.flowID) == ["new", "old"])
    }

    // MARK: - CFM: Import / Export

    @Test func importCopiesTheCatAndFixtures() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        // A picked folder: one .cat plus fixtures (audio, text, a nested subfolder).
        let source = base.appendingPathComponent("My Flow", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try validCat.write(to: source.appendingPathComponent("My Flow.cat"),
                           atomically: true, encoding: .utf8)
        try Data([0, 1, 2]).write(to: source.appendingPathComponent("clip.wav"))
        try "draft".write(to: source.appendingPathComponent("draft.txt"),
                          atomically: true, encoding: .utf8)
        let nested = source.appendingPathComponent("vacation", isDirectory: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try Data([9, 9]).write(to: nested.appendingPathComponent("photo-01.png"))

        let entry = try UserFlowStore.importFlow(from: source, workspace: ws)
        #expect(entry.title == "My Flow")
        #expect(entry.parseIssue == nil)
        let dir = ws.directory(for: entry.flowID)
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("My Flow.cat").path))
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("clip.wav").path))
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("draft.txt").path))
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("vacation/photo-01.png").path))
        // The shelf picks the imported flow up by scanning.
        let entries = UserFlowStore.scan(workspace: ws, bundledFlowIDs: [])
        #expect(entries.count == 1)
        #expect(entries[0].title == "My Flow")
    }

    @Test func importRefusesAFolderWithoutExactlyOneCat() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        let empty = base.appendingPathComponent("empty", isDirectory: true)
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        #expect(throws: UserFlowStoreError.importNeedsOneFlowFile(count: 0)) {
            try UserFlowStore.importFlow(from: empty, workspace: ws)
        }
        let two = base.appendingPathComponent("two", isDirectory: true)
        try FileManager.default.createDirectory(at: two, withIntermediateDirectories: true)
        try validCat.write(to: two.appendingPathComponent("A.cat"), atomically: true, encoding: .utf8)
        try validCat.write(to: two.appendingPathComponent("B.cat"), atomically: true, encoding: .utf8)
        #expect(throws: UserFlowStoreError.importNeedsOneFlowFile(count: 2)) {
            try UserFlowStore.importFlow(from: two, workspace: ws)
        }
    }

    @Test func exportCopiesTheWholeFolderAndRoundTrips() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        let id = try makeFlow(root: ws.root, name: "Voiceover", text: validCat)
        let flowDir = ws.directory(for: id)
        try Data([7, 7, 7]).write(to: flowDir.appendingPathComponent("voice.wav"))

        let destination = base.appendingPathComponent("exports", isDirectory: true)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let exported = try UserFlowStore.export(flowID: id, name: "Voiceover",
                                                to: destination, workspace: ws)
        #expect(exported.lastPathComponent == "Voiceover")
        #expect(FileManager.default.fileExists(atPath: exported.appendingPathComponent("Voiceover.cat").path))
        #expect(FileManager.default.fileExists(atPath: exported.appendingPathComponent("voice.wav").path))
        // The exported folder is importable verbatim.
        let entry = try UserFlowStore.importFlow(from: exported, workspace: ws)
        #expect(entry.title == "Voiceover")
        #expect(entry.parseIssue == nil)
        #expect(UserFlowStore.scan(workspace: ws, bundledFlowIDs: []).count == 2)
    }

    @Test func exportRefusesAnExistingDestination() throws {
        let (ws, base) = try makeRoot()
        defer { teardown(base) }
        let id = try makeFlow(root: ws.root, name: "Mine", text: validCat)
        let destination = base.appendingPathComponent("exports", isDirectory: true)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: destination.appendingPathComponent("Mine", isDirectory: true),
                                                withIntermediateDirectories: true)
        #expect(throws: UserFlowStoreError.exportDestinationExists("Mine")) {
            try UserFlowStore.export(flowID: id, name: "Mine", to: destination, workspace: ws)
        }
    }

    private let validCat = """
    catflow 0.8
    1. Read Text   memo.txt
    2. Save Text   out.md
    """

    private let validPipeline = """
    catpipeline 0.8
    1. Generate Image   Z-Image Turbo; "a tree"
    2. Save Image       tree.png
    """
}
