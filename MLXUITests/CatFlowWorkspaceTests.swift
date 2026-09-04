import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R1-4: `FlowWorkspace` (working directory, first-run asset copy, path
/// security boundary, Reveal in Finder) and `GalleryLoader`. The workspace is constructed
/// with an injected temp base directory — the real Application Support directory is never
/// touched (`RSI/policies.md` puts user data off-limits). See
/// `RSI/DelegateMergeBacklog.md` CFM-R1-4.
struct CatFlowWorkspaceTests {

    private func makeWorkspace() throws -> (FlowWorkspace, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-workspace-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return (FlowWorkspace(root: base.appendingPathComponent("flows")), base)
    }

    private func teardown(_ base: URL) {
        try? FileManager.default.removeItem(at: base)
    }

    // MARK: - prepare

    @Test func prepareCreatesDirectoryAndCopiesInputs() throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        // A fake "bundle" source dir with the flow's assets.
        let src = base.appendingPathComponent("bundle")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try Data("memo".utf8).write(to: src.appendingPathComponent("memo.m4a"))

        try ws.prepare(flowID: "01-SpokenSummary", sourceDir: src,
                       bundledAssets: [("memo.m4a", "memo.m4a")])
        let flowDir = ws.directory(for: "01-SpokenSummary")
        #expect(FileManager.default.fileExists(atPath: flowDir.path))
        #expect(FileManager.default.fileExists(atPath: flowDir.appendingPathComponent("memo.m4a").path))
    }

    @Test func prepareIsIdempotentAndKeepsModifiedFiles() throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let src = base.appendingPathComponent("bundle")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try Data("original".utf8).write(to: src.appendingPathComponent("memo.m4a"))

        try ws.prepare(flowID: "01-SpokenSummary", sourceDir: src,
                       bundledAssets: [("memo.m4a", "memo.m4a")])
        let dest = ws.directory(for: "01-SpokenSummary").appendingPathComponent("memo.m4a")

        // User edits the copied file.
        try Data("edited-by-user".utf8).write(to: dest)

        // A second prepare must NOT overwrite the modified file.
        try Data("original".utf8).write(to: src.appendingPathComponent("memo.m4a"))
        try ws.prepare(flowID: "01-SpokenSummary", sourceDir: src,
                       bundledAssets: [("memo.m4a", "memo.m4a")])
        let contents = try String(contentsOf: dest, encoding: .utf8)
        #expect(contents == "edited-by-user")
    }

    @Test func prepareCopiesNestedDestinations() throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let src = base.appendingPathComponent("bundle")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try Data("photo".utf8).write(to: src.appendingPathComponent("photo-01.png"))

        try ws.prepare(flowID: "21-PhotoWebPrep", sourceDir: src,
                       bundledAssets: [("photo-01.png", "vacation/photo-01.png")])
        let dest = ws.directory(for: "21-PhotoWebPrep").appendingPathComponent("vacation/photo-01.png")
        #expect(FileManager.default.fileExists(atPath: dest.path))
    }

    // MARK: - resolve (the security boundary)

    @Test func resolveLandsInsideFlowDirectory() throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let url = try ws.resolve("memo.m4a", flowID: "01-SpokenSummary")
        #expect(url.path.hasPrefix(ws.directory(for: "01-SpokenSummary").path + "/"))
    }

    @Test func resolveRejectsDotDotTraversal() throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        #expect(throws: FlowWorkspaceError.self) {
            _ = try ws.resolve("../../../etc/passwd", flowID: "01-SpokenSummary")
        }
    }

    @Test func resolveRejectsAbsolutePath() throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        #expect(throws: FlowWorkspaceError.self) {
            _ = try ws.resolve("/etc/passwd", flowID: "01-SpokenSummary")
        }
    }

    @Test func resolveRejectsSymlinkEscape() throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        // A symlink inside the flow dir pointing at a directory outside it.
        let flowDir = ws.directory(for: "01-SpokenSummary")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let outside = base.appendingPathComponent("outside")
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: flowDir.appendingPathComponent("evil"),
                                                   withDestinationURL: outside)

        #expect(throws: FlowWorkspaceError.self) {
            _ = try ws.resolve("evil/secret.txt", flowID: "01-SpokenSummary")
        }
    }

    @Test func resolveRejectsDanglingSymlink() throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        // A symlink whose target does not yet exist. `fileExists` *traverses* links and
        // would report false for it, letting a later write escape the workspace through it
        // (H6) — the `.isSymbolicLinkKey` probe must catch it.
        let flowDir = ws.directory(for: "01-SpokenSummary")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let notYet = base.appendingPathComponent("not-yet-created")
        try FileManager.default.createSymbolicLink(at: flowDir.appendingPathComponent("bed.wav"),
                                                   withDestinationURL: notYet)

        #expect(throws: FlowWorkspaceError.self) {
            _ = try ws.resolve("bed.wav", flowID: "01-SpokenSummary")
        }
    }

    // MARK: - GalleryLoader (uses the real bundle, which ships the three flows)

    @Test func galleryLoaderFindsAllFlows() throws {
        let metadata = GalleryLoader.loadMetadata()
        // The full gallery ships: 71 entries (68 .cat + 3 .catpipeline).
        #expect(metadata.count == 71)
        let spoken = try #require(metadata.first { $0.flowID == "01-SpokenSummary" })
        #expect(spoken.title == "Spoken Summary")
        #expect(spoken.number == 1)
    }

    @Test func galleryLoaderLoadsDocuments() throws {
        let doc = try GalleryLoader.loadDocument(flowID: "01-SpokenSummary")
        #expect(doc.rows.count == 6)
        let raw = try GalleryLoader.rawCatText(flowID: "01-SpokenSummary")
        #expect(raw.contains("1. Read Audio         memo.m4a"))
    }
}
