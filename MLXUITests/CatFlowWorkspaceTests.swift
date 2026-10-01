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
        // The full gallery ships: 77 entries (74 .cat + 3 .catpipeline).
        #expect(metadata.count == 77)
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

    // MARK: - LY-9-FIX-1: flow 77's assets actually materialise (the smoke-97 regression)

    /// A bundled asset's flat source name, resolved the way `Bundle.main.url(forResource:
    /// withExtension:)` — and therefore the synchronized root group's flattened
    /// `Contents/Resources/` — actually looks it up: name and extension split apart.
    private func resolvesInBundle(_ flatName: String) -> Bool {
        let name = (flatName as NSString).deletingPathExtension
        let ext = (flatName as NSString).pathExtension
        return Bundle.main.url(forResource: name, withExtension: ext) != nil
    }

    @Test func ticketRouterBundledAssetsResolveInTheBundle() throws {
        let assets = GalleryLoader.bundledAssets(flowID: "77-TicketRouter")
        #expect(assets.count == 5)
        #expect(assets.map(\.destination) == [
            "tickets/ticket-01.txt", "tickets/ticket-02.txt", "tickets/ticket-03.txt",
            "tickets/ticket-04.txt", "tickets/ticket-05.txt",
        ])
        for asset in assets {
            #expect(resolvesInBundle(asset.source), "\(asset.source) doesn't resolve in the app bundle")
        }
    }

    /// The general guard LY-9-FIX-1 asked for: every gallery flow's declared bundled assets
    /// actually resolve in the shipped bundle, and no two flows' flat source names collide
    /// **destructively** — exactly the two ways `77-TicketRouter`'s original
    /// `tickets/ticket-0N.txt` shape (no `bundledAssets` case at all, so nothing was ever
    /// copied) could have been caught, plus the collision class `GalleryLoader`'s own doc
    /// comment warns about.
    ///
    /// "Collide" here means the **same flat name resolves to different destinations** across
    /// flows — that's two flows disagreeing about what a shared physical file is *for*, a real
    /// bug class. It deliberately does **not** flag the seven flows that share
    /// `library-chunks.jsonl`/`library-manifest.json`/`library-vectors.bin`, all mapping to
    /// the identical `library.index/...` destination on purpose (`GalleryLoader`'s own doc
    /// comment: "seven share the prebuilt `library.index`") — a first cut of this test flagged
    /// those 18 pairs as "collisions," which they are not; reusing the exact same file for the
    /// exact same purpose across flows is the documented, intended shape, not a bug to report.
    /// Reports every real problem found, rather than stopping at the first — LY-9-FIX-1 said
    /// report, don't fix, if this surfaces anything beyond flow 77.
    @Test func everyGalleryFlowsBundledAssetsResolveAndDontCollide() throws {
        var destinationForSource: [String: String] = [:]   // flat source -> its first-seen destination
        var unresolved: [String] = []
        var collisions: [String] = []

        for meta in GalleryLoader.loadMetadata() {
            for asset in GalleryLoader.bundledAssets(flowID: meta.flowID) {
                if !resolvesInBundle(asset.source) {
                    unresolved.append("\(meta.flowID): \(asset.source)")
                }
                if let existingDestination = destinationForSource[asset.source] {
                    if existingDestination != asset.destination {
                        collisions.append(
                            "\(asset.source): \(meta.flowID) wants '\(asset.destination)', "
                                + "another flow already wants '\(existingDestination)'")
                    }
                } else {
                    destinationForSource[asset.source] = asset.destination
                }
            }
        }
        #expect(unresolved.isEmpty, "bundled assets missing from the bundle: \(unresolved.joined(separator: "; "))")
        #expect(collisions.isEmpty, "flat source names that disagree on their destination: \(collisions.joined(separator: "; "))")
    }

    /// The actual regression from smoke row 97: drives the **same** `FlowWorkspace.prepare`
    /// call `FlowEditRoute.duplicateAndEdit`/`editOpenedCopy` and the plain "Open" path all
    /// go through, with the **real** bundle as `sourceDir` (not a fake one, unlike the
    /// `prepare*` tests above) — so a missing `bundledAssets` case or a source name that
    /// doesn't survive the synchronized group's flattening shows up here exactly as it would
    /// for a real Run.
    @Test func ticketRouterMaterialisesItsTicketFilesOnOpen() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-ly9-fix1-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }

        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        try ws.prepare(flowID: "77-TicketRouter", sourceDir: GalleryLoader.resourcesDirectory,
                       bundledAssets: GalleryLoader.bundledAssets(flowID: "77-TicketRouter"))

        let flowDir = ws.directory(for: "77-TicketRouter")
        for n in 1 ... 5 {
            let ticket = flowDir.appendingPathComponent("tickets/ticket-0\(n).txt")
            #expect(FileManager.default.fileExists(atPath: ticket.path), "\(ticket.lastPathComponent) was never materialised")
        }
    }
}
