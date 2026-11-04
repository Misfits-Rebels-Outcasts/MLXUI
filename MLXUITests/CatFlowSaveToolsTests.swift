import Testing
import Foundation
import CoreGraphics
@testable import MLXUI

/// CFM-R12-5 — `Save Image` / `Save Images` / `Save Video`: file-backed results land in the
/// flow folder at the resolved path, atomically, with the Python's status sentences.
struct CatFlowSaveToolsTests {

    private func makeWorkspace() throws -> (FlowWorkspace, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-save-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return (FlowWorkspace(root: base.appendingPathComponent("flows")), base)
    }

    private func makePNG() throws -> Data {
        let ctx = CGContext(data: nil, width: 2, height: 2, bitsPerComponent: 8, bytesPerRow: 8,
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        ctx?.setFillColor(CGColor(srgbRed: 1, green: 0, blue: 0, alpha: 1))
        ctx?.fill(CGRect(x: 0, y: 0, width: 2, height: 2))
        let image = try #require(ctx?.makeImage())
        return try #require(PNGEncoder.pngData(from: image))
    }

    private func teardown(_ base: URL) {
        try? FileManager.default.removeItem(at: base)
    }

    @Test func saveImageCopiesTheFileAndSaysSavedTo() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let src = flowDir.appendingPathComponent("blob.png")
        let data = try makePNG()
        try data.write(to: src)

        let tool = SaveImageTool(workspace: ws, flowID: "f", settings: "tree.png")
        let out = try await tool.run(Asset(items: [Item(kind: .image, value: nil, path: src, sourceText: nil)])) { _ in }
        let status = try #require(out.items.first)
        #expect(status.value == "saved to tree.png")
        let dest = flowDir.appendingPathComponent("tree.png")
        #expect(FileManager.default.fileExists(atPath: dest.path))
        #expect(try Data(contentsOf: dest) == data)
    }

    @Test func saveImageRefusesWithoutAnInputPath() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let tool = SaveImageTool(workspace: ws, flowID: "f", settings: "tree.png")
        await #expect(throws: FlowError.self) {
            _ = try await tool.run(Asset(items: [])) { _ in }
        }
    }

    @Test func saveImagesNamesByIndexIntoTheFolder() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let a = flowDir.appendingPathComponent("a.png")
        let b = flowDir.appendingPathComponent("b.png")
        try makePNG().write(to: a)
        try makePNG().write(to: b)

        let tool = SaveImagesTool(workspace: ws, flowID: "f",
                                  settings: "out/; naming=shot-{n}{ext}")
        let input = Asset(items: [Item(kind: .image, value: nil, path: a, sourceText: nil),
                                  Item(kind: .image, value: nil, path: b, sourceText: nil)])
        let out = try await tool.run(input) { _ in }
        let status = try #require(out.items.first)
        #expect(status.value == "saved 2 image(s) to out")
        let folder = flowDir.appendingPathComponent("out")
        #expect(FileManager.default.fileExists(atPath: folder.appendingPathComponent("shot-1.png").path))
        #expect(FileManager.default.fileExists(atPath: folder.appendingPathComponent("shot-2.png").path))
    }

    /// Gallery 21 (Photo Web Prep) names its output `{name}-web{ext}`. The reference's gallery file says `{name}-web`, which drops
    /// every photo's extension; MLXUI fixed it in `a274785`. This runs the flow's own Save Images row settings on `photo.jpg`.
    // SPEC-Q245
    @Test func flow21SaveImagesKeepsTheExtension() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let doc = try GalleryLoader.loadDocument(flowID: "21-PhotoWebPrep")
        let row = try #require(doc.rows.last)
        #expect(row.task == "Save Images")
        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let photo = flowDir.appendingPathComponent("photo.jpg")
        try makePNG().write(to: photo)

        let tool = SaveImagesTool(workspace: ws, flowID: "f", settings: row.settings ?? "")
        _ = try await tool.run(Asset(items: [Item(kind: .image, value: nil, path: photo, sourceText: nil)])) { _ in }
        let folder = flowDir.appendingPathComponent("vacation-web")
        let written = (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []
        #expect(written == ["photo-web.jpg"], "wrote \(written)")
    }

    @Test func saveVideoCopiesTheFile() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let src = flowDir.appendingPathComponent("clip.bin")
        try Data([0x00, 0x01, 0x02]).write(to: src)

        let tool = SaveVideoTool(workspace: ws, flowID: "f", settings: "saved/clip.mp4")
        let out = try await tool.run(Asset(items: [Item(kind: .video, value: nil, path: src, sourceText: nil)])) { _ in }
        let status = try #require(out.items.first)
        #expect(status.value == "saved to saved/clip.mp4")
        #expect(FileManager.default.fileExists(atPath: flowDir.appendingPathComponent("saved/clip.mp4").path))
    }

    /// CFM-R12-FIX-1/12 honesty check: Save Image made the *tools* runnable, but of flows
    /// 60–66 only **8-VoiceoverBed**, **7-UpscaleSmall** (SeedVR2 3B now in catalog,
    /// SV-AM1) and — since CFM-R15-1 — **6-CutOutSubject** (SAM Base bridges onto
    /// `mlx-community/sam3-4bit`) have a model the bridge can resolve. The rest refuse at the
    /// live gate (Generate Image / Edit Image / Segment / the latent rows have no bridge model).
    @Test func flowsUnblockedBySaveImageAreRunnableNow() throws {
        let catalog = try makeCatalog()
        func runs(_ fid: String) -> String? {
            guard let doc = try? GalleryLoader.loadDocument(flowID: fid) else { return "no doc" }
            return FlowRunnability.refusalReason(for: doc, catalog: catalog,
                                                 installed: [], totalRAMGB: 32, claimableModelIDs: [])
        }
        // These flows have a resolvable model in the catalog (not blocked at the bridge gate).
        #expect(runs("8-VoiceoverBed") == nil)
        #expect(runs("7-UpscaleSmall") == nil, "SeedVR2 3B is in the catalog (SV-AM1)")
        #expect(runs("6-CutOutSubject") == nil, "SAM Base bridges onto sam3-4bit (CFM-R15-1)")
        // MP7-W-f (owner ruling, Boo, 2026-11-01): 60 and 66 write "Z-Image Turbo (4-bit)", now a `.substitute` bridge row onto FLUX.2 Klein 4B; with claimableModelIDs empty
        // here the substitute still resolves (readiness is install-only), so they pass the live gate.
        #expect(runs("60-GenerateProductShot") == nil)
        #expect(runs("66-SeedSweep") == nil)
        // These still refuse: Edit Image is unserved (the preflight names the task), Depth Pro has no bridge model.
        for fid in ["61-EditInPlace", "62-SeeDepth"] {
            let reason = runs(fid)
            #expect(reason != nil, "\(fid) should refuse at the live gate")
        }
    }

    private func makeCatalog() throws -> [ModelEntry] {
        let url = try #require(Bundle.main.url(forResource: "browser", withExtension: "json"))
        let browser = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
        return browser.domains.flatMap { $0.allModels }
    }

    /// CFM-R12-FIX-11: an overwrite moves the previous file to `.trash`, never deletes it.
    @Test func saveImageTrashesTheOverwrite() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let src = flowDir.appendingPathComponent("blob.png")
        try PNGEncoder.pngData(from: try ImageTools.load(try ImageTools.save(redPixel(), format: nil, blobDirectory: flowDir, row: "t")))?.write(to: src)
        let tool = SaveImageTool(workspace: ws, flowID: "f", settings: "out.png")
        _ = try await tool.run(Asset(items: [Item(kind: .image, value: nil, path: src, sourceText: nil)])) { _ in }
        _ = try await tool.run(Asset(items: [Item(kind: .image, value: nil, path: src, sourceText: nil)])) { _ in }
        let trash = flowDir.appendingPathComponent(".trash")
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: trash.path)) ?? []
        #expect(entries.contains { $0.hasPrefix("out.png.") })
    }

    private func redPixel() -> CGImage {
        let ctx = CGContext(data: nil, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(srgbRed: 1, green: 0, blue: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: 1, height: 1))
        return ctx.makeImage()!
    }

}
