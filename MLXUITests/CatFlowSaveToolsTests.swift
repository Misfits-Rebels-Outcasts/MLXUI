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

    /// R12-5's done-when + the unhide audit: porting `Save Image` unblocks every flow whose
    /// only blocker was it. 60/66 stay blocked (Watermark; Range+Contact Sheet), so the
    /// unhide decision (AppState.hiddenFlowNumbers) is handed to the owner in the journal.
    @Test func flowsUnblockedBySaveImageAreRunnableNow() throws {
        let runnable = ["61-EditInPlace", "62-SeeDepth", "63-CutOutSubject", "64-UpscaleSmall", "65-VoiceoverBed"]
        for fid in runnable {
            let doc = try GalleryLoader.loadDocument(flowID: fid)
            #expect(FlowRunner.canRun(doc) == .runnable, "\(fid) should be runnable after Save Image")
        }
        let stillBlocked = ["60-GenerateProductShot", "66-SeedSweep"]
        for fid in stillBlocked {
            let doc = try GalleryLoader.loadDocument(flowID: fid)
            guard case .notRunnable = FlowRunner.canRun(doc) else {
                Issue.record("\(fid) should still be blocked")
                continue
            }
        }
    }
}
