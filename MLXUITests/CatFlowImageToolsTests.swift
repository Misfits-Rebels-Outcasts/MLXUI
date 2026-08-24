import Testing
import Foundation
import CoreGraphics
import AppKit
@testable import MLXUI

/// CFM-R12-7 group d — the image tools: Resize, Crop, Convert, Watermark, Overlay Text,
/// Contact Sheet. Each produces a file-backed `.image` item that decodes.
struct CatFlowImageToolsTests {

    private func makeWorkspace() throws -> (FlowWorkspace, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-img-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return (FlowWorkspace(root: base.appendingPathComponent("flows")), base)
    }

    private func teardown(_ base: URL) {
        try? FileManager.default.removeItem(at: base)
    }

    /// A 100×50 solid-color CGImage.
    private func makeSource(width: Int = 100, height: Int = 50, red: CGFloat = 1) -> CGImage {
        let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                            bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(srgbRed: red, green: 0, blue: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return ctx.makeImage()!
    }

    private func writeSource(_ ws: FlowWorkspace) throws -> URL {
        let dir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return try ImageTools.save(makeSource(), format: nil, blobDirectory: dir, row: "test")
    }

    /// The decoded width of a file-backed output, or 0 on failure (for `#expect`).
    private func width(of out: Asset) -> Int {
        guard let path = out.items.first?.path,
              let cg = try? ImageTools.load(path) else { return 0 }
        return cg.width
    }

    @Test func resizeThumbnailsToTheMaxDimension() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let src = try writeSource(ws)
        let tool = ResizeTool(workspace: ws, flowID: "f", settings: "max=50")
        let out = try await tool.run(Asset(items: [Item(kind: .image, value: nil, path: src, sourceText: nil)])) { _ in }
        let cg = try ImageTools.load(try out.items.first?.path ?? { throw FlowError.stageFailure(row: "t", message: "no path") }())
        #expect(cg.width == 50)
        #expect(cg.height == 25)
    }

    @Test func cropTakesTheBox() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let src = try writeSource(ws)
        let tool = CropTool(workspace: ws, flowID: "f", settings: "box=0,0,40,30")
        let out = try await tool.run(Asset(items: [Item(kind: .image, value: nil, path: src, sourceText: nil)])) { _ in }
        let cg = try ImageTools.load(try out.items.first?.path ?? { throw FlowError.stageFailure(row: "t", message: "no path") }())
        #expect(cg.width == 40)
        #expect(cg.height == 30)
    }

    @Test func convertChangesTheEncoding() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let src = try writeSource(ws)
        let tool = ConvertTool(workspace: ws, flowID: "f", settings: "format=jpeg")
        let out = try await tool.run(Asset(items: [Item(kind: .image, value: nil, path: src, sourceText: nil)])) { _ in }
        let url = try out.items.first?.path ?? { throw FlowError.stageFailure(row: "t", message: "no path") }()
        #expect(url.pathExtension == "jpeg")
        #expect(try ImageTools.load(url).width == 100)
    }

    @Test func watermarkCompositesTheMark() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let src = try writeSource(ws)
        let dir = ws.directory(for: "f")
        let markURL = dir.appendingPathComponent("mark.png")
        try PNGEncoder.pngData(from: makeSource(width: 10, height: 10, red: 0))?.write(to: markURL)
        let tool = WatermarkTool(workspace: ws, flowID: "f", settings: "asset=mark.png; position=top-left")
        let out = try await tool.run(Asset(items: [Item(kind: .image, value: nil, path: src, sourceText: nil)])) { _ in }
        #expect(width(of: out) == 100)
    }

    @Test func overlayTextDrawsOnTheImage() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let src = try writeSource(ws)
        let tool = OverlayTextTool(workspace: ws, flowID: "f", settings: "text=\"Hello\"")
        let out = try await tool.run(Asset(items: [Item(kind: .image, value: nil, path: src, sourceText: nil)])) { _ in }
        #expect(width(of: out) == 100)
    }

    @Test func contactSheetTilesImagesWithLabels() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let dir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let a = try ImageTools.save(makeSource(), format: nil, blobDirectory: dir, row: "test")
        let b = try ImageTools.save(makeSource(red: 0), format: nil, blobDirectory: dir, row: "test")
        let tool = ContactSheetTool(workspace: ws, flowID: "f", settings: "cols=2; size=32")
        let input = Asset(items: [
            Item(kind: .image, value: nil, path: a, sourceText: nil),
            Item(kind: .image, value: nil, path: b, sourceText: nil),
            Item(kind: .text, value: "one", path: nil, sourceText: nil),
            Item(kind: .text, value: "two", path: nil, sourceText: nil),
        ])
        let out = try await tool.run(inputs: [input])
        // Two 32px thumbs in one row → roughly 64px wide.
        #expect(width(of: out) >= 64)
    }
}
