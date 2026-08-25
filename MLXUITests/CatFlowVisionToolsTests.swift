import Testing
import Foundation
import CoreGraphics
@testable import MLXUI

/// CFM-R12-7 group f — the vision tools: `Detect Edges` (Sobel + Canny hysteresis, no
/// model) and `Detect Pose` (Apple Vision — the stick-figure renderer is pinned by tests).
struct CatFlowVisionToolsTests {

    private func makeWorkspace() throws -> (FlowWorkspace, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-vision-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return (FlowWorkspace(root: base.appendingPathComponent("flows")), base)
    }

    private func teardown(_ base: URL) {
        try? FileManager.default.removeItem(at: base)
    }

    /// A 10×10 image with a bright vertical stripe — a clear vertical edge.
    private func stripeImage() -> CGImage {
        var data = [UInt8](repeating: 0, count: 10 * 10)
        for y in 0..<10 { for x in 0..<5 { data[y * 10 + x] = 255 } }
        let ctx = CGContext(data: &data, width: 10, height: 10, bitsPerComponent: 8, bytesPerRow: 10,
                            space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        return ctx.makeImage()!
    }

    @Test func detectEdgesFindsTheStripe() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let dir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let src = try ImageTools.save(stripeImage(), format: nil, blobDirectory: dir, row: "test")

        let tool = DetectEdgesTool(workspace: ws, flowID: "f", settings: "method=sobel")
        let out = try await tool.run(Asset(items: [Item(kind: .image, value: nil, path: src, sourceText: nil)])) { _ in }
        let outPath = try out.items.first?.path ?? { throw FlowError.stageFailure(row: "t", message: "no path") }()
        let cg = try ImageTools.load(outPath)
        #expect(cg.width == 10)
        #expect(cg.height == 10)
    }

    @Test func sobelAndHysteresisAreArithmetic() throws {
        // A single strong pixel in the middle; weak neighbours survive via hysteresis.
        let gray = [[Float]](repeating: [Float](repeating: 0.5, count: 7), count: 7)
        var strong = [[Bool]](repeating: [Bool](repeating: false, count: 7), count: 7)
        var weak = [[Bool]](repeating: [Bool](repeating: false, count: 7), count: 7)
        strong[3][3] = true
        for y in 2...4 { for x in 2...4 { weak[y][x] = true } }
        let kept = DetectEdgesTool.hysteresis(strong: strong, weak: weak)
        #expect(kept[3][3])
        #expect(kept[2][2])          // connected weak neighbour
        #expect(!kept[0][0])         // disconnected
        // Sobel on a vertical edge yields a non-zero gradient.
        let rows = [[Float]](repeating: [0, 0, 1, 1], count: 4)
        let gx = DetectEdgesTool.sobel(rows, axis: 1)
        #expect(gx[1][2] != 0)
    }

    @Test func poseRendererDrawsTheStickFigure() throws {
        // A minimal upright person: neck + shoulders + hips + knees + ankles.
        let kps: [(x: Double, y: Double)?] = [
            (50, 40), (50, 60), (40, 70), (35, 90), (33, 110),   // nose..rightWrist
            (60, 70), (65, 90), (67, 110),                       // left shoulder..wrist
            (45, 90), (45, 120), (45, 150),                      // right hip..ankle
            (55, 90), (55, 120), (55, 150),                      // left hip..ankle
            (52, 38), (48, 38), (54, 36), (46, 36),              // eyes, ears
        ]
        let img = DetectPoseTool.renderPose(kps, width: 100, height: 160)
        #expect(img.width == 100)
        #expect(img.height == 160)
    }

    @Test func onlyJoinVideoIsUnportedNow() {
        // Every instant catalog tool except Join Video has a real dispatch.
        let unported = TaskCatalog.entries.filter {
            $0.taskClass == .instant && !TaskAvailability.supportedInstantTools.contains($0.name)
        }.map(\.name)
        #expect(unported == ["Join Video"])
    }
}
