import Testing
import Foundation
import AVFoundation
import CoreGraphics
@testable import MLXUI

/// CFM-R12-7 group e — the AVFoundation subset: Read Video, Extract Frame, Extract Audio,
/// Trim, Mux. A tiny real video (20 frames + silence) is generated with AVAssetWriter, so
/// the exports run against actual media.
struct CatFlowVideoToolsTests {

    private func makeWorkspace() throws -> (FlowWorkspace, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-video-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return (FlowWorkspace(root: base.appendingPathComponent("flows")), base)
    }

    private func teardown(_ base: URL) {
        try? FileManager.default.removeItem(at: base)
    }

    /// A 1-second 64×64 mp4 (20 frames of solid color, no audio).
    private func makeVideo(at url: URL) async throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: 64,
            AVVideoHeightKey: 64,
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input,
                                                           sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
            kCVPixelBufferWidthKey as String: 64,
            kCVPixelBufferHeightKey as String: 64,
        ])
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        let fps = 20
        for i in 0..<fps {
            var attempts = 0
            while !input.isReadyForMoreMediaData {
                attempts += 1
                if attempts > 200 { break }   // ~1s; give up rather than hang
                try await Task.sleep(for: .milliseconds(5))
            }
            var pb: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &pb)
            guard let buffer = pb else { continue }
            CVPixelBufferLockBaseAddress(buffer, [])
            if let base = CVPixelBufferGetBaseAddress(buffer) {
                memset(base, 0x80, CVPixelBufferGetDataSize(buffer))
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            let time = CMTime(value: CMTimeValue(i), timescale: CMTimeScale(fps))
            adaptor.append(buffer, withPresentationTime: time)
        }
        input.markAsFinished()
        await writer.finishWriting()
        if writer.status == .failed { throw writer.error! }
        // Ensure the file is fully flushed before the consumer reads it.
        try? await Task.sleep(for: .milliseconds(20))
    }

    @Test func readVideoIsFileBacked() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let video = flowDir.appendingPathComponent("clip.mp4")
        try await makeVideo(at: video)
        let tool = ReadVideoTool(workspace: ws, flowID: "f", settings: "clip.mp4")
        let out = try await tool.run(Asset(items: [])) { _ in }
        #expect(out.items.first?.kind == .video)
        #expect(out.items.first?.path?.lastPathComponent == "clip.mp4")
    }

    @Test func parseTimeHandlesHMSAndSeconds() {
        #expect(VideoTools.parseTime("00:00:05") == 5)
        #expect(VideoTools.parseTime("1:30") == 90)
        #expect(VideoTools.parseTime("1.5") == 1.5)
        #expect(VideoTools.parseTime("0") == 0)
        #expect(VideoTools.parseTime("x") == nil)
    }

    @Test func extractFrameGrabsARealFrame() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let video = flowDir.appendingPathComponent("clip.mp4")
        try await makeVideo(at: video)
        let item = Item(kind: .video, value: nil, path: video, sourceText: nil)
        let tool = ExtractFrameTool(workspace: ws, flowID: "f", settings: "at=00:00:01")
        let out = try await tool.run(Asset(items: [item])) { _ in }
        let imagePath = try out.items.first?.path ?? { throw FlowError.stageFailure(row: "t", message: "no path") }()
        let cg = try ImageTools.load(imagePath)
        #expect(cg.width == 64)
        #expect(cg.height == 64)
    }

    @Test func twentyFourClipNotesIsRunnableNow() throws {
        // 24-ClipNotes used Read Video + Extract Audio + Extract Frame + Save Image.
        let doc = try GalleryLoader.loadDocument(flowID: "24-ClipNotes")
        #expect(FlowRunner.canRun(doc) == .runnable)
    }
}
