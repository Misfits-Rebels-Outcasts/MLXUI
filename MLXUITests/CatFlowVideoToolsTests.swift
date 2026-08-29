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

    // MARK: - Join Video (R13-3, the last unported instant tool)

    /// A 1-second 64×64 mp4 with a distinctive pixel color, so the joined output can be
    /// distinguished by size (two clips ≈ two seconds).
    private func makeColorVideo(at url: URL, color: UInt8) async throws {
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
                if attempts > 200 { break }
                try await Task.sleep(for: .milliseconds(5))
            }
            var pb: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &pb)
            guard let buffer = pb else { continue }
            CVPixelBufferLockBaseAddress(buffer, [])
            if let base = CVPixelBufferGetBaseAddress(buffer) {
                memset(base, Int32(color), CVPixelBufferGetDataSize(buffer))
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            let time = CMTime(value: CMTimeValue(i), timescale: CMTimeScale(fps))
            adaptor.append(buffer, withPresentationTime: time)
        }
        input.markAsFinished()
        await writer.finishWriting()
        if writer.status == .failed { throw writer.error! }
        try? await Task.sleep(for: .milliseconds(20))
    }

    @Test func joinVideoConcatenatesTwoClips() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let a = flowDir.appendingPathComponent("a.mp4")
        let b = flowDir.appendingPathComponent("b.mp4")
        try await makeColorVideo(at: a, color: 0x60)
        try await makeColorVideo(at: b, color: 0xA0)

        let tool = JoinVideoTool(workspace: ws, flowID: "f")
        let out = try await tool.run(inputs: [Asset(items: [
            Item(kind: .video, value: nil, path: a, sourceText: nil),
            Item(kind: .video, value: nil, path: b, sourceText: nil),
        ])])

        let joinedPath = try out.items.first?.path ?? { throw FlowError.stageFailure(row: "Join Video", message: "no path") }()
        let joined = AVURLAsset(url: joinedPath)
        let duration = try await joined.load(.duration).seconds
        #expect(abs(duration - 2.0) < 0.35, "two 1s clips should join to ~2s, got \(duration)")
        let tracks = try await joined.loadTracks(withMediaType: .video)
        #expect(!tracks.isEmpty)
    }

    @Test func joinVideoRefusesMismatchedSizesHonestly() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let a = flowDir.appendingPathComponent("a.mp4")
        let big = flowDir.appendingPathComponent("big.mp4")
        try await makeColorVideo(at: a, color: 0x60)

        // A 128×128 clip — different display size than the 64×64 first clip.
        let writer = try AVAssetWriter(outputURL: big, fileType: .mp4)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: 128,
            AVVideoHeightKey: 128,
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input,
                                                           sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
            kCVPixelBufferWidthKey as String: 128,
            kCVPixelBufferHeightKey as String: 128,
        ])
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
        var attempts = 0
        while !input.isReadyForMoreMediaData {
            attempts += 1
            if attempts > 200 { break }
            try await Task.sleep(for: .milliseconds(5))
        }
        var pb: CVPixelBuffer?
        guard let pool = adaptor.pixelBufferPool else {
            input.markAsFinished()
            await writer.finishWriting()
            throw FlowError.stageFailure(row: "test-setup", message: "pixelBufferPool unavailable for big clip")
        }
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb)
        if let buffer = pb {
            CVPixelBufferLockBaseAddress(buffer, [])
            if let baddr = CVPixelBufferGetBaseAddress(buffer) {
                memset(baddr, 0x60, CVPixelBufferGetDataSize(buffer))
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            adaptor.append(buffer, withPresentationTime: .zero)
        }
        input.markAsFinished()
        await writer.finishWriting()
        if writer.status == .failed { throw writer.error! }
        try? await Task.sleep(for: .milliseconds(20))

        let tool = JoinVideoTool(workspace: ws, flowID: "f")
        do {
            _ = try await tool.run(inputs: [Asset(items: [
                Item(kind: .video, value: nil, path: a, sourceText: nil),
                Item(kind: .video, value: nil, path: big, sourceText: nil),
            ])])
            Issue.record("expected a mismatch refusal")
        } catch FlowError.stageFailure(_, let message) {
            #expect(message.contains("don't match"))
            #expect(message.contains("64"))
        }
    }
}
