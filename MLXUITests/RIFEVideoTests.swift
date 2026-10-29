import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import MLX
import Testing
@testable import MLXUI

/// RIFE-3c: the video pipeline. The planner/limit tests need nothing; the end-to-end tests need the weights (see `RIFERealLoad`).
enum RIFETestVideo {
    /// A small H.264 MP4 of a square moving right, `frames` frames at `fps`, optionally with `seconds` of AAC (a 440 Hz tone).
    static func make(frames: Int, fps: Int32, width: Int = 64, height: Int = 48, audioSeconds: Double? = nil) async throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("rife-src-\(UUID().uuidString).mp4")
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let videoIn = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoIn, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
        ])
        writer.add(videoIn)
        var audioIn: AVAssetWriterInput?
        var asbd = AudioStreamBasicDescription(mSampleRate: 44100, mFormatID: kAudioFormatLinearPCM,
                                               mFormatFlags: kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked,
                                               mBytesPerPacket: 2, mFramesPerPacket: 1, mBytesPerFrame: 2,
                                               mChannelsPerFrame: 1, mBitsPerChannel: 16, mReserved: 0)
        var format: CMAudioFormatDescription?
        CMAudioFormatDescriptionCreate(allocator: nil, asbd: &asbd, layoutSize: 0, layout: nil, magicCookieSize: 0,
                                       magicCookie: nil, extensions: nil, formatDescriptionOut: &format)
        if audioSeconds != nil {
            let a = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 44100, AVNumberOfChannelsKey: 1, AVEncoderBitRateKey: 64000,
            ], sourceFormatHint: format)
            writer.add(a)
            audioIn = a
        }
        guard writer.startWriting() else { throw RIFEVideoError.writerFailed(writer.error?.localizedDescription ?? "start") }
        writer.startSession(atSourceTime: .zero)
        for i in 0 ..< frames {
            while !videoIn.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(2)) }
            var buffer: CVPixelBuffer?
            CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, nil, &buffer)
            guard let buffer else { throw RIFEVideoError.writerFailed("buffer") }
            CVPixelBufferLockBaseAddress(buffer, [])
            if let base = CVPixelBufferGetBaseAddress(buffer) {
                let stride = CVPixelBufferGetBytesPerRow(buffer)
                for y in 0 ..< height {
                    let row = base.advanced(by: y * stride).assumingMemoryBound(to: UInt8.self)
                    for x in 0 ..< width {
                        let inSquare = abs(x - (10 + i * 3)) < 8 && abs(y - height / 2) < 8
                        row[x * 4 + 0] = inSquare ? 40 : 200
                        row[x * 4 + 1] = inSquare ? 60 : 120
                        row[x * 4 + 2] = inSquare ? 230 : 90
                        row[x * 4 + 3] = 255
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            adaptor.append(buffer, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: fps))
        }
        videoIn.markAsFinished()
        if let audioIn, let seconds = audioSeconds {
            let count = Int(44100 * seconds)
            var samples = [Int16](repeating: 0, count: count)
            for n in 0 ..< count { samples[n] = Int16(8000 * sin(2 * Double.pi * 440 * Double(n) / 44100)) }
            let bytes = samples.withUnsafeBytes { Data($0) }
            var block: CMBlockBuffer?
            CMBlockBufferCreateWithMemoryBlock(allocator: nil, memoryBlock: nil, blockLength: bytes.count, blockAllocator: nil,
                                               customBlockSource: nil, offsetToData: 0, dataLength: bytes.count, flags: 0, blockBufferOut: &block)
            if let block { CMBlockBufferReplaceDataBytes(with: [UInt8](bytes), blockBuffer: block, offsetIntoDestination: 0, dataLength: bytes.count) }
            var sample: CMSampleBuffer?
            if let block, let format {
                CMAudioSampleBufferCreateReadyWithPacketDescriptions(allocator: nil, dataBuffer: block, formatDescription: format,
                                                                     sampleCount: count, presentationTimeStamp: .zero,
                                                                     packetDescriptions: nil, sampleBufferOut: &sample)
            }
            while !audioIn.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(2)) }
            if let sample { audioIn.append(sample) }
            audioIn.markAsFinished()
        }
        await writer.finishWriting()
        guard writer.status == .completed else { throw RIFEVideoError.writerFailed(writer.error?.localizedDescription ?? "finish") }
        return url
    }

    static func duration(_ url: URL, _ type: AVMediaType) async throws -> Double? {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: type).first else { return nil }
        return CMTimeGetSeconds(try await track.load(.timeRange).duration)
    }

    static func frameCount(_ url: URL) async throws -> Int {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { return 0 }
        let reader = try AVAssetReader(asset: asset)
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        reader.add(out)
        reader.startReading()
        var n = 0
        while out.copyNextSampleBuffer() != nil { n += 1 }
        return n
    }
}

struct RIFEVideoPlanTests {

    @Test func optionsAreRestrictedToTheRuledChoices() throws {
        try RIFEVideoPipeline.validate(RIFEVideoOptions(times: 2, scale: 1))
        try RIFEVideoPipeline.validate(RIFEVideoOptions(times: 4, scale: 0.5))
        #expect(throws: RIFEVideoError.self) { try RIFEVideoPipeline.validate(RIFEVideoOptions(times: 3, scale: 1)) }
        #expect(throws: RIFEVideoError.self) { try RIFEVideoPipeline.validate(RIFEVideoOptions(times: 2, scale: 0.25)) }
    }

    // RIFE-5b (owner ruling 2026-10-29): above 1440p Scale is 0.5; the pipeline refuses 1.0 there, not only the Run screen.
    @Test func fullScaleIsRefusedAbove1440p() throws {
        func info(_ w: Int, _ h: Int) -> RIFEVideoInfo {
            RIFEVideoInfo(width: w, height: h, frameRate: 30, duration: 2, estimatedFrames: 60, hasAudio: false)
        }
        let full = RIFEVideoOptions(times: 2, scale: 1), half = RIFEVideoOptions(times: 2, scale: 0.5)
        try RIFEVideoPipeline.validate(info(1920, 1080), options: full)
        try RIFEVideoPipeline.validate(info(2560, 1440), options: full)
        try RIFEVideoPipeline.validate(info(3840, 2160), options: half)
        #expect(throws: RIFEVideoError.fullScaleTooLarge(width: 3840, height: 2160)) {
            try RIFEVideoPipeline.validate(info(3840, 2160), options: full)
        }
        #expect(throws: RIFEVideoError.self) { try RIFEVideoPipeline.validate(info(2560, 1600), options: full) }
        #expect(RIFEVideoError.fullScaleTooLarge(width: 3840, height: 2160).description.contains("Scale 0.5"))
    }

    // Owner ruling 2026-10-29: say so when the source has no audio, instead of promising it is copied.
    @Test func theOutputLineFollowsTheSourceAudio() {
        let withAudio = RIFEVideoInfo(width: 1280, height: 720, frameRate: 30, duration: 2, estimatedFrames: 60, hasAudio: true)
        var silent = withAudio; silent.hasAudio = false
        #expect(RIFERunModel.outputLine(withAudio, times: 2) == "Output: 60 fps, about 119 frames, the same length as the original. The audio is copied unchanged.")
        #expect(RIFERunModel.outputLine(silent, times: 2) == "Output: 60 fps, about 119 frames, the same length as the original. The clip has no audio.")
        #expect(!RIFERunModel.outputLine(silent, times: 4).contains("copied"))
    }

    @MainActor @Test func theRunModelFixesScaleAtHalfAbove1440p() {
        let model = RIFERunModel()
        model.scale = 1.0
        #expect(!model.scaleIsFixed && model.options.scale == 1.0)
        model.info = RIFEVideoInfo(width: 3840, height: 2160, frameRate: 30, duration: 2, estimatedFrames: 60, hasAudio: false)
        #expect(model.scaleIsFixed && model.options.scale == 0.5)
        model.info = RIFEVideoInfo(width: 2560, height: 1440, frameRate: 30, duration: 2, estimatedFrames: 60, hasAudio: false)
        #expect(!model.scaleIsFixed && model.options.scale == 1.0)
    }

    @Test func limitsRefuseALongOrHugeClipAndATrivialOne() {
        let ok = RIFEVideoInfo(width: 1920, height: 1080, frameRate: 30, duration: 10, estimatedFrames: 300, hasAudio: true)
        #expect(throws: Never.self) { try RIFEVideoPipeline.validate(ok) }
        var long = ok; long.estimatedFrames = RIFEVideoOptions.maxSourceFrames + 1
        #expect(throws: RIFEVideoError.tooManyFrames(count: long.estimatedFrames, limit: RIFEVideoOptions.maxSourceFrames)) {
            try RIFEVideoPipeline.validate(long)
        }
        var huge = ok; huge.width = 7680; huge.height = 4320
        #expect(throws: RIFEVideoError.tooLarge(width: 7680, height: 4320)) { try RIFEVideoPipeline.validate(huge) }
        var single = ok; single.estimatedFrames = 1
        #expect(throws: RIFEVideoError.noFrames) { try RIFEVideoPipeline.validate(single) }
        var uhd = ok; uhd.width = 3840; uhd.height = 2160
        #expect(throws: Never.self) { try RIFEVideoPipeline.validate(uhd) }
    }

    @Test func outputFrameAndRateArithmetic() {
        let info = RIFEVideoInfo(width: 64, height: 48, frameRate: 24, duration: 1, estimatedFrames: 24, hasAudio: false)
        #expect(info.outputFrames(times: 2) == 47)
        #expect(info.outputFrames(times: 4) == 93)
        #expect(info.outputFrameRate(times: 4) == 96)
    }

    @Test func insertedTimesAreEvenlySpacedAndKeepVariablePacing() {
        let a = CMTime(value: 0, timescale: 600), b = CMTime(value: 40, timescale: 600)
        #expect((1 ..< 4).map { CMTimeGetSeconds(RIFEVideoPipeline.insertedTime(from: a, to: b, step: $0, of: 4)) }
            == [10, 20, 30].map { Double($0) / 600 })
        let c = CMTime(value: 100, timescale: 600), d = CMTime(value: 130, timescale: 600)   // a shorter gap
        #expect(CMTimeGetSeconds(RIFEVideoPipeline.insertedTime(from: c, to: d, step: 1, of: 2)) == 115.0 / 600)
    }

    @Test func inspectReadsASyntheticClip() async throws {
        let url = try await RIFETestVideo.make(frames: 12, fps: 12, audioSeconds: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        let info = try await RIFEVideoPipeline.inspect(url)
        #expect(info.width == 64 && info.height == 48)
        #expect(abs(info.frameRate - 12) < 0.5)
        #expect(info.hasAudio)
        #expect(info.estimatedFrames >= 11 && info.estimatedFrames <= 13)
    }

    @Test func aFileWithNoVideoIsRefusedPlainly() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("rife-not-video-\(UUID().uuidString).mp4")
        try Data("not a video".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        await #expect(throws: RIFEVideoError.self) { _ = try await RIFEVideoPipeline.inspect(url) }
    }

    @Test func pixelConversionRoundTripsRGB() throws {
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, 5, 3, kCVPixelFormatType_32BGRA, nil, &buffer)
        let pb = try #require(buffer)
        let frame = MLXArray([0.0, 0.5, 1.0] as [Float]).reshaped(1, 1, 1, 3) * MLXArray.ones([1, 3, 5, 3])
        try RIFEVideoPipeline.write(frame, into: pb)
        let back = RIFEVideoPipeline.read(pb)
        #expect(back.shape == [1, 3, 5, 3])
        let first = back[0, 0, 0].asArray(Float.self)
        #expect(abs(first[0] - 0) < 0.005 && abs(first[1] - 0.5) < 0.005 && abs(first[2] - 1) < 0.005)
    }
}

@Suite(.enabled(if: RIFERealLoad.enabled, "needs MLXUI_TEST_RIFE_DIR or an installed mlx-community/RIFE-4.25"))
struct RIFEVideoRunTests {

    private func run(_ src: URL, times: Int, scale: Float = 1) async throws -> URL {
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("rife-out-\(UUID().uuidString).mp4")
        let dir = try #require(RIFERealLoad.directory)
        try await RIFEVideoPipeline.run(input: src, output: out, modelDirectory: dir,
                                        options: RIFEVideoOptions(times: times, scale: scale)) { _ in }
        return out
    }

    @Test func twoTimesGivesTwoNMinusOneFramesSameDurationAndKeepsAudio() async throws {
        let src = try await RIFETestVideo.make(frames: 12, fps: 12, audioSeconds: 1)
        defer { try? FileManager.default.removeItem(at: src) }
        let out = try await run(src, times: 2)
        defer { try? FileManager.default.removeItem(at: out) }
        #expect(try await RIFETestVideo.frameCount(out) == 23)
        let sourceVideo = try #require(try await RIFETestVideo.duration(src, .video))
        let outVideo = try #require(try await RIFETestVideo.duration(out, .video))
        #expect(abs(outVideo - sourceVideo) < 0.1, "interpolation adds frames between the originals; the clip doesn't get longer")
        let sourceAudio = try #require(try await RIFETestVideo.duration(src, .audio))
        let outAudio = try #require(try await RIFETestVideo.duration(out, .audio))
        #expect(abs(outAudio - sourceAudio) < 0.05, "audio is copied, not re-encoded")
        let info = try await RIFEVideoPipeline.inspect(out)
        #expect(abs(info.frameRate - 24) < 1)
    }

    @Test func fourTimesGivesFourNMinusThreeFramesPlusOne() async throws {
        let src = try await RIFETestVideo.make(frames: 6, fps: 10)
        defer { try? FileManager.default.removeItem(at: src) }
        let out = try await run(src, times: 4, scale: 0.5)
        defer { try? FileManager.default.removeItem(at: out) }
        #expect(try await RIFETestVideo.frameCount(out) == 21)   // (6 - 1) * 4 + 1
        #expect(try await RIFETestVideo.duration(out, .audio) == nil)
    }

    @Test func cancellingStopsAndLeavesNoPartialFile() async throws {
        let src = try await RIFETestVideo.make(frames: 90, fps: 30, width: 320, height: 240)
        defer { try? FileManager.default.removeItem(at: src) }
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("rife-cancel-\(UUID().uuidString).mp4")
        let dir = try #require(RIFERealLoad.directory)
        let task = Task {
            try await RIFEVideoPipeline.run(input: src, output: out, modelDirectory: dir, options: RIFEVideoOptions()) { _ in }
        }
        try await Task.sleep(for: .milliseconds(400))
        task.cancel()
        await #expect(throws: (any Error).self) { try await task.value }
        #expect(!FileManager.default.fileExists(atPath: out.path))
    }

    @Test func theMainActorStaysResponsiveWhileItRuns() async throws {
        let src = try await RIFETestVideo.make(frames: 30, fps: 30, width: 256, height: 192)
        defer { try? FileManager.default.removeItem(at: src) }
        let dir = try #require(RIFERealLoad.directory)
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("rife-main-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: out) }
        let work = Task { try await RIFEVideoPipeline.run(input: src, output: out, modelDirectory: dir, options: RIFEVideoOptions()) { _ in } }
        // A hop to the main actor must complete promptly while the run is in flight (the MP5-3g gate).
        var worst = 0.0
        for _ in 0 ..< 5 {
            let start = ContinuousClock.now
            await MainActor.run {}
            worst = max(worst, Double((ContinuousClock.now - start).components.attoseconds) / 1e18)
            try await Task.sleep(for: .milliseconds(100))
        }
        try await work.value
        #expect(worst < 0.5, "the main actor was blocked for \(worst) s")
    }
}

// RIFE-4: the stage the Run screen runs (text path in, text path out), end to end.
@Suite(.enabled(if: RIFERealLoad.enabled, "needs MLXUI_TEST_RIFE_DIR or an installed mlx-community/RIFE-4.25"))
struct RIFEStageRunTests {
    @Test func theStageTurnsAVideoPathIntoANewVideoPath() async throws {
        let src = try await RIFETestVideo.make(frames: 8, fps: 8, audioSeconds: 1)
        defer { try? FileManager.default.removeItem(at: src) }
        let out = RIFEStage.temporaryOutputURL()
        defer { try? FileManager.default.removeItem(at: out) }
        let stage = RIFEStage(modelDirectory: try #require(RIFERealLoad.directory), options: RIFEVideoOptions(times: 2, scale: 1), outputURL: out)
        var last = 0.0
        let result = try await stage.run(.text(src.path), progress: { last = max(last, $0) })
        guard case let .text(path) = result else { Issue.record("expected text"); return }
        #expect(path == out.path)
        #expect(try await RIFETestVideo.frameCount(out) == 15)
        #expect(last > 0)
    }
}
