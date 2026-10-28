import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import MLX

/// What a clip may be and how it is processed. The limits are provisional (RIFE-5 measures memory at 1080p and 4K); they exist so a
/// long or huge clip is refused up front with a plain sentence, never half-processed (owner ruling 2026-10-28: "a maximum clip
/// length or frame count, cancellable, never on the main actor", the MP5-3g gate).
nonisolated struct RIFEVideoOptions: Sendable, Equatable {
    /// Output frame-rate multiplier: 2 inserts one frame between neighbours, 4 inserts three.
    var times: Int = 2
    /// Pyramid scale of the flow estimate: 1.0, or 0.5 for 4K (less memory, a coarser flow).
    var scale: Float = 1.0

    static let allowedTimes = [2, 4]
    static let allowedScales: [Float] = [1.0, 0.5]
    /// Source frames: 60 s at 30 fps. The output has `times`× as many.
    static let maxSourceFrames = 1800
    /// 3840 x 2160. Anything larger is refused rather than tried.
    static let maxPixels = 3840 * 2160
}

nonisolated enum RIFEVideoError: Error, LocalizedError, Equatable {
    case notAVideo
    case unreadable(String)
    case badOptions(String)
    case tooManyFrames(count: Int, limit: Int)
    case tooLarge(width: Int, height: Int)
    case writerFailed(String)
    case noFrames

    var errorDescription: String? {
        switch self {
        case .notAVideo:                       return "That file has no video track."
        case .unreadable(let why):             return "Couldn't read that video (\(why))."
        case .badOptions(let why):             return "RIFE: \(why)"
        case .tooManyFrames(let n, let limit): return "That clip has about \(n) frames; the limit is \(limit) (about 60 seconds at 30 fps). Trim it and try again."
        case .tooLarge(let w, let h):          return "That video is \(w) × \(h); the largest supported size is 3840 × 2160."
        case .writerFailed(let why):           return "Couldn't write the new video (\(why))."
        case .noFrames:                        return "That video has fewer than two frames, so there is nothing to interpolate between."
        }
    }
}

nonisolated struct RIFEVideoInfo: Sendable, Equatable {
    var width: Int
    var height: Int
    var frameRate: Double
    var duration: Double
    var estimatedFrames: Int
    var hasAudio: Bool

    /// Frames in the output: each of the `n - 1` neighbour pairs gains `times - 1` frames.
    func outputFrames(times: Int) -> Int { max(0, (estimatedFrames - 1) * times + 1) }
    func outputFrameRate(times: Int) -> Double { frameRate * Double(times) }
}

nonisolated enum RIFEVideoPipeline {

    // MARK: Pure helpers (unit-tested without a model)

    static func validate(_ options: RIFEVideoOptions) throws {
        guard RIFEVideoOptions.allowedTimes.contains(options.times) else {
            throw RIFEVideoError.badOptions("the frame-rate multiplier must be 2 or 4")
        }
        guard RIFEVideoOptions.allowedScales.contains(options.scale) else {
            throw RIFEVideoError.badOptions("the scale must be 1.0 or 0.5")
        }
    }

    static func validate(_ info: RIFEVideoInfo) throws {
        guard info.width * info.height <= RIFEVideoOptions.maxPixels else {
            throw RIFEVideoError.tooLarge(width: info.width, height: info.height)
        }
        guard info.estimatedFrames <= RIFEVideoOptions.maxSourceFrames else {
            throw RIFEVideoError.tooManyFrames(count: info.estimatedFrames, limit: RIFEVideoOptions.maxSourceFrames)
        }
        guard info.estimatedFrames >= 2 else { throw RIFEVideoError.noFrames }
    }

    /// The presentation time of the `step`-th inserted frame (1 ... times-1) between two source frames: evenly spaced, so a
    /// variable-frame-rate clip keeps its own pacing and the total duration is unchanged.
    static func insertedTime(from a: CMTime, to b: CMTime, step: Int, of times: Int) -> CMTime {
        a + CMTimeMultiplyByRatio(b - a, multiplier: Int32(step), divisor: Int32(times))
    }

    // MARK: Inspect

    static func inspect(_ url: URL) async throws -> RIFEVideoInfo {
        let asset = AVURLAsset(url: url)
        let videoTracks: [AVAssetTrack]
        let audioTracks: [AVAssetTrack]
        let duration: CMTime
        do {
            videoTracks = try await asset.loadTracks(withMediaType: .video)
            audioTracks = try await asset.loadTracks(withMediaType: .audio)
            duration = try await asset.load(.duration)
        } catch {
            throw RIFEVideoError.unreadable(error.localizedDescription)
        }
        guard let track = videoTracks.first else { throw RIFEVideoError.notAVideo }
        let size: CGSize
        let rate: Float
        do {
            size = try await track.load(.naturalSize)
            rate = try await track.load(.nominalFrameRate)
        } catch {
            throw RIFEVideoError.unreadable(error.localizedDescription)
        }
        let fps = rate > 0 ? Double(rate) : 30
        let seconds = max(0, CMTimeGetSeconds(duration))
        return RIFEVideoInfo(width: Int(size.width), height: Int(size.height), frameRate: fps, duration: seconds,
                             estimatedFrames: Int((seconds * fps).rounded()), hasAudio: !audioTracks.isEmpty)
    }

    // MARK: Run

    /// Interpolates `input` into `output` (an MP4 that does not exist yet). The whole run, MLX work included, executes off the
    /// caller's actor (`@concurrent`), and every frame's MLX result is evaluated before any suspension, so no lazy array crosses a
    /// thread hop (memory: mlx-swift-032-thread-bound-streams). Cancelling the task stops between frame pairs, removes the partial
    /// file and throws `CancellationError`. `progress` reports 0...1.
    @concurrent
    static func run(input: URL, output: URL, modelDirectory: URL, options: RIFEVideoOptions,
                    progress: @Sendable @escaping (Double) -> Void) async throws {
        try validate(options)
        let info = try await inspect(input)
        try validate(info)
        let loaded = try RIFEEngine.load(directory: modelDirectory)
        try? FileManager.default.removeItem(at: output)
        do {
            try await process(input: input, output: output, loaded: loaded, info: info, options: options, progress: progress)
        } catch {
            try? FileManager.default.removeItem(at: output)
            throw error
        }
    }

    @concurrent
    private static func process(input: URL, output: URL, loaded: RIFEEngine.Loaded, info: RIFEVideoInfo,
                                options: RIFEVideoOptions, progress: @Sendable @escaping (Double) -> Void) async throws {
        let asset = AVURLAsset(url: input)
        let videoTrack = try await asset.loadTracks(withMediaType: .video).first.unwrap(or: RIFEVideoError.notAVideo)
        let audioTrack = try await asset.loadTracks(withMediaType: .audio).first
        let transform = try await videoTrack.load(.preferredTransform)

        let reader = try AVAssetReader(asset: asset)
        let videoOut = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        reader.add(videoOut)
        // Audio: compressed samples straight through (outputSettings nil), so there is no re-encode and the duration is unchanged.
        var audioOut: AVAssetReaderTrackOutput?
        if let audioTrack {
            let out = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: nil)
            reader.add(out)
            audioOut = out
        }

        let writer = try AVAssetWriter(outputURL: output, fileType: .mp4)
        let videoIn = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: info.width,
            AVVideoHeightKey: info.height,
        ])
        videoIn.transform = transform
        videoIn.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoIn, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: info.width,
            kCVPixelBufferHeightKey as String: info.height,
        ])
        writer.add(videoIn)
        var audioIn: AVAssetWriterInput?
        if let audioTrack, audioOut != nil {
            let formats = try await audioTrack.load(.formatDescriptions)
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: nil, sourceFormatHint: formats.first)
            input.expectsMediaDataInRealTime = false
            writer.add(input)
            audioIn = input
        }

        guard reader.startReading() else { throw RIFEVideoError.unreadable(reader.error?.localizedDescription ?? "the reader didn't start") }
        guard writer.startWriting() else { throw RIFEVideoError.writerFailed(writer.error?.localizedDescription ?? "the writer didn't start") }
        writer.startSession(atSourceTime: .zero)

        func waitUntilReady(_ input: AVAssetWriterInput) async throws {
            while !input.isReadyForMoreMediaData {
                try Task.checkCancellation()
                if writer.status == .failed { throw RIFEVideoError.writerFailed(writer.error?.localizedDescription ?? "unknown") }
                try await Task.sleep(for: .milliseconds(2))
            }
        }

        func append(_ frame: MLXArray, at time: CMTime) async throws {
            try await waitUntilReady(videoIn)
            guard let pool = adaptor.pixelBufferPool else { throw RIFEVideoError.writerFailed("no pixel buffer pool") }
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
            guard let buffer else { throw RIFEVideoError.writerFailed("couldn't allocate a frame") }
            try RIFEVideoPipeline.write(frame, into: buffer)
            guard adaptor.append(buffer, withPresentationTime: time) else {
                throw RIFEVideoError.writerFailed(writer.error?.localizedDescription ?? "a frame was refused")
            }
        }

        var previous: (frame: MLXArray, time: CMTime)?
        var sourceIndex = 0
        let expectedPairs = max(1, info.estimatedFrames - 1)
        while let sample = videoOut.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let pixels = CMSampleBufferGetImageBuffer(sample) else { continue }
            sourceIndex += 1
            guard sourceIndex <= RIFEVideoOptions.maxSourceFrames else {
                throw RIFEVideoError.tooManyFrames(count: sourceIndex, limit: RIFEVideoOptions.maxSourceFrames)
            }
            let time = CMSampleBufferGetPresentationTimeStamp(sample)
            let current = RIFEVideoPipeline.read(pixels)
            try MLXErrorGuard.evaluate(current)
            if let previous {
                try await append(previous.frame, at: previous.time)
                for step in 1 ..< options.times {
                    let t = Float(step) / Float(options.times)
                    let mid = try RIFEEngine.interpolate(loaded, previous.frame, current, timestep: t, scale: options.scale)
                    try MLXErrorGuard.evaluate(mid)
                    try await append(mid, at: insertedTime(from: previous.time, to: time, step: step, of: options.times))
                }
                progress(min(0.98, Double(sourceIndex - 1) / Double(expectedPairs)))
            }
            previous = (current, time)
        }
        guard let last = previous, sourceIndex >= 2 else { throw RIFEVideoError.noFrames }
        try await append(last.frame, at: last.time)
        videoIn.markAsFinished()

        if let audioOut, let audioIn {
            while let sample = audioOut.copyNextSampleBuffer() {
                try await waitUntilReady(audioIn)
                guard audioIn.append(sample) else {
                    throw RIFEVideoError.writerFailed(writer.error?.localizedDescription ?? "an audio packet was refused")
                }
            }
            audioIn.markAsFinished()
        }
        if reader.status == .failed { throw RIFEVideoError.unreadable(reader.error?.localizedDescription ?? "read failed") }
        await writer.finishWriting()
        guard writer.status == .completed else { throw RIFEVideoError.writerFailed(writer.error?.localizedDescription ?? "unknown") }
        progress(1)
    }

    // MARK: Pixel conversion

    /// BGRA pixel buffer → `[1, H, W, 3]` float32 RGB in 0...1.
    static func read(_ buffer: CVPixelBuffer) -> MLXArray {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return MLXArray.zeros([1, h, w, 3]) }
        var packed = [UInt8](repeating: 0, count: w * h * 4)
        for row in 0 ..< h {
            packed.withUnsafeMutableBytes { dst in
                dst.baseAddress?.advanced(by: row * w * 4).copyMemory(from: base.advanced(by: row * stride), byteCount: w * 4)
            }
        }
        let bgra = MLXArray(packed).reshaped(1, h, w, 4)
        // Channels 2, 1, 0 → R, G, B.
        let rgb = concatenated([bgra[0..., 0..., 0..., 2 ..< 3], bgra[0..., 0..., 0..., 1 ..< 2], bgra[0..., 0..., 0..., 0 ..< 1]], axis: -1)
        return rgb.asType(.float32) / 255
    }

    /// `[1, H, W, 3]` float RGB in 0...1 → BGRA bytes in `buffer` (alpha 255), rounded and clamped.
    static func write(_ frame: MLXArray, into buffer: CVPixelBuffer) throws {
        let h = frame.dim(1), w = frame.dim(2)
        let bytes = round(clip(frame, min: 0, max: 1) * 255).asType(.uint8)
        let bgra = concatenated([bytes[0..., 0..., 0..., 2 ..< 3], bytes[0..., 0..., 0..., 1 ..< 2], bytes[0..., 0..., 0..., 0 ..< 1],
                                 MLXArray.full([1, h, w, 1], values: MLXArray(UInt8(255)))], axis: -1)
        let packed = try MLXErrorGuard.read(bgra.reshaped(-1), as: UInt8.self)   // evaluates under withError: a failed array throws, never segfaults
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return }
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        packed.withUnsafeBytes { src in
            guard let from = src.baseAddress else { return }
            for row in 0 ..< h {
                base.advanced(by: row * stride).copyMemory(from: from.advanced(by: row * w * 4), byteCount: w * 4)
            }
        }
    }
}

private extension Optional {
    func unwrap(or error: Error) throws -> Wrapped {
        guard let self else { throw error }
        return self
    }
}
