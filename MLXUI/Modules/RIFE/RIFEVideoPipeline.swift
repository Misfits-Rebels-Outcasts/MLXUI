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
    /// Source frames: 30 minutes at 30 fps. The output has `times`× as many. This is an app limit, not a model limit: memory stays flat
    /// over a run (RIFE-L: a 5-minute 1080p 2x run peaked within 10% of the 8 s clip's), so only time bounds it, and the Run screen
    /// shows the estimate before Interpolate (owner ruling, Boo, 2026-10-29).
    static let maxSourceFrames = 54_000
    /// 3840 x 2160. Anything larger is refused rather than tried.
    static let maxPixels = 3840 * 2160
    /// 2560 x 1440. Above this the flow is always estimated at Scale 0.5: Scale 1.0 at 4K ran 30 minutes and peaked near 13 GiB
    /// (RIFE-5, owner ruling 2026-10-29).
    static let fullScaleMaxPixels = 2560 * 1440

    /// Whether Scale 1.0 may be used on a clip of this size.
    static func allowsFullScale(width: Int, height: Int) -> Bool { width * height <= fullScaleMaxPixels }}

nonisolated enum RIFEVideoError: Error, LocalizedError, Equatable {
    case notAVideo
    case unreadable(String)
    case badOptions(String)
    case tooManyFrames(count: Int, limit: Int)
    case tooLarge(width: Int, height: Int)
    case fullScaleTooLarge(width: Int, height: Int)
    case writerFailed(String)
    case noFrames

    var errorDescription: String? {
        switch self {
        case .notAVideo:                       return "That file has no video track."
        case .unreadable:                        return "Couldn't read that video."
        case .badOptions(let why):             return "RIFE: \(why)"
        case .tooManyFrames(let n, let limit): return "That clip has about \(n) frames; the limit is \(limit) (about 30 minutes at 30 fps). Trim it and try again."
        case .tooLarge(let w, let h):          return "That video is \(w) × \(h); the largest supported size is 3840 × 2160."
        case .fullScaleTooLarge(let w, let h): return "That video is \(w) × \(h). Above 2560 × 1440 the motion is always estimated at half size (Scale 0.5)."
        case .writerFailed(let why):           return "Couldn't write the new video (\(why))."
        case .noFrames:                        return "That video has fewer than two frames, so there is nothing to interpolate between."
        }
    }
}

extension RIFEVideoError: CustomStringConvertible {
    /// The flow interpreter and `StageError` stringify with `String(describing:)`; give them the sentence, not the case name.
    nonisolated var description: String { errorDescription ?? "RIFE failed." }
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

/// Decides frame by frame whether a pair of neighbouring frames is a hard cut, from the pair differences seen so far (no look-ahead).
/// A cut is a difference over 0.25, **or** a difference of at least 0.10 that is at least 5× the median of the previous 15 pairs
/// (a jump cut in a quiet shot is small in absolute terms but huge against its surroundings); the relative rule waits for 3 earlier
/// pairs. Owner ruling, Boo, 2026-10-29; validated on App_Preview_179.mp4 (cuts at 36.07 s: 0.139, 6.9×, and 60.03 s: 0.172, 16.3×; the
/// next highest pair is 4.0×, 0.085).
nonisolated struct RIFESceneCutDetector: Sendable {
    static let absoluteThreshold: Float = 0.25
    static let relativeFloor: Float = 0.10
    static let relativeFactor: Float = 5
    static let windowSize = 15
    static let minimumHistory = 3

    private var history: [Float] = []

    /// Feed the next pair's difference; true when that pair is a cut. Every difference, cut or not, joins the window.
    mutating func isCut(difference: Float) -> Bool {
        var cut = difference > Self.absoluteThreshold
        if !cut, history.count >= Self.minimumHistory, difference >= Self.relativeFloor {
            cut = difference >= Self.relativeFactor * Self.median(history)
        }
        history.append(difference)
        if history.count > Self.windowSize { history.removeFirst(history.count - Self.windowSize) }
        return cut
    }

    static func median(_ values: [Float]) -> Float {
        let s = values.sorted()
        guard !s.isEmpty else { return 0 }
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }
}

/// Keeps the Mac awake while a run works: a 30-minute clip takes hours, and idle sleep would stall it. Begun when a run starts, ended
/// on finish, cancel or error (owner ruling, Boo, 2026-10-29).
nonisolated enum RIFEKeepAwake {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var active = 0

    /// How many runs currently hold the Mac awake (tests read this).
    static var activeCount: Int { lock.lock(); defer { lock.unlock() }; return active }

    static func hold<T>(reason: String, _ body: () async throws -> T) async rethrows -> T {
        let token = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .idleSystemSleepDisabled], reason: reason)
        lock.lock(); active += 1; lock.unlock()
        defer {
            ProcessInfo.processInfo.endActivity(token)
            lock.lock(); active -= 1; lock.unlock()
        }
        return try await body()
    }
}

/// How long a run takes, before it starts: inserted frames × megapixels × a per-megapixel rate fitted to the measured runs (owner ruling,
/// Boo, 2026-10-29; M3 Pro: 720p 2x ≈ 8 s per second of video, 1080p 2x ≈ 15, 1080p 4x ≈ 45, 4K 2x at Scale 0.5 ≈ 85 -- all about 0.30 s per
/// inserted megapixel at Scale 1.0, 0.34 at Scale 0.5; then "err long": RIFE-L measured 0.285-0.295 s/MP at Scale 1.0 on 1080p and 1440p runs, up to the
/// 5-minute 1080p run (17.8 s per second of video at 30 fps), so 0.31 is above every measured run). An estimate for the Output line, never a promise.
nonisolated enum RIFEEstimate {
    static let secondsPerMegapixelFull = 0.31
    static let secondsPerMegapixelHalf = 0.34

    static func seconds(for info: RIFEVideoInfo, times: Int, scale: Float) -> Double {
        let inserted = Double(max(0, info.estimatedFrames - 1) * max(0, times - 1))
        let megapixels = Double(info.width * info.height) / 1_000_000
        return inserted * megapixels * (scale < 1 ? secondsPerMegapixelHalf : secondsPerMegapixelFull)
    }

    /// "under a minute", "about 15 minutes", "about 7.5 hours".
    static func phrase(seconds: Double) -> String {
        if seconds < 60 { return "under a minute" }
        let minutes = Int((seconds / 60).rounded())
        if minutes == 1 { return "about 1 minute" }
        if minutes < 120 { return "about \(minutes) minutes" }
        let hours = (seconds / 1800).rounded() / 2   // nearest half hour
        return hours == hours.rounded() ? "about \(Int(hours)) hours" : "about \(hours) hours"
    }
}

nonisolated enum RIFEVideoPipeline {

    // MARK: Scene cuts

    /// 0 for identical frames, 1 for black against white. Frames are `[1, H, W, 3]` floats in 0...1.
    static func sceneDifference(_ a: MLXArray, _ b: MLXArray) throws -> Float {
        func thinned(_ x: MLXArray) -> MLXArray {
            let h = x.dim(1) / 8 * 8, w = x.dim(2) / 8 * 8
            return x[0..., 0 ..< h, 0 ..< w, 0...].reshaped(1, h / 8, 8, w / 8, 8, 3)[0..., 0..., 0, 0..., 0, 0...]
        }
        return try MLXErrorGuard.read(mean(abs(thinned(a) - thinned(b))), as: Float.self).first ?? 0
    }

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

    /// The Scale rule, enforced here as well as in the Run screen.
    static func validate(_ info: RIFEVideoInfo, options: RIFEVideoOptions) throws {
        guard options.scale < 1 || RIFEVideoOptions.allowsFullScale(width: info.width, height: info.height) else {
            throw RIFEVideoError.fullScaleTooLarge(width: info.width, height: info.height)
        }
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
        try await RIFEKeepAwake.hold(reason: "RIFE is interpolating a video") {
            try await runAwake(input: input, output: output, modelDirectory: modelDirectory, options: options, progress: progress)
        }
    }

    @concurrent
    private static func runAwake(input: URL, output: URL, modelDirectory: URL, options: RIFEVideoOptions,
                                 progress: @Sendable @escaping (Double) -> Void) async throws {
        try validate(options)
        let info = try await inspect(input)
        try validate(info)
        try validate(info, options: options)
        let loaded = try RIFEEngine.load(directory: modelDirectory)
        try? FileManager.default.removeItem(at: output)
        // MLX keeps freed buffers in a pool as large as the memory limit; at 4K that pool alone was 8 GiB of the footprint (RIFE-5a:
        // 11.3 GiB peak with the default, 5.9 GiB with this cap, same speed). Same scoped cap the Local Server uses.
        ServeMLXMemory.shared.begin()
        defer { ServeMLXMemory.shared.end() }
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

        // The writer interleaves by time: with a long clip it stops accepting video until the audio has caught up, so the audio
        // is fed alongside the video (up to just past the frame being written), never after it.
        var pendingAudio: CMSampleBuffer?
        var audioFinished = false
        func pumpAudio(upTo limit: CMTime?) throws {
            guard let audioOut, let audioIn, !audioFinished else { return }
            while audioIn.isReadyForMoreMediaData {
                if pendingAudio == nil { pendingAudio = audioOut.copyNextSampleBuffer() }
                // The source ran dry: the input must be closed now, or the writer keeps waiting for more audio and holds the video.
                guard let sample = pendingAudio else {
                    audioIn.markAsFinished()
                    audioFinished = true
                    return
                }
                // A buffer with no samples (or no valid timestamp) carries nothing to write, and an invalid time compares as
                // later than every limit, which would park it here forever and starve the video.
                if CMSampleBufferGetNumSamples(sample) == 0 { pendingAudio = nil; continue }
                if let limit, CMTimeCompare(CMSampleBufferGetPresentationTimeStamp(sample), limit) > 0,
                   CMSampleBufferGetPresentationTimeStamp(sample).isValid { return }
                guard audioIn.append(sample) else {
                    throw RIFEVideoError.writerFailed(writer.error?.localizedDescription ?? "an audio packet was refused")
                }
                pendingAudio = nil
            }
        }

        func waitUntilReady(_ input: AVAssetWriterInput, audioUpTo time: CMTime? = nil) async throws {
            while !input.isReadyForMoreMediaData {
                try Task.checkCancellation()
                if writer.status == .failed { throw RIFEVideoError.writerFailed(writer.error?.localizedDescription ?? "unknown") }
                try pumpAudio(upTo: time)
                try await Task.sleep(for: .milliseconds(2))
            }
        }

        func append(_ frame: MLXArray, at time: CMTime) async throws {
            try pumpAudio(upTo: time + CMTime(seconds: 1, preferredTimescale: 600))
            try await waitUntilReady(videoIn, audioUpTo: time + CMTime(seconds: 1, preferredTimescale: 600))
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
        var cutDetector = RIFESceneCutDetector()
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
                // A hard cut: interpolating across it only draws a ghost of two unrelated pictures, so the earlier frame is repeated.
                let cut = cutDetector.isCut(difference: try sceneDifference(previous.frame, current))
                for step in 1 ..< options.times {
                    let t = Float(step) / Float(options.times)
                    let mid: MLXArray
                    if cut {
                        mid = previous.frame
                    } else {
                        mid = try RIFEEngine.interpolate(loaded, previous.frame, current, timestep: t, scale: options.scale)
                        try MLXErrorGuard.evaluate(mid)
                    }
                    try await append(mid, at: insertedTime(from: previous.time, to: time, step: step, of: options.times))
                }
                progress(min(0.98, Double(sourceIndex - 1) / Double(expectedPairs)))
            }
            previous = (current, time)
        }
        guard let last = previous, sourceIndex >= 2 else { throw RIFEVideoError.noFrames }
        try await append(last.frame, at: last.time)
        videoIn.markAsFinished()

        while audioIn != nil, !audioFinished {
            try Task.checkCancellation()
            if writer.status == .failed { throw RIFEVideoError.writerFailed(writer.error?.localizedDescription ?? "unknown") }
            try pumpAudio(upTo: nil)
            if !audioFinished { try await Task.sleep(for: .milliseconds(2)) }
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
