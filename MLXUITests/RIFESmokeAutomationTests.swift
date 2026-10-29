import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import MLX
import Testing
@testable import MLXUI

/// RIFE-4 / RIFE-5: smoke rows 137 and 138, automated as far as a test can go. Gated on **both** `MLXUI_TEST_RIFE_DIR` (the weights)
/// and `MLXUI_TEST_RIFE_SAMPLES` (the folder of `RSI/notes/claude-code/rife/rife-smoke/` clips, made by `make_samples.sh`); a skipped
/// run does not count. It drives `RIFERunModel` (the Run screen's logic) and the real stage on those clips, and checks the saved
/// files frame by frame. What it cannot do: look at the SwiftUI window, play the preview, or click Save (a modal panel).
enum RIFESmoke {
    nonisolated static let samples: URL? = {
        guard let path = ProcessInfo.processInfo.environment["MLXUI_TEST_RIFE_SAMPLES"] else { return nil }
        return URL(fileURLWithPath: path)
    }()
    nonisolated static let enabled = RIFERealLoad.enabled && samples != nil
    nonisolated static func clip(_ name: String) throws -> URL {
        let dir = try #require(samples)
        let url = dir.appendingPathComponent(name)
        try #require(FileManager.default.fileExists(atPath: url.path), "missing sample \(name)")
        return url
    }

    struct Frame { let time: Double; let center: (x: Double, y: Double)? }

    /// Every frame's presentation time and the centre of the white square (the largest cluster of near-white sample points).
    nonisolated static func frames(of url: URL, step: Int = 6) async throws -> [Frame] {
        let asset = AVURLAsset(url: url)
        let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
        let reader = try AVAssetReader(asset: asset)
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        reader.add(out)
        reader.startReading()
        var frames: [Frame] = []
        while let sample = out.copyNextSampleBuffer() {
            guard let buffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            frames.append(Frame(time: CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample)), center: squareCenter(buffer, step: step)))
        }
        return frames
    }

    nonisolated static func squareCenter(_ buffer: CVPixelBuffer, step: Int) -> (x: Double, y: Double)? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer), stride = CVPixelBufferGetBytesPerRow(buffer)
        let cols = w / step, rows = h / step
        var white = [Bool](repeating: false, count: cols * rows)
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        for r in 0 ..< rows {
            for c in 0 ..< cols {
                let p = bytes + (r * step) * stride + (c * step) * 4
                white[r * cols + c] = p[0] >= 235 && p[1] >= 235 && p[2] >= 235
            }
        }
        var seen = [Bool](repeating: false, count: cols * rows)
        var best: [Int] = []
        for start in 0 ..< white.count where white[start] && !seen[start] {
            var stack = [start], cluster: [Int] = []
            seen[start] = true
            while let i = stack.popLast() {
                cluster.append(i)
                let r = i / cols, c = i % cols
                for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let nr = r + dr, nc = c + dc
                    guard nr >= 0, nr < rows, nc >= 0, nc < cols else { continue }
                    let j = nr * cols + nc
                    if white[j], !seen[j] { seen[j] = true; stack.append(j) }
                }
            }
            if cluster.count > best.count { best = cluster }
        }
        guard best.count >= 20 else { return nil }
        let sx = best.reduce(0) { $0 + ($1 % cols) }, sy = best.reduce(0) { $0 + ($1 / cols) }
        return (Double(sx) / Double(best.count) * Double(step), Double(sy) / Double(best.count) * Double(step))
    }

    /// Where `make_samples.sh` puts the square's centre at time `t` (it is `overlay`'s own expression).
    nonisolated static func expectedCenter(t: Double, width w: Double, height h: Double) -> (x: Double, y: Double) {
        let b = Double(Int(h) / 9)
        return (0.5 * w + 0.30 * w * sin(2 * Double.pi * t / 5), 0.80 * h + b / 2 - 0.55 * h * abs(sin(Double.pi * t)))
    }

    /// Decoded mono PCM of the first audio track, or nil for a silent file.
    nonisolated static func pcm(of url: URL) async throws -> [Float]? {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else { return nil }
        let reader = try AVAssetReader(asset: asset)
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false, AVLinearPCMIsNonInterleaved: false, AVNumberOfChannelsKey: 1, AVSampleRateKey: 44100,
        ])
        reader.add(out)
        reader.startReading()
        var samples: [Float] = []
        while let sample = out.copyNextSampleBuffer(), let block = CMSampleBufferGetDataBuffer(sample) {
            var length = 0
            var pointer: UnsafeMutablePointer<Int8>?
            CMBlockBufferGetDataPointer(block, atOffset: 0, lengthAtOffsetOut: nil, totalLengthOut: &length, dataPointerOut: &pointer)
            if let pointer {
                pointer.withMemoryRebound(to: Float.self, capacity: length / 4) { samples.append(contentsOf: UnsafeBufferPointer(start: $0, count: length / 4)) }
            }
        }
        return samples
    }

    /// Drives the Run screen's model the way the buttons do and waits for the run to end.
    @MainActor static func waitUntil(timeout: Double, _ condition: () -> Bool) async -> Bool {
        let end = ContinuousClock.now + .seconds(timeout)
        while ContinuousClock.now < end {
            if condition() { return true }
            try? await Task.sleep(for: .milliseconds(20))
        }
        return condition()
    }
}

@Suite(.enabled(if: RIFESmoke.enabled, "needs MLXUI_TEST_RIFE_DIR and MLXUI_TEST_RIFE_SAMPLES"), .serialized)
@MainActor
struct RIFESmokeRow137Tests {

    private func choose(_ name: String) async throws -> RIFERunModel {
        let model = RIFERunModel()
        model.choose(try RIFESmoke.clip(name))
        let settled = await RIFESmoke.waitUntil(timeout: 15) { model.info != nil || model.limitMessage != nil }
        #expect(settled, "the file was never inspected")
        return model
    }

    private func run(_ model: RIFERunModel, times: Int, scale: Float) async throws -> URL {
        model.times = times
        model.scale = scale
        let out = RIFEStage.temporaryOutputURL()
        let stage = RIFEStage(modelDirectory: try #require(RIFERealLoad.directory), options: model.options, outputURL: out)
        model.start(stage: stage, output: out)
        let finished = await RIFESmoke.waitUntil(timeout: 600) { !model.isRunning }
        #expect(finished, "the run never finished")
        return out
    }

    // Steps 1, 2: the summary line, the Output line, the arithmetic.
    @Test func chooseShowsTheSummaryAndTheOutputArithmetic() async throws {
        let m720 = try await choose("01-720p-30fps-12s.mp4")
        let info = try #require(m720.info)
        #expect(m720.sourceSummary(info) == "1280 × 720 · 30 fps · 12.0 s · audio")
        #expect(info.outputFrames(times: 2) == 719 && abs(info.outputFrameRate(times: 2) - 60) < 0.01)
        #expect(info.outputFrames(times: 4) == 1437 && abs(info.outputFrameRate(times: 4) - 120) < 0.01)
        #expect(m720.limitMessage == nil && m720.canRun)
        let m1080 = try await choose("02-1080p-24fps-8s.mp4")
        #expect(m1080.sourceSummary(try #require(m1080.info)) == "1920 × 1080 · 24 fps · 8.0 s · audio")
    }

    // Steps 3-5: a 2x run; the file; the audio; the picture.
    @Test func twoTimesRunProducesTheRightFileAudioAndPicture() async throws {
        let model = try await choose("01-720p-30fps-12s.mp4")
        let out = try await run(model, times: 2, scale: 1.0)
        defer { try? FileManager.default.removeItem(at: out) }
        #expect(model.errorText == nil, "error bar: \(model.errorText ?? "")")
        #expect(model.hasResult)
        #expect(model.resultSummary == "Done: 719 frames at 60 fps")
        #expect(model.progress == 1)
        // The saved file (Save Video… copies this file byte for byte).
        let info = try await RIFEVideoPipeline.inspect(out)
        #expect(abs(info.frameRate - 60) < 0.5 && info.hasAudio)
        #expect(abs(info.duration - 12) < 0.1)
        let frames = try await RIFESmoke.frames(of: out)
        #expect(frames.count == 719)
        // Audio: identical samples, so identical beeps.
        let source = try #require(try await RIFESmoke.pcm(of: RIFESmoke.clip("01-720p-30fps-12s.mp4")))
        let result = try #require(try await RIFESmoke.pcm(of: out))
        #expect(source.count == result.count, "audio length \(source.count) vs \(result.count)")
        let drift = zip(source, result).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        #expect(drift == 0, "decoded audio differs by \(drift)")
        // Picture: the square follows its own analytic path, and the inserted frames sit between the originals.
        let w = 1280.0, h = 720.0
        var originals: [Double] = [], inserted: [Double] = []
        for (i, frame) in frames.enumerated() {
            guard let c = frame.center else { Issue.record("no square in frame \(i) at \(frame.time)"); continue }
            let e = RIFESmoke.expectedCenter(t: frame.time, width: w, height: h)
            let error = hypot(c.x - e.x, c.y - e.y)
            if i % 2 == 0 { originals.append(error) } else { inserted.append(error) }
        }
        func stats(_ v: [Double]) -> (median: Double, p90: Double, max: Double) {
            let s = v.sorted(); return (s[s.count / 2], s[Int(Double(s.count) * 0.9)], s.last ?? 0)
        }
        let o = stats(originals), n = stats(inserted)
        print(String(format: "RIFESMOKE 137 picture error (px, box 80): originals median %.1f p90 %.1f max %.1f | inserted median %.1f p90 %.1f max %.1f",
                     o.median, o.p90, o.max, n.median, n.p90, n.max))
        // Measured on the 720p clip: originals median 3.3 px (H.264 re-encode plus box centring), so 3 was too tight a guess.
        #expect(o.median < 4, "the detector is miscalibrated: originals should sit on the path")
        #expect(n.median < 6 && n.p90 < 20, "inserted frames wander off the path: median \(n.median) p90 \(n.p90)")
        #expect(frames.first?.time == 0)
    }

    // RIFE-5b: the stage itself refuses Scale 1.0 on the 4K clip, before any frame is read.
    @Test func theStageRefusesFullScaleOnThe4KClip() async throws {
        let out = RIFEStage.temporaryOutputURL()
        let stage = RIFEStage(modelDirectory: try #require(RIFERealLoad.directory), options: RIFEVideoOptions(times: 2, scale: 1.0), outputURL: out)
        do {
            _ = try await stage.run(.text(try RIFESmoke.clip("03-4k-30fps-6s.mp4").path), progress: { _ in })
            Issue.record("Scale 1.0 on 4K should have been refused")
        } catch let error as RIFEVideoError {
            #expect(error == .fullScaleTooLarge(width: 3840, height: 2160))
        }
        #expect(!FileManager.default.fileExists(atPath: out.path))
    }

    // Step 6.
    @Test func fourTimesAtHalfScaleGivesQuadrupleFps() async throws {
        let model = try await choose("01-720p-30fps-12s.mp4")
        let out = try await run(model, times: 4, scale: 0.5)
        defer { try? FileManager.default.removeItem(at: out) }
        #expect(model.errorText == nil, "error bar: \(model.errorText ?? "")")
        #expect(model.resultSummary == "Done: 1437 frames at 120 fps")
        let info = try await RIFEVideoPipeline.inspect(out)
        #expect(abs(info.frameRate - 120) < 1 && abs(info.duration - 12) < 0.1 && info.hasAudio)
        #expect(try await RIFESmoke.frames(of: out).count == 1437)
    }

    // Step 7.
    @Test func cancellingMidRunShowsNoErrorNoResultAndLeavesNoFile() async throws {
        let model = try await choose("02-1080p-24fps-8s.mp4")
        model.times = 2
        let out = RIFEStage.temporaryOutputURL()
        model.start(stage: RIFEStage(modelDirectory: try #require(RIFERealLoad.directory), options: model.options, outputURL: out), output: out)
        let started = await RIFESmoke.waitUntil(timeout: 120) { model.progress >= 0.2 }
        #expect(started, "never reached 20% (progress \(model.progress))")
        model.cancel()
        #expect(await RIFESmoke.waitUntil(timeout: 60) { !model.isRunning }, "Cancel never ended the run")
        #expect(model.errorText == nil, "a cancelled run must not show an error: \(model.errorText ?? "")")
        #expect(!model.hasResult && model.player == nil)
        #expect(model.progress == 0)
        #expect(!FileManager.default.fileExists(atPath: out.path))
        #expect(model.canRun, "after Cancel the user can run again")
    }

    // Step 8.
    @Test func theLongClipIsRefusedWithAPlainSentenceAndRunIsDisabled() async throws {
        let model = try await choose("04-too-long-31min.mp4")
        let message = try #require(model.limitMessage)
        print("RIFESMOKE 137 long clip message: \(message)")
        #expect(message.contains("54000") && message.contains("about 30 minutes at 30 fps"))
        #expect(!model.canRun)
    }

    // Step 9.
    @Test func aNonVideoIsRefusedWithAPlainSentence() async throws {
        let model = try await choose("06-not-a-video.mp4")
        let message = try #require(model.limitMessage)
        print("RIFESMOKE 137 not-a-video message: \(message)")
        #expect(!message.isEmpty && !model.canRun)
        #expect(!message.contains("Error Domain") && !message.contains("AVFoundation"), "raw framework text reached the user: \(message)")
    }

    // Step 10.
    @Test func aSilentClipRunsAndHasNoAudio() async throws {
        let model = try await choose("05-1080p-30fps-silent-6s.mp4")
        let info = try #require(model.info)
        #expect(!info.hasAudio && !model.sourceSummary(info).contains("audio"))
        let out = try await run(model, times: 2, scale: 1.0)
        defer { try? FileManager.default.removeItem(at: out) }
        #expect(model.errorText == nil)
        let produced = try await RIFEVideoPipeline.inspect(out)
        #expect(!produced.hasAudio && abs(produced.frameRate - 60) < 0.5)
    }

    // Step 3's "the window stays responsive": the longest main-actor stall while a 1080p run is going.
    @Test func theMainActorNeverStallsDuringA1080pRun() async throws {
        let model = try await choose("02-1080p-24fps-8s.mp4")
        model.times = 2
        let out = RIFEStage.temporaryOutputURL()
        defer { try? FileManager.default.removeItem(at: out) }
        model.start(stage: RIFEStage(modelDirectory: try #require(RIFERealLoad.directory), options: model.options, outputURL: out), output: out)
        var worst = 0.0
        var last = ContinuousClock.now
        while model.isRunning {
            try await Task.sleep(for: .milliseconds(10))
            let now = ContinuousClock.now
            let gap = Double((now - last).components.seconds) + Double((now - last).components.attoseconds) / 1e18
            worst = max(worst, gap)
            last = now
        }
        print(String(format: "RIFESMOKE 137 longest main-actor gap during a 1080p run: %.3f s (ticker asked for 0.010)", worst))
        #expect(worst < 0.5, "the main actor stalled for \(worst) s")
        #expect(model.errorText == nil)
    }
}

/// Row 138 (RIFE-5): peak `phys_footprint` through the real stage. One case per process, so the peak is that case's own:
/// `MLXUI_TEST_RIFE_MEMORY=A|B|C|D|E|F|G` selects it (A 1080p 2x s1.0, B 4K 2x s0.5, C 4K 2x s1.0, D 1080p 4x s1.0). A test host is smaller than
/// the app (no windows, no model list), so the figures are a floor for the app's Activity Monitor number, not above it.
@Suite(.enabled(if: RIFESmoke.enabled && ProcessInfo.processInfo.environment["MLXUI_TEST_RIFE_MEMORY"] != nil, "needs MLXUI_TEST_RIFE_MEMORY=A|B|C|D|E|F|G"))
struct RIFEMemoryMeasurementTests {
    @Test func measure() async throws {
        let which = try #require(ProcessInfo.processInfo.environment["MLXUI_TEST_RIFE_MEMORY"])
        let cases: [String: (clip: String, times: Int, scale: Float, label: String)] = [
            "A": ("02-1080p-24fps-8s.mp4", 2, 1.0, "1080p 2x scale 1.0"),
            "B": ("03-4k-30fps-6s.mp4", 2, 0.5, "4K 2x scale 0.5"),
            "C": ("03-4k-30fps-6s.mp4", 2, 1.0, "4K 2x scale 1.0"),
            "D": ("02-1080p-24fps-8s.mp4", 4, 1.0, "1080p 4x scale 1.0"),
            "F": ("07-1440p-30fps-6s.mp4", 2, 1.0, "1440p 2x scale 1.0"),
            "G": ("08-1080p-30fps-5min.mp4", 2, 1.0, "1080p 5 min 2x scale 1.0"),
            "E": ("03-4k-30fps-6s.mp4", 4, 0.5, "4K 4x scale 0.5"),
        ]
        let c = try #require(cases[which])
        let src = try RIFESmoke.clip(c.clip)
        let out = RIFEStage.temporaryOutputURL()
        defer { try? FileManager.default.removeItem(at: out) }
        let tracker = FootprintTracker()
        let atPeak = PeakSnapshot()
        let started0 = Date()
        nonisolated(unsafe) var lastPrint = Date()
        let sampler = Task.detached {
            while !Task.isCancelled {
                let v = FootprintTracker.phys()
                tracker.note(v)
                atPeak.note(v, active: Memory.activeMemory, cache: Memory.cacheMemory)
                if Date().timeIntervalSince(lastPrint) >= 120 {   // a trend line for long runs: does anything grow?
                    lastPrint = Date()
                    print(String(format: "RIFEMEM %@ trend +%.0f s: footprint %.0f MiB, MLX active %.0f, cache %.0f", which,
                                 Date().timeIntervalSince(started0), Double(v) / 1_048_576, Double(Memory.activeMemory) / 1_048_576, Double(Memory.cacheMemory) / 1_048_576))
                }
                try? await Task.sleep(for: .milliseconds(20))
            }
        }
        let baseline = FootprintTracker.phys()
        tracker.note(baseline)
        Memory.peakMemory = 0
        let started = Date()
        let stage = RIFEStage(modelDirectory: try #require(RIFERealLoad.directory),
                              options: RIFEVideoOptions(times: c.times, scale: c.scale), outputURL: out)
        var failure: String?
        do { _ = try await stage.run(.text(src.path), progress: { _ in }) } catch { failure = MLXErrorGuard.sentence(for: error) }
        let wall = Date().timeIntervalSince(started)
        sampler.cancel()
        let peak = tracker.runPeak
        let info = failure == nil ? try? await RIFEVideoPipeline.inspect(out) : nil
        print(String(format: "RIFEMEM %@ (%@): %@ in %.1f s; peak phys_footprint %.0f MiB (%.2f GiB), baseline before the run %.0f MiB, added %.0f MiB; MLX peak %.0f MiB; output %@",
                     which, c.label, failure ?? "finished", wall, Double(peak) / 1_048_576, Double(peak) / 1_073_741_824,
                     Double(baseline) / 1_048_576, Double(peak - baseline) / 1_048_576, Double(Memory.peakMemory) / 1_048_576,
                     info.map { "\($0.width)x\($0.height) \(Int($0.frameRate)) fps" } ?? "none"))
        let mib = 1_048_576.0
        print(String(format: "RIFEMEM %@ at the footprint peak: MLX active %.0f MiB, MLX cache %.0f MiB, so outside MLX %.0f MiB",
                     which, Double(atPeak.active) / mib, Double(atPeak.cache) / mib,
                     Double(peak) / mib - Double(atPeak.active + atPeak.cache) / mib))
        #expect(failure == nil, "the run failed: \(failure ?? "")")
    }
}

/// MLX active and cache memory at the instant `phys_footprint` was highest.
final class PeakSnapshot: @unchecked Sendable {
    private let lock = NSLock()
    private var best: Int64 = 0
    private(set) var active = 0, cache = 0
    func note(_ v: Int64, active a: Int, cache c: Int) {
        lock.lock(); defer { lock.unlock() }
        if v > best { best = v; active = a; cache = c }
    }
}


// RIFE-L: scene cuts. The first test measures the neighbouring-frame difference on every sample (the threshold's evidence); the second
// runs the real model over a clip with a hard cut and checks that nothing is drawn across it.
@Suite(.enabled(if: RIFESmoke.samples != nil, "needs MLXUI_TEST_RIFE_SAMPLES"), .serialized)
struct RIFESceneCutTests {

    nonisolated static func differences(of url: URL) async throws -> [Float] {
        let asset = AVURLAsset(url: url)
        let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
        let reader = try AVAssetReader(asset: asset)
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        reader.add(out)
        reader.startReading()
        var previous: MLXArray?
        var result: [Float] = []
        while let sample = out.copyNextSampleBuffer(), let pixels = CMSampleBufferGetImageBuffer(sample) {
            let frame = RIFEVideoPipeline.read(pixels)
            try MLXErrorGuard.evaluate(frame)
            if let previous { result.append(try RIFEVideoPipeline.sceneDifference(previous, frame)) }
            previous = frame
        }
        return result
    }

    @Test func measureTheThresholdOnTheSamples() async throws {
        for name in ["01-720p-30fps-12s.mp4", "02-1080p-24fps-8s.mp4", "03-4k-30fps-6s.mp4", "05-1080p-30fps-silent-6s.mp4"] {
            let d = try await Self.differences(of: try RIFESmoke.clip(name))
            let s = d.sorted()
            print(String(format: "RIFECUT %@: %d pairs, median %.4f, p99 %.4f, max %.4f", name, d.count, s[s.count / 2], s[Int(Double(s.count) * 0.99)], s.last ?? 0))
            #expect((s.last ?? 1) < RIFEVideoPipeline.sceneCutThreshold, "\(name): ordinary motion must never read as a cut")
        }
        let cut = try await Self.differences(of: try RIFESmoke.clip("09-hard-cut-640x360-30fps-4s.mp4"))
        let sorted = cut.enumerated().sorted { $0.element > $1.element }
        print(String(format: "RIFECUT hard-cut clip: cut pair %d = %.4f; next largest pair %d = %.4f; threshold %.2f",
                     sorted[0].offset, sorted[0].element, sorted[1].offset, sorted[1].element, RIFEVideoPipeline.sceneCutThreshold))
        #expect(sorted[0].offset == 59, "the cut is between source frames 59 and 60")
        #expect(sorted[0].element > RIFEVideoPipeline.sceneCutThreshold)
        #expect(sorted[1].element < RIFEVideoPipeline.sceneCutThreshold, "only the cut pair crosses the threshold")
    }

    /// Mean absolute pixel difference (0...255) between two frames of the output, every 8th pixel.
    nonisolated static func thumbnails(of url: URL) async throws -> [[UInt8]] {
        let asset = AVURLAsset(url: url)
        let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
        let reader = try AVAssetReader(asset: asset)
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        reader.add(out)
        reader.startReading()
        var frames: [[UInt8]] = []
        while let sample = out.copyNextSampleBuffer(), let buffer = CMSampleBufferGetImageBuffer(sample) {
            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
            guard let base = CVPixelBufferGetBaseAddress(buffer) else { continue }
            let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer), stride = CVPixelBufferGetBytesPerRow(buffer)
            let bytes = base.assumingMemoryBound(to: UInt8.self)
            var thumb: [UInt8] = []
            for y in Swift.stride(from: 0, to: h, by: 8) { for x in Swift.stride(from: 0, to: w, by: 8) {
                let p = bytes + y * stride + x * 4
                thumb.append(contentsOf: [p[0], p[1], p[2]])
            } }
            frames.append(thumb)
        }
        return frames
    }

    nonisolated static func meanDiff(_ a: [UInt8], _ b: [UInt8]) -> Double {
        zip(a, b).reduce(0.0) { $0 + abs(Double($1.0) - Double($1.1)) } / Double(max(1, a.count))
    }

    @Test(.enabled(if: RIFERealLoad.enabled, "needs the model")) func aHardCutIsRepeatedNotInterpolated() async throws {
        let out = RIFEStage.temporaryOutputURL()
        defer { try? FileManager.default.removeItem(at: out) }
        _ = try await RIFEStage(modelDirectory: try #require(RIFERealLoad.directory), options: RIFEVideoOptions(times: 2, scale: 1), outputURL: out)
            .run(.text(try RIFESmoke.clip("09-hard-cut-640x360-30fps-4s.mp4").path), progress: { _ in })
        let frames = try await Self.thumbnails(of: out)
        #expect(frames.count == 239)
        // Output index 2n is source frame n: source 59 is output 118, source 60 is output 120, and the frame inserted across the cut is 119.
        let before = frames[118], inserted = frames[119], after = frames[120]
        let toEarlier = Self.meanDiff(inserted, before), toLater = Self.meanDiff(inserted, after)
        print(String(format: "RIFECUT inserted frame: %.2f from the earlier frame, %.2f from the later one (0-255 scale)", toEarlier, toLater))
        #expect(toEarlier < 3, "the frame at the cut repeats the earlier frame")
        #expect(toLater > 30, "and is nothing like the later one: no ghost of both pictures")
        // An ordinary pair still gets a real in-between: it differs from both neighbours.
        let ordinary = Self.meanDiff(frames[41], frames[40])
        #expect(ordinary > 0, "frames away from the cut are interpolated, not repeated")
    }
}
