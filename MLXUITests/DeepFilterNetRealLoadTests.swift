import Foundation
import MLX
import Testing
@testable import MLXUI

/// MP5-3b -- the real-weights tests for DeepFilterNet 3 (KICKOFF rule 13b, rule 17c, the memory harness). Owner/lead-run, gated
/// on the weights (about 26 MB for v1 to v3): skipped unless `mlx-community--DeepFilterNet-mlx/.installed` exists under the
/// directory this process resolves, `ModelStore`'s or `MLXUI_TEST_DEEPFILTERNET_DIR=<model folder>` (passed to the host as
/// `TEST_RUNNER_MLXUI_TEST_DEEPFILTERNET_DIR`). The exact commands are in the MP5-3b journal. A skipped run does not count.
///
/// Every test calls `DeepFilterNetEngine.enhance` (directly, or through `DeepFilterNetStage(modelDirectory:)`, which is what the Run
/// UI builds with `init(modelID:)`), on a temp copy of the model folder.
enum DeepFilterNetRealLoad {
    nonisolated static let env = ProcessInfo.processInfo.environment

    nonisolated static let directory: URL? = {
        let url: URL
        if let override = env["MLXUI_TEST_DEEPFILTERNET_DIR"] {
            url = URL(fileURLWithPath: override)
        } else {
            url = ModelStore.shared.directory(forHFModelID: "mlx-community/DeepFilterNet-mlx")
        }
        return FileManager.default.fileExists(atPath: url.appendingPathComponent(".installed").path) ? url : nil
    }()

    /// A copy of the model folder in a fresh temp directory (the Qwen3-ASR / Sortformer precedent). The source is never touched.
    struct WorkingCopy {
        nonisolated let url: URL
        nonisolated let root: URL?
        nonisolated func remove() { if let root { try? FileManager.default.removeItem(at: root) } }
    }

    nonisolated static func workingCopy() throws -> WorkingCopy {
        let source = try #require(directory)
        guard env["MLXUI_TEST_DEEPFILTERNET_DIR"] != nil else { return WorkingCopy(url: source, root: nil) }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("deepfilternet-model-\(UUID().uuidString)")
        let copy = root.appendingPathComponent(source.lastPathComponent)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        do { try FileManager.default.copyItem(at: source, to: copy) } catch {
            try? FileManager.default.removeItem(at: root)
            throw error
        }
        return WorkingCopy(url: copy, root: root)
    }

    // MARK: Thresholds -- PROPOSED from the first real run

    // First real run, 2026-10-25 (lead, MLXUI scheme, journal 2026-597/598), parity clip mixed at 5 dB, chunked 8 s / 1 s unless noted:
    // noisy: snrOut 18.13 dB, improvement 13.13 dB, rmsRatio 0.958; 44.1 kHz input 12.96 dB / 0.956; 16 kHz input 10.38 dB / 0.949;
    // clean input: snrOutVsInput 32.51 dB, rmsRatio 0.981; chunk sweep: gain within 0.07 dB of the whole file at every setting.
    // Each constant below sits well under its measurement. CONFIRMED by the owner (Boo, 2026-10-25): "Confirm the test thresholds as
    // proposed: SNR gain >= 8 dB (>= 6 dB for 16 kHz input), output/clean RMS 0.85-1.10, clean-input SNR >= 25 dB with RMS 0.90-1.05."
    // The chunked-vs-whole slack (1 dB) was proposed with them and is the implementer's call, pending owner confirmation.

    /// Owner-confirmed 2026-10-25 (measured 13.13 dB). Minimum SNR gain (dB) on the parity clip.
    nonisolated static let snrImprovementThresholdDB = 8.0
    /// Owner-confirmed 2026-10-25 (measured 10.38 dB). Minimum SNR gain for the 16 kHz input.
    nonisolated static let snrImprovementThreshold16kDB = 6.0
    /// Owner-confirmed 2026-10-25 (measured 0.949 to 0.958). RMS of the cleaned clip over the clean reference's.
    nonisolated static let rmsRatioRange = 0.85...1.10
    /// Owner-confirmed 2026-10-25 (measured 32.51 dB). Clean speech: SNR of output against input (dB).
    nonisolated static let cleanInputMinSNRDB = 25.0
    /// Owner-confirmed 2026-10-25 (measured 0.981). Clean speech: RMS of output over input.
    nonisolated static let cleanInputRMSRatioRange = 0.90...1.05
    /// Implementer's call, pending owner confirmation (measured at most 0.07 dB). How far below the whole-file gain a chunked run may fall (dB).
    nonisolated static let chunkedImprovementSlackDB = 1.0
    /// The SNR the parity clip is mixed at (dB). Implementer's call, pending owner confirmation.
    nonisolated static let mixSNRDB = 5.0
    nonisolated static let noiseSeed: UInt64 = 20_261_025

    // MARK: Parity clip (generated, never checked in)

    /// A ~25 s passage; `say` reads it at roughly 170 words a minute.
    nonisolated static let passage = """
    The harbour was quiet before sunrise. A few fishing boats rocked against their ropes, and the only sound was the slow \
    knock of wood on wood. Maria walked down the stone steps with a thermos of coffee and a small notebook. She had promised \
    herself that this year she would write down everything the old fishermen said, every proverb, every warning about the \
    weather, every story about the storms that never made the newspapers.
    """

    struct Clip {
        nonisolated let clean: [Float]   // 48 kHz mono, RMS normalised to 0.1
        nonisolated let noisy: [Float]   // clean + seeded broadband noise and a 50 Hz hum at `mixSNRDB`
        nonisolated var seconds: Double { Double(clean.count) / 48_000 }
    }

    enum ClipError: Error, CustomStringConvertible {
        case sayUnavailable(String)
        var description: String {
            switch self { case .sayUnavailable(let why): return "the parity clip could not be generated: \(why)" }
        }
    }

    /// The clean 48 kHz reference: `MLXUI_TEST_DEEPFILTERNET_CLEAN_WAV` (a clean speech file you made yourself) when set, else a
    /// macOS `say` voice (`MLXUI_TEST_DEEPFILTERNET_VOICE`, default Samantha, falling back to the system voice), cached in the temp
    /// folder. Throws `ClipError` when neither works (the tests report that and return; a skip, not a failure).
    nonisolated static func cleanReference() throws -> [Float] {
        let buffer: AudioBuffer
        if let path = env["MLXUI_TEST_DEEPFILTERNET_CLEAN_WAV"] {
            buffer = try AudioFileReader.read(URL(fileURLWithPath: path))
        } else {
            let cache = FileManager.default.temporaryDirectory.appendingPathComponent("dfn-parity-\(abs(passage.hashValue)).wav")
            if !FileManager.default.fileExists(atPath: cache.path) { try say(passage, to: cache) }
            buffer = try AudioFileReader.read(cache)
        }
        var samples = try AudioResampler.resample(buffer, to: 48_000).samples
        let level = AudioMetrics.rms(samples)
        guard level > 0 else { throw ClipError.sayUnavailable("the reference is silent") }
        let gain = Float(0.1 / level)
        samples = samples.map { $0 * gain }
        return samples
    }

    nonisolated private static func say(_ text: String, to url: URL) throws {
        let voices = [env["MLXUI_TEST_DEEPFILTERNET_VOICE"] ?? "Samantha", nil]
        var lastFailure = "no `say` output"
        for voice in voices {
            guard FileManager.default.isExecutableFile(atPath: "/usr/bin/say") else { throw ClipError.sayUnavailable("/usr/bin/say is missing") }
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
            process.arguments = (voice.map { ["-v", $0] } ?? []) + ["--file-format=WAVE", "--data-format=LEI16@48000", "-o", url.path, text]
            let errors = Pipe()
            process.standardError = errors
            do { try process.run() } catch { lastFailure = "\(error)"; continue }
            process.waitUntilExit()
            let message = String(decoding: errors.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            if process.terminationStatus == 0, (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int).flatMap({ $0 }) ?? 0 > 44 { return }
            lastFailure = "say exited \(process.terminationStatus): \(message)"
        }
        throw ClipError.sayUnavailable(lastFailure)
    }

    nonisolated static func parityClip() throws -> Clip {
        let clean = try cleanReference()
        let noise = AudioMetrics.noise(count: clean.count, sampleRate: 48_000, seed: noiseSeed)
        return Clip(clean: clean, noisy: AudioMetrics.mix(reference: clean, noise: noise, snrDB: mixSNRDB))
    }

    /// Runs `body` with the clip, or prints why it can't and returns (a skip with a message, not a failure).
    nonisolated static func withClip(_ body: (Clip) async throws -> Void) async throws {
        let clip: Clip
        do { clip = try parityClip() } catch let error as ClipError {
            print("DFNSKIP \(error)")
            return
        }
        try await body(clip)
    }

    nonisolated static func finite(_ samples: [Float]) -> Bool { !samples.contains { !$0.isFinite } }

    /// One greppable line: `DFNPARITY <label> key=value ...`.
    nonisolated static func report(_ label: String, noisy: [Float], cleaned: [Float], clean: [Float]) -> (improvement: Double, rmsRatio: Double) {
        let snrIn = AudioMetrics.snrDB(estimate: noisy, reference: clean)
        let snrOut = AudioMetrics.snrDB(estimate: cleaned, reference: clean)
        let ratio = AudioMetrics.rms(cleaned) / AudioMetrics.rms(clean)
        let lag = AudioMetrics.bestLag(estimate: cleaned, reference: clean, maxLag: 960, window: 192_000)
        print(String(format: "DFNPARITY %@ snrIn=%.2f dB snrOut=%.2f dB improvement=%.2f dB rmsRatio=%.3f lag=%d samples seconds=%.1f",
                     label, snrIn, snrOut, snrOut - snrIn, ratio, lag, Double(clean.count) / 48_000))
        return (snrOut - snrIn, ratio)
    }
}

struct DeepFilterNetRealLoadTests {
    private static let enabled = DeepFilterNetRealLoad.directory != nil

    /// Rule 13b: the real v3 checkpoint loads through the app's engine path and cleans the parity clip. Length preserved, finite,
    /// not silent. Success proves the files `InstallManager` fetched (v1, v2, v3 each with config + weights) cover the loader.
    @Test(.enabled(if: enabled))
    func theCheckpointLoadsAndCleansTheParityClip() async throws {
        let copy = try DeepFilterNetRealLoad.workingCopy()
        defer { copy.remove() }
        try await DeepFilterNetRealLoad.withClip { clip in
            Memory.clearCache()
            Memory.peakMemory = 0
            let started = Date()
            let cleaned = try await DeepFilterNetEngine.enhance(clip.noisy, modelDirectory: copy.url)
            print(String(format: "DFNLOAD %.1f s for %.1f s of audio, MLX peak %.3f GiB", Date().timeIntervalSince(started), clip.seconds,
                         Double(Memory.snapshot().peakMemory) / 1_073_741_824))
            #expect(cleaned.count == clip.noisy.count)
            #expect(DeepFilterNetRealLoad.finite(cleaned))
            #expect(AudioMetrics.rms(cleaned) > 1e-3, "the output is silent")
            #expect(FileManager.default.fileExists(atPath: copy.url.appendingPathComponent(".installed").path))
        }
    }

    /// The parity check (owner ruling 2026-10-25): SNR improvement against the clean reference and the RMS ratio, with a
    /// tolerance, never bit for bit. Thresholds are PLACEHOLDERS in `DeepFilterNetRealLoad` until the first real run.
    @Test(.enabled(if: enabled))
    func noisySpeechGainsSNRAndKeepsItsLevel() async throws {
        let copy = try DeepFilterNetRealLoad.workingCopy()
        defer { copy.remove() }
        try await DeepFilterNetRealLoad.withClip { clip in
            let cleaned = try await DeepFilterNetEngine.enhance(clip.noisy, modelDirectory: copy.url)
            let r = DeepFilterNetRealLoad.report("noisy-5dB", noisy: clip.noisy, cleaned: cleaned, clean: clip.clean)
            #expect(r.improvement >= DeepFilterNetRealLoad.snrImprovementThresholdDB)
            #expect(DeepFilterNetRealLoad.rmsRatioRange.contains(r.rmsRatio))
        }
    }

    /// Clean speech comes out nearly unchanged.
    @Test(.enabled(if: enabled))
    func cleanSpeechComesOutNearlyUnchanged() async throws {
        let copy = try DeepFilterNetRealLoad.workingCopy()
        defer { copy.remove() }
        try await DeepFilterNetRealLoad.withClip { clip in
            let cleaned = try await DeepFilterNetEngine.enhance(clip.clean, modelDirectory: copy.url)
            let snr = AudioMetrics.snrDB(estimate: cleaned, reference: clip.clean)
            let ratio = AudioMetrics.rms(cleaned) / AudioMetrics.rms(clip.clean)
            let lag = AudioMetrics.bestLag(estimate: cleaned, reference: clip.clean, maxLag: 960, window: 192_000)
            print(String(format: "DFNCLEAN snrOutVsInput=%.2f dB rmsRatio=%.3f lag=%d samples", snr, ratio, lag))
            #expect(cleaned.count == clip.clean.count)
            #expect(snr >= DeepFilterNetRealLoad.cleanInputMinSNRDB)
            #expect(DeepFilterNetRealLoad.cleanInputRMSRatioRange.contains(ratio))
        }
    }

    /// 44.1 kHz and 16 kHz inputs through the real resample path (the stage), the stage the Run UI builds: output is 48 kHz and
    /// about as long as the input. The reference is compared at 48 kHz; a 16 kHz input has lost everything above 8 kHz, so its
    /// SNR gain threshold is lower.
    @Test(.enabled(if: enabled), arguments: [44_100, 16_000])
    func otherInputRatesComeOutAt48kHz(inputRate: Int) async throws {
        let copy = try DeepFilterNetRealLoad.workingCopy()
        defer { copy.remove() }
        try await DeepFilterNetRealLoad.withClip { clip in
            let input = try AudioResampler.resample(AudioBuffer(samples: clip.noisy, sampleRate: 48_000), to: inputRate)
            let stage = DeepFilterNetStage(modelDirectory: copy.url)
            let output = try await stage.run(.audio(input)) { _ in }
            guard case let .audio(buffer) = output else { Issue.record("not audio"); return }
            #expect(buffer.sampleRate == 48_000)
            #expect(abs(Double(buffer.samples.count) / 48_000 - clip.seconds) < 0.1)
            #expect(DeepFilterNetRealLoad.finite(buffer.samples))
            let r = DeepFilterNetRealLoad.report("input-\(inputRate)Hz", noisy: clip.noisy, cleaned: buffer.samples, clean: clip.clean)
            #expect(r.improvement >= (inputRate == 16_000 ? DeepFilterNetRealLoad.snrImprovementThreshold16kDB : DeepFilterNetRealLoad.snrImprovementThresholdDB))
            let ratio = AudioMetrics.rms(buffer.samples) / AudioMetrics.rms(clip.clean)
            #expect(DeepFilterNetRealLoad.rmsRatioRange.contains(ratio), "rms ratio \(ratio)")
        }
    }
}

/// Rule 17c: the shape the Run UI gives the engine -- awaited from the main actor inside `MLXErrorGuard.guardedRun`
/// (`DeepFilterNetRunModel`). Real output, twice, and no MLX error recorded by the process-wide fallback. If it reports a stream
/// error, flip `DeepFilterNetEngine.usesAudioThread` (route through `MLXAudioThread.run`) and run it again.
@MainActor
struct DeepFilterNetMainActorProbeTests {
    private static let enabled = DeepFilterNetRealLoad.directory != nil

    @Test(.enabled(if: enabled))
    func enhanceFromTheMainActorReturnsCleanedAudio() async throws {
        let copy = try DeepFilterNetRealLoad.workingCopy()
        defer { copy.remove() }
        try await DeepFilterNetRealLoad.withClip { clip in
            MLXErrorGuard.installFallback()
            MLXErrorGuard.resetFallbackMessage()
            for pass in 1...2 {
                let cleaned = try await MLXErrorGuard.guardedRun {
                    try await DeepFilterNetEngine.enhance(clip.noisy, modelDirectory: copy.url)
                }
                let r = DeepFilterNetRealLoad.report("probe-pass\(pass)", noisy: clip.noisy, cleaned: cleaned, clean: clip.clean)
                #expect(cleaned.count == clip.noisy.count)
                #expect(DeepFilterNetRealLoad.finite(cleaned))
                #expect(r.improvement >= DeepFilterNetRealLoad.snrImprovementThresholdDB)
            }
            #expect(MLXErrorGuard.lastFallbackMessage() == nil)
        }
    }

    /// The whole Run UI path from the main actor: the stage the UI builds, resampling a 44.1 kHz input, inside `guardedRun`.
    @Test(.enabled(if: enabled))
    func theStageFromTheMainActorWith44k1Input() async throws {
        let copy = try DeepFilterNetRealLoad.workingCopy()
        defer { copy.remove() }
        try await DeepFilterNetRealLoad.withClip { clip in
            MLXErrorGuard.installFallback()
            MLXErrorGuard.resetFallbackMessage()
            let input = try AudioResampler.resample(AudioBuffer(samples: clip.noisy, sampleRate: 48_000), to: 44_100)
            let stage = DeepFilterNetStage(modelDirectory: copy.url)
            let output = try await MLXErrorGuard.guardedRun { try await stage.run(.audio(input)) { _ in } }
            guard case let .audio(buffer) = output else { Issue.record("not audio"); return }
            #expect(buffer.sampleRate == 48_000 && AudioMetrics.rms(buffer.samples) > 1e-3)
            #expect(MLXErrorGuard.lastFallbackMessage() == nil)
        }
    }

    /// Chunked against whole-file on the parity clip, from the main actor: the evidence for choosing chunk length and overlap
    /// (MP5-3c). Prints one `DFNCHUNK` line per setting: SNR gain of each, the RMS of their difference overall, within 0.25 s of
    /// every overlap centre (the seams) and away from them. Asserts only that the default-candidate setting (8 s, 1 s overlap)
    /// keeps the length, is finite, and gains within `chunkedImprovementSlackDB` of the whole-file run.
    @Test(.enabled(if: enabled))
    func chunkedRunsAgreeWithTheWholeFileRun() async throws {
        let copy = try DeepFilterNetRealLoad.workingCopy()
        defer { copy.remove() }
        try await DeepFilterNetRealLoad.withClip { clip in
            MLXErrorGuard.installFallback()
            MLXErrorGuard.resetFallbackMessage()
            let whole = try await MLXErrorGuard.guardedRun { try await DeepFilterNetEngine.enhance(clip.noisy, modelDirectory: copy.url, chunkSeconds: nil) }
            let wholeGain = DeepFilterNetRealLoad.report("whole", noisy: clip.noisy, cleaned: whole, clean: clip.clean).improvement
            let settings: [(chunk: Double, overlap: Double)] = [(4, 0.5), (4, 1), (8, 0.5), (8, 1), (8, 2), (16, 1)]
            for setting in settings {
                let chunked = try await MLXErrorGuard.guardedRun {
                    try await DeepFilterNetEngine.enhance(clip.noisy, modelDirectory: copy.url,
                                                          chunkSeconds: setting.chunk, overlapSeconds: setting.overlap)
                }
                let gain = AudioMetrics.snrDB(estimate: chunked, reference: clip.clean) - AudioMetrics.snrDB(estimate: clip.noisy, reference: clip.clean)
                let plan = DeepFilterNetEngine.chunkPlan(sampleCount: clip.noisy.count, chunkSamples: Int(setting.chunk * 48_000),
                                                         overlapSamples: Int(setting.overlap * 48_000))
                let centres = zip(plan.windows, plan.windows.dropFirst()).map { $0.1.lowerBound + plan.overlap / 2 }
                let near = Int(0.25 * 48_000)
                var nearSeam = [Float](), away = [Float](), all = [Float]()
                for i in 0..<min(whole.count, chunked.count) {
                    let d = chunked[i] - whole[i]
                    all.append(d)
                    if centres.contains(where: { abs($0 - i) < near }) { nearSeam.append(d) } else { away.append(d) }
                }
                print(String(format: "DFNCHUNK chunk=%.0fs overlap=%.1fs windows=%d gainWhole=%.2f dB gainChunked=%.2f dB diffRMS=%.5f diffRMSNearSeams=%.5f diffRMSAway=%.5f",
                             setting.chunk, setting.overlap, plan.windows.count, wholeGain, gain,
                             AudioMetrics.rms(all), AudioMetrics.rms(nearSeam), AudioMetrics.rms(away)))
                #expect(chunked.count == clip.noisy.count && DeepFilterNetRealLoad.finite(chunked))
                if setting.chunk == 8 && setting.overlap == 1 {
                    #expect(gain >= wholeGain - DeepFilterNetRealLoad.chunkedImprovementSlackDB)
                }
            }
            #expect(MLXErrorGuard.lastFallbackMessage() == nil)
        }
    }
}

extension DeepFilterNetMainActorProbeTests {
    /// Owner-requested (2026-10-25): no click at the seams. On the parity clip at the app's default (8 s windows, 1 s overlap), the
    /// largest sample-to-sample jump inside each overlap region (and so within 5 ms of every fade-in start, centre and fade-out end)
    /// is compared with the largest jump everywhere else. One `DFNSEAM` line: `seamMaxJump` (all seams), `elsewhereMaxJump`,
    /// `ratio`, and `perSeam=[...]`. The assertion allows 10 % over the elsewhere maximum (`seamJumpTolerance`, implementer's call,
    /// pending owner confirmation).
    @Test(.enabled(if: DeepFilterNetRealLoad.directory != nil))
    func noClickAtTheChunkSeams() async throws {
        let copy = try DeepFilterNetRealLoad.workingCopy()
        defer { copy.remove() }
        try await DeepFilterNetRealLoad.withClip { clip in
            MLXErrorGuard.installFallback()
            MLXErrorGuard.resetFallbackMessage()
            let cleaned = try await MLXErrorGuard.guardedRun { try await DeepFilterNetEngine.enhance(clip.noisy, modelDirectory: copy.url) }
            let plan = DeepFilterNetEngine.chunkPlan(sampleCount: clip.noisy.count,
                                                     chunkSamples: Int((DeepFilterNetEngine.defaultChunkSeconds ?? 8) * 48_000),
                                                     overlapSamples: Int(DeepFilterNetEngine.defaultOverlapSeconds * 48_000))
            let near = 240   // 5 ms at 48 kHz
            let regions = zip(plan.windows, plan.windows.dropFirst()).map { ($0.1.lowerBound - near)..<($0.0.upperBound + near) }
            let perSeam = regions.map { AudioMetrics.maxJump(cleaned, in: $0) }
            var elsewhere: Float = 0
            for i in 1..<cleaned.count where !regions.contains(where: { $0.contains(i) }) {
                elsewhere = max(elsewhere, abs(cleaned[i] - cleaned[i - 1]))
            }
            let seam = perSeam.max() ?? 0
            print(String(format: "DFNSEAM chunk=%.0fs overlap=%.1fs seams=%d seamMaxJump=%.5f elsewhereMaxJump=%.5f ratio=%.3f perSeam=[%@]",
                         DeepFilterNetEngine.defaultChunkSeconds ?? 0, DeepFilterNetEngine.defaultOverlapSeconds, regions.count, seam, elsewhere,
                         elsewhere > 0 ? Double(seam / elsewhere) : 0, perSeam.map { String(format: "%.5f", $0) }.joined(separator: ", ")))
            #expect(regions.count >= 1)
            #expect(seam <= elsewhere * 1.10, "a click at a seam: \(seam) vs \(elsewhere)")
            #expect(MLXErrorGuard.lastFallbackMessage() == nil)
        }
    }
}

/// The memory harness (MP4-3d's shape). Opt-in: `MLXUI_TEST_DEEPFILTERNET_FOOTPRINT=1`. For each length in
/// `MLXUI_TEST_DEEPFILTERNET_MINUTES` (comma list, default `1,10,60`, run in ascending order) it writes a 48 kHz mono 16-bit WAV
/// of that length (the parity clip, tiled, streamed to disk one tile at a time), then does what the Run UI does: decode with
/// `DeepFilterNetInput.read`, run `DeepFilterNetStage(modelDirectory:)` (resample, `DeepFilterNetEngine.enhance`), and build the
/// WAV bytes the Save button writes. A 20 ms sampler keeps the `phys_footprint` peak of the whole run. Prints one
/// `DFNFOOTPRINT` line per length. `MLXUI_TEST_DEEPFILTERNET_CHUNK_SECONDS` (a number, or `none` = whole file; default the app's 8 s) and
/// `..._OVERLAP_SECONDS` (default 1) choose the engine setting, so whole-file and chunked runs are separate invocations. Guard:
/// a whole-file run of more than 10 minutes is skipped unless `MLXUI_TEST_DEEPFILTERNET_ALLOW_WHOLE_LONG=1` (its memory is
/// expected to grow with length and can exhaust the machine).
struct DeepFilterNetFootprintTests {
    private static let enabled = DeepFilterNetRealLoad.directory != nil
        && DeepFilterNetRealLoad.env["MLXUI_TEST_DEEPFILTERNET_FOOTPRINT"] == "1"

    @Test(.enabled(if: enabled))
    func measureFootprintAtOneTenAndSixtyMinutes() async throws {
        let env = DeepFilterNetRealLoad.env
        let minutes = (env["MLXUI_TEST_DEEPFILTERNET_MINUTES"] ?? "1,10,60").split(separator: ",").compactMap { Double($0) }.sorted()
        // Default: the app's chunked setting (8 s / 1 s). `none` runs the whole file.
        let chunk: Double? = env["MLXUI_TEST_DEEPFILTERNET_CHUNK_SECONDS"].map { $0 == "none" ? nil : Double($0) } ?? DeepFilterNetEngine.defaultChunkSeconds
        let overlap = env["MLXUI_TEST_DEEPFILTERNET_OVERLAP_SECONDS"].flatMap(Double.init) ?? DeepFilterNetEngine.defaultOverlapSeconds
        let allowLongWhole = env["MLXUI_TEST_DEEPFILTERNET_ALLOW_WHOLE_LONG"] == "1"
        let copy = try DeepFilterNetRealLoad.workingCopy()
        defer { copy.remove() }
        try await DeepFilterNetRealLoad.withClip { clip in
            let tile = clip.noisy
            let chunkLabel = chunk.map { String(format: "%.0fs/%.1fs", $0, overlap) } ?? "whole"
            for length in minutes {
                if chunk == nil && length > 10 && !allowLongWhole {
                    print("DFNFOOTPRINT minutes=\(length) chunk=whole SKIPPED (set MLXUI_TEST_DEEPFILTERNET_ALLOW_WHOLE_LONG=1)")
                    continue
                }
                let file = try Self.makeWav(tile: tile, seconds: length * 60)
                defer { try? FileManager.default.removeItem(at: file) }
                Memory.clearCache()
                Memory.peakMemory = 0
                let tracker = FootprintTracker()
                let sampler = Task.detached {
                    while !Task.isCancelled { tracker.note(FootprintTracker.phys()); try? await Task.sleep(for: .milliseconds(20)) }
                }
                let baseline = FootprintTracker.phys()
                tracker.note(baseline)
                let started = Date()
                let buffer = try DeepFilterNetInput.read(file)
                let afterDecode = tracker.takePhasePeak()
                let stage = DeepFilterNetStage(modelDirectory: copy.url, chunkSeconds: chunk, overlapSeconds: overlap)
                let output = try await stage.run(.audio(buffer)) { _ in }
                let afterStage = tracker.takePhasePeak()
                guard case let .audio(cleaned) = output else { Issue.record("not audio"); sampler.cancel(); return }
                let wavBytes = AudioWriter.wavData(cleaned).count   // what Save Cleaned Audio writes
                let wall = Date().timeIntervalSince(started)
                tracker.note(FootprintTracker.phys())
                sampler.cancel()
                let mib = 1_048_576.0
                print(String(format: "DFNFOOTPRINT minutes=%.0f chunk=%@ peakGB=%.3f peakMiB=%.0f baselineMiB=%.0f decodePhasePeakMiB=%.0f stagePhasePeakMiB=%.0f mlxPeakGB=%.3f wall=%.1f s samples=%d wavMiB=%.0f",
                             length, chunkLabel, Double(tracker.runPeak) / 1e9, Double(tracker.runPeak) / mib, Double(baseline) / mib,
                             Double(afterDecode) / mib, Double(afterStage) / mib, Double(Memory.snapshot().peakMemory) / 1e9, wall,
                             cleaned.samples.count, Double(wavBytes) / mib))
                #expect(cleaned.sampleRate == 48_000)
                #expect(abs(Double(cleaned.samples.count) - length * 60 * 48_000) < 48_000)
            }
        }
    }

    /// A 48 kHz mono 16-bit WAV of `seconds`, `tile` repeated, written to the temp folder one tile at a time so the test process
    /// stays small (only the file survives).
    nonisolated static func makeWav(tile: [Float], seconds: Double) throws -> URL {
        let total = Int(seconds * 48_000)
        var header = AudioWriter.wavData(AudioBuffer(samples: [], sampleRate: 48_000))
        func put(_ offset: Int, _ n: Int) { var v = UInt32(n).littleEndian; withUnsafeBytes(of: &v) { header.replaceSubrange(offset..<offset + 4, with: $0) } }
        put(4, 36 + total * 2); put(40, total * 2)
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("dfn-footprint-\(UUID().uuidString).wav")
        FileManager.default.createFile(atPath: file.path, contents: header)
        let handle = try FileHandle(forWritingTo: file)
        defer { try? handle.close() }
        try handle.seekToEnd()
        let piece = AudioWriter.wavData(AudioBuffer(samples: tile, sampleRate: 48_000)).dropFirst(44)
        var written = 0
        while written < total {
            let count = min(tile.count, total - written)
            try handle.write(contentsOf: piece.prefix(count * 2))
            written += count
        }
        return file
    }
}

/// Not gated on weights: the generators the real-weights tests stand on.
struct DeepFilterNetHarnessTests {
    @Test func theParityClipIsAboutHalfAMinuteAndMixedAtTheStatedSNR() async throws {
        try await DeepFilterNetRealLoad.withClip { clip in
            print(String(format: "DFNCLIP %.1f s, clean RMS %.4f", clip.seconds, AudioMetrics.rms(clip.clean)))
            #expect((12...45).contains(clip.seconds), "\(clip.seconds) s")
            #expect(abs(AudioMetrics.rms(clip.clean) - 0.1) < 1e-3)
            #expect(abs(AudioMetrics.snrDB(estimate: clip.noisy, reference: clip.clean) - DeepFilterNetRealLoad.mixSNRDB) < 0.01)
            // Deterministic: a second build mixes identical noise.
            let again = AudioMetrics.mix(reference: clip.clean,
                                         noise: AudioMetrics.noise(count: clip.clean.count, sampleRate: 48_000, seed: DeepFilterNetRealLoad.noiseSeed),
                                         snrDB: DeepFilterNetRealLoad.mixSNRDB)
            #expect(again == clip.noisy)
        }
    }

    @Test func theFootprintWavIsTheTileRepeatedToTheRequestedLength() throws {
        let tile: [Float] = (0..<4_800).map { 0.25 * Float(sin(Double($0) * 0.1)) }
        let url = try DeepFilterNetFootprintTests.makeWav(tile: tile, seconds: 1.0)   // 48,000 samples: ten tiles
        defer { try? FileManager.default.removeItem(at: url) }
        let back = try DeepFilterNetInput.read(url)
        #expect(back.sampleRate == 48_000 && back.samples.count == 48_000)
        #expect(AudioMetrics.snrDB(estimate: Array(back.samples[4_800..<9_600]), reference: tile) > 60)
        let partial = try DeepFilterNetFootprintTests.makeWav(tile: tile, seconds: 0.15)   // 7,200: one and a half tiles
        defer { try? FileManager.default.removeItem(at: partial) }
        #expect(try DeepFilterNetInput.read(partial).samples.count == 7_200)
    }
}
