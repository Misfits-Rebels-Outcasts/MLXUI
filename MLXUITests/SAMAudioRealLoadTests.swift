import Foundation
import Testing
import MLX
import MLXAudioSTS
@testable import MLXUI

/// MC1-4 — the real-weights tests for SAM-Audio Base (KICKBOX rule 13b, rule 17c). Owner/lead-run, gated on the weights
/// (about 5.9 GB): skipped unless `MLXUI_TEST_SAMAUDIO_DIR` points at an installed copy — normally a `/private/tmp` copy,
/// because the sandboxed test host cannot read `~/Library/Application Support`. Without the variable, `ModelStore`'s own
/// folder is tried.
///
/// Every test calls `SAMAudioEngine.separate` (the app's engine path), on the folder itself: the tests never write to it.
/// A skipped run does not count as evidence.
///
/// **Opt-in (R5):** these also need `MLXUI_TEST_SAMAUDIO=1` (see ``optedIn``). Without it every test here reports as
/// skipped, never as passed.
enum SAMAudioRealLoad {
    nonisolated static let env = ProcessInfo.processInfo.environment

    nonisolated static let directory: URL? = {
        let url: URL
        if let override = env["MLXUI_TEST_SAMAUDIO_DIR"] ?? env["TEST_RUNNER_MLXUI_TEST_SAMAUDIO_DIR"] {
            url = URL(fileURLWithPath: override)
        } else {
            url = ModelStore.shared.directory(forHFModelID: "mlx-community/sam-audio-base")
        }
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.appendingPathComponent("config.json").path),
              fm.fileExists(atPath: url.appendingPathComponent("t5/model.safetensors").path) else { return nil }
        return url
    }()

    /// R5 (MC1-8): every suite that loads the real weights needs this explicit opt-in as well as the weights — a stray
    /// `ModelStore` install must never start a 12+ GB run by itself. `MLXUI_TEST_SAMAUDIO=1`, which reaches the test
    /// host as `TEST_RUNNER_MLXUI_TEST_SAMAUDIO=1` through `xcodebuild`. Same shape as the footprint suite's switch.
    nonisolated static var optedIn: Bool {
        (env["MLXUI_TEST_SAMAUDIO"] ?? env["TEST_RUNNER_MLXUI_TEST_SAMAUDIO"]) == "1"
    }

    /// True only when the opt-in is set **and** the weights are installed. Every real-weights `.enabled(if:)` uses this
    /// (the footprint suite, which also needs its own switch, goes through it too).
    nonisolated static var canRun: Bool { optedIn && directory != nil }

    /// A ~25 s speech passage, read by `say` at roughly 170 words a minute.
    nonisolated static let passage = """
    The harbour was quiet before sunrise. A few fishing boats rocked against their ropes, and the only sound was the slow \
    knock of wood on wood. Maria walked down the stone steps with a thermos of coffee and a small notebook. She had promised \
    herself that this year she would write down everything the old fishermen said, every proverb, every warning about the \
    weather, every story about the storms that never made the newspapers.
    """

    enum ClipError: Error, CustomStringConvertible {
        case sayUnavailable(String)
        var description: String {
            switch self { case .sayUnavailable(let why): return "the reference clip could not be generated: \(why)" }
        }
    }

    /// Clean 48 kHz speech, RMS-normalised to 0.1 (the DeepFilterNet harness's level, so the two compare).
    nonisolated static func speechReference() throws -> [Float] {
        let cache = FileManager.default.temporaryDirectory
            .appendingPathComponent("sam-speech-\(abs(passage.hashValue)).wav")
        if !FileManager.default.fileExists(atPath: cache.path) {
            // Two tests (or suites) can race here; each `say` writes its own file and the first complete one is
            // moved into place, so a concurrent writer never truncates the clip the other is reading.
            let unique = FileManager.default.temporaryDirectory
                .appendingPathComponent("sam-speech-\(UUID().uuidString).wav")
            defer { try? FileManager.default.removeItem(at: unique) }
            try say(passage, to: unique)
            if !FileManager.default.fileExists(atPath: cache.path) {
                try? FileManager.default.moveItem(at: unique, to: cache)
            }
            guard FileManager.default.fileExists(atPath: cache.path) else {
                throw ClipError.sayUnavailable("the clip could not be moved into \(cache.path)")
            }
        }
        let samples = try AudioResampler.resample(AudioFileReader.read(cache), to: 48_000).samples
        let level = AudioMetrics.rms(samples)
        guard level > 0 else { throw ClipError.sayUnavailable("the reference is silent") }
        return samples.map { $0 * Float(0.1 / level) }
    }

    /// One `say` render to `url`. Throws ``ClipError`` with `say`'s own message.
    nonisolated private static func say(_ text: String, to url: URL) throws {
        let voice = env["MLXUI_TEST_SAMAUDIO_VOICE"] ?? "Samantha"
        guard FileManager.default.isExecutableFile(atPath: "/usr/bin/say") else {
            throw ClipError.sayUnavailable("/usr/bin/say is missing")
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        process.arguments = ["-v", voice, "--file-format=WAVE", "--data-format=LEI16@48000", "-o", url.path, text]
        let errors = Pipe()
        process.standardError = errors
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw ClipError.sayUnavailable("say exited \(process.terminationStatus): \(String(decoding: errors.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self))")
        }
    }

    /// A deterministic music bed: three tones plus a little seeded noise, well below the speech, so "speech" is the
    /// thing a separator should find.
    nonisolated static func musicBed(count: Int, seed: UInt64 = 560_1) -> [Float] {
        let noise = AudioMetrics.noise(count: count, sampleRate: 48_000, seed: seed)
        return (0..<count).map { i in
            let t = Double(i) / 48_000
            let tones = 0.05 * Float(sin(2 * Double.pi * 220 * t))
                + 0.04 * Float(sin(2 * Double.pi * 277.18 * t))
                + 0.03 * Float(sin(2 * Double.pi * 329.63 * t))
            return tones + 0.01 * noise[i]
        }
    }

    struct Clip: Sendable {
        let mix: [Float]
        let speech: [Float]
    }

    /// Speech over the music bed, `seconds` long (the speech tiled if it is shorter).
    nonisolated static func voiceOverMusic(seconds: Double) throws -> Clip {
        let speech = try speechReference()
        let count = Int(seconds * 48_000)
        var mix = [Float](repeating: 0, count: count)
        let bed = musicBed(count: count)
        for i in 0..<count {
            let voice = speech[i % speech.count]
            mix[i] = max(-1, min(1, voice + bed[i]))
        }
        let tiled = (0..<count).map { speech[$0 % speech.count] }
        return Clip(mix: mix, speech: tiled)
    }

    /// Speech over the music bed, `seconds` long (the speech tiled if it is shorter), for the footprint phases: one
    /// array only, so a 30-minute mix does not hold a second copy of itself.
    nonisolated static func voiceOverMusicMix(seconds: Double) throws -> [Float] {
        let speech = try speechReference()
        let count = Int(seconds * 48_000)
        let bed = musicBed(count: count)
        var mix = [Float](repeating: 0, count: count)
        for i in 0..<count {
            mix[i] = max(-1, min(1, speech[i % speech.count] + bed[i]))
        }
        return mix
    }

    /// Normalised correlation at lag 0 — how much of `reference` is in `estimate`. 1 = the same signal.
    nonisolated static func correlation(_ estimate: [Float], _ reference: [Float]) -> Double {
        let n = min(estimate.count, reference.count)
        guard n > 0 else { return 0 }
        var dot = 0.0, ea = 0.0, er = 0.0
        for i in 0..<n {
            let a = Double(estimate[i]), r = Double(reference[i])
            dot += a * r; ea += a * a; er += r * r
        }
        guard ea > 0, er > 0 else { return 0 }
        return dot / (ea.squareRoot() * er.squareRoot())
    }

    nonisolated static func finite(_ samples: [Float]) -> Bool { !samples.contains { !$0.isFinite } }

    /// The header keys of a safetensors file (the JSON after the 8-byte length), without reading the tensors.
    nonisolated static func safetensorsKeys(at url: URL) -> [String] {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return [] }
        defer { try? handle.close() }
        guard let lengthData = try? handle.read(upToCount: 8), lengthData.count == 8 else { return [] }
        let length = lengthData.withUnsafeBytes { $0.load(as: UInt64.self) }
        guard length > 0, length < 50_000_000, let header = try? handle.read(upToCount: Int(length)),
              let json = try? JSONSerialization.jsonObject(with: header) as? [String: Any] else { return [] }
        return json.keys.filter { $0 != "__metadata__" }
    }

    /// Runs `body` with a voice-over-music clip, or prints why it can't and returns (a skip with a message, not a failure).
    nonisolated static func withClip(seconds: Double, _ body: (Clip) async throws -> Void) async throws {
        let clip: Clip
        do { clip = try voiceOverMusic(seconds: seconds) } catch let error as ClipError {
            print("SAMSKIP \(error)")
            return
        }
        try await body(clip)
    }

    nonisolated static func report(_ label: String, result: SAMAudioEngine.Result, speech: [Float],
                                   wall: Double, audioSeconds: Double) {
        let isoCorrelation = correlation(result.isolated, speech)
        let otherCorrelation = correlation(result.other, speech)
        print(String(format: "SAMPARITY %@ seconds=%.1f wall=%.1f s speed=%.2fx isolatedRms=%.4f otherRms=%.4f isolatedVsSpeech=%.3f otherVsSpeech=%.3f peakGB=%.3f",
                     label, audioSeconds, wall, audioSeconds / max(wall, 0.001),
                     AudioMetrics.rms(result.isolated), AudioMetrics.rms(result.other),
                     isoCorrelation, otherCorrelation,
                     Double(Memory.snapshot().peakMemory) / 1e9))
    }
}

@Suite(.serialized)
struct SAMAudioRealLoadTests {
    private static var enabled: Bool { SAMAudioRealLoad.canRun }

    /// The installed checkpoint loads through the app's engine path and separates a short clip in one pass: 48 kHz in
    /// and out, both tracks the recording's length, finite, not silent.
    @Test(.enabled(if: SAMAudioRealLoadTests.enabled))
    func theCheckpointLoadsAndSeparatesAShortClip() async throws {
        let dir = try #require(SAMAudioRealLoad.directory)
        try await SAMAudioRealLoad.withClip(seconds: 10) { clip in
            Memory.clearCache()
            Memory.peakMemory = 0
            MLXErrorGuard.installFallback()
            MLXErrorGuard.resetFallbackMessage()
            let started = Date()
            let result = try await SAMAudioEngine.separate(clip.mix, description: "speech", modelDirectory: dir)
            let wall = Date().timeIntervalSince(started)
            SAMAudioRealLoad.report("10s-one-pass", result: result, speech: clip.speech, wall: wall, audioSeconds: 10)
            #expect(result.sampleRate == 48_000, "the model's own rate, not 16 kHz")
            #expect(result.isolated.count == clip.mix.count, "the isolated track keeps the recording's length")
            #expect(result.other.count == clip.mix.count, "the second track keeps the recording's length")
            #expect(SAMAudioRealLoad.finite(result.isolated) && SAMAudioRealLoad.finite(result.other))
            #expect(AudioMetrics.rms(result.isolated) > 1e-4, "the isolated track is silent")
            #expect(AudioMetrics.rms(result.other) > 1e-4, "the second track is silent")
            #expect(MLXErrorGuard.lastFallbackMessage() == nil, "a recorded MLX error fails the run")
        }
    }

    /// The isolated track follows the speech reference more closely than "everything else" does — the separation is
    /// doing what the description asked, not returning two copies of the mix.
    @Test(.enabled(if: SAMAudioRealLoadTests.enabled))
    func speechIsolatedFromAMusicBedFollowsTheSpeech() async throws {
        let dir = try #require(SAMAudioRealLoad.directory)
        try await SAMAudioRealLoad.withClip(seconds: 20) { clip in
            let result = try await SAMAudioEngine.separate(clip.mix, description: "speech", modelDirectory: dir)
            let iso = SAMAudioRealLoad.correlation(result.isolated, clip.speech)
            let other = SAMAudioRealLoad.correlation(result.other, clip.speech)
            print(String(format: "SAMCORRELATION isolatedVsSpeech=%.3f otherVsSpeech=%.3f", iso, other))
            #expect(iso > other, "the isolated track must be the speech one (\(iso) vs \(other))")
            #expect(iso > 0.3, "the isolated track barely follows the speech (\(iso))")
            #expect(SAMAudioRealLoad.correlation(result.isolated, result.other) < 0.95,
                    "the two tracks must differ")
        }
    }

    /// Trap 2: the installed `t5/model.safetensors` carries the decoder / lm_head keys of the full T5 encoder-decoder,
    /// and loading it succeeds — which it only can because the library's `T5Encoder.sanitize` drops them (`verify:
    /// .noUnusedKeys` would otherwise throw). Nothing is stripped or rewritten on disk.
    @Test(.enabled(if: SAMAudioRealLoadTests.enabled))
    func theT5FileCarriesDecoderKeysAndTheLoaderDropsThem() async throws {
        let dir = try #require(SAMAudioRealLoad.directory)
        let keys = SAMAudioRealLoad.safetensorsKeys(at: dir.appendingPathComponent("t5/model.safetensors"))
        let decoderKeys = keys.filter { $0.hasPrefix("decoder.") || $0.hasPrefix("lm_head.") }
        print("SAMT5 keys=\(keys.count) decoderKeys=\(decoderKeys.count) sample=\(decoderKeys.prefix(3))")
        #expect(!keys.isEmpty, "the T5 safetensors header must be readable")
        #expect(!decoderKeys.isEmpty, "google-t5/t5-base ships the full encoder-decoder; the decoder keys must be there")
        // Loading the encoder is the proof the sanitize step drops them: `update(verify: .noUnusedKeys)` would throw.
        let loaded = try SAMAudioEngine.load(modelDirectory: dir)
        let encoded = try await loaded.model.textEncoder.encode(["speech"])
        let features = try MLXErrorGuard.read(encoded.features, as: Float.self)
        #expect(features.count > 0)
        #expect(loaded.sampleRate == 48_000, "the codec's own rate")
    }

    /// A clip over 30 s takes the windowed path (30 s windows every 27 s: a 65 s clip is 3 windows): progress rises per
    /// window, both tracks keep the recording's length, and the whole thing finishes. The line prints wall time and the measured speed.
    @Test(.enabled(if: SAMAudioRealLoadTests.enabled))
    func aClipOverThirtySecondsGoesThroughTheWindows() async throws {
        let dir = try #require(SAMAudioRealLoad.directory)
        try await SAMAudioRealLoad.withClip(seconds: 65) { clip in
            let log = SamProgressLog()
            let started = Date()
            let result = try await SAMAudioEngine.separate(clip.mix, description: "speech", modelDirectory: dir) { log.add($0) }
            let wall = Date().timeIntervalSince(started)
            SAMAudioRealLoad.report("65s-windowed", result: result, speech: clip.speech, wall: wall, audioSeconds: 65)
            print("SAMWINDOWS values=\(log.values)")
            #expect(result.isolated.count == clip.mix.count)
            #expect(log.values.count > 1, "a 65 s clip must run several (3) windows")
            #expect(log.values == log.values.sorted() && log.values.last == 1.0)
            // MC1-9 (R3): the windowed path must separate as well as the single pass does (0.975 / 0.032 measured).
            let iso = SAMAudioRealLoad.correlation(result.isolated, clip.speech)
            let other = SAMAudioRealLoad.correlation(result.other, clip.speech)
            #expect(iso > other, "the isolated track must be the speech one (\(iso) vs \(other))")
            #expect(iso > 0.3, "the isolated track barely follows the speech (\(iso))")
            #expect(other < 0.3, "the other track still carries the speech (\(other))")
        }
    }

    /// Stop, honoured between windows (R4): the run is cancelled from inside the **first progress callback** — the first
    /// window has just finished, so the load is over and the loop is at the check before window 2 — and must end with a
    /// `CancellationError` and no audio within one window's time plus a margin. Not a fixed delay: that would race the
    /// ~15-60 s load and prove nothing about the windows.
    ///
    /// The bound, `stopBound` = 45 s from the first window finishing to the run ending: a constant of its own, not
    /// derived from the engine's time estimate (which can change for the Run screen's sake). The loop checks
    /// cancellation before the next window, so the real stop is expected in well under a second; one window measured
    /// about 11 s warm, so 45 s still covers a window that was in flight on a much slower Mac and fails a run that went
    /// on to finish the file. MC1-10: windows are 30 s now, one takes about 21 s warm (65 s clips ran at 0.96-1.0x with
    /// the load included), so 45 s is about two windows of slack; the clip is 150 s (6 windows, so over 100 s of work
    /// is left when the first window finishes) to keep a run that carries on well past the bound. Two timestamps
    /// are printed, relative to the test start, so the journal shows the stop came after the first window and not
    /// during the load. (The loop's Stop logic itself is covered without weights by `SAMAudioWindowLoopTests`.)
    static let stopBound: Double = 45

    @Test(.enabled(if: SAMAudioRealLoadTests.enabled))
    func cancelStopsBetweenWindows() async throws {
        let dir = try #require(SAMAudioRealLoad.directory)
        try await SAMAudioRealLoad.withClip(seconds: 150) { clip in
            let started = Date()
            let holder = SamTaskHolder()
            let firstProgressAt = SamTimestamp()
            let task = Task {
                try await SAMAudioEngine.separate(clip.mix, description: "speech", modelDirectory: dir) { _ in
                    if firstProgressAt.setIfFirst(Date().timeIntervalSince(started)) { holder.cancel() }
                }
            }
            holder.set(task)
            do {
                _ = try await task.value
                Issue.record("a cancelled run must not return audio")
            } catch is CancellationError {
                let ended = Date().timeIntervalSince(started)
                let first = try #require(firstProgressAt.value, "the run ended before any window finished")
                print(String(format: "SAMCANCEL firstProgressAt=%.1f s runEndedAt=%.1f s (stop took %.2f s after window 1; 150 s clip, 6 windows)",
                             first, ended, ended - first))
                #expect(ended - first < Self.stopBound,
                        "the run must stop within \(Self.stopBound) s of the first window finishing")
            } catch {
                Issue.record("\(error)")
            }
        }
    }
}

/// Rule 17c — the main-actor probe. The engine is called from the main actor inside `MLXErrorGuard.guardedRun`, and
/// every returned array is read there; thread-bound arrays under mlx-swift 0.32 would raise (or segfault) instead.
@Suite(.serialized)
struct SAMAudioMainActorProbeTests {
    @MainActor
    @Test(.enabled(if: SAMAudioRealLoad.canRun))
    func separatingFromTheMainActorReturnsReadableAudio() async throws {
        let dir = try #require(SAMAudioRealLoad.directory)
        try await SAMAudioRealLoad.withClip(seconds: 10) { clip in
            MLXErrorGuard.installFallback()
            MLXErrorGuard.resetFallbackMessage()
            let result = try await MLXErrorGuard.guardedRun {
                try await SAMAudioEngine.separate(clip.mix, description: "speech", modelDirectory: dir)
            }
            #expect(result.isolated.count == clip.mix.count)
            #expect(result.isolated.contains { $0 != 0 }, "not silence")
            #expect(result.other.contains { $0 != 0 }, "not silence")
            #expect(MLXErrorGuard.lastFallbackMessage() == nil)
        }
    }
}

/// The owner's "listen to these" deliverable: two WAVs from a real run on a voice-over-music clip. The test writes them
/// and prints the paths; a skipped run writes nothing.
@Suite(.serialized)
struct SAMAudioListenableOutputTests {
    @Test(.enabled(if: SAMAudioRealLoad.canRun))
    func writesTheTwoTracksForTheOwnerToListen() async throws {
        let dir = try #require(SAMAudioRealLoad.directory)
        try await SAMAudioRealLoad.withClip(seconds: 20) { clip in
            let result = try await SAMAudioEngine.separate(clip.mix, description: "speech", modelDirectory: dir)
            // The sandboxed test host cannot write to /private/tmp; the process's own temporary directory (the
            // app container's `tmp/`) is writable and openable from Finder. `SAM4D_LISTEN_DIR` overrides it.
            let out = SAMAudioRealLoad.env["SAM4D_LISTEN_DIR"].map { URL(fileURLWithPath: $0, isDirectory: true) }
                ?? FileManager.default.temporaryDirectory.appendingPathComponent("sam-audio-mc1", isDirectory: true)
            try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
            let isolated = out.appendingPathComponent("voice-over-music-isolated.wav")
            let other = out.appendingPathComponent("voice-over-music-everything-else.wav")
            let input = out.appendingPathComponent("voice-over-music-input.wav")
            try AudioWriter.writeWAV(AudioBuffer(samples: result.isolated, sampleRate: result.sampleRate), to: isolated)
            try AudioWriter.writeWAV(AudioBuffer(samples: result.other, sampleRate: result.sampleRate), to: other)
            try AudioWriter.writeWAV(AudioBuffer(samples: clip.mix, sampleRate: 48_000), to: input)
            print("SAMLISTEN \(isolated.path)")
            print("SAMLISTEN \(other.path)")
            print("SAMLISTEN \(input.path)")
            #expect(FileManager.default.fileExists(atPath: isolated.path))
            #expect(FileManager.default.fileExists(atPath: other.path))
        }
    }
}

private final class SamProgressLog: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [Double] = []
    func add(_ v: Double) { lock.lock(); stored.append(v); lock.unlock() }
    var values: [Double] { lock.lock(); defer { lock.unlock() }; return stored }
}

private final class SamTimestamp: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Double?
    /// Records `v` if nothing is recorded yet; true for the first caller only.
    func setIfFirst(_ v: Double) -> Bool { lock.lock(); defer { lock.unlock() }; if stored != nil { return false }; stored = v; return true }
    var value: Double? { lock.lock(); defer { lock.unlock() }; return stored }
}

/// Lets the progress callback cancel the task that is running it (the task does not exist yet when the closure is made).
private final class SamTaskHolder: @unchecked Sendable {
    private let lock = NSLock()
    private var task: Task<SAMAudioEngine.Result, Error>?
    private var cancelRequested = false
    func set(_ t: Task<SAMAudioEngine.Result, Error>) {
        lock.lock(); task = t; let pending = cancelRequested; lock.unlock()
        if pending { t.cancel() }
    }
    func cancel() {
        lock.lock(); cancelRequested = true; let t = task; lock.unlock()
        t?.cancel()
    }
}
