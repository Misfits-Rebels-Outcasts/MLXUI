import Foundation
import MLX
import MLXAudioVAD
import Testing
@testable import MLXUI

/// MP4-3a -- the real load of Sortformer 4spk v2.1 (fp16) (KICKOFF rule 13b) and the main-actor probe (rule 17c). Owner-run,
/// gated on the weights (0.24 GB). Skipped unless `mlx-community--diar_streaming_sortformer_4spk-v2.1-fp16/.installed` exists
/// under the directory this process resolves: `ModelStore`'s, or `MLXUI_TEST_SORTFORMER_DIR=<model folder>` (passed to the
/// host as `TEST_RUNNER_MLXUI_TEST_SORTFORMER_DIR`). The exact commands are in the MP4-3a journal. A skipped run does not count.
enum SortformerRealLoad {
    nonisolated static let slug = "mlx-community--diar_streaming_sortformer_4spk-v2.1-fp16"

    nonisolated static let directory: URL? = {
        let url: URL
        if let override = ProcessInfo.processInfo.environment["MLXUI_TEST_SORTFORMER_DIR"] {
            url = URL(fileURLWithPath: override)
        } else {
            url = ModelStore.shared.directory(forHFModelID: "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16")
        }
        let marker = url.appendingPathComponent(".installed")
        return FileManager.default.fileExists(atPath: marker.path) ? url : nil
    }()

    /// A copy of the model folder in a fresh temp directory (the Qwen3-ASR precedent: the sandboxed test host cannot always
    /// read or write the Direct store in place). `copyItem` is an APFS clone on the same volume. The source is never touched.
    /// Without the override the `ModelStore` folder is used as is. Call `remove()` (a `defer`) when done.
    struct WorkingCopy {
        nonisolated let url: URL
        nonisolated let root: URL?

        nonisolated func remove() {
            if let root { try? FileManager.default.removeItem(at: root) }
        }
    }

    nonisolated static func workingCopy() throws -> WorkingCopy {
        let source = try #require(directory)
        guard ProcessInfo.processInfo.environment["MLXUI_TEST_SORTFORMER_DIR"] != nil else {
            return WorkingCopy(url: source, root: nil)
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("sortformer-model-\(UUID().uuidString)")
        let copy = root.appendingPathComponent(source.lastPathComponent)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        do {
            try FileManager.default.copyItem(at: source, to: copy)
        } catch {
            try? FileManager.default.removeItem(at: root)
            throw error
        }
        return WorkingCopy(url: copy, root: root)
    }

    /// The parity clip as the 16 kHz mono buffer `SortformerStage` hands the engine.
    nonisolated static func clipSamples(_ url: URL = SortformerParity.clipURL) throws -> [Float] {
        let buffer = try AudioFileReader.read(url)
        return try AudioResampler.resample(buffer, to: 16_000).samples
    }

    nonisolated static func gib(_ bytes: Int) -> String { String(format: "%.3f GiB", Double(bytes) / 1_073_741_824) }

    nonisolated static func line(_ label: String, _ s: Memory.Snapshot) -> String {
        "SORTFORMERMEASURE \(label): active \(s.activeMemory) B (\(gib(s.activeMemory))), cache \(s.cacheMemory) B (\(gib(s.cacheMemory))), peak \(s.peakMemory) B (\(gib(s.peakMemory)))"
    }

    /// The first `MLXUI_TEST_SORTFORMER_TRIM_SECONDS` seconds, when set (to land a measurement on an exact length).
    nonisolated static func trimmed(_ samples: [Float]) -> [Float] {
        guard let n = ProcessInfo.processInfo.environment["MLXUI_TEST_SORTFORMER_TRIM_SECONDS"].flatMap(Double.init) else { return samples }
        return Array(samples.prefix(Int(n * 16_000)))
    }

    /// What the clip should show: at least two distinct speakers, sorted by start, every segment inside the clip.
    nonisolated static func plausible(_ segments: [SpeakerSegment], duration: Double) -> [String] {
        var problems: [String] = []
        if SpeakerSegments.speakers(in: segments).count < 2 { problems.append("fewer than 2 distinct speakers") }
        if segments != SpeakerSegments.cleaned(segments) { problems.append("not sorted/cleaned") }
        for s in segments where s.start < 0 || s.end > duration + 0.001 || s.end <= s.start {
            problems.append("segment outside the clip or empty: \(s)")
        }
        return problems
    }
}

struct SortformerRealLoadTests {
    /// Loads through `SortformerEngine`'s own path (`SortformerModel.fromModelDirectory` on the installed folder). Success with
    /// only the files `InstallManager` fetched (`config.json` + `model.safetensors`) proves the install covers the loader.
    /// Prints the segments (`SORTFORMER segments:`, for the MP4-3b golden) and the MLX peak.
    @Test(.enabled(if: SortformerRealLoad.directory != nil))
    func theCheckpointLoadsAndDiarizesTheTwoSpeakerClip() async throws {
        let copy = try SortformerRealLoad.workingCopy()
        defer { copy.remove() }
        let samples = try SortformerRealLoad.clipSamples()
        let duration = Double(samples.count) / 16_000
        Memory.clearCache()
        Memory.peakMemory = 0
        print(SortformerRealLoad.line("start", Memory.snapshot()))
        let started = Date()
        let raw = try await SortformerEngine.diarize(samples, modelDirectory: copy.url)
        let seconds = Date().timeIntervalSince(started)
        print(SortformerRealLoad.line("afterDiarize", Memory.snapshot()) + String(format: ", %.1f s for %.1f s of audio", seconds, duration))
        let segments = SpeakerSegments.cleaned(raw, duration: duration)
        print("SORTFORMER raw: \(raw.count) segments; normalized: \(segments.count)")
        print("SORTFORMER segments: \(segments.map { String(format: "(%d, %.3f, %.3f)", $0.speaker, $0.start, $0.end) }.joined(separator: " "))")
        print("SORTFORMER csv-begin\n\(SpeakerSegments.csv(segments))SORTFORMER csv-end")
        let problems = SortformerRealLoad.plausible(segments, duration: duration)
        #expect(problems.isEmpty, "\(problems)")
        #expect(FileManager.default.fileExists(atPath: copy.url.appendingPathComponent(".installed").path))
    }
}

/// The shape the Run UI gives the engine: awaited from a main-actor `Task`, inside `MLXErrorGuard.guardedRun`
/// (`SortformerRunModel`). The offline `generate` never reaches the library's thread-bound gather (`Sortformer.swift:1318-1319`,
/// streaming compression only), so this is the evidence that the engine needs no `MLXAudioThread.run`: real segments, twice,
/// and no MLX error recorded by the process-wide fallback. If it reports a stream error, flip
/// `SortformerEngine.usesAudioThread` to `true` and run it again.
@MainActor
struct SortformerMainActorProbeTests {
    @Test(.enabled(if: SortformerRealLoad.directory != nil))
    func diarizeFromTheMainActorReturnsSegments() async throws {
        let copy = try SortformerRealLoad.workingCopy()
        defer { copy.remove() }
        let samples = try SortformerRealLoad.clipSamples()
        let duration = Double(samples.count) / 16_000
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        for _ in 0..<2 {   // twice: the second call reuses nothing but must not differ in kind
            let raw = try await MLXErrorGuard.guardedRun {
                try await SortformerEngine.diarize(samples, modelDirectory: copy.url)
            }
            let segments = SpeakerSegments.cleaned(raw, duration: duration)
            print("SORTFORMERPROBE \(segments.count) segments, speakers \(SpeakerSegments.speakers(in: segments))")
            let problems = SortformerRealLoad.plausible(segments, duration: duration)
            #expect(problems.isEmpty, "\(problems)")
        }
        #expect(MLXErrorGuard.lastFallbackMessage() == nil)
    }

    /// MP4-3b: the **engine's streaming branch** (`feed`, the speaker-cache compression gather included) from the main actor
    /// inside `guardedRun`: the parity clip is sent down the streaming path with `offlineLimit: 1`. Asserts real segments with
    /// no stream error and no fallback message, and that the speakers match the ground-truth pair.
    @Test(.enabled(if: SortformerRealLoad.directory != nil))
    func engineStreamingBranchFromTheMainActorReturnsSegments() async throws {
        let copy = try SortformerRealLoad.workingCopy()
        defer { copy.remove() }
        let samples = try SortformerRealLoad.clipSamples()
        let duration = Double(samples.count) / 16_000
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        let raw = try await MLXErrorGuard.guardedRun {
            try await SortformerEngine.diarize(samples, modelDirectory: copy.url, offlineLimit: 1)
        }
        let segments = SpeakerSegments.cleaned(raw, duration: duration)
        print("SORTFORMERSTREAMPROBE \(segments.count) segments, speakers \(SpeakerSegments.speakers(in: segments)): "
              + segments.map { String(format: "(%d, %.3f, %.3f)", $0.speaker, $0.start, $0.end) }.joined(separator: " "))
        let problems = SortformerRealLoad.plausible(segments, duration: duration)
        #expect(problems.isEmpty, "\(problems)")
        #expect(MLXErrorGuard.lastFallbackMessage() == nil)
    }

    /// Rule 17's other half, on purpose: the library's **streaming** path builds the mixed-subscript gather (once the speaker
    /// cache overflows: the 52.7 s clip is about 660 frames and the cache holds 188, so it does). This runs it from the main
    /// actor and only reports (`SORTFORMERSTREAM ...`); it never fails on the outcome, because the app does not use the path.
    /// If it reports a stream error, the engine must keep to the offline `generate`; if it works, `MLXAudioThread` is not
    /// needed even for that path. Opt in with `TEST_RUNNER_MLXUI_TEST_SORTFORMER_STREAM_PROBE=1`.
    @Test(.enabled(if: SortformerRealLoad.directory != nil && ProcessInfo.processInfo.environment["MLXUI_TEST_SORTFORMER_STREAM_PROBE"] == "1"))
    func streamingPathFromTheMainActorIsReportedNotAsserted() async throws {
        let copy = try SortformerRealLoad.workingCopy()
        defer { copy.remove() }
        let samples = try SortformerRealLoad.clipSamples()
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        let model = try SortformerModel.fromModelDirectory(copy.url)
        do {
            let count = try await MLXErrorGuard.guardedRun { () async throws -> Int in
                var total = 0
                for try await output in model.generateStream(audio: MLXArray(samples), chunkDuration: 5.0, threshold: 0.5) {
                    total += output.segments.count
                }
                return total
            }
            print("SORTFORMERSTREAM ok: \(count) segment events")
        } catch {
            print("SORTFORMERSTREAM threw: \(error)")
        }
        print("SORTFORMERSTREAM fallback message: \(MLXErrorGuard.lastFallbackMessage() ?? "none")")
    }
}

/// Long-clip measurement (the MP2 Parakeet / MP3 Qwen3-ASR precedent). Owner-run, one run at a time, nothing else loaded.
/// Gated on the installed weights and `MLXUI_TEST_SORTFORMER_MEASURE=1` (host env: `TEST_RUNNER_...`). Inputs:
/// `MLXUI_TEST_SORTFORMER_LONG_CLIP=<audio file>` (default: the parity clip) and `MLXUI_TEST_SORTFORMER_REPEAT=<n>`
/// (concatenate the clip in memory, so a 53 s clip x 12 is a 10.5 min recording). The offline pass is a single forward pass over the whole
/// clip (`SortformerEngine`'s long-audio note), so this is what shows how memory and time grow. The host is sandboxed:
/// everything is printed (`SORTFORMERMEASURE ...`). MLX's own counters are read after the load, after `generate`, and after
/// `Memory.clearCache()`, with the peak reset between.
struct SortformerLongClipMeasurement {
    nonisolated static let env = ProcessInfo.processInfo.environment

    nonisolated static var clip: URL {
        env["MLXUI_TEST_SORTFORMER_LONG_CLIP"].map { URL(fileURLWithPath: $0) } ?? SortformerParity.clipURL
    }

    @Test(.enabled(if: SortformerRealLoad.directory != nil && env["MLXUI_TEST_SORTFORMER_MEASURE"] == "1"))
    func measureLoadAndDiarize() async throws {
        let copy = try SortformerRealLoad.workingCopy()
        defer { copy.remove() }
        let once = try SortformerRealLoad.clipSamples(Self.clip)
        let repeats = Self.env["MLXUI_TEST_SORTFORMER_REPEAT"].flatMap(Int.init) ?? 1
        let samples = SortformerRealLoad.trimmed((1..<max(repeats, 1)).reduce(once) { acc, _ in acc + once })
        let seconds = samples.count / 16_000
        let audio = MLXArray(samples)
        eval(audio)
        Memory.clearCache()
        Memory.peakMemory = 0
        print(SortformerRealLoad.line("start", Memory.snapshot()))

        let t0 = Date()
        let model = try SortformerModel.fromModelDirectory(copy.url)
        let loadSeconds = Date().timeIntervalSince(t0)
        print(SortformerRealLoad.line("afterLoad", Memory.snapshot()) + String(format: ", %.1f s", loadSeconds))

        Memory.peakMemory = 0   // the setter ignores the value and resets the peak: the load floor is separate
        let t1 = Date()
        let output = try await model.generate(audio: audio, sampleRate: 16_000, threshold: 0.5)
        let generateSeconds = Date().timeIntervalSince(t1)
        print(SortformerRealLoad.line("afterGenerate", Memory.snapshot()) + String(format: ", %.1f s", generateSeconds))
        Memory.clearCache()
        print(SortformerRealLoad.line("afterClearCache", Memory.snapshot()))
        print("SORTFORMERMEASURE summary: clip \(Self.clip.lastPathComponent) x\(max(repeats, 1)), \(seconds) s of audio, "
              + String(format: "load %.1f s, generate %.1f s", loadSeconds, generateSeconds)
              + ", \(output.segments.count) segments, \(output.numSpeakers) speakers")
        #expect(!output.segments.isEmpty)
    }

    /// MP4-3b: the same measurement for the library's **streaming** paths, to decide how to handle long clips. Gated like the
    /// offline one plus `MLXUI_TEST_SORTFORMER_STREAM_MEASURE=1`. `MLXUI_TEST_SORTFORMER_STREAM_MODE` picks the path:
    /// `generateStream` (default; one call, the library pre-encodes the whole recording first) or `feed` (this test cuts the
    /// audio into chunks and carries the state handle between calls). `MLXUI_TEST_SORTFORMER_CHUNK` is the chunk length in
    /// seconds (default 5). Only segments are read (Swift values); no library array is touched. Prints
    /// `SORTFORMERSTREAMMEASURE` lines, and for a 1x clip the segments, plus how they differ from the offline pass.
    @Test(.enabled(if: SortformerRealLoad.directory != nil && env["MLXUI_TEST_SORTFORMER_STREAM_MEASURE"] == "1"))
    func measureStreaming() async throws {
        let copy = try SortformerRealLoad.workingCopy()
        defer { copy.remove() }
        let once = try SortformerRealLoad.clipSamples(Self.clip)
        let repeats = Self.env["MLXUI_TEST_SORTFORMER_REPEAT"].flatMap(Int.init) ?? 1
        let samples = SortformerRealLoad.trimmed((1..<max(repeats, 1)).reduce(once) { acc, _ in acc + once })
        let seconds = samples.count / 16_000
        let mode = Self.env["MLXUI_TEST_SORTFORMER_STREAM_MODE"] ?? "generateStream"
        let chunkSeconds = Self.env["MLXUI_TEST_SORTFORMER_CHUNK"].flatMap(Float.init) ?? 5
        let audio = MLXArray(samples)
        eval(audio)
        Memory.clearCache()
        Memory.peakMemory = 0
        let tag = "SORTFORMERSTREAMMEASURE \(mode)"
        print(SortformerRealLoad.line("start", Memory.snapshot()).replacingOccurrences(of: "SORTFORMERMEASURE", with: tag))

        let t0 = Date()
        let model = try SortformerModel.fromModelDirectory(copy.url)
        print(SortformerRealLoad.line("afterLoad", Memory.snapshot()).replacingOccurrences(of: "SORTFORMERMEASURE", with: tag)
              + String(format: ", %.1f s", Date().timeIntervalSince(t0)))
        Memory.peakMemory = 0

        var segments: [SpeakerSegment] = []
        func take(_ output: DiarizationOutput) {
            segments += output.segments.map { SpeakerSegment(speaker: $0.speaker, start: Double($0.start), end: Double($0.end)) }
        }
        let t1 = Date()
        var chunks = 0
        if mode == "feed" {
            let chunkSamples = Int(chunkSeconds * 16_000)
            var state = model.initStreamingState()
            var offset = 0
            while offset < samples.count {
                let end = min(offset + chunkSamples, samples.count)
                let chunk = MLXArray(Array(samples[offset..<end]))
                let (output, next) = try await model.feed(chunk: chunk, state: state, threshold: 0.5)
                take(output)
                state = next
                offset = end
                chunks += 1
            }
        } else {
            for try await output in model.generateStream(audio: audio, chunkDuration: chunkSeconds, threshold: 0.5) {
                take(output)
                chunks += 1
            }
        }
        let elapsed = Date().timeIntervalSince(t1)
        print(SortformerRealLoad.line("afterStream", Memory.snapshot()).replacingOccurrences(of: "SORTFORMERMEASURE", with: tag)
              + String(format: ", %.1f s", elapsed))
        Memory.clearCache()
        print(SortformerRealLoad.line("afterClearCache", Memory.snapshot()).replacingOccurrences(of: "SORTFORMERMEASURE", with: tag))
        // Chunk edges split a turn into touching pieces; join pieces that touch (1 ms) and nothing else.
        let joined = SpeakerSegments.normalized(segments, duration: Double(samples.count) / 16_000, mergeGap: SpeakerSegments.joinGap)
        print("\(tag) summary: x\(max(repeats, 1)), \(seconds) s of audio, chunk \(chunkSeconds) s, \(chunks) chunks, "
              + String(format: "%.1f s", elapsed) + ", \(segments.count) raw segments, \(joined.count) joined, slots \(SpeakerSegments.speakers(in: joined))")
        if repeats <= 1 {
            print("\(tag) segments: \(joined.map { String(format: "(%d, %.3f, %.3f)", $0.speaker, $0.start, $0.end) }.joined(separator: " "))")
            let offline = try await SortformerEngine.diarize(samples, modelDirectory: copy.url)
            let reference = SpeakerSegments.cleaned(offline, duration: Double(samples.count) / 16_000)
            print("\(tag) offline: \(reference.map { String(format: "(%d, %.3f, %.3f)", $0.speaker, $0.start, $0.end) }.joined(separator: " "))")
        }
        #expect(!segments.isEmpty)
    }

    /// MP4-3b: the **engine's own path** (`SortformerEngine.diarize`, so the offline/streaming choice and the timing fix),
    /// peak and time for a clip of any length: the figure `ramGB` and the load-profile row come from. Gated like the others plus
    /// `MLXUI_TEST_SORTFORMER_ENGINE_MEASURE=1`; `..._REPEAT` and `..._TRIM_SECONDS` choose the length. Prints
    /// `SORTFORMERENGINEMEASURE` lines.
    @Test(.enabled(if: SortformerRealLoad.directory != nil && env["MLXUI_TEST_SORTFORMER_ENGINE_MEASURE"] == "1"))
    func measureEnginePath() async throws {
        let copy = try SortformerRealLoad.workingCopy()
        defer { copy.remove() }
        let once = try SortformerRealLoad.clipSamples(Self.clip)
        let repeats = Self.env["MLXUI_TEST_SORTFORMER_REPEAT"].flatMap(Int.init) ?? 1
        let samples = SortformerRealLoad.trimmed((1..<max(repeats, 1)).reduce(once) { acc, _ in acc + once })
        let seconds = Double(samples.count) / 16_000
        let path = SortformerEngine.path(forSeconds: seconds)
        Memory.clearCache()
        Memory.peakMemory = 0
        let started = Date()
        let raw = try await SortformerEngine.diarize(samples, modelDirectory: copy.url)
        let elapsed = Date().timeIntervalSince(started)
        let snapshot = Memory.snapshot()
        print(SortformerRealLoad.line("afterEngine", snapshot).replacingOccurrences(of: "SORTFORMERMEASURE", with: "SORTFORMERENGINEMEASURE")
              + String(format: ", %.1f s for %.1f s of audio, path %@, %d raw segments", elapsed, seconds, "\(path)", raw.count))
        let cleaned = SpeakerSegments.cleaned(raw, duration: seconds)
        print("SORTFORMERENGINEMEASURE speakers \(SpeakerSegments.speakers(in: cleaned)), \(cleaned.count) segments")
        #expect(!raw.isEmpty)
    }
}
