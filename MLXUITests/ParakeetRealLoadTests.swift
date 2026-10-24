import Foundation
import MLX
import MLXAudioSTT
import Testing
@testable import MLXUI

/// MP2-3 — the real load of Parakeet TDT 0.6B v3 (KICKOFF rule 13b) and the main-actor probe (rule 17c,
/// owner ruling K, 2026-10-24). Owner-run, gated on the weights (2.51 GB; one real model at a time).
/// Skipped unless `mlx-community--parakeet-tdt-0.6b-v3/.installed` exists under the directory this process
/// resolves: `ModelStore`'s, or `MLXUI_TEST_PARAKEET_DIR=<model folder>` (passed to the host as
/// `TEST_RUNNER_MLXUI_TEST_PARAKEET_DIR`). Run the host unsandboxed. The exact command is in the MP2-3
/// journal. A skipped run does not count.
enum ParakeetRealLoad {
    nonisolated static let directory: URL? = {
        let url: URL
        if let override = ProcessInfo.processInfo.environment["MLXUI_TEST_PARAKEET_DIR"] {
            url = URL(fileURLWithPath: override)
        } else {
            url = ModelStore.shared.directory(forHFModelID: "mlx-community/parakeet-tdt-0.6b-v3")
        }
        let marker = url.appendingPathComponent(".installed")
        return FileManager.default.fileExists(atPath: marker.path) ? url : nil
    }()

    /// The bundled English sample, as the 16 kHz mono buffer `ASRStage` hands the engine.
    nonisolated static func sampleSamples() throws -> [Float] {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("MLXUI/Resources/Samples/sample-audio.m4a")
        let buffer = try AudioFileReader.read(url)
        return try AudioResampler.resample(buffer, to: 16_000).samples
    }

    /// The sample is the `say`-generated voice memo from catflow-mlx's `01-SpokenSummary`
    /// ("Quick voice memo. Follow up with the design team …").
    nonisolated static func readsTheMemo(_ transcript: String) -> Bool {
        transcript.lowercased().contains("memo")
    }
}

struct ParakeetRealLoadTests {
    /// Loads through `ParakeetEngine`'s own path (`ParakeetModel.fromDirectory` on the installed folder).
    /// Success here with `tokenizer.vocab` and `vocab.txt` absent from the install proves T2: the loader
    /// needs only config.json + the safetensors.
    @Test(.enabled(if: ParakeetRealLoad.directory != nil))
    func theCheckpointLoadsAndTranscribesTheSample() async throws {
        let dir = try #require(ParakeetRealLoad.directory)
        let samples = try ParakeetRealLoad.sampleSamples()
        let text = try await ParakeetEngine.transcribe(samples, modelDirectory: dir)
        print("PARAKEET transcript: \(text)")
        #expect(!text.isEmpty)
        #expect(ParakeetRealLoad.readsTheMemo(text), "\(text)")
    }
}

/// The shape the Run UI gives the engines: awaited from a main-actor `Task`, inside
/// `MLXErrorGuard.guardedRun` (`ASRRunView`). 5c4c12b's Parakeet uses `MLX.compile` and reads its own
/// arrays with bare `.item` / `.asArray`, which our guard cannot wrap, so this checks real output and
/// that no MLX error escaped through the process-wide fallback.
@MainActor
struct ParakeetMainActorProbeTests {
    @Test(.enabled(if: ParakeetRealLoad.directory != nil))
    func transcribeFromTheMainActorReturnsTheTranscript() async throws {
        let dir = try #require(ParakeetRealLoad.directory)
        let samples = try ParakeetRealLoad.sampleSamples()
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        for _ in 0..<2 {   // twice: the second call reuses nothing but must not differ
            let text = try await MLXErrorGuard.guardedRun {
                try await ParakeetEngine.transcribe(samples, modelDirectory: dir)
            }
            print("PARAKEETPROBE transcript: \(text)")
            #expect(!text.isEmpty)
            #expect(ParakeetRealLoad.readsTheMemo(text), "\(text)")
        }
        #expect(MLXErrorGuard.lastFallbackMessage() == nil)
    }
}

/// MP2-3-FIX-2 measurement (owner rulings X, Y, Z of 2026-10-24). Owner-run, one run at a time, nothing else loaded.
/// Gated on the installed weights and `MLXUI_TEST_PARAKEET_MEASURE=1` (host env: `TEST_RUNNER_…`). Inputs:
/// `MLXUI_TEST_PARAKEET_LONG_CLIP=<audio file>` (default: the bundled sample) and `MLXUI_TEST_PARAKEET_CHUNK=<seconds>`
/// (default: `ParakeetEngine.chunkSeconds`; a test-only override, the engine constant is unchanged).
/// The host is sandboxed, so nothing is written to disk: everything is printed to the log (`PARAKEETMEASURE …`).
/// MLX's own counters are read (a) right after `ParakeetModel.fromDirectory` and (b) right after `generate`, after
/// `Memory.clearCache()` and a peak reset, so the load floor is separate from the transcription. `peakMemory` is the
/// peak of *active* MLX memory since the reset (it excludes the buffer cache).
struct ParakeetLongClipMeasurement {
    nonisolated static let env = ProcessInfo.processInfo.environment

    nonisolated static var clip: URL {
        if let path = env["MLXUI_TEST_PARAKEET_LONG_CLIP"] { return URL(fileURLWithPath: path) }
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("MLXUI/Resources/Samples/sample-audio.m4a")
    }

    nonisolated static func line(_ label: String, _ s: Memory.Snapshot) -> String {
        func gib(_ b: Int) -> String { String(format: "%.3f GiB", Double(b) / 1_073_741_824) }
        return "PARAKEETMEASURE \(label): active \(s.activeMemory) B (\(gib(s.activeMemory))), cache \(s.cacheMemory) B (\(gib(s.cacheMemory))), peak \(s.peakMemory) B (\(gib(s.peakMemory)))"
    }

    /// Direct path: load, generate, then `Memory.clearCache()`, with the three counters at each point. The model is a
    /// local here, so it is released when this returns.
    nonisolated static func directPhase(dir: URL, audio: MLXArray, chunk: Float, seconds: Int) throws -> String {
        let t0 = Date()
        let model = try ParakeetModel.fromDirectory(dir)
        let loadSeconds = Date().timeIntervalSince(t0)
        print(line("afterLoad", Memory.snapshot()) + String(format: ", %.1f s", loadSeconds))

        let base = model.defaultGenerationParameters
        let params = STTGenerateParameters(
            maxTokens: base.maxTokens, temperature: base.temperature, topP: base.topP, topK: base.topK,
            verbose: base.verbose, language: base.language, chunkDuration: chunk,
            minChunkDuration: base.minChunkDuration, repetitionPenalty: base.repetitionPenalty,
            repetitionContextSize: base.repetitionContextSize, kvBits: base.kvBits,
            kvGroupSize: base.kvGroupSize, quantizedKVStart: base.quantizedKVStart)
        let t1 = Date()
        let output = model.generate(audio: audio, generationParameters: params)
        let text = output.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let generateSeconds = Date().timeIntervalSince(t1)
        print(line("afterGenerate", Memory.snapshot()) + String(format: ", %.1f s", generateSeconds))
        Memory.clearCache()
        print(line("afterClearCache", Memory.snapshot()))

        print("PARAKEETMEASURE summary: clip \(clip.lastPathComponent), \(seconds) s of audio, chunk \(chunk) s, "
              + String(format: "load %.1f s, generate %.1f s", loadSeconds, generateSeconds)
              + ", \(text.split(separator: " ").count) words")
        // Sentence timing from the model, to place chunk boundaries (every `chunk - 2` s of audio) in the transcript.
        for (i, seg) in (output.segments ?? []).enumerated() {
            print("PARAKEETMEASURE seg \(i): " + seg.map { "\($0.key)=\($0.value)" }.sorted().joined(separator: " | "))
        }
        print("PARAKEETMEASURE transcript-begin")
        print(text)
        print("PARAKEETMEASURE transcript-end")
        return text
    }

    @Test(.enabled(if: ParakeetRealLoad.directory != nil && env["MLXUI_TEST_PARAKEET_MEASURE"] == "1"))
    func measureLoadAndTranscribe() async throws {
        let dir = try #require(ParakeetRealLoad.directory)
        let buffer = try AudioFileReader.read(Self.clip)
        let once = try AudioResampler.resample(buffer, to: 16_000).samples
        // MLXUI_TEST_PARAKEET_REPEAT=3 concatenates the clip in memory (~29 min for the 588 s clip); no new file.
        let repeats = Self.env["MLXUI_TEST_PARAKEET_REPEAT"].flatMap(Int.init) ?? 1
        let samples = (1..<max(repeats, 1)).reduce(once) { acc, _ in acc + once }
        let audio = MLXArray(samples)
        let chunk = Self.env["MLXUI_TEST_PARAKEET_CHUNK"].flatMap(Float.init) ?? ParakeetEngine.chunkSeconds

        eval(audio)
        Memory.clearCache()
        Memory.peakMemory = 0   // the setter ignores the value and resets the peak
        print(Self.line("start", Memory.snapshot()))
        let text = try Self.directPhase(dir: dir, audio: audio, chunk: chunk, seconds: samples.count / 16_000)
        #expect(!text.isEmpty)

        // The engine's own path (owner ruling AB: it returns the cache itself), after the model above is released:
        // what the app keeps after one transcription.
        if Self.env["MLXUI_TEST_PARAKEET_ENGINE_PHASE"] == "1" {
            Memory.clearCache()
            Memory.peakMemory = 0
            print(Self.line("engineStart", Memory.snapshot()))
            let t2 = Date()
            let engineText = try await ParakeetEngine.transcribe(samples, modelDirectory: dir)
            print(Self.line("engineAfterTranscribe", Memory.snapshot()) + String(format: ", %.1f s", Date().timeIntervalSince(t2)))
            print("PARAKEETMEASURE engineWords \(engineText.split(separator: " ").count)")
            #expect(!engineText.isEmpty)
        }
    }
}
