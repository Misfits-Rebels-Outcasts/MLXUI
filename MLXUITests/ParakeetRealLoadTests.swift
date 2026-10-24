import Foundation
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
