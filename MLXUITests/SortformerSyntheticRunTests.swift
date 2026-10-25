import Foundation
import MLX
import MLXNN
import MLXAudioVAD
import Testing
@testable import MLXUI

/// MP4-3a: runs the real `SortformerEngine` path (`SortformerModel.fromModelDirectory`, `MLXArray` input, the library's offline
/// `generate`, `predsToSegments`) on a **tiny randomly initialised model** written to a temp folder, from the main actor inside
/// `MLXErrorGuard.guardedRun` -- the way the Run UI calls it. It needs no weights, so it runs in every suite. It proves the
/// loader keys, the threading (no "no Stream in current thread" error from the offline path, KICKOFF rule 17) and the engine's
/// segment mapping; it says nothing about accuracy (the weights are noise). The real checkpoint is `SortformerRealLoadTests`.
@MainActor
struct SortformerSyntheticRunTests {

    /// The pinned config's structure at toy sizes (mel bins, subsampling factor, speakers and processor settings unchanged).
    private static let tinyConfig = """
    {
      "model_type": "sortformer",
      "num_speakers": 4,
      "fc_encoder_config": {
        "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
        "intermediate_size": 64, "num_mel_bins": 128, "conv_kernel_size": 9, "subsampling_factor": 8,
        "subsampling_conv_channels": 16, "subsampling_conv_kernel_size": 3, "subsampling_conv_stride": 2,
        "max_position_embeddings": 5000, "attention_bias": true, "scale_input": true
      },
      "tf_encoder_config": {
        "d_model": 16, "encoder_layers": 2, "encoder_attention_heads": 4, "encoder_ffn_dim": 32,
        "max_source_positions": 1500, "k_proj_bias": true
      },
      "modules_config": {
        "num_speakers": 4, "fc_d_model": 32, "tf_d_model": 16, "subsampling_factor": 8, "chunk_len": 188, "fifo_len": 0,
        "spkcache_len": 188, "spkcache_update_period": 188, "chunk_left_context": 1, "chunk_right_context": 1,
        "spkcache_sil_frames_per_spk": 3, "pred_score_threshold": 0.25, "max_index": 99999, "scores_boost_latest": 0.05,
        "sil_threshold": 0.2, "strong_boost_rate": 0.75, "weak_boost_rate": 1.5, "min_pos_scores_rate": 0.5, "use_aosc": true
      },
      "processor_config": {
        "feature_size": 128, "sampling_rate": 16000, "hop_length": 160, "n_fft": 512, "win_length": 400, "preemphasis": 0.97
      }
    }
    """

    /// A temp model folder: `config.json` + the random model's own parameters as `model.safetensors`.
    private func writeTinyModel() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("sortformer-tiny-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Self.tinyConfig.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let config = try JSONDecoder().decode(SortformerConfig.self, from: Data(Self.tinyConfig.utf8))
        let model = SortformerModel(config)
        eval(model.parameters())
        let arrays = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))
        return dir
    }

    @Test func theEnginePathRunsFromTheMainActorOnATinyRandomModel() async throws {
        let dir = try writeTinyModel()
        defer { try? FileManager.default.removeItem(at: dir) }
        let clip = try SortformerRealLoad.clipSamples()
        let duration = Double(clip.count) / 16_000
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        for _ in 0..<2 {
            let raw = try await MLXErrorGuard.guardedRun {
                try await SortformerEngine.diarize(clip, modelDirectory: dir)
            }
            let segments = SpeakerSegments.normalized(raw, duration: duration)
            // Random weights: no claim about the content, only about shape and bounds.
            for s in segments {
                #expect(s.speaker >= 0 && s.speaker < 4)
                #expect(s.start >= 0 && s.end <= duration + 0.001 && s.end > s.start)
            }
            #expect(segments == SpeakerSegments.normalized(segments))
        }
        #expect(MLXErrorGuard.lastFallbackMessage() == nil)
    }

    @Test func aStageBuiltOnTheTinyFolderReturnsParsableCSV() async throws {
        let dir = try writeTinyModel()
        defer { try? FileManager.default.removeItem(at: dir) }
        let clip = try SortformerRealLoad.clipSamples()
        let stage = SortformerStage(id: "tiny", name: "tiny") { samples in
            try await SortformerEngine.diarize(samples, threshold: 0.5, modelDirectory: dir)
        }
        let out = try await MLXErrorGuard.guardedRun {
            try await stage.run(.audio(AudioBuffer(samples: clip, sampleRate: 16_000)), progress: { _ in })
        }
        guard case let .text(csv) = out else { Issue.record("expected text"); return }
        _ = try SpeakerSegments.parseCSV(csv)
    }

    @Test func aFolderWithoutWeightsFailsAsAStageError() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("sortformer-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.tinyConfig.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        await #expect(throws: StageError.self) {
            _ = try await SortformerEngine.diarize([0.1, 0.2, 0.3], modelDirectory: dir)
        }
    }
    /// MP4-3b: the engine's **streaming branch** (`feed`, chunk by chunk, state carried between calls) on the tiny model, from the
    /// main actor inside `guardedRun`. `offlineLimit: 1` sends the 52 s clip down it (11 chunks of 5 s, the last folded). Shape
    /// and bounds only (random weights): no stream error, no fallback message, sorted cleaned output inside the clip.
    @Test func theStreamingBranchRunsFromTheMainActorOnATinyRandomModel() async throws {
        let dir = try writeTinyModel()
        defer { try? FileManager.default.removeItem(at: dir) }
        let clip = try SortformerRealLoad.clipSamples()
        let duration = Double(clip.count) / 16_000
        #expect(SortformerEngine.path(forSeconds: duration, offlineLimit: 1) == .streaming)
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        for _ in 0..<2 {
            let raw = try await MLXErrorGuard.guardedRun {
                try await SortformerEngine.diarize(clip, modelDirectory: dir, offlineLimit: 1)
            }
            let segments = SpeakerSegments.cleaned(raw, duration: duration)
            for s in segments {
                #expect(s.speaker >= 0 && s.speaker < 4)
                #expect(s.start >= 0 && s.end <= duration + 0.001 && s.end > s.start)
            }
            #expect(SpeakerSegments.speakers(in: segments) == Array(0..<SpeakerSegments.speakers(in: segments).count), "compact numbering")
        }
        #expect(MLXErrorGuard.lastFallbackMessage() == nil)
    }

    @Test func aRecordingPastTheLimitIsRefusedBeforeAnyModelLoads() async throws {
        // No model folder at all: the limit check must come first. 3601 s of silence is 230 MB of floats; use a sparse stand-in
        // by asking the pure check, and the engine with a length just over the limit.
        let long = [Float](repeating: 0, count: Int(SortformerEngine.maxSeconds) * 16_000 + 16_000)
        await #expect(throws: SortformerEngine.LimitError.self) {
            _ = try await SortformerEngine.diarize(long, modelDirectory: URL(fileURLWithPath: "/nonexistent"))
        }
    }
}
