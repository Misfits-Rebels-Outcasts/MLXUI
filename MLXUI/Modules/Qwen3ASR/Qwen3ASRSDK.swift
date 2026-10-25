import Foundation

/// `ModelSDK` for MLX-native Qwen3-ASR, backed by `mlx-audio-swift`'s `Qwen3ASRModel` via `Qwen3ASREngine`.
/// Claims `.asr` + `source == .mlx` + Qwen3-ASR-family entries at `.exact`; the Whisper, Voxtral and Parakeet SDKs
/// decline that family and this one declines theirs (readme T3). Reuses the existing `ASRStage` (resamples to 16 kHz
/// mono) with a Qwen3-ASR-backed transcribe closure.
nonisolated struct Qwen3ASRSDK: ModelSDK {
    let id = "qwen3-asr"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .asr,
              model.source == .mlx,
              model.family.lowercased().contains("qwen3-asr") else {
            return .no
        }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        let directory = Qwen3ASREngine.installedModelDirectory(id: model.id)
        return ASRStage(id: "qwen3-asr.\(model.id)", name: "Qwen3-ASR") { samples in
            try await Qwen3ASREngine.transcribe(samples, modelDirectory: directory)
        }
    }
}
