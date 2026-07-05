import Foundation

/// `ModelSDK` for MLX-native Whisper ASR, backed by `mlx-audio-swift` via
/// `MLXWhisperEngine`. Claims `.asr` + `source == .mlx` + Whisper-family entries so it wins
/// the `.exact` tie over the WhisperKit `WhisperSDK` (which stays the fallback for any
/// non-mlx whisper) — provided `MLXWhisperModule` is registered first (see `ModelModules`).
/// Reuses the existing `ASRStage` (resamples to 16 kHz) with an MLX-backed transcribe closure.
nonisolated struct MLXWhisperSDK: ModelSDK {
    let id = "mlx-whisper"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .asr,
              model.source == .mlx,
              model.family.lowercased().contains("whisper") else {
            return .no
        }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        let directory = MLXWhisperEngine.installedModelDirectory(id: model.id)
        let language = config.language
        return ASRStage(id: "mlx-whisper.\(model.id)", name: "Whisper ASR (MLX)") { samples in
            try await MLXWhisperEngine.transcribe(samples, modelDirectory: directory, language: language)
        }
    }
}
