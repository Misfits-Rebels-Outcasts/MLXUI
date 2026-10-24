import Foundation

/// `ModelSDK` for MLX-native Parakeet ASR, backed by `mlx-audio-swift`'s `ParakeetModel` via
/// `ParakeetEngine`. Claims `.asr` + `source == .mlx` + Parakeet-family entries at `.exact`; the
/// Whisper and Voxtral SDKs decline that family and this one declines theirs. Reuses the existing
/// `ASRStage` (resamples to 16 kHz mono) with a Parakeet-backed transcribe closure.
nonisolated struct ParakeetSDK: ModelSDK {
    let id = "parakeet"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .asr,
              model.source == .mlx,
              model.family.lowercased().contains("parakeet") else {
            return .no
        }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        let directory = ParakeetEngine.installedModelDirectory(id: model.id)
        return ASRStage(id: "parakeet.\(model.id)", name: "Parakeet ASR") { samples in
            try await ParakeetEngine.transcribe(samples, modelDirectory: directory)
        }
    }
}
