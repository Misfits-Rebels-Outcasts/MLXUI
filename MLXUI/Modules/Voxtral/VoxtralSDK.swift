import Foundation

/// `ModelSDK` for MLX-native Voxtral ASR, backed by `mlx-audio-swift`'s `VoxtralRealtimeModel`
/// via `VoxtralEngine`. Claims `.asr` + `source == .mlx` + Voxtral-family entries at `.exact`.
/// The MLX-Whisper SDK declines non-whisper families, so there's no conflict. Reuses the
/// existing `ASRStage` (resamples to 16 kHz) with a Voxtral-backed transcribe closure.
nonisolated struct VoxtralSDK: ModelSDK {
    let id = "voxtral"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .asr,
              model.source == .mlx,
              model.family.lowercased().contains("voxtral") else {
            return .no
        }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        let directory = VoxtralEngine.installedModelDirectory(id: model.id)
        return ASRStage(id: "voxtral.\(model.id)", name: "Voxtral ASR") { samples in
            try await VoxtralEngine.transcribe(samples, modelDirectory: directory)
        }
    }
}
