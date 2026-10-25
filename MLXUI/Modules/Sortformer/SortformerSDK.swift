import Foundation

/// `ModelSDK` for MLX-native Sortformer speaker diarization, backed by `mlx-audio-swift`'s `SortformerModel` via
/// `SortformerEngine`. Claims `.diarization` + `source == .mlx` + Sortformer-family entries at `.exact`. No other SDK claims
/// that kind, and this one claims nothing else; `taskFamilies` stays empty and no flow task names the kind (MP-0 D).
nonisolated struct SortformerSDK: ModelSDK {
    let id = "sortformer"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .diarization,
              model.source == .mlx,
              model.family.lowercased().contains("sortformer") else {
            return .no
        }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        SortformerStage(modelID: model.id)
    }
}
