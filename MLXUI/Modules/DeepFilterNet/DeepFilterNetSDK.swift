import Foundation

/// `ModelSDK` for MLX-native DeepFilterNet speech enhancement, backed by `mlx-audio-swift`'s `DeepFilterNetModel` via
/// `DeepFilterNetEngine`. Claims `.enhance` + `source == .mlx` + DeepFilterNet-family entries at `.exact`. No other SDK claims
/// that kind, and this one claims nothing else; `taskFamilies` stays empty and no flow task names the kind (MP-0 D).
nonisolated struct DeepFilterNetSDK: ModelSDK {
    let id = "deepfilternet"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .enhance,
              model.source == .mlx,
              model.family.lowercased().contains("deepfilternet") else {
            return .no
        }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        DeepFilterNetStage(modelID: model.id)
    }
}
