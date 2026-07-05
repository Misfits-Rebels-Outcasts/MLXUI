import Foundation

/// `ModelSDK` for MLX text embedding, backed by `mlx-swift-lm`'s `MLXEmbedders` via
/// `EmbeddingStage`/`EmbeddingEngine`. Claims `.embedding` + `source == .mlx` entries
/// (all-MiniLM, embeddinggemma, nomic, bge-m3). `makeStage` binds the installed directory.
nonisolated struct EmbeddingSDK: ModelSDK {
    let id = "embedding"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .embedding, model.source == .mlx else { return .no }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        EmbeddingStage(modelID: model.id)
    }
}
