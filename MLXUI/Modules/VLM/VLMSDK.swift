import Foundation

/// `ModelSDK` for MLX vision-language models, backed by `mlx-swift-lm`'s `MLXVLM` via
/// `VLMStage`/`VLMEngine`. Claims `.vision` + `source == .mlx` entries (LFM2-VL, gemma-3-VL,
/// Qwen3-VL). `makeStage` binds the installed model directory; the prompt comes from
/// `StageConfig.prompt` (the run view supplies the user's question per run).
nonisolated struct VLMSDK: ModelSDK {
    let id = "vlm"

    /// Default question when none is supplied (e.g. via the pipeline factory).
    static let defaultPrompt = "Describe this image in detail."

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .vision, model.source == .mlx else { return .no }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        VLMStage(modelID: model.id,
                 prompt: config.prompt ?? Self.defaultPrompt,
                 maxTokens: config.maxTokens)
    }
}
