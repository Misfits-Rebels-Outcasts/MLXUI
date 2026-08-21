import Foundation

/// `ModelSDK` for MLX LLMs, backed by `mlx-swift-lm` via `LLMStage`/`LLMEngine`.
/// Claims every `.llm` entry (there is no other LLM SDK, so no tie can form).
/// `makeStage` binds the installed model directory. This is what makes every
/// CAT Flow frame task (`Summarize`, `Merge`, `Title`, `Rewrite`, …) and every
/// decider reachable from FlowKit: LLM rows previously had no registry path.
nonisolated struct ChatSDK: ModelSDK {
    let id = "chat"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .llm else { return .no }
        return .exact
    }

    /// H1: `LLMStage`'s convenience init is `@MainActor` while this protocol
    /// method is not. Mirror what it does rather than changing the protocol:
    /// capture the `Sendable` values (`dir`, `maxTokens`, `systemPrompt`)
    /// ahead of time and build the stage via its `nonisolated` designated init.
    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        let dir = LLMEngine.modelDirectory(for: model.id)
        let maxTokens = config.maxTokens
        return LLMStage(
            id: model.id,
            name: model.displayName,
            systemPrompt: config.systemPrompt,
            generate: { prompt in
                try await LLMEngine.generate(prompt: prompt, modelDir: dir, maxTokens: maxTokens)
            }
        )
    }
}
