import Foundation

/// `ModelSDK` for MLX-native sound isolation, backed by `mlx-audio-swift`'s `SAMAudio` via ``SAMAudioEngine``.
/// Claims **`.separate` + `source == .mlx` + the SAM-Audio family only** (Trap 6: by family, never by architecture) at
/// `.exact`. No other SDK claims that kind, so the first-match tie-break cannot hand this model to another module.
nonisolated struct SAMAudioSDK: ModelSDK {
    let id = "samaudio"

    /// The isolation description travels on `StageConfig.prompt`; `speech` is the default a generic caller gets.
    var promptSupport: PromptSupport { .freeText(default: Self.defaultDescription) }

    static let defaultDescription = "speech"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .separate,
              model.source == .mlx,
              model.family.lowercased() == "sam-audio" else {
            return .no
        }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        SAMAudioStage(modelID: model.id, description: config.prompt ?? Self.defaultDescription)
    }
}
