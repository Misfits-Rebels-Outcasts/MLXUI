import Foundation

/// `ModelSDK` for MLX-native Qwen3 forced alignment, backed by `mlx-audio-swift`'s `Qwen3ForcedAlignerModel` via
/// `ForcedAlignerEngine`. Claims `.align` + `source == .mlx` + the **exact** `Qwen3-ForcedAligner` family (trap 3): the
/// repo is tagged `qwen3_asr` like Qwen3-ASR 1.7B, and Qwen3-ASR's SDK claims families containing `qwen3-asr`, which
/// `Qwen3-ForcedAligner` does not contain — so neither SDK can take the other's entry. `taskFamilies` stays empty.
nonisolated struct ForcedAlignerSDK: ModelSDK {
    let id = "forced-aligner"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .align,
              model.source == .mlx,
              model.family.lowercased() == "qwen3-forcedaligner" else {
            return .no
        }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        // A Run sheet or a flow row always supplies its own stage (the transcript is per-run data); this is the builder
        // those two replace, and it refuses an empty transcript on the run with a plain sentence.
        ForcedAlignerStage(modelID: model.id,
                           transcript: config.transcript ?? "",
                           language: config.language ?? ForcedAlignerEngine.defaultLanguageCode)
    }
}
