import Foundation

/// `ModelSDK` for MLX-native Qwen3 forced alignment, backed by `mlx-audio-swift`'s `Qwen3ForcedAlignerModel` via
/// `ForcedAlignerEngine`. Claims `.align` + `source == .mlx` + the **exact** `Qwen3-ForcedAligner` family (trap 3): the
/// repo is tagged `qwen3_asr` like Qwen3-ASR 1.7B, and Qwen3-ASR's SDK claims families containing `qwen3-asr`, which
/// `Qwen3-ForcedAligner` does not contain — so neither SDK can take the other's entry. `taskFamilies` stays empty.
///
/// `StageBooksOwnWeights` (FA-R20): the engine books each run's memory on the shared budget itself, so `EngineCache`
/// books 0 for the stage and the "make room?" question asks for the files' charge plus the working set.
nonisolated struct ForcedAlignerSDK: ModelSDK, StageBooksOwnWeights {
    let id = "forced-aligner"

    /// The catalog's `ramGB` in bytes: what the booking uses for the weights when there are no files to measure.
    static func footprintBytes(for model: ModelEntry) -> Int64 { Int64(model.ramGB * 1_073_741_824) }

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .align,
              model.source == .mlx,
              model.family.lowercased() == "qwen3-forcedaligner" else {
            return .no
        }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        // Built from the language alone; each run supplies its transcript through `AlignTranscript` (ruling 17).
        ForcedAlignerStage(modelID: model.id,
                           language: config.language ?? ForcedAlignerEngine.defaultLanguageCode,
                           footprintBytes: Self.footprintBytes(for: model))
    }
}
