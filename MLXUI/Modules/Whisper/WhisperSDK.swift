import Foundation

/// `ModelSDK` for OpenAI Whisper ASR models, backed by WhisperKit via `ASRStage`. Claims
/// only `.asr` models whose family is Whisper, so Voxtral — also `.asr` but **not** Whisper
/// — returns `.no` and falls through to the unsupported path.
nonisolated struct WhisperSDK: ModelSDK {
    let id = "whisper"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .asr,
              model.family.lowercased().contains("whisper") else {
            return .no
        }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        // WhisperKit manages its own CoreML model; the catalog entry only selects the size.
        ASRStage(model: WhisperKitSize.from(model))
    }
}
