import Foundation

/// `ModelSDK` for LFM2.5-Audio, backed by `mlx-audio-swift`'s `LFM2AudioModel` via `LFMAudioEngine`. Claims `.voice` +
/// `source == .mlx` + LFM2.5-Audio-family entries at `.exact`; nothing else claims that kind and this claims nothing else.
///
/// `makeStage` returns the **speech-to-text** stage, the one mode that is a linear `audio -> text` step. The other modes are
/// built by ``LFMAudioStages/tts(modelID:style:)`` and driven by the Run UI (voice turn, MP6-3b). `taskFamilies` stays empty and
/// no flow task names the kind until MP6-W.
nonisolated struct LFMAudioSDK: ModelSDK {
    let id = "lfmaudio"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .voice,
              model.source == .mlx,
              model.family.lowercased().contains("lfm2.5-audio") else {
            return .no
        }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        LFMAudioStages.asr(modelID: model.id)
    }
}
