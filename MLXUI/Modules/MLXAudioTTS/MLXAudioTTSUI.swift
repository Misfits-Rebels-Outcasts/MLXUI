import SwiftUI

/// `ModelUI` for non-Kokoro MLX TTS: presents `MLXAudioTTSRunView`. `claim` mirrors
/// `MLXAudioTTSSDK` so UI resolution matches SDK resolution.
struct MLXAudioTTSUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .tts,
              model.source == .mlx,
              !model.family.lowercased().contains("kokoro") else {
            return .no
        }
        return .generic
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(MLXAudioTTSRunView(modelDisplayName: model.displayName,
                                   license: model.license,
                                   modelID: model.id,
                                   family: model.family))
    }
}
