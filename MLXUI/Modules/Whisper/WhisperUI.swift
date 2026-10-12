import SwiftUI

/// `ModelUI` for Whisper: presents the reusable `ASRRunView` for a resolved Whisper stage.
/// `claim` mirrors `WhisperSDK` so UI resolution stays consistent with SDK resolution.
struct WhisperUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .asr,
              model.family.lowercased().contains("whisper") else {
            return .no
        }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        // MLX Whisper conditions on a language token; give it a language picker that rebuilds
        // the stage per run. Gated to the MLX Whisper family (this UI is also reused by Voxtral
        // and WhisperKit, whose engines don't take a language) — matches `MLXWhisperSDK.claim`.
        var rebuild: ((String?) -> any PipelineStage)? = nil
        if model.source == .mlx, model.family.lowercased().contains("whisper") {
            let sdk = MLXWhisperSDK()
            // SC-2b: each rebuild returns the previous stage's lease and leases the new one.
            rebuild = { [stage] language in
                LeasingStage.rebuilding(stage, model: model, sdk: sdk, config: StageConfig(language: language)) ?? stage
            }
        }
        return AnyView(ASRRunView(modelDisplayName: model.displayName,
                                  license: model.license,
                                  stage: stage,
                                  rebuildStage: rebuild))
    }
}
