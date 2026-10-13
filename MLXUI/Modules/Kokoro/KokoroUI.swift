import SwiftUI

/// `ModelUI` for Kokoro: presents the reusable `TTSRunView` for a resolved Kokoro stage.
/// `claim` mirrors `KokoroSDK` so UI resolution matches SDK resolution.
struct KokoroUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .tts,
              model.family.lowercased().contains("kokoro") else {
            return .no
        }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        // The view builds a TTSStage per run with the chosen voice/speed and runs it through the handed
        // `stage` (SC-5c: ask, lease, booking).
        AnyView(TTSRunView(modelDisplayName: model.displayName,
                           license: model.license,
                           modelID: model.id,
                           voices: KokoroEngine.availableVoices(modelID: model.id),
                           handedStage: stage))
    }
}
