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
        // The view builds a fresh TTSStage per run with the chosen voice/speed, so it needs
        // the model id (and the installed voice list for the picker) rather than the
        // pre-built `stage`.
        AnyView(TTSRunView(modelDisplayName: model.displayName,
                           license: model.license,
                           modelID: model.id,
                           voices: KokoroEngine.availableVoices(modelID: model.id)))
    }
}
