import SwiftUI

/// `ModelUI` for MLX VLMs: presents the reusable `ImageQARunView`. `claim` mirrors `VLMSDK`
/// so UI resolution matches SDK resolution.
struct VLMUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .vision, model.source == .mlx else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        // The view builds a fresh VLMStage per run with the typed prompt, so it needs the
        // model id (and display metadata) rather than the pre-built `stage` — mirrors KokoroUI.
        AnyView(ImageQARunView(modelDisplayName: model.displayName,
                               license: model.license,
                               modelID: model.id))
    }
}
