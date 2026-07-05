import SwiftUI

/// `ModelUI` for MLX embedders: presents `EmbeddingRunView`. `claim` mirrors `EmbeddingSDK`
/// so UI resolution matches SDK resolution.
struct EmbeddingUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .embedding, model.source == .mlx else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(EmbeddingRunView(modelDisplayName: model.displayName,
                                 license: model.license,
                                 modelID: model.id,
                                 family: model.family))
    }
}
