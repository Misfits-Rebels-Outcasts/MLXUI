import SwiftUI

/// `ModelUI` for Laya. `claim` mirrors `LayaSDK` so UI resolution matches SDK resolution. The
/// real Run surface is `Views/Run/LayaRunView.swift` (LY-5) — mirrors `EmbeddingUI`/
/// `ImageQARunView`'s split (the SDK/module file just resolves and hands off).
struct LayaUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .decision, model.source == .mlx else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(LayaRunView(modelDisplayName: model.displayName, license: model.license, modelID: model.id))
    }
}
