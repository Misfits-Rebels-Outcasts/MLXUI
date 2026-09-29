import SwiftUI

/// `ModelUI` for CLM. `claim` mirrors `CLMSDK` so UI resolution matches SDK resolution. The
/// real Run surface is `Views/Run/CLMRunView.swift` (CL-5, gate G "a duplicate Run screen") —
/// mirrors `LayaUI`'s split (the SDK/module file just resolves and hands off). CL-1's
/// `ModelSupport` gap flag is removed now that this real Run screen exists.
struct CLMUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .decision, model.source == .mlx, model.family == "CLM" else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(CLMRunView(modelDisplayName: model.displayName, license: model.license, modelID: model.id))
    }
}
