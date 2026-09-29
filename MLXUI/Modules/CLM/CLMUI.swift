import SwiftUI

/// `ModelUI` for CLM. `claim` mirrors `CLMSDK` so UI resolution matches SDK resolution.
/// `makeRunView` returns the generic `UnsupportedModelView` for now — CL-1's `ModelSupport`
/// gap flag still holds (the owner's ruling, 2026-09-28: land the engine/SDK/UI wiring in
/// CL-4b without a real Run UI yet, since the dedicated `CLMRunView` is explicitly CL-5's
/// job) — so a caller never actually reaches this in practice; ModelDetailView/Run short-
/// circuit to the gap flag's own "not yet supported" messaging first. CL-5 replaces this.
struct CLMUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .decision, model.source == .mlx, model.family == "CLM" else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(UnsupportedModelView(model: model))
    }
}
