import SwiftUI

/// `ModelUI` for Clef Flash. `claim` mirrors `ClefSDK` so UI resolution matches SDK resolution. The
/// Run surface is `Views/Run/ClefRunView.swift` (CF-6, gate E1 — "a separate `ClefRunView` copied from
/// `CLMRunView`"), so this file only resolves and hands off, exactly as `LayaUI` and `CLMUI` do.
struct ClefUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .decision, model.source == .mlx, model.family == "Clef" else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(ClefRunView(
            modelDisplayName: model.displayName, license: model.license, modelID: model.id))
    }
}
