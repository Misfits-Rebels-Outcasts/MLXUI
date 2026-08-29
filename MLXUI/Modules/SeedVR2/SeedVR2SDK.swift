import Foundation
import SwiftUI

/// `ModelSDK` for SeedVR2 3B image super-resolution.
/// Claims models with `runnerKind == .upscale` and "seedvr2" in the id/hfModelId.
nonisolated struct SeedVR2SDK: ModelSDK {
    let id = "seedvr2"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .upscale else { return .no }
        let hay = (model.id + " " + model.hfModelId).lowercased()
        guard hay.contains("seedvr2") else { return .no }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        SeedVR2Stage(modelID: model.id, scale: 2)
    }
}

/// `ModelUI` for SeedVR2 — routes to `SeedVR2RunView` (SV-AM5).
struct SeedVR2UI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .upscale else { return .no }
        let hay = (model.id + " " + model.hfModelId).lowercased()
        guard hay.contains("seedvr2") else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(SeedVR2RunView(modelDisplayName: model.displayName, modelID: model.id))
    }
}
