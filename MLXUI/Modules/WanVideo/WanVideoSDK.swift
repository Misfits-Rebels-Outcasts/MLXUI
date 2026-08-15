import Foundation
import SwiftUI

/// `ModelSDK` for WAN 2.1 text → video (T2V-1.3B). Claims `runnerKind == .video`.
nonisolated struct WanVideoSDK: ModelSDK {
    let id = "wanvideo"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .video else { return .no }
        let haystack = (model.id + " " + model.hfModelId).lowercased()
        guard haystack.contains("wan") else { return .no }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        WanVideoStage(modelID: model.id)
    }
}

/// `ModelUI` for WAN 2.1 — displays the FLUX-style run view (prompt + progress + first frame preview).
/// A dedicated `WanVideoRunView` with video playback is AM5.
struct WanVideoUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .video else { return .no }
        let haystack = (model.id + " " + model.hfModelId).lowercased()
        guard haystack.contains("wan") else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(WanVideoRunView(modelDisplayName: model.displayName, modelID: model.id))
    }
}
