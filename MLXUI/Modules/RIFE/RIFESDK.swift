import Foundation
import SwiftUI

/// `ModelSDK` for RIFE 4.25 frame interpolation. Claims `.interpolate` + `source == .mlx` + RIFE-family entries at `.exact`. No other SDK
/// claims that kind and this one claims nothing else; `taskFamilies` stays empty and no flow task names the kind (RIFE-W skipped).
nonisolated struct RIFESDK: ModelSDK {
    let id = "rife"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .interpolate,
              model.source == .mlx,
              model.family.lowercased() == "rife" else {
            return .no
        }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        RIFEStage(modelID: model.id)
    }
}

/// `ModelUI` for RIFE: `RIFERunView` (pick a video, 2× or 4×, Scale, interpolate, preview, save).
struct RIFEUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore { RIFESDK().claim(model) }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(RIFERunView(modelDisplayName: model.displayName, license: model.license, modelID: model.id, handedStage: stage))
    }
}
