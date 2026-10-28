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

/// `ModelUI` for RIFE. Until RIFE-4 builds the video Run screen this returns the standard "not supported yet" view, which is exactly what
/// the app showed before the module was registered, so registering the engine (RIFE-3d) changes no behaviour a user can reach.
struct RIFEUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore { RIFESDK().claim(model) }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(UnsupportedModelView(model: model))
    }
}
