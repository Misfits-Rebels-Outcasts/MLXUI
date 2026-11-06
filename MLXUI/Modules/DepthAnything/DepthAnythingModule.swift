import Foundation
import SwiftUI

/// `ModelSDK` for Depth Anything V2 Small monocular depth estimation.
///
/// Trap 8: claim **by family, never by architecture** — `.depth` + `source == .mlx` + family
/// `depth-anything`, at `.exact`. No other SDK claims the `depth` kind, and this one claims nothing else.
nonisolated struct DepthAnythingSDK: ModelSDK {
    let id = "depth-anything"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .depth,
              model.source == .mlx,
              model.family.lowercased() == "depth-anything" else {
            return .no
        }
        return .exact
    }

    /// The row takes no settings (owner decision 7); every setting is refused by name in `RealExecutor`.
    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        DepthAnythingStage(modelID: model.id)
    }
}

/// `ModelUI` for Depth Anything: `DepthAnythingRunView` (pick a photo, Estimate depth, Grayscale/Colour
/// preview, Save PNG).
struct DepthAnythingUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore { DepthAnythingSDK().claim(model) }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(DepthAnythingRunView(modelDisplayName: model.displayName, license: model.license,
                                     modelID: model.id, handedStage: stage))
    }
}

/// Bundles the Depth Anything SDK and UI and registers them. A from-scratch Swift/MLX port of the official
/// Apache-2.0 `depth-anything/Depth-Anything-V2-Small-hf` checkpoint (DINOv2 ViT-S/14 + DPT head), loading the
/// published fp32 safetensors the installer downloads — never a download at run time.
enum DepthAnythingModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "depth-anything",
        displayName: "Depth Anything V2 Small (monocular depth, MLX)",
        modalities: ["image"],
        modelTypes: [.depth],
        backingPackage: "depth-anything/Depth-Anything-V2-Small-hf (Swift/MLX port)",
        packageLicense: "Apache-2.0",
        maintainers: ["Lihe Yang et al.", "depth-anything"],
        notes: """
        Monocular depth estimation: one photo in, an 8-bit grayscale relative-depth map out at the same pixel \
        size (near = bright, min–max normalised per image). A Swift/MLX port of the official Small checkpoint \
        — DINOv2 ViT-S/14 backbone + DPT neck and head, fp32 as published, PyTorch layouts converted in the \
        loader. Not metric depth. Claims only depth-anything-family .depth models; no other module claims that \
        kind. The AI Workflows row "Estimate Depth" runs it; gallery flow 62's "Depth Pro" resolves to it with a \
        shown substitution.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: DepthAnythingSDK(), ui: DepthAnythingUI(), descriptor: descriptor)
    }
}
