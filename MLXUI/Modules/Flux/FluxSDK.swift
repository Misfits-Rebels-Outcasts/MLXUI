import Foundation

/// `ModelSDK` for text → image (FLUX.1-family diffusion), ported from
/// `ml-explore/mlx-examples/flux` (`RSI/plan-flux.md`, AM4). Claims `.image` +
/// `source == .mlx` + flux-family entries.
nonisolated struct FluxSDK: ModelSDK {
    let id = "flux"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .image, model.source == .mlx else { return .no }
        let haystack = (model.id + " " + model.hfModelId).lowercased()
        guard haystack.contains("flux") else { return .no }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        FluxStage(modelID: model.id)
    }
}
