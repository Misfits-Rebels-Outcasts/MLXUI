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
        // CFM-R16-1: `seed` reaches the stage (FluxStage already takes it). `width`/`height`/
        // `steps` are read off a diffusion row by `RealExecutor.stageConfig`, but `FluxEngine`
        // renders a fixed 512×512 image over a fixed 50 steps — so a row naming one of them
        // must refuse with a sentence naming the setting, never silently drop it (the whole
        // lesson of the phase). A future engine that honours sizes builds its own SDK.
        if config.width != nil || config.height != nil {
            throw StageError.unsupportedSetting(setting: config.width != nil ? "width" : "height")
        }
        if config.steps != nil {
            throw StageError.unsupportedSetting(setting: "steps")
        }
        return FluxStage(modelID: model.id, seed: config.seed)
    }
}
