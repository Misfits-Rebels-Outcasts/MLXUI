import Foundation

/// `ModelSDK` for text → image SDXL-Turbo, ported from `ml-explore/mlx-examples/stable_diffusion`
/// (SD-MOD1). Claims `.image` + `source == .mlx` + sdxl-turbo-family entries. The haystack
/// predicate (`"sdxl-turbo"` / `"stable-diffusion-xl-turbo"`) is disjoint from `FluxSDK`'s
/// `"flux"`, so registration order vs Flux is not load-bearing.
nonisolated struct SDXLTurboSDK: ModelSDK {
    let id = "sdxl-turbo"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .image, model.source == .mlx else { return .no }
        let haystack = (model.id + " " + model.hfModelId).lowercased()
        guard haystack.contains("sdxl-turbo") || haystack.contains("stable-diffusion-xl-turbo") else { return .no }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        SDXLTurboStage(modelID: model.id)
    }
}
