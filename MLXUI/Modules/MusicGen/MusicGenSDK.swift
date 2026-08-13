import Foundation

/// `ModelSDK` for MusicGen text → music, ported from `ml-explore/mlx-examples/musicgen`
/// (MG-MOD1). Claims `.music` + `source == .mlx` + musicgen-family entries. The haystack
/// predicate (`"musicgen"`) is disjoint from every other SDK, so registration order is not
/// load-bearing.
nonisolated struct MusicGenSDK: ModelSDK {
    let id = "musicgen"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .music, model.source == .mlx else { return .no }
        let haystack = (model.id + " " + model.hfModelId).lowercased()
        guard haystack.contains("musicgen") else { return .no }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        MusicGenStage(modelID: model.id)
    }
}
