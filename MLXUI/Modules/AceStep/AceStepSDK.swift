import Foundation

/// `ModelSDK` for ACE-Step 1.5 text -> music (MC5-3i). Claims `.music` + `.mlx` + the `ace-step` family at `.exact`. The MusicGen SDK claims only ids
/// containing "musicgen", so the two are disjoint whichever registers first (`AceStepModuleTests`).
nonisolated struct AceStepSDK: ModelSDK {
    let id = "acestep"

    func claim(_ model: ModelEntry) -> ClaimScore { Self.claimsAceStep(model) ? .exact : .no }

    nonisolated static func claimsAceStep(_ model: ModelEntry) -> Bool {
        guard model.runnerKind == .music, model.source == .mlx else { return false }
        return model.family == "ace-step" || (model.id + " " + model.hfModelId).lowercased().contains("ace-step")
    }

    /// Steps, size and quantize are not settings of this model (turbo is fixed at 8 steps; the output is audio): a run that names one is refused by name,
    /// never run without it (CLAUDE.md, the CFM-R16-1 lesson).
    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        if config.steps != nil { throw StageError.unsupportedSetting(setting: "steps") }
        if config.width != nil { throw StageError.unsupportedSetting(setting: "width") }
        if config.height != nil { throw StageError.unsupportedSetting(setting: "height") }
        if config.quantize != nil { throw StageError.unsupportedSetting(setting: "quantize") }
        if config.scale != nil { throw StageError.unsupportedSetting(setting: "scale") }
        return AceStepStage(modelID: model.id, settings: AceStepRunSettings(seed: config.seed))
    }
}
