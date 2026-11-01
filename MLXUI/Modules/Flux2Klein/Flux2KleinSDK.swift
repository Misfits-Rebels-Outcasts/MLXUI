import Foundation
import SwiftUI

/// `ModelSDK` for FLUX.2 Klein 4B text -> image (MP7-4). Claims `.image` + `.mlx` + the Klein 4B id at `.exact`. `FluxSDK.claim` returns `.no` for the same ids, so
/// the FLUX.1 engine never sees them whichever module registers first (`Flux2KleinRegistryRoutingTests`).
///
/// The Local Server never serves this model: nothing here is reachable from `Services/Server`.
nonisolated struct Flux2KleinSDK: ModelSDK {
    let id = "flux2-klein"

    /// What the stage's refusals call a row when the caller has no better name (a Run window, `.default` probes).
    static let defaultRowLabel = Flux2KleinStage.runWindowLabel

    func claim(_ model: ModelEntry) -> ClaimScore { Self.claimsKlein(model) ? .exact : .no }

    nonisolated static func claimsKlein(_ model: ModelEntry) -> Bool {
        guard model.runnerKind == .image, model.source == .mlx else { return false }
        return (model.id + " " + model.hfModelId).lowercased().contains("flux.2-klein-4b")
    }

    /// The Run sheet's probe (and `.default` probes). The sheet must open on a Mac too small for the 1024 default: the Run UI offers only sizes that fit and shows the cap
    /// sentence when none does (MP7-4-FIX-1, owner ruling 3), so with no size named nothing is checked against the cap here; `Flux2KleinStage.run` still checks the size
    /// it is given before any weight loads. A config that names a size is validated as a flow row's is.
    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        if config.width == nil && config.height == nil {
            if config.scale != nil { throw StageError.unsupportedSetting(setting: "scale") }
            let settings = Flux2KleinRunSettings(steps: config.steps ?? Flux2KleinRunSettings.defaultSteps, seed: config.seed)
            return Flux2KleinStage(modelID: model.id, settings: settings, rowLabel: Self.defaultRowLabel)
        }
        return try makeStage(for: model, config: config, quantize: Flux2KleinRunSettings.defaultQuantize, rowLabel: Self.defaultRowLabel)
    }

    /// The same with the row's `quantize` and its real label (a flow row's "Row 3"): what MP7-W's executor calls. `config.width/height/steps/seed` fall back to the manifest's
    /// 1024 / 1024 / 4 / random. A request the engine would refuse (quantize outside {4, 8}, a size that is not a multiple of 16 (SPEC-Q242), a size over this Mac's cap,
    /// no steps) throws here, naming `rowLabel`, before any weight loads.
    func makeStage(for model: ModelEntry, config: StageConfig, quantize: Int, rowLabel: String) throws -> any PipelineStage {
        // `scale` is an upscale setting; a Klein row naming it must say so, not run unscaled (the CFM-R16-1 lesson).
        if config.scale != nil { throw StageError.unsupportedSetting(setting: "scale") }
        let settings = Flux2KleinRunSettings(width: config.width ?? Flux2KleinRunSettings.defaultSize,
                                             height: config.height ?? Flux2KleinRunSettings.defaultSize,
                                             steps: config.steps ?? Flux2KleinRunSettings.defaultSteps,
                                             seed: config.seed, quantize: quantize)
        try Flux2KleinEngine.validate(settings.request(prompt: "", seed: 0, row: rowLabel), budgetBytes: Flux2KleinMemoryPlan.defaultBudgetBytes)
        return Flux2KleinStage(modelID: model.id, settings: settings, rowLabel: rowLabel)
    }

    // MARK: Settings the stage cannot honour

    /// The curated manifest (`flux2-klein-4b.json`, never edited) declares settings this build cannot honour; a row that names one must be refused, never silently run
    /// without it (owner, 2026-10-31 ruling 4). `value(key)` reads the row's setting, nil when absent. Returns the sentence for the first one, or nil. `StageConfig` carries none
    /// of these, so MP7-W's executor calls this before dispatch (as `scale` is checked in `RealExecutor.stageConfig`).
    ///
    /// * `guidance`: Klein 4B is guidance-distilled and has no guidance path (`Flux2Model.usesGuidanceEmbeds == false`), so only the manifest default 2.0 is accepted.
    /// * `negative`, `lora`, `controlnet`: any non-empty value. `preview`: anything but an explicit off.
    /// * `quantize`: 4 or 8 (the engine's own sentence).
    static func refusal(forRowSetting value: (String) -> String?, rowLabel: String) -> String? {
        func present(_ key: String) -> String? {
            guard let raw = value(key)?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else { return nil }
            return raw
        }
        if let raw = present("guidance"), Double(raw) != 2.0 {
            return "\(rowLabel): FLUX.2 Klein 4B has no guidance setting (it is distilled to 4 steps), so guidance can only be left at 2.0, not \(raw)."
        }
        for (key, what) in [("negative", "a negative prompt"), ("lora", "LoRA weights"), ("controlnet", "a ControlNet")] where present(key) != nil {
            return "\(rowLabel): FLUX.2 Klein 4B can't use \(what) in this build; remove the \(key) setting."
        }
        if let raw = present("preview"), !["false", "no", "off", "0"].contains(raw.lowercased()) {
            return "\(rowLabel): FLUX.2 Klein 4B can't show a preview while it denoises in this build; remove the preview setting."
        }
        if let raw = present("quantize") {
            guard let bits = Int(raw) else { return "\(rowLabel): quantize must be a whole number, 4 or 8; the row says \(raw)." }
            do { try Flux2KleinEngine.validate(quantize: bits, row: rowLabel) } catch { return (error as? LocalizedError)?.errorDescription ?? "\(rowLabel): unsupported quantize \(bits)." }
        }
        return nil
    }
}

/// `ModelUI` for FLUX.2 Klein 4B: `Flux2KleinRunView`.
struct Flux2KleinUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore { Flux2KleinSDK().claim(model) }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(Flux2KleinRunView(modelDisplayName: model.displayName, license: model.license, modelID: model.id, handedStage: stage))
    }
}

/// What the shared `MemoryBudget` books for a Klein run (read by `EngineCache.footprint`). The generic `.image` rule books `max(files x 1.10, catalog ramGB)` = 17.6 GB for the
/// 16 GB of bf16 files, which no Mac could ever grant, although the staged engine never holds more than one component (about 5.9 GB at 1024 x 1024).
nonisolated enum Flux2KleinBooking {
    static func isKlein(_ model: ModelEntry) -> Bool { Flux2KleinSDK.claimsKlein(model) }

    /// The largest whole-process `phys_footprint` peak measured for the default run (1024 x 1024, 4 steps, quantize 8): 6.05 GB over the six runs of MP7-3h and MP7-4 (5.61 to 6.05;
    /// the plan's busiest stage, `Flux2KleinMemoryPlan`, says 5.91). It is also the catalog's `ramGB`, so the Browse figure and the ledger agree (a test pins that). It is only the
    /// fallback for a caller that has no run settings; a run books `bytes(width:height:quantize:)`.
    static let measuredFootprintGB = 6.05
    static var bookedBytes: Int64 { Int64((measuredFootprintGB * 1e9).rounded()) }

    /// MP7-4-FIX-1 (owner, 2026-11-01): what ONE run books: the busiest staged peak at this size and quantize, times the same 1.15 margin the size cap uses. A 1344 x 1344 run
    /// (8.25 GB measured) therefore cannot share the Mac with a model `MemoryBudget` admitted against the 6.05 GB figure.
    static func bytes(width: Int, height: Int, quantize: Int) -> Int64 {
        let busiest = Flux2KleinMemoryPlan.peaks(width: width, height: height, quantize: quantize).busiest
        return Int64((busiest * Flux2KleinMemoryPlan.safetyMargin * 1e9).rounded())
    }

    static func bytes(for model: ModelEntry) -> Int64? { isKlein(model) ? bookedBytes : nil }
}
