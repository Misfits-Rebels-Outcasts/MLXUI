import Foundation

/// `ModelSDK` for TranslateGemma 4B, backed by `mlx-swift-lm`'s `gemma3` text runner via ``TranslateGemmaEngine``.
///
/// Claims **`.translate` + `source == .mlx` + the `translategemma` family only** (Trap 5: by family, never by
/// architecture — the catalog's Gemma 3 4B/12B entries carry the same `Gemma3ForConditionalGeneration` string). No other
/// SDK claims `.translate`, and `ChatSDK` claims only `.llm`, so the entry can never resolve to chat.
///
/// `StageBooksOwnWeights`: the engine leases its container from the shared `ModelContainerPool`, so the `EngineCache`
/// must not book the weights a second time (the VLM's arrangement).
nonisolated struct TranslateGemmaSDK: ModelSDK, StageBooksOwnWeights {
    let id = "translategemma"

    /// The translator takes no free-text prompt: its whole request is the source text plus the two language codes, both
    /// carried by the stage, never by `StageConfig.prompt`. A quoted instruction in a row is therefore stranded, and
    /// the flight editor's existing warning machinery handles that.
    var promptSupport: PromptSupport { .none }

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .translate,
              model.source == .mlx,
              model.family.lowercased() == "translategemma" else {
            return .no
        }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        let footprint = Int64(model.ramGB * 1_073_741_824)
        if let source = config.sourceLanguage, let target = config.targetLanguage {
            return TranslateGemmaStage(modelID: model.id, displayName: model.displayName,
                                       sourceCode: source, targetCode: target, footprintBytes: footprint)
        }
        // The Run sheet builds a probe stage (`.default` config) before the person has picked languages; the Translate
        // screen replaces it with a per-run stage that carries both (MC2-5). Running the probe without languages fails
        // with a sentence rather than translating from a guess.
        return TranslateGemmaStage(id: model.id, name: model.displayName, sourceCode: "", targetCode: "") { _ in
            throw TranslateGemmaError.missingLanguages
        }
    }
}
