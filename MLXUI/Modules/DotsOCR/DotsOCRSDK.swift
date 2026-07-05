import Foundation

/// `ModelSDK` for **dots.ocr / dots.mocr** (SUP-3). Its `dots_ocr` architecture (a custom
/// `dots_vit` vision tower + a Qwen2 LM) isn't in MLXVLM's stock type registry, so the generic
/// `OCRSDK`/`VLMStage` path fails with `unsupportedModelType`. This SDK routes dots checkpoints to
/// `DotsOCRStage`, which runs a from-reference Swift/MLX port registered into MLXVLM's factory.
///
/// Claims `.exact` for `source == .mlx` entries whose id names `dots`. It's registered **before**
/// `OCRModule` in `App/ModelModules.swift`; both would score `.exact`, and the registry keeps the
/// earliest-registered on a tie, so dots.ocr resolves here while every other OCR repo (olmOCR, …)
/// still resolves to `OCRSDK`.
nonisolated struct DotsOCRSDK: ModelSDK {
    let id = "dots-ocr"

    /// OCR over a full page can be long; allow more tokens than the LLM default.
    static let ocrMaxTokens = 2048

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.source == .mlx else { return .no }
        let haystack = (model.id + " " + model.hfModelId).lowercased()
        guard haystack.contains("dots") else { return .no }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        DotsOCRStage(modelID: model.id, maxTokens: max(config.maxTokens, Self.ocrMaxTokens))
    }
}
