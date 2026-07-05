import Foundation

/// `ModelSDK` for **DeepSeek-OCR-2** (SUP-2). Its `deepseekocr_2` architecture (SAM ViT → Qwen2-0.5B
/// decoder-as-encoder → DeepSeek-V2 MoE LM) isn't in MLXVLM's stock type registry, so the generic
/// `OCRSDK`/`VLMStage` path fails with `unsupportedModelType`. This SDK routes DeepSeek-OCR checkpoints
/// to `DeepSeekOCRStage`, which runs a from-reference Swift/MLX port registered into MLXVLM's factory
/// (lands in a later slice — see `RSI/plan-deepseek-ocr.md`).
///
/// Claims `.exact` for `source == .mlx` entries whose id names `deepseek-ocr`. Registered **before**
/// `OCRModule`; both would score `.exact`, and the registry keeps the earliest-registered on a tie, so
/// DeepSeek-OCR resolves here while every other OCR repo (olmOCR, …) still resolves to `OCRSDK`.
nonisolated struct DeepSeekOCRSDK: ModelSDK {
    let id = "deepseek-ocr"

    /// OCR over a full page can be long; allow more tokens than the LLM default.
    static let ocrMaxTokens = 2048

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.source == .mlx else { return .no }
        let haystack = (model.id + " " + model.hfModelId).lowercased()
        guard haystack.contains("deepseek-ocr") else { return .no }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        DeepSeekOCRStage(modelID: model.id, maxTokens: max(config.maxTokens, Self.ocrMaxTokens))
    }
}
