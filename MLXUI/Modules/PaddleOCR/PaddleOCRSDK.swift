import Foundation

/// `ModelSDK` for **PaddleOCR-VL**. Its architecture (`paddleocr_vl`) isn't in `MLXVLM`'s
/// type registry, so the generic `OCRSDK`/`VLMStage` path fails with `unsupportedModelType`.
/// This SDK routes PaddleOCR-VL to the vendored `PaddleOCRVL` pipeline instead (`PaddleOCRStage`).
///
/// Claims `.exact` for `source == .mlx` entries whose id names `paddleocr`. It's registered
/// **before** `OCRModule` in `App/ModelModules.swift`; both would score `.exact`, and the
/// registry keeps the earliest-registered on a tie, so PaddleOCR-VL resolves here while every
/// other OCR repo (olmOCR, …) still resolves to `OCRSDK`.
nonisolated struct PaddleOCRSDK: ModelSDK {
    let id = "paddleocr"

    /// OCR over a full page can be long; allow more tokens than the LLM default.
    static let ocrMaxTokens = 2048

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.source == .mlx else { return .no }
        let haystack = (model.id + " " + model.hfModelId).lowercased()
        guard haystack.contains("paddleocr") else { return .no }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        PaddleOCRStage(modelID: model.id, maxTokens: max(config.maxTokens, Self.ocrMaxTokens))
    }
}
