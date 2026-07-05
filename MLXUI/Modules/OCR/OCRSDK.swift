import Foundation

/// `ModelSDK` for OCR. The catalog's OCR repos (PaddleOCR-VL, DeepSeek-OCR, dots.ocr,
/// olmOCR) are MLX VLMs, so OCR runs on the same `MLXVLM` engine as `VLMSDK` — via a
/// `VLMStage` with a **fixed transcription prompt** (OCR has no per-run prompt). No Apple
/// Vision. Claims `.ocr` + `source == .mlx`.
nonisolated struct OCRSDK: ModelSDK {
    let id = "ocr"

    /// Fixed instruction for text extraction.
    static let ocrPrompt =
        "Transcribe all text in this image exactly, preserving reading order. Output only the text."

    /// OCR over a full page can be long; allow more tokens than the LLM default.
    static let ocrMaxTokens = 2048

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .ocr, model.source == .mlx else { return .no }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        VLMStage(modelID: model.id,
                 prompt: config.prompt ?? Self.ocrPrompt,
                 maxTokens: max(config.maxTokens, Self.ocrMaxTokens))
    }
}
