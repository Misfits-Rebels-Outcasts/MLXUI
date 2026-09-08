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

    /// OCP-0: PaddleOCR-VL's four recognition modes. These mirror `PaddleOCRTask.allCases`
    /// in the vendored package (`Vendor/PaddleOCRVL/Sources/PaddleOCRVL/Configuration.swift`) —
    /// kept as bare strings here so the `PaddleOCRVL` import stays isolated to `PaddleOCREngine`
    /// (rule 6: no vendored edit). `.table` returns an OTSL grid (`RSI/journal/2026-228`);
    /// `.ocr` plain text, `.formula` LaTeX, `.chart` chart markup.
    static let modes = ["ocr", "table", "formula", "chart"]

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.source == .mlx else { return .no }
        let haystack = (model.id + " " + model.hfModelId).lowercased()
        guard haystack.contains("paddleocr") else { return .no }
        return .exact
    }

    /// A fixed-mode picker, not a free-text box: `mode=table` is checkable against this set in
    /// a way `"extract tables as CSV"` never will be (`DelegateOCRPromptBacklog.md` §0.1).
    var promptSupport: PromptSupport { .modes(values: Self.modes, default: "ocr") }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        // `config.prompt` carries the recognition mode (the single-carrier doctrine — no
        // parallel `mode` field). An unrecognized value must fail loudly rather than fall
        // through to `.ocr`: a wrong output nobody notices is worse than an error
        // (CFM-R16-1 / MoC-3-2, mirrored from `FluxSDK`).
        if let mode = config.prompt, !mode.isEmpty, !Self.modes.contains(mode) {
            throw StageError.unsupportedSetting(setting: "mode '\(mode)'")
        }
        return PaddleOCRStage(modelID: model.id,
                              maxTokens: max(config.maxTokens, Self.ocrMaxTokens),
                              mode: config.prompt)
    }
}
