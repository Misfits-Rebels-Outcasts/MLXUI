import Foundation

/// Flags catalog models whose **architecture has no MLX runner yet**, so the UI can warn
/// before a Run that would only produce an `unsupportedModelType(...)` error. These are
/// tracked for iterative porting in `RSI/backlog.md` (§ "Model support gaps"). The backing
/// packages' type registries (`VLMTypeRegistry`, `EmbedderTypeRegistry`) enumerate what's
/// runnable; the entries below are the browser.json models whose `model_type` isn't in them.
///
/// Keyed on `hfModelId`/`id` substrings (precise to the catalog entries) so we don't need the
/// model installed to flag it. Update when an architecture is ported (remove its entry).
enum ModelSupport {
    private struct Gap {
        let idFragment: String   // matched case-insensitively against id / hfModelId
        let reason: String
    }

    private static let gaps: [Gap] = [
        // All catalog OCR / embedder architectures now have MLX runners:
        // PaddleOCR-VL → vendored PaddleOCRVL pipeline (SUP-1, journal/2026-42).
        // dots.ocr / dots.mocr → custom `dots_ocr` model registered into MLXVLM (SUP-3, journal/2026-44/45).
        // DeepSeek-OCR-2 → custom `deepseekocr_2` model registered into MLXVLM (SUP-2, journal/2026-47…51).
        // ModernBERT (nomicai-modernbert-embed) → standalone runner (SUP-4, journal/2026-41).
    ]

    /// A human-readable reason this model can't run yet, or `nil` if it should run.
    static func unsupportedReason(for model: ModelEntry) -> String? {
        let haystack = (model.id + " " + model.hfModelId).lowercased()
        return gaps.first { haystack.contains($0.idFragment) }?.reason
    }
}
