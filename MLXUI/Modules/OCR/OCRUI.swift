import SwiftUI

/// `ModelUI` for OCR: presents the reusable `OCRRunView`. OCR has no per-run prompt, so —
/// unlike `VLMUI` — it uses the pre-built `stage` directly (mirrors `WhisperUI`/`ASRRunView`).
struct OCRUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .ocr, model.source == .mlx else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(OCRRunView(modelDisplayName: model.displayName,
                           license: model.license,
                           stage: stage))
    }
}
