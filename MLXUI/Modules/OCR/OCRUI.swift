import SwiftUI

/// `ModelUI` for OCR: presents the reusable `OCRRunView`. Most OCR models have a frozen
/// prompt and use the pre-built `stage` directly (mirrors `WhisperUI`/`ASRRunView`). When the
/// registered SDK declares a non-`.none` `promptSupport` (OCP-0: PaddleOCR-VL's mode picker),
/// the view is handed that support plus a closure to rebuild the stage per run.
struct OCRUI: ModelUI {
    /// The SDK registered alongside this UI, so the run surface can read `promptSupport`.
    /// `nil` ⇒ the legacy prompt-less surface (dots.ocr, DeepSeek-OCR, olmOCR in this phase).
    var sdk: (any ModelSDK)?

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .ocr, model.source == .mlx else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        let support = sdk?.promptSupport ?? .none
        let rebuild: ((String) -> (any PipelineStage)?)?
        switch support {
        case .none:
            rebuild = nil
        case .freeText, .modes:
            let sdk = sdk
            rebuild = { prompt in
                try? sdk?.makeStage(for: model, config: StageConfig(prompt: prompt))
            }
        }
        return AnyView(OCRRunView(modelDisplayName: model.displayName,
                                  license: model.license,
                                  stage: stage,
                                  promptSupport: support,
                                  rebuildStage: rebuild))
    }
}
