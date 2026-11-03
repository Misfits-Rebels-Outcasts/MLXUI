import SwiftUI

/// `ModelUI` for TranslateGemma. MC2-4 lands the module and its routing; MC2-5 replaces this placeholder with the
/// Translate screen (from/to languages, Auto detection, a text box, Translate/Stop, Copy, the 2K count).
struct TranslateGemmaUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        TranslateGemmaSDK().claim(model)
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(UnsupportedModelView(model: model))
    }
}
