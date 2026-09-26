import SwiftUI

/// Placeholder Run surface — LY-4's scope is the model running, not the UI (LY-5 replaces
/// this with the real question-type picker, options list, and per-option probability bars).
/// Mirrors `RerankRunView`'s "headless for now" shape.
struct LayaRunView: View {
    let modelDisplayName: String
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "arrow.triangle.branch")
                .font(.system(size: 40))
                .foregroundStyle(.secondary)
            Text(modelDisplayName)
                .font(.headline)
            Text("Laya answers one typed question — choice, score, or yes/no — about a piece of text: a label, a probability per option, and a confidence. It never generates text. The full Run UI (question type, options, probability bars) lands in a later cycle.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button("Done") { dismiss() }
                .keyboardShortcut(.cancelAction)
        }
        .padding(32)
        .frame(width: 360)
    }
}

/// `ModelUI` for Laya. `claim` mirrors `LayaSDK` so UI resolution matches SDK resolution.
struct LayaUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .decision, model.source == .mlx else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(LayaRunView(modelDisplayName: model.displayName))
    }
}
