import SwiftUI

/// Rerank has no standalone Run surface — it's a headless flow task, scoring a whole list
/// against a query, not a chat-style exchange (MoC-4-3, `RSI/DelegateMoCBacklog.md`). This
/// view is what a user actually sees if they reach the registry-driven Run sheet directly
/// (e.g. the model card's Run button) — it explains why, rather than pretending to be an
/// interactive surface it isn't.
struct RerankRunView: View {
    let modelDisplayName: String
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "arrow.up.arrow.down.square")
                .font(.system(size: 40))
                .foregroundStyle(.secondary)
            Text(modelDisplayName)
                .font(.headline)
            Text("Rerank scores a list of candidates against a query. It only runs as part of an AI Workflow — a Rerank row in a flow — not as a standalone Run.")
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

/// `ModelUI` for the Qwen3-Reranker scorer. `claim` mirrors `RerankSDK` so UI resolution
/// matches SDK resolution.
struct RerankUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .rerank, model.source == .mlx else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(RerankRunView(modelDisplayName: model.displayName))
    }
}
