import SwiftUI

/// `ModelUI` for LFM2.5-Audio. **MP6-3a placeholder (implementer's call, pending owner confirmation):** the voice Run UI (hold to
/// talk, streamed reply, transcript) is MP6-3b. Until it lands this shows one line saying so, so a model that is installed and
/// claimed never opens an empty window. Replace `LFMAudioPlaceholderView` with the real run view in MP6-3b; not for release as is.
struct LFMAudioUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .voice,
              model.source == .mlx,
              model.family.lowercased().contains("lfm2.5-audio") else {
            return .no
        }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(LFMAudioPlaceholderView(modelDisplayName: model.displayName))
    }
}

struct LFMAudioPlaceholderView: View {
    @Environment(\.dismiss) private var dismiss
    let modelDisplayName: String

    var body: some View {
        VStack(spacing: 12) {
            Text(modelDisplayName).font(.headline)
            Text("The voice conversation screen arrives in the next update.")
                .foregroundStyle(.secondary)
            Button("Done") { dismiss() }
        }
        .padding(32)
        .frame(minWidth: 360, minHeight: 160)
    }
}
