import SwiftUI

/// Placeholder Run view for SAM3 image segmentation — replaced in SA-AM5 with the full
/// interactive canvas (image picker + point/box overlay + mask rendering).
struct SegmentAnythingRunView: View {
    let modelDisplayName: String
    let modelID: String

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "lasso")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)

            Text(modelDisplayName)
                .font(.headline)

            Text("Interactive segmentation UI is being built.\nThis model's engine is ready — point and box prompt support will arrive in the next update.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .frame(maxWidth: 400)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

/// `ModelUI` adapter for the SAM3 segmentation module.
struct SegmentAnythingUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .segmentation else { return .no }
        let haystack = (model.id + " " + model.hfModelId).lowercased()
        guard haystack.contains("sam") else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(SegmentAnythingRunView(
            modelDisplayName: model.displayName,
            modelID: model.id
        ))
    }
}
