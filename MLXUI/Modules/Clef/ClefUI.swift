import SwiftUI

/// `ModelUI` for Clef Flash. `claim` mirrors `ClefSDK` so UI resolution matches SDK resolution. The
/// real Run surface is `Views/Run/ClefRunView.swift` (CF-6, gate E1 — "a separate `ClefRunView`
/// copied from `CLMRunView`"), so this file only resolves and hands off, exactly as `LayaUI` and
/// `CLMUI` do. CF-2's `ModelSupport` gap is removed in CF-5b: this is a real Run screen's SDK.
struct ClefUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .decision, model.source == .mlx, model.family == "Clef" else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(ClefRunPlaceholderView(model: model))
    }
}

/// Until CF-6 lands `ClefRunView`, the Run surface says what it is rather than showing a blank
/// screen — the same placeholder discipline `ModelRunner.showUnsupported` uses for a runner gap.
private struct ClefRunPlaceholderView: View {
    let model: ModelEntry

    var body: some View {
        VStack(spacing: 12) {
            Text(model.displayName)
                .font(.title2.weight(.semibold))
            Text("Clef Flash's Run screen lands in CF-6. The engine and its SDK are installed:")
                .foregroundStyle(.secondary)
            Text("Choose a state, then ask a choice, score or yes/no question.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .padding(32)
        .frame(minWidth: 480, minHeight: 320)
    }
}
