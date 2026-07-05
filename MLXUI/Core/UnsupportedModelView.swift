import SwiftUI

/// Graceful fallback shown when a model has no registered module (or resolution fails):
/// the model name plus a "not yet supported" note and a Done button. Keeps the registry
/// run path self-contained so `UnsupportedModelView(model:)` always resolves.
struct UnsupportedModelView: View {
    let model: ModelEntry
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "questionmark.circle")
                .font(.system(size: 40))
                .foregroundStyle(.secondary)
            Text(model.displayName)
                .font(.headline)
            Text("Running this model isn’t supported yet.")
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
