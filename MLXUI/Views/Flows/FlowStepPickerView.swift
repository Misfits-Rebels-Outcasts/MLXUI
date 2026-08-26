import SwiftUI

/// CFM-R8-1 — the step picker: the task catalog filtered by the *selected row's output
/// shape* (else the last row's, else starting nodes on an empty flow), the same lookup
/// `catflow tasks --accepts K` provides. "Add from Full Catalog" is the escape hatch
/// (answer `l`) that reveals the hidden primitives (`Generate`/`Decide`) and every
/// shape-mismatched task (a row added that way turns yellow until its input is pointed).
struct FlowStepPickerView: View {
    /// The filtered tasks for the current context.
    let steps: [TaskDescriptor]
    /// Whether the "Add from Full Catalog" mode is on (reveals the hidden primitives).
    @Binding var showFullCatalog: Bool
    /// Called with the chosen task name.
    let onPick: (String) -> Void
    var onCancel: () -> Void = {}
    /// CFM-R14-2 — the derived model pool's inputs: a model task is marked "needs newer
    /// support" exactly when its registry-claimable pool is empty, so the picker and the
    /// runtime can't disagree about a task that has no runnable model.
    var catalog: [ModelEntry] = []
    var claimableModelIDs: Set<String> = []

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Add a step")
                    .font(.headline)
                Spacer()
                Toggle("Add from Full Catalog", isOn: $showFullCatalog)
                    .toggleStyle(.checkbox)
                    .controlSize(.small)
            }
            .padding(12)
            Divider()
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(steps, id: \.name) { task in
                        let marker = TaskAvailability.marker(for: task, catalog: catalog,
                                                             claimableModelIDs: claimableModelIDs)
                        Button {
                            onPick(task.name)
                        } label: {
                            HStack(alignment: .firstTextBaseline, spacing: 8) {
                                Text(task.name)
                                    .font(.body.weight(.medium))
                                    .foregroundStyle(marker == nil ? Color.primary : Color.secondary)
                                Spacer()
                                if let marker {
                                    Text(marker)
                                        .font(.caption)
                                        .foregroundStyle(.orange)
                                }
                                Text(signature(task))
                                    .font(.caption.monospaced())
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .help(marker.map { "\(task.name): \($0)" } ?? task.name)
                        Divider().opacity(0.3)
                    }
                }
                .padding(.vertical, 4)
            }
            .frame(minHeight: 200, maxHeight: 420)
            Divider()
            HStack {
                if steps.isEmpty {
                    Text("No tasks fit this row's output — use Full Catalog.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Cancel", role: .cancel, action: onCancel)
            }
            .padding(10)
        }
        .frame(width: 440)
    }

    /// The task's `accepts → gives` signature line.
    private func signature(_ task: TaskDescriptor) -> String {
        "\(task.accepts.signatureText) → \(task.gives.signatureText)"
    }
}
