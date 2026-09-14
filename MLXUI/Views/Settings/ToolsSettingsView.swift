import SwiftUI

/// SET-1's Tools tab. Content moved verbatim from the old `SettingsView`'s "Agent Tools"
/// section — copy, grouping (none yet), and bindings are unchanged; only the container
/// changed. The rename to "Tools", per-tool descriptions, grouping, folder grants and the
/// Audit Log link are SET-3's job (`RSI/DelegateSettingsBacklog.md` §5).
struct ToolsSettingsView: View {
    @Environment(AppState.self) private var appState

    private var runner: ModelRunner { appState.modelRunner }

    var body: some View {
        Form {
            Section("Agent Tools") {
                Text("Tools a chat model may call. Side-effecting tools also have a standing approval policy: ask each time, always allow, or never allow.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                ForEach(runner.availableTools, id: \.name) { tool in
                    HStack {
                        Toggle(tool.name, isOn: toolEnabledBinding(tool.name))
                        if tool.requiresApproval {
                            Spacer()
                            Picker("", selection: toolPolicyBinding(tool.name)) {
                                Text("Ask each time").tag(ToolApprovalPolicy.ask)
                                Text("Always allow").tag(ToolApprovalPolicy.always)
                                Text("Never allow").tag(ToolApprovalPolicy.never)
                            }
                            .labelsHidden()
                            .fixedSize()
                        }
                    }
                }

                Picker("Tool call limit per reply", selection: toolCallLimitBinding) {
                    ForEach(toolCallLimitOptions, id: \.self) { limit in
                        Text("\(limit)").tag(limit)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 520)
    }

    private func toolEnabledBinding(_ name: String) -> Binding<Bool> {
        Binding(
            get: { runner.enabledToolNames.contains(name) },
            set: { runner.setToolEnabled($0, for: name) }
        )
    }

    private func toolPolicyBinding(_ name: String) -> Binding<ToolApprovalPolicy> {
        Binding(
            get: { runner.toolPolicy(for: name) },
            set: { runner.setToolPolicy($0, for: name) }
        )
    }

    private var toolCallLimitBinding: Binding<Int> {
        Binding(
            get: { runner.toolCallLimit },
            set: { runner.setToolCallLimit($0) }
        )
    }

    /// Picker choices for the per-reply budget; a persisted off-list value stays selectable.
    private var toolCallLimitOptions: [Int] {
        let options = [2, 4, 8, 16, 32]
        return options.contains(runner.toolCallLimit)
            ? options
            : (options + [runner.toolCallLimit]).sorted()
    }
}
