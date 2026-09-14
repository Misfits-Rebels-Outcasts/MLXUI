import SwiftUI

/// SET-3 (`RSI/DelegateSettingsBacklog.md`) — the Tools pane: tools grouped by family (§0,
/// `AgentToolGroup`) with their own `description` as a caption (never rewritten — §8), the
/// raw tool id, per-tool approval, why an off-by-default tool is off, folder grants
/// (sandboxed edition), the tool-call limit, and the Audit Log. Renamed from "Agent Tools" to
/// "Tools" per OG-5 — "Agent" is the codebase's word, not the user's.
///
/// Per D4's rule ("the wrench is the in-session switch; Settings is the standing
/// configuration, and Settings is complete"), everything here also exists in the wrench menu
/// today; SET-4 trims the wrench down to on/off + approval + Audit Log.
struct ToolsSettingsView: View {
    @Environment(AppState.self) private var appState
    @State private var showAuditLog = false

    private var runner: ModelRunner { appState.modelRunner }
    private var groups: [(group: AgentToolGroup, tools: [any AgentTool])] {
        AgentToolGroup.grouping(runner.availableTools)
    }

    var body: some View {
        Form {
            Text("What a model may do on this Mac. Anything that changes something asks first, unless you say otherwise.")
                .font(.caption)
                .foregroundStyle(.secondary)

            ForEach(groups, id: \.group.rawValue) { entry in
                Section(entry.group.title) {
                    ForEach(entry.tools, id: \.name) { tool in
                        toolRow(tool)
                    }
                    if let note = entry.group.offByDefaultNote {
                        Text(note)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            #if !DIRECT_BUILD
            Section("File access") {
                Text("These tools can only reach folders you grant below.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                ForEach(runner.folderGrantPaths, id: \.self) { path in
                    HStack {
                        Text(path)
                        Spacer()
                        Button("Revoke") { runner.revokeFolderAccess(path: path) }
                    }
                }
                Button("Grant Folder Access…") { FolderAccessPanel.presentAndGrant(using: runner) }
            }
            #endif

            Section("Limits") {
                Text("How many tools a model may call while answering once. Lower is safer and faster; higher lets it finish longer jobs without stopping.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Picker("Tool call limit per reply", selection: toolCallLimitBinding) {
                    ForEach(toolCallLimitOptions, id: \.self) { limit in
                        Text("\(limit)").tag(limit)
                    }
                }
            }

            Section {
                Button("Open Audit Log…") { showAuditLog = true }
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 520)
        .sheet(isPresented: $showAuditLog) {
            AuditLogView(runner: runner)
        }
    }

    private func toolRow(_ tool: any AgentTool) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Toggle(isOn: toolEnabledBinding(tool.name)) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(AgentToolCopy.title(for: tool.name))
                    Text(tool.toolDescription)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(tool.name)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.tertiary)
                }
            }
            if tool.requiresApproval {
                Picker("Approval", selection: toolPolicyBinding(tool.name)) {
                    Text("Ask each time").tag(ToolApprovalPolicy.ask)
                    Text("Always allow").tag(ToolApprovalPolicy.always)
                    Text("Never allow").tag(ToolApprovalPolicy.never)
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .fixedSize()
            }
        }
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
