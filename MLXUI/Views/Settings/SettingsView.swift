import SwiftUI

/// Settings sheet: HuggingFace access token (stored in the Keychain via `KeychainHelper`)
/// plus the consolidated agent-tool configuration (AG6b-2) — per-tool enable, standing
/// approval policy for side-effecting tools, and the per-reply tool-call limit.
struct SettingsView: View {
    @Environment(AppState.self) private var appState
    @Environment(\.dismiss) private var dismiss
    @State private var tokenInput: String = ""
    @State private var hasToken: Bool = KeychainHelper.getToken() != nil
    @State private var status: String = ""

    private var runner: ModelRunner { appState.modelRunner }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Settings")
                    .font(.headline)
                Spacer()
                Button("Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
            .padding(12)
            Divider()

            Form {
                Section("HuggingFace Access Token") {
                    Text("Required to download gated or private models. Stored securely in your macOS Keychain — never shared.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    SecureField("Paste your HuggingFace token", text: $tokenInput)
                        .textFieldStyle(.roundedBorder)

                    HStack {
                        Button("Save") { save() }
                            .disabled(!HFTokenValidator.isPlausible(tokenInput))
                        Button("Clear", role: .destructive) { clear() }
                            .disabled(!hasToken)
                        Spacer()
                        if !status.isEmpty {
                            Text(status)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    Label(hasToken ? "A token is saved." : "No token saved.",
                          systemImage: hasToken ? "checkmark.seal.fill" : "exclamationmark.triangle")
                        .font(.caption)
                        .foregroundStyle(hasToken ? Color.green : Color.secondary)

                    Link("Get a token on huggingface.co",
                         destination: URL(string: "https://huggingface.co/settings/tokens")!)
                        .font(.caption)
                }

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
        }
        .frame(width: 460, height: 560)
        .onAppear { hasToken = KeychainHelper.getToken() != nil }
    }

    // MARK: - Agent tools (AG6b-2)

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

    private func save() {
        KeychainHelper.saveToken(HFTokenValidator.normalized(tokenInput))
        hasToken = true
        tokenInput = ""
        status = "Saved."
    }

    private func clear() {
        KeychainHelper.deleteToken()
        hasToken = false
        tokenInput = ""
        status = "Cleared."
    }
}
