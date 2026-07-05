import SwiftUI

/// Settings sheet for entering a HuggingFace access token, needed to install gated or
/// private models. The token is stored in the Keychain via `KeychainHelper`.
struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var tokenInput: String = ""
    @State private var hasToken: Bool = KeychainHelper.getToken() != nil
    @State private var status: String = ""

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
            }
            .formStyle(.grouped)
        }
        .frame(width: 460, height: 360)
        .onAppear { hasToken = KeychainHelper.getToken() != nil }
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
