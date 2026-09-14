import SwiftUI

/// SET-1's Models tab. Content moved verbatim from the old `SettingsView`'s "HuggingFace
/// Access Token" section — copy and layout are unchanged; only the container changed (a real
/// `Settings` window pane instead of a sheet section). Storage location, size on disk, and the
/// flow-cache row are SET-6's job (`RSI/DelegateSettingsBacklog.md` §5).
struct ModelsSettingsView: View {
    @State private var tokenInput: String = ""
    @State private var hasToken: Bool = KeychainHelper.getToken() != nil
    @State private var status: String = ""

    var body: some View {
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
        .frame(minWidth: 520)
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
