import SwiftUI

/// SET-1's Providers tab, content moved verbatim from the old `SettingsView`'s "Model & Search
/// Providers" section. SET-6 removed the empty-state branch (`RSI/DelegateSettingsBacklog.md`
/// §0: unreachable in the shipping bundle — five manifests name credentials today).
struct ProvidersSettingsView: View {
    /// KEY-2 — scanned from installed manifests, never hardcoded (see
    /// `CuratedManifest.installedCredentialNames`).
    private var providerNames: [String] { CuratedManifest.installedCredentialNames() }

    /// KC-3.2: set once the rows have queried the Keychain, never as "No key set".
    @State private var keychainUnavailable = false

    var body: some View {
        Form {
            Section("Model & Search Providers") {
                if keychainUnavailable {
                    Text("This build can't use the Keychain — check signing")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
                Text("Keys are stored in your macOS Keychain. They are never written into a flow file, and a flow you send someone does not carry them.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                ForEach(providerNames, id: \.self) { name in
                    ProviderCredentialRowView(providerName: name)
                }
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 520)
        .task {
            // Let the rows' own `.task` presence checks run first.
            try? await Task.sleep(for: .milliseconds(300))
            keychainUnavailable = KeychainHelper.hasEntitlementFailure
        }
    }
}
