import SwiftUI

/// SET-1's Providers tab, content moved verbatim from the old `SettingsView`'s "Model & Search
/// Providers" section. SET-6 removed the empty-state branch (`RSI/DelegateSettingsBacklog.md`
/// §0: unreachable in the shipping bundle — five manifests name credentials today).
struct ProvidersSettingsView: View {
    /// KEY-2 — scanned from installed manifests, never hardcoded (see
    /// `CuratedManifest.installedCredentialNames`).
    private var providerNames: [String] { CuratedManifest.installedCredentialNames() }

    var body: some View {
        Form {
            Section("Model & Search Providers") {
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
    }
}
