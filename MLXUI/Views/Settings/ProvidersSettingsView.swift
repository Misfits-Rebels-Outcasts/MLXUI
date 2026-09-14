import SwiftUI

/// SET-1's Providers tab. Content moved verbatim from the old `SettingsView`'s "Model & Search
/// Providers" section — copy, the empty-state text, and `ProviderCredentialRowView` are
/// unchanged; only the container changed. Renaming/rewording per §8 is SET-6+'s job.
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

                if providerNames.isEmpty {
                    Text("No installed model or search provider needs a key yet.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(providerNames, id: \.self) { name in
                        ProviderCredentialRowView(providerName: name)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 520)
    }
}
