import SwiftUI

/// SET-1's Providers tab. KC-5.2 splits it by manifest `kind` into **Search Providers** and
/// **AI Model Providers**; each lists only providers that have a key (via `CredentialPresence`,
/// so opening the tab never reads a secret) and offers an Add button for the rest.
/// SET-6 removed the empty-state branch (`RSI/DelegateSettingsBacklog.md` §0: unreachable in the
/// shipping bundle — five manifests name credentials today).
struct ProvidersSettingsView: View {
    /// KEY-2 — scanned from installed manifests, never hardcoded.
    private let groups = CuratedManifest.credentialGroups()

    /// KC-3.2: set once the rows have queried the Keychain, never as "No key set".
    @State private var keychainUnavailable = false
    @State private var configured: Set<String> = []
    @State private var sheet: KeySheetTarget?

    private enum Kind { case search, provider }

    private struct KeySheetTarget: Identifiable {
        let kind: Kind
        /// `nil` = Add (choose among unconfigured); otherwise Change for that provider.
        let provider: String?
        var id: String { "\(kind)-\(provider ?? "add")" }
    }

    private func names(_ kind: Kind) -> [String] {
        kind == .search ? groups.search : groups.provider
    }

    var body: some View {
        Form {
            if keychainUnavailable {
                Text("This build can't use the Keychain — check signing")
                    .font(.caption)
                    .foregroundStyle(.red)
            }
            Text("Keys are stored in your macOS Keychain. They are never written into a flow file, and a flow you send someone does not carry them.")
                .font(.caption)
                .foregroundStyle(.secondary)

            section("Search Providers", kind: .search, addLabel: "Add Search Provider")
            section("AI Model Providers", kind: .provider, addLabel: "Add AI Model Provider")
        }
        .formStyle(.grouped)
        .frame(minWidth: 520)
        .task {
            configured = Set((groups.search + groups.provider).filter {
                CredentialPresence.shared.isPresent(account: KeychainHelper.providerAccount($0))
            })
            keychainUnavailable = KeychainHelper.hasEntitlementFailure
        }
        .sheet(item: $sheet) { target in keySheet(for: target) }
    }

    @ViewBuilder
    private func section(_ title: String, kind: Kind, addLabel: String) -> some View {
        let all = names(kind)
        let set = all.filter { configured.contains($0) }
        let unset = all.filter { !configured.contains($0) }
        Section(title) {
            ForEach(set, id: \.self) { name in
                ProviderCredentialRowView(
                    providerName: name,
                    onChange: { sheet = KeySheetTarget(kind: kind, provider: name) },
                    onRemove: { remove(name) })
            }
            Button(addLabel) { sheet = KeySheetTarget(kind: kind, provider: nil) }
                .disabled(unset.isEmpty)
        }
    }

    private func keySheet(for target: KeySheetTarget) -> some View {
        let choices = target.provider.map { [$0] }
            ?? names(target.kind).filter { !configured.contains($0) }
        return CredentialKeySheet(
            title: target.provider == nil
                ? (target.kind == .search ? "Add Search Provider" : "Add AI Model Provider")
                : "Change…",
            choices: choices,
            placeholder: { "Paste your \($0) key" },
            signupLink: { name in
                ProviderCredentialInfo.known[name].map {
                    ("Get a \(name) key (\($0.freeTierNote))", $0.signupURL)
                }
            },
            onSave: { name, key in
                KeychainHelper.save(key, account: KeychainHelper.providerAccount(name))
                configured.insert(name)
            })
    }

    private func remove(_ name: String) {
        KeychainHelper.delete(account: KeychainHelper.providerAccount(name))
        configured.remove(name)
    }
}
