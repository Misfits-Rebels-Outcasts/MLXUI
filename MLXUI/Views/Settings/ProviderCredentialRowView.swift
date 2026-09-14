import SwiftUI

/// KEY-2 — one row of Settings' "Model & Search Providers" section: a provider's name,
/// a `SecureField`, Save/Test/Remove, and whether a key is already set. The key's
/// *value* never appears anywhere but the field the user just typed into — `hasKey`
/// renders as a label, `status` only ever holds "Saved."/"Removed."/a tester's plain
/// success-or-failure sentence, never the key itself (the constraint that runs through
/// this whole phase — see `KeychainHelperTests`).
struct ProviderCredentialRowView: View {
    let providerName: String
    @State private var keyInput: String = ""
    // Read on .task, not here: a synchronous Keychain read in every row's init fired the
    // instant `TabView` built this tab's content (macOS builds every tab up front, not just
    // the selected one) — for a saved credential, that means a macOS Keychain access prompt
    // for every provider at once, the moment Settings opens, regardless of which tab is
    // showing. Deferring to .task means a row only touches its own Keychain item once it's
    // actually on screen.
    @State private var hasKey: Bool = false
    @State private var status: String = ""
    @State private var isTesting: Bool = false

    init(providerName: String) {
        self.providerName = providerName
    }

    private var account: String { KeychainHelper.providerAccount(providerName) }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(providerName.capitalized)
                .font(.subheadline.bold())

            SecureField("Paste your \(providerName) key", text: $keyInput)
                .textFieldStyle(.roundedBorder)

            HStack {
                Button("Save") { save() }
                    .disabled(keyInput.isEmpty)
                Button("Test") { Task { await test() } }
                    .disabled(!hasKey || isTesting)
                Button("Remove", role: .destructive) { remove() }
                    .disabled(!hasKey)
                Spacer()
                if !status.isEmpty {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Label(hasKey ? "A key is set." : "No key set.",
                  systemImage: hasKey ? "checkmark.seal.fill" : "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(hasKey ? Color.green : Color.secondary)

            if let info = ProviderCredentialInfo.known[providerName] {
                Link("Get a \(providerName) key (\(info.freeTierNote))", destination: info.signupURL)
                    .font(.caption)
            }
        }
        .padding(.vertical, 4)
        .task { hasKey = KeychainHelper.get(account: account) != nil }
    }

    private func save() {
        KeychainHelper.save(keyInput, account: account)
        hasKey = true
        keyInput = ""
        status = "Saved."
    }

    private func remove() {
        KeychainHelper.delete(account: account)
        hasKey = false
        status = "Removed."
    }

    private func test() async {
        isTesting = true
        status = "Testing…"
        defer { isTesting = false }
        guard let key = KeychainHelper.get(account: account) else {
            status = "No key set."
            return
        }
        switch await ProviderKeyTester.test(providerName: providerName, key: key) {
        case .success:
            status = "Success."
        case .failure(let message):
            status = message
        }
    }
}
