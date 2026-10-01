import SwiftUI

/// One **configured** provider in Settings' Providers pane (KC-5.2): its name, Test, Change…
/// and Remove. Providers with no key don't get a row — they're reached from the section's Add
/// button. The key's *value* never appears here: `status` only ever holds "Saved."/"Removed."/a
/// tester's plain success-or-failure sentence, never the key itself (the constraint that runs
/// through this whole phase — see `KeychainHelperTests`). `Test` is the only read of the secret,
/// and only when the user presses it.
struct ProviderCredentialRowView: View {
    let providerName: String
    let onChange: () -> Void
    let onRemove: () -> Void
    @State private var status: String = ""
    @State private var isTesting: Bool = false

    private var account: String { KeychainHelper.providerAccount(providerName) }

    var body: some View {
        HStack {
            Text(providerName.capitalized)
                .font(.subheadline.bold())
            Label("A key is set.", systemImage: "checkmark.seal.fill")
                .font(.caption)
                .foregroundStyle(Color.green)
            Spacer()
            if !status.isEmpty {
                Text(status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Button("Test") { Task { await test() } }
                .disabled(isTesting)
            Button("Change…") { onChange() }
            Button("Remove", role: .destructive) { onRemove() }
        }
        .padding(.vertical, 2)
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
