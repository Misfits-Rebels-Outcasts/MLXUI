import SwiftUI

/// SET-5 (`RSI/DelegateSettingsBacklog.md`) — the Privacy pane: `RM-6-privacy-disclosure.md`
/// §1, rendered rather than left in the repo where no user will ever see it. The HuggingFace
/// row, the web-fetch row, and the "what does not leave" block are static prose taken from §1
/// verbatim (their "Since" column is development bookkeeping, not disclosure content, and is
/// dropped here). The provider/search rows are **derived** from the installed catalog via
/// `PrivacyInventory`, so a new provider manifest updates this automatically — never hand-add
/// a row here for one.
struct PrivacyView: View {
    private var rows: [PrivacyRow] {
        PrivacyInventory.rows(from: CuratedManifest.installedManifests(bundle: .main))
    }

    var body: some View {
        Form {
            Text("What leaves this Mac, and what does not.")
                .font(.caption)
                .foregroundStyle(.secondary)

            Section("What leaves this Mac") {
                row(
                    destination: "HuggingFace",
                    carrying: "the model id being fetched; the user's HF token when set",
                    account: "the user's own"
                )
                row(
                    destination: "Web Fetch · HTTP Get · Fetch Feed · Download File",
                    carrying: "the URL the row names",
                    account: "none"
                )
                ForEach(rows, id: \.destination) { entry in
                    row(destination: entry.destination, carrying: entry.carrying, account: entry.account)
                }
            }

            Section("What does not leave") {
                Text("Apple Foundation Models runs on-device. Private Cloud Compute is not used.")
                Text("The app has no server. No analytics, no telemetry, no crash reporting, no phone-home. There is nothing for the developer to receive, because there is nothing that receives.")
                Text("Keys live in the macOS Keychain, are never written into a .cat file, and are never logged.")
                Text("A flow that binds a provider cannot hide it. It must declare offdevice on line one or fail check — whatever its egress, LAN included.")
            }
            .font(.caption)
            .foregroundStyle(.secondary)

            Section {
                Text(editionLine)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                SettingsOpener(pane: .providers) { Text("Open Providers") }
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 520)
    }

    private func row(destination: String, carrying: String, account: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(destination)
                .fontWeight(.medium)
            Text(carrying)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text("Under: \(account)")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    /// OG-7 — no General tab; the edition line lives here, where it changes what every other
    /// pane's promises mean (a sandboxed edition's file tools are scoped; the Direct
    /// edition's are not).
    private var editionLine: String {
        #if DIRECT_BUILD
        "AI Browser Pro — Developer ID, notarized, no App Sandbox (Hardened Runtime only)."
        #else
        "AI Browser — Mac App Store, sandboxed."
        #endif
    }
}
