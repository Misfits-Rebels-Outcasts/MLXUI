import Foundation

/// SET-5's G2 seam (`RSI/DelegateSettingsBacklog.md`) — one row of the Privacy pane's
/// disclosure table, derived from a `CuratedManifest` rather than hand-typed, so a new
/// provider or search manifest updates the disclosure automatically.
nonisolated struct PrivacyRow: Sendable, Equatable {
    /// The manifest's own display name (e.g. `"claude-sonnet @ anthropic"`, `"Tavily"`).
    let destination: String
    let carrying: String
    let account: String
    /// The Keychain credential name this row's manifest declares (`CuratedManifest
    /// .credentials`), or `nil` for a credential-less LAN endpoint. Cross-referenced against
    /// `CuratedManifest.installedCredentialNames` by `PrivacyInventoryTests` — every name
    /// here must have a matching Providers-pane row.
    let credentialsName: String?
}

/// SET-5 (`RSI/DelegateSettingsBacklog.md`) — derives the Privacy pane's per-manifest rows
/// from the installed catalog (`RM-6-privacy-disclosure.md` §1 is the reference table this
/// mirrors, one row per manifest instead of one row per provider family, so a manifest for a
/// brand-new provider needs no grouping list updated to show up).
nonisolated enum PrivacyInventory {
    /// A manifest leaves the machine when its `kind` is `"provider"` (a remote API or LAN
    /// endpoint, RM) or `"search"` (a `.net`-class tool's key, WS) — `"local"` (a
    /// `browser.json` download) and `"system"` (Apple Foundation Models) never do
    /// (`CuratedManifest.kind`'s own doc comment, `CatalogBridge.swift`).
    static func isOffMachine(_ manifest: CuratedManifest) -> Bool {
        manifest.kind == "provider" || manifest.kind == "search"
    }

    /// One row per off-machine manifest, sorted by display name for a stable order.
    static func rows(from manifests: [CuratedManifest]) -> [PrivacyRow] {
        manifests
            .filter(isOffMachine)
            .sorted { $0.display < $1.display }
            .map { manifest in
                PrivacyRow(
                    destination: manifest.display,
                    carrying: carrying(for: manifest),
                    account: account(for: manifest),
                    credentialsName: manifest.credentials
                )
            }
    }

    private static func carrying(for manifest: CuratedManifest) -> String {
        if manifest.kind == "search" {
            return "the query text"
        }
        return manifest.credentials != nil
            ? "the row's prompt — the user's actual document content"
            : "the row's prompt"
    }

    private static func account(for manifest: CuratedManifest) -> String {
        manifest.credentials != nil
            ? "the user's own API key"
            : "none — the user's own machine"
    }
}
