import Foundation

/// KEY-2 — the "where do I get one" facts for a provider's credential row: the signup
/// URL and a plain free-tier note. This is Settings-only prose, never consulted by
/// FlowKit's own resolution — Registry §7/M6 only cares about the *account name*
/// (`credentials: "tavily"` → Keychain account `provider-tavily`), not where a human
/// gets the key. A `credentials` name with no entry here (any provider RM/WS ships
/// before this table is updated) still gets a full row from `installedCredentialNames`
/// — missing signup info is a cosmetic gap, never a reason to hide the row.
nonisolated struct ProviderCredentialInfo: Sendable, Equatable {
    let signupURL: URL
    let freeTierNote: String

    /// Keyed by the manifest's `credentials` name. Two facts confirmed by the owner's
    /// own KEY-2 instructions (`RSI/DelegateOffMachineBacklog.md`): Tavily's free tier
    /// is 1,000 searches/month; Brave's search API is paid only.
    static let known: [String: ProviderCredentialInfo] = [
        "tavily": ProviderCredentialInfo(
            signupURL: URL(string: "https://tavily.com")!,
            freeTierNote: "1,000 searches/month free"),
        "brave": ProviderCredentialInfo(
            signupURL: URL(string: "https://api.search.brave.com")!,
            freeTierNote: "paid only"),
    ]
}

/// KEY-2's **Test** button: one cheap live call per provider, reporting plainly and
/// never surfacing the key itself (success/failure only — see `KeychainHelperTests`'
/// header for why that constraint runs through this whole phase).
///
/// Deliberately **not** a plugin registry. No manifest on disk names a `credentials`
/// value yet (`CuratedManifest.installedCredentialNames()` returns `[]` today,
/// verified at journal `2026-259`), so there is no real endpoint a live call could hit
/// — building an extensibility mechanism for zero current cases is exactly the
/// "design for hypothetical future requirements" `CLAUDE.md` rules against. RM-1/WS-1
/// add their own `case` here, alongside the manifest that makes the row exist, the same
/// single-PR shape as every other "one line" plugin point this phase leaves behind
/// (`ProviderCredential.readiness`, `SetupAction.openSettings`).
nonisolated enum ProviderKeyTester {
    /// Not `Result<Void, String>` — `String` doesn't conform to `Error`, and wrapping
    /// it in one just to satisfy that would invent an error type nothing ever throws.
    enum Outcome: Sendable, Equatable {
        case success
        case failure(String)
    }

    static func test(providerName: String, key: String) async -> Outcome {
        switch providerName {
        default:
            return .failure("\(providerName) doesn't have a live check wired up yet. The key is saved.")
        }
    }
}
