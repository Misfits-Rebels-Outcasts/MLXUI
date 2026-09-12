import Testing
import Foundation
@testable import MLXUI

/// KEY-2/KEY-3 — the manifest-driven credential scan (`CuratedManifest
/// .installedCredentialNames`), the Keychain-backed readiness helper
/// (`ProviderCredential.readiness`), and `R910`'s catalog entry. Every Keychain value
/// here is a throwaway, UUID-suffixed test string — never printed, never a real key,
/// per the phase's standing constraint (see `KeychainHelperTests`' header).
struct CatFlowProviderCredentialTests {

    // MARK: - CuratedManifest.installedCredentialNames

    private func write(_ json: String, to directory: URL, name: String) throws -> URL {
        let url = directory.appendingPathComponent(name)
        try json.data(using: .utf8)!.write(to: url)
        return url
    }

    /// A manifest with both `kind` and `credentials` contributes its credential name.
    @Test func manifestWithKindAndCredentialsContributesARow() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let url = try write("""
        {"id": "anthropic/claude-sonnet", "display": "claude-sonnet @ anthropic",
         "kind": "provider", "credentials": "anthropic", "settings": {}}
        """, to: dir, name: "claude.json")

        #expect(CuratedManifest.installedCredentialNames(manifestURLs: [url]) == ["anthropic"])
    }

    /// Two manifests naming the same credential collapse to one row, not two.
    @Test func twoManifestsSharingACredentialProduceOneRow() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let a = try write("""
        {"id": "anthropic/claude-sonnet", "display": "claude-sonnet @ anthropic",
         "kind": "provider", "credentials": "anthropic", "settings": {}}
        """, to: dir, name: "claude-sonnet.json")
        let b = try write("""
        {"id": "anthropic/claude-haiku", "display": "claude-haiku @ anthropic",
         "kind": "provider", "credentials": "anthropic", "settings": {}}
        """, to: dir, name: "claude-haiku.json")

        #expect(CuratedManifest.installedCredentialNames(manifestURLs: [a, b]) == ["anthropic"])
    }

    /// A manifest with no `credentials` field (every local MLX manifest today) contributes
    /// nothing — this is the common case, not an edge case.
    @Test func manifestWithNoCredentialsContributesNoRow() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let url = try write("""
        {"id": "mlx-community/Kokoro-82M-4bit", "display": "Kokoro 82M",
         "kind": "local", "settings": {}}
        """, to: dir, name: "kokoro.json")

        #expect(CuratedManifest.installedCredentialNames(manifestURLs: [url]).isEmpty)
    }

    /// Unrelated bundled JSON (a `browser.json`/gallery-`_metadata.json` shape) fails to
    /// decode as a `CuratedManifest` and is silently skipped, not crashed on or misread —
    /// this is what makes bundle-root scanning safe despite everything flattening together.
    @Test func nonManifestJSONIsSkippedNotMisread() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let url = try write("""
        {"domains": [], "models": [], "version": "6.0-mvp"}
        """, to: dir, name: "browser.json")

        #expect(CuratedManifest.installedCredentialNames(manifestURLs: [url]).isEmpty)
    }

    /// The production entry point, run against the real bundle: RM-1 ported three keyed
    /// provider manifests (`claude-sonnet-4`/`gpt-5.6-luna`/`deepseek-v4-flash`), each
    /// naming its own `credentials`; the fourth (`macstudio-qwen3-32b`, the keyless LAN
    /// endpoint) correctly contributes no row. `[]` was the verified answer at KEY —
    /// this is the verified answer now, updated rather than loosened, per that journal's
    /// own note that it "will legitimately need updating the day RM or WS lands the first
    /// `credentials`-bearing manifest, which is the entire point."
    @Test func shippedManifestsNameExactlyRMsThreeKeyedProviders() {
        #expect(CuratedManifest.installedCredentialNames() == ["anthropic", "deepseek", "openai"])
    }

    // MARK: - ProviderCredential.readiness

    @Test func readinessIsReadyOnceAKeyIsSaved() {
        let name = "test-provider-\(UUID().uuidString)"
        let account = KeychainHelper.providerAccount(name)
        defer { KeychainHelper.delete(account: account) }

        #expect(ProviderCredential.readiness(providerName: name) == .needsSetup(
            reason: "Add your \(name) key in Settings", action: .openSettings(.providers)))

        KeychainHelper.save("test-key-\(UUID().uuidString)", account: account)
        #expect(ProviderCredential.readiness(providerName: name) == .ready)
    }

    @Test func readinessGoesBackToNeedsSetupAfterRemoval() {
        let name = "test-provider-\(UUID().uuidString)"
        let account = KeychainHelper.providerAccount(name)
        KeychainHelper.save("test-key-\(UUID().uuidString)", account: account)
        #expect(ProviderCredential.readiness(providerName: name) == .ready)

        KeychainHelper.delete(account: account)
        guard case .needsSetup(_, let action) = ProviderCredential.readiness(providerName: name) else {
            Issue.record("expected .needsSetup once the key is removed")
            return
        }
        #expect(action == .openSettings(.providers))
    }

    // MARK: - R910

    @Test func r910MatchesTheEstablishedVoiceAndFillsCleanly() throws {
        let v07 = try ErrorCatalog.fill(code: "R910", values: ["n": "3", "provider": "tavily"], isV08: false)
        #expect(v07 == "Row 3 needs a tavily key, and none is set. Nothing has run yet — add the key in Settings, then start the run again.")

        let v08 = try ErrorCatalog.fill(code: "R910", values: ["n": "3", "provider": "tavily"], isV08: true)
        #expect(v08 == v07)
    }

    @Test func r910IsTheNextFreeCodeAfterR909() {
        #expect(ErrorCatalog.catalogV08["R910"] != nil)
        #expect(ErrorCatalog.catalogV07["R910"] != nil)
        for code in ["R901", "R902", "R903", "R904", "R905", "R906", "R907", "R908", "R909"] {
            #expect(ErrorCatalog.catalogV08[code] != nil)
        }
    }

    // MARK: - ProviderKeyTester

    /// No provider has a live check wired up yet (none exist — RM/WS haven't landed);
    /// the tester says so plainly and never fabricates a success.
    @Test func testerReportsPlainlyWhenNoLiveCheckExists() async {
        let outcome = await ProviderKeyTester.test(providerName: "tavily", key: "irrelevant-\(UUID().uuidString)")
        guard case .failure(let message) = outcome else {
            Issue.record("expected .failure — no tester is registered for any provider yet")
            return
        }
        #expect(message.contains("tavily"))
        #expect(!message.isEmpty)
    }
}
