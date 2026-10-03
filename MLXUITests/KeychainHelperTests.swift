import Testing
import Foundation
@testable import MLXUI

/// KEY-1 — `KeychainHelper` generalized from one hardcoded HF-token account to
/// `account:`-parameterized `get`/`save`/`delete`. Every value here is a throwaway,
/// UUID-suffixed test string — never a real secret, never printed, and this file never
/// asserts on a key's *contents* in a way that would need one logged to debug a failure.
struct KeychainHelperTests {

    /// The HF-token wrappers round-trip. KC-1.3: this runs against a throwaway account via the
    /// wrappers' `account:` seam, so it never reads, overwrites or deletes the developer's
    /// real `huggingface-token` item (which itself raised a Keychain prompt in `xcodebuild test`).
    @Test func hfTokenWrappersRoundTrip() {
        let account = KeychainHelper.providerAccount("test-hf-\(UUID().uuidString)")
        defer { KeychainHelper.deleteToken(account: account) }
        let probe = "test-\(UUID().uuidString)"
        #expect(!KeychainHelper.hasToken(account: account))
        KeychainHelper.saveToken(probe, account: account)
        #expect(KeychainHelper.hasToken(account: account))
        #expect(KeychainHelper.getToken(account: account) == probe)
        KeychainHelper.deleteToken(account: account)
        #expect(!KeychainHelper.hasToken(account: account))
        #expect(KeychainHelper.getToken(account: account) == nil)
    }

    /// The wrappers' default account is still the HF item — pinned by name, no Keychain access.
    @Test func hfWrappersDefaultToTheHuggingFaceAccount() {
        #expect(KeychainHelper.hfAccount == "huggingface-token")
    }

    /// KC-1.1: `exists` tracks save/delete and is false for a never-set account.
    @Test func existsTracksSaveAndDelete() {
        let account = KeychainHelper.providerAccount("test-exists-\(UUID().uuidString)")
        defer { KeychainHelper.delete(account: account) }
        #expect(!KeychainHelper.exists(account: account))
        KeychainHelper.save("value", account: account)
        #expect(KeychainHelper.exists(account: account))
        KeychainHelper.delete(account: account)
        #expect(!KeychainHelper.exists(account: account))
    }

    /// Two provider accounts (KEY-1's actual reason for existing) don't collide, and
    /// deleting one leaves the other untouched.
    @Test func twoProviderAccountsDoNotCollide() {
        let accountA = KeychainHelper.providerAccount("test-a-\(UUID().uuidString)")
        let accountB = KeychainHelper.providerAccount("test-b-\(UUID().uuidString)")
        defer {
            KeychainHelper.delete(account: accountA)
            KeychainHelper.delete(account: accountB)
        }
        KeychainHelper.save("key-a", account: accountA)
        KeychainHelper.save("key-b", account: accountB)
        #expect(KeychainHelper.get(account: accountA) == "key-a")
        #expect(KeychainHelper.get(account: accountB) == "key-b")

        KeychainHelper.delete(account: accountA)
        #expect(KeychainHelper.get(account: accountA) == nil)
        #expect(KeychainHelper.get(account: accountB) == "key-b")   // untouched by the delete
    }

    /// `providerAccount` names the Registry §7 / M6 convention exactly: `provider-<name>`.
    @Test func providerAccountNamingMatchesTheRegistryConvention() {
        #expect(KeychainHelper.providerAccount("anthropic") == "provider-anthropic")
        #expect(KeychainHelper.providerAccount("tavily") == "provider-tavily")
    }

    /// Saving overwrites rather than duplicates — a second `save` for the same account must
    /// still read back exactly one value (`SecItemAdd` on an existing item fails
    /// `errSecDuplicateItem` without the `SecItemDelete` `save(_:account:)` does first).
    @Test func savingTwiceOverwritesRatherThanDuplicates() {
        let account = KeychainHelper.providerAccount("test-overwrite-\(UUID().uuidString)")
        defer { KeychainHelper.delete(account: account) }
        KeychainHelper.save("first", account: account)
        KeychainHelper.save("second", account: account)
        #expect(KeychainHelper.get(account: account) == "second")
    }

    @Test func deletingAnUnsetAccountIsANoOp() {
        let account = KeychainHelper.providerAccount("test-never-set-\(UUID().uuidString)")
        KeychainHelper.delete(account: account)   // must not crash
        #expect(KeychainHelper.get(account: account) == nil)
    }
}
