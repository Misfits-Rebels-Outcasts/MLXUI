import Testing
import Foundation
@testable import MLXUI

/// KEY-1 — `KeychainHelper` generalized from one hardcoded HF-token account to
/// `account:`-parameterized `get`/`save`/`delete`. Every value here is a throwaway,
/// UUID-suffixed test string — never a real secret, never printed, and this file never
/// asserts on a key's *contents* in a way that would need one logged to debug a failure.
struct KeychainHelperTests {

    /// The existing HF-token wrappers still round-trip, and restore whatever was there
    /// before the test ran (a developer's real token, or nothing) rather than clobbering it.
    @Test func hfTokenStillRoundTrips() {
        let original = KeychainHelper.getToken()
        defer {
            if let original {
                KeychainHelper.saveToken(original)
            } else {
                KeychainHelper.deleteToken()
            }
        }
        let probe = "test-\(UUID().uuidString)"
        KeychainHelper.saveToken(probe)
        #expect(KeychainHelper.getToken() == probe)
        KeychainHelper.deleteToken()
        #expect(KeychainHelper.getToken() == nil)
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
