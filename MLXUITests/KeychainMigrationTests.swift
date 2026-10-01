import Testing
import Foundation
import Security
@testable import MLXUI

/// KC-3 — data-protection keychain + lazy migration from the legacy login keychain.
/// Legacy items are planted with raw `SecItem*` calls under a UUID account and a throwaway
/// value; nothing here reads or asserts on a real secret.
struct KeychainMigrationTests {

    private let service = "com.ai-browser"

    private func legacyQuery(_ account: String) -> [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: service,
         kSecAttrAccount as String: account,
         kSecUseDataProtectionKeychain as String: false]   // file keychain only
    }

    private func plantLegacy(_ value: String, account: String) {
        var q = legacyQuery(account)
        q[kSecValueData as String] = Data(value.utf8)
        SecItemAdd(q as CFDictionary, nil)
    }

    private func legacyExists(_ account: String) -> Bool {
        var q = legacyQuery(account)
        q[kSecMatchLimit as String] = kSecMatchLimitOne
        return SecItemCopyMatching(q as CFDictionary, nil) == errSecSuccess
    }

    private func cleanUp(_ account: String) {
        KeychainHelper.delete(account: account)
        SecItemDelete(legacyQuery(account) as CFDictionary)
        UserDefaults.standard.removeObject(forKey: "keychain.migrated.\(account)")
    }

    private func freshAccount() -> String {
        KeychainHelper.providerAccount("test-mig-\(UUID().uuidString)")
    }

    /// `exists` sees an unmigrated legacy item without reading it, and does not move it.
    @Test func existsSeesALegacyItemWithoutMigratingIt() {
        let account = freshAccount()
        defer { cleanUp(account) }
        plantLegacy("legacy-value", account: account)
        #expect(KeychainHelper.exists(account: account))
        #expect(legacyExists(account))   // untouched
    }

    /// First `get` moves the value to the data-protection keychain and deletes the legacy copy.
    @Test func getMigratesLegacyItemOnce() {
        let account = freshAccount()
        defer { cleanUp(account) }
        plantLegacy("legacy-value", account: account)
        #expect(KeychainHelper.get(account: account) == "legacy-value")
        #expect(!legacyExists(account))
        #expect(UserDefaults.standard.bool(forKey: "keychain.migrated.\(account)"))
        // Served from the data-protection keychain from now on.
        #expect(KeychainHelper.get(account: account) == "legacy-value")
        #expect(KeychainHelper.exists(account: account))
    }

    /// Once flagged migrated, a legacy item that reappears is ignored.
    @Test func migratedFlagSkipsTheLegacyLookup() {
        let account = freshAccount()
        defer { cleanUp(account) }
        UserDefaults.standard.set(true, forKey: "keychain.migrated.\(account)")
        plantLegacy("stray", account: account)
        #expect(KeychainHelper.get(account: account) == nil)
        #expect(!KeychainHelper.exists(account: account))
    }

    /// Saving a new value supersedes (and removes) an unmigrated legacy copy.
    @Test func saveDropsTheLegacyCopy() {
        let account = freshAccount()
        defer { cleanUp(account) }
        plantLegacy("old", account: account)
        KeychainHelper.save("new", account: account)
        #expect(!legacyExists(account))
        #expect(KeychainHelper.get(account: account) == "new")
    }

    /// Delete removes both the data-protection item and an unmigrated legacy one.
    @Test func deleteRemovesBothKeychains() {
        let account = freshAccount()
        defer { cleanUp(account) }
        plantLegacy("old", account: account)
        KeychainHelper.delete(account: account)
        #expect(!legacyExists(account))
        #expect(!KeychainHelper.exists(account: account))
    }

    /// Update-in-place: saving twice leaves one current value.
    @Test func saveUpdatesInPlace() {
        let account = freshAccount()
        defer { cleanUp(account) }
        KeychainHelper.save("one", account: account)
        KeychainHelper.save("two", account: account)
        #expect(KeychainHelper.get(account: account) == "two")
    }

    /// The hosted test run can use the data-protection keychain; if this fails the build is
    /// unsigned for it and KC-3.2's "check signing" state is what users would see.
    @Test func hostedRunHasNoMissingEntitlement() {
        let account = freshAccount()
        defer { cleanUp(account) }
        KeychainHelper.save("probe", account: account)
        #expect(!KeychainHelper.hasEntitlementFailure)
        #expect(KeychainHelper.exists(account: account))
    }
}
