import Testing
import Foundation
import os
@testable import MLXUI

/// KC-2 — presence is cached in memory; `KeychainHelper.save`/`delete` keep the cache current.
/// Throwaway UUID accounts only; no secret contents are read, logged or asserted.
struct CredentialPresenceTests {

    private func counted(_ answer: Bool) -> (CredentialPresence, OSAllocatedUnfairLock<Int>) {
        let hits = OSAllocatedUnfairLock(initialState: 0)
        let cache = CredentialPresence(probe: { _ in hits.withLock { $0 += 1 }; return answer })
        return (cache, hits)
    }

    @Test func probesOncePerAccountThenServesFromMemory() {
        let (cache, hits) = counted(true)
        #expect(cache.isPresent(account: "a"))
        #expect(cache.isPresent(account: "a"))
        #expect(cache.isPresent(account: "a"))
        #expect(hits.withLock { $0 } == 1)
        #expect(cache.isPresent(account: "b"))
        #expect(hits.withLock { $0 } == 2)
    }

    @Test func absenceIsCachedToo() {
        let (cache, hits) = counted(false)
        #expect(!cache.isPresent(account: "a"))
        #expect(!cache.isPresent(account: "a"))
        #expect(hits.withLock { $0 } == 1)
    }

    @Test func recordOverridesWithoutProbing() {
        let (cache, hits) = counted(false)
        cache.record(account: "a", present: true)
        #expect(cache.isPresent(account: "a"))
        #expect(hits.withLock { $0 } == 0)
    }

    @Test func invalidateForcesARequery() {
        let (cache, hits) = counted(true)
        _ = cache.isPresent(account: "a")
        cache.invalidate(account: "a")
        _ = cache.isPresent(account: "a")
        #expect(hits.withLock { $0 } == 2)
        cache.invalidate()
        _ = cache.isPresent(account: "a")
        #expect(hits.withLock { $0 } == 3)
    }

    /// Save → ready, delete → needsSetup, through the real Keychain and the shared cache,
    /// with the cache primed first so a stale "absent" would be caught.
    @Test func saveAndDeleteFlipReadinessThroughTheSharedCache() {
        let name = "test-presence-\(UUID().uuidString)"
        let account = KeychainHelper.providerAccount(name)
        defer { KeychainHelper.delete(account: account) }

        guard case .needsSetup = ProviderCredential.readiness(providerName: name) else {
            Issue.record("expected .needsSetup before any key is saved"); return
        }
        KeychainHelper.save("value", account: account)
        guard case .ready = ProviderCredential.readiness(providerName: name) else {
            Issue.record("expected .ready after save"); return
        }
        KeychainHelper.delete(account: account)
        guard case .needsSetup = ProviderCredential.readiness(providerName: name) else {
            Issue.record("expected .needsSetup after delete"); return
        }
    }
}
