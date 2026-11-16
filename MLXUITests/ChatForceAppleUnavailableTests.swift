import Foundation
import Testing
@testable import MLXUI

/// CD-DEBUG-1 — the DEBUG-only `-ChatForceAppleUnavailable YES` launch argument.
@Suite("ChatForceAppleUnavailable")
struct ChatForceAppleUnavailableTests {

    private func defaults() -> UserDefaults {
        let name = "ChatForceAppleUnavailableTests-\(UUID().uuidString)"
        let suite = UserDefaults(suiteName: name) ?? .standard
        suite.removePersistentDomain(forName: name)
        return suite
    }

    @Test func overrideIsCompiledInExactlyWhenDebug() {
        #if DEBUG
        #expect(ChatController.debugOverrideIsCompiledIn)
        #else
        #expect(!ChatController.debugOverrideIsCompiledIn)
        #endif
    }

    #if DEBUG
    @Test func theArgumentMakesAppleIntelligenceUnavailable() {
        let d = defaults()
        d.set(true, forKey: ChatController.debugForceAppleUnavailableKey)
        guard case .unavailable? = ChatController.defaultReadiness(defaults: d) else {
            Issue.record("expected .unavailable")
            return
        }
    }

    @Test func withoutTheArgumentTheRealReadinessIsUsed() {
        let d = defaults()
        #expect(ChatController.defaultReadiness(defaults: d) == AppleFoundationAvailability.currentReadiness())
    }

    @Test func launchArgumentSpellingParsesAsTrue() {
        let d = defaults()
        d.set("YES", forKey: ChatController.debugForceAppleUnavailableKey)
        #expect(d.bool(forKey: ChatController.debugForceAppleUnavailableKey))
    }
    #endif
}
