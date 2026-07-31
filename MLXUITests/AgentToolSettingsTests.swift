import Foundation
import Testing
@testable import MLXUI

/// AG6b — per-tool approval policy. The policy model, gate decision, mutation, and persistence
/// round-trip are pure and tested synchronously; the wrench-menu UI is verified by build + smoke.
struct AgentToolSettingsTests {

    @Test func defaultPolicyIsAsk() {
        let settings = AgentToolSettings()
        #expect(settings.policy(for: "write_clipboard") == .ask)
        #expect(settings.decision(for: "write_clipboard") == .prompt)
    }

    @Test func decisionMapsEachPolicy() {
        let ask = AgentToolSettings().setting(.ask, for: "t")
        let always = AgentToolSettings().setting(.always, for: "t")
        let never = AgentToolSettings().setting(.never, for: "t")
        #expect(ask.decision(for: "t") == .prompt)
        #expect(always.decision(for: "t") == .autoApprove)
        #expect(never.decision(for: "t") == .autoDeny)
    }

    @Test func settingUpdatesPolicy() {
        let settings = AgentToolSettings().setting(.always, for: "write_clipboard")
        #expect(settings.policy(for: "write_clipboard") == .always)
        // An unrelated tool is unaffected.
        #expect(settings.policy(for: "write_file") == .ask)
    }

    @Test func settingBackToAskClearsTheOverride() {
        // `.ask` is the default and is stored as an absence, keeping the dict minimal.
        let toggled = AgentToolSettings()
            .setting(.never, for: "t")
            .setting(.ask, for: "t")
        #expect(toggled == AgentToolSettings())
        #expect(toggled.policy(for: "t") == .ask)
    }

    @Test func roundTripsThroughJSON() throws {
        let settings = AgentToolSettings().setting(.always, for: "write_clipboard")
        let data = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(AgentToolSettings.self, from: data)
        #expect(decoded == settings)
    }

    @Test func persistsThroughUserDefaults() {
        let suite = "AgentToolSettingsTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }

        AgentToolSettings().setting(.never, for: "write_clipboard").save(to: defaults)
        let loaded = AgentToolSettings.load(from: defaults)
        #expect(loaded.policy(for: "write_clipboard") == .never)
    }

    @Test func loadReturnsDefaultsWhenNothingStored() {
        let suite = "AgentToolSettingsTests-empty-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        #expect(AgentToolSettings.load(from: defaults) == AgentToolSettings())
    }
}
