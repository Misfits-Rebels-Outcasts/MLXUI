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

    // MARK: - Enabled overrides (AG6b-2)

    @Test func enabledToolNamesStartsFromDefaults() {
        let defaults: Set<String> = ["fetch_url", "read_file"]
        #expect(AgentToolSettings().enabledToolNames(defaults: defaults) == defaults)
    }

    @Test func enabledOverridesApplyOnTopOfDefaults() {
        let settings = AgentToolSettings()
            .settingEnabled(false, for: "fetch_url", defaultEnabled: true)   // turn a default off
            .settingEnabled(true, for: "write_file", defaultEnabled: false)  // opt into write_file
        let names = settings.enabledToolNames(defaults: ["fetch_url", "read_file"])
        #expect(names == ["read_file", "write_file"])
    }

    @Test func togglingBackToTheDefaultClearsTheOverride() {
        // A value matching the build default is stored as an absence, so a future build's
        // defaults (and brand-new tools) keep applying.
        let roundTripped = AgentToolSettings()
            .settingEnabled(false, for: "fetch_url", defaultEnabled: true)
            .settingEnabled(true, for: "fetch_url", defaultEnabled: true)
        #expect(roundTripped == AgentToolSettings())
    }

    // MARK: - Tool-call limit (AG6b-2)

    @Test func toolCallLimitDefaultsAndClamps() {
        #expect(AgentToolSettings().toolCallLimit == AgentToolSettings.defaultToolCallLimit)
        #expect(AgentToolSettings().settingToolCallLimit(0).toolCallLimit == 1)
        #expect(AgentToolSettings().settingToolCallLimit(1000).toolCallLimit == 32)
        #expect(AgentToolSettings().settingToolCallLimit(16).toolCallLimit == 16)
    }

    // MARK: - Field preservation + back-compat (AG6b-2)

    @Test func mutationsPreserveTheOtherFields() {
        // Each `setting…` returns a copy; none may drop the fields it doesn't touch.
        let settings = AgentToolSettings()
            .setting(.always, for: "write_clipboard")
            .settingEnabled(true, for: "write_file", defaultEnabled: false)
            .settingToolCallLimit(16)
            .setting(.never, for: "write_file")
        #expect(settings.policy(for: "write_clipboard") == .always)
        #expect(settings.policy(for: "write_file") == .never)
        #expect(settings.enabledToolNames(defaults: []) == ["write_file"])
        #expect(settings.toolCallLimit == 16)
    }

    @Test func decodesAG6b1BlobsWithoutTheNewFields() throws {
        // Blobs persisted before AG6b-2 carry only `policies`.
        let legacy = Data(#"{"policies":{"write_clipboard":"always"}}"#.utf8)
        let decoded = try JSONDecoder().decode(AgentToolSettings.self, from: legacy)
        #expect(decoded.policy(for: "write_clipboard") == .always)
        #expect(decoded.enabledToolNames(defaults: ["fetch_url"]) == ["fetch_url"])
        #expect(decoded.toolCallLimit == AgentToolSettings.defaultToolCallLimit)
    }

    @Test func allFieldsPersistThroughUserDefaults() {
        let suite = "AgentToolSettingsTests-all-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }

        AgentToolSettings()
            .setting(.never, for: "write_clipboard")
            .settingEnabled(true, for: "write_file", defaultEnabled: false)
            .settingToolCallLimit(4)
            .save(to: defaults)
        let loaded = AgentToolSettings.load(from: defaults)
        #expect(loaded.policy(for: "write_clipboard") == .never)
        #expect(loaded.enabledToolNames(defaults: []) == ["write_file"])
        #expect(loaded.toolCallLimit == 4)
    }
}
