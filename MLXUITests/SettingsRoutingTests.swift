import Testing
@testable import MLXUI

/// SET-2's G2 seam (`RSI/DelegateSettingsBacklog.md`, D3) — `settingsPane(for:)` is the pure
/// decision behind `FlowListView`'s fix-it buttons landing on the pane they actually name,
/// instead of Settings' first tab regardless of which key was missing.
struct SettingsRoutingTests {

    @Test func openSettingsRoutesToItsOwnPane() {
        #expect(settingsPane(for: .openSettings(.providers)) == .providers)
        #expect(settingsPane(for: .openSettings(.models)) == .models)
    }

    @Test func enableAppleIntelligenceRoutesToModels() {
        #expect(settingsPane(for: .enableAppleIntelligence) == .models)
    }

    @Test func installModelHasNoPane() {
        let entry = makeEntry()
        #expect(settingsPane(for: .installModel(entry)) == nil)
    }
}
