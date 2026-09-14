import Testing
@testable import MLXUI

/// SET-1's G2 seam (`RSI/DelegateSettingsBacklog.md`) — `SettingsPane` gained `CaseIterable`,
/// `title`, `systemImage`, and a `String` raw value when Settings became a real window with a
/// `TabView`. SET-2 wires the raw value up as the persisted `@AppStorage("settingsPane")` tab
/// key and adds `.agentTools`/`.privacy`, so once a case ships its raw value must never
/// change; `allCases` order is the tab order, so a case added out of order silently reorders
/// someone's window.
struct SettingsPaneTests {

    @Test func everyCaseHasANonEmptyTitle() {
        for pane in SettingsPane.allCases {
            #expect(!pane.title.isEmpty)
        }
    }

    @Test func everyCaseHasANonEmptySystemImage() {
        for pane in SettingsPane.allCases {
            #expect(!pane.systemImage.isEmpty)
        }
    }

    @Test func rawValuesAreStableStrings() {
        #expect(SettingsPane.models.rawValue == "models")
        #expect(SettingsPane.providers.rawValue == "providers")
        #expect(SettingsPane.agentTools.rawValue == "agentTools")
        #expect(SettingsPane.privacy.rawValue == "privacy")
    }

    @Test func allCasesOrderIsTheTabOrder() {
        #expect(SettingsPane.allCases == [.models, .providers, .agentTools, .privacy])
    }

    @Test func existingSetupActionEqualityStillHolds() {
        // The three tests in CatFlowProviderCredentialTests, CatFlowNetToolsTests and
        // CatFlowRemoteModelsTests compare `.openSettings(.providers)` by equality — pinned
        // here so a future change to SettingsPane's conformances can't silently break them.
        #expect(SetupAction.openSettings(.providers) == .openSettings(.providers))
    }
}
