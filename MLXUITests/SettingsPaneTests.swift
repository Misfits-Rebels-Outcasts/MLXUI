import Testing
@testable import MLXUI

/// SET-1's G2 seam (`RSI/DelegateSettingsBacklog.md`) — `SettingsPane` gained `CaseIterable`,
/// `title`, `systemImage`, and a `String` raw value when Settings became a real window with a
/// `TabView`. The raw value becomes the persisted `@AppStorage` tab key in SET-2, so once a
/// case ships its raw value must never change; `allCases` order is the tab order, so a case
/// added out of order silently reorders someone's window.
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
    }

    @Test func allCasesOrderIsTheTabOrder() {
        #expect(SettingsPane.allCases == [.models, .providers])
    }

    @Test func existingSetupActionEqualityStillHolds() {
        // The three tests in CatFlowProviderCredentialTests, CatFlowNetToolsTests and
        // CatFlowRemoteModelsTests compare `.openSettings(.providers)` by equality — pinned
        // here so a future change to SettingsPane's conformances can't silently break them.
        #expect(SetupAction.openSettings(.providers) == .openSettings(.providers))
    }
}
