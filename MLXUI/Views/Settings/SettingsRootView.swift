import SwiftUI

/// SET-1 (`RSI/DelegateSettingsBacklog.md`) — the real `Settings` scene root, replacing the
/// 460×560 unresizable sheet `SettingsView` used to be. A `TabView` is the macOS-14-budget way
/// to switch panes (`Tab`/`.sidebarAdaptable` are 15.0+ and out of budget — see backlog §3.1).
///
/// Four tabs: Models, Providers, Tools, Privacy (SET-5) — Providers and Privacy omitted from
/// the tab bar entirely when `AppState.hideProvidersPrivacy` is `true`; Models and Tools always
/// render. `selectedPane` is `SettingsPane`'s own `String` raw value under
/// `@AppStorage("settingsPane")`, the same key every `SettingsOpener` writes to before it opens
/// the window.
///
/// Each tab wires its content view directly — `if selectedPane == <pane> { View() }` in place of
/// the view was tried to defer `ProvidersSettingsView`'s per-row Keychain reads until that tab is
/// actually selected (`TabView` on macOS otherwise builds every tab's view up front regardless of
/// which one is selected — see `ProviderCredentialRowView`'s note), but that gate broke `TabView`'s
/// own tag-to-content matching outright: **every** tab, including the unconditional Models and
/// Tools ones, rendered blank. Reverted; the Keychain-burst concern is moot in practice now that
/// `hideProvidersPrivacy` keeps `ProvidersSettingsView` out of the tab bar for most builds.
struct SettingsRootView: View {
    @AppStorage("settingsPane") private var selectedPane: SettingsPane = .models

    var body: some View {
        TabView(selection: $selectedPane) {
            ModelsSettingsView()
                .tabItem { Label(SettingsPane.models.title, systemImage: SettingsPane.models.systemImage) }
                .tag(SettingsPane.models)

            if !AppState.hideProvidersPrivacy {
                ProvidersSettingsView()
                    .tabItem { Label(SettingsPane.providers.title, systemImage: SettingsPane.providers.systemImage) }
                    .tag(SettingsPane.providers)
            }

            ToolsSettingsView()
                .tabItem { Label(SettingsPane.agentTools.title, systemImage: SettingsPane.agentTools.systemImage) }
                .tag(SettingsPane.agentTools)

            if !AppState.hideProvidersPrivacy {
                PrivacyView()
                    .tabItem { Label(SettingsPane.privacy.title, systemImage: SettingsPane.privacy.systemImage) }
                    .tag(SettingsPane.privacy)
            }
        }
        .onAppear {
            // A prior launch (before this flag was set, or with it toggled back) may have left
            // `selectedPane` pointed at a tab that's hidden now — land on Models instead of a
            // pane the TabView no longer offers.
            if AppState.hideProvidersPrivacy,
               selectedPane == .providers || selectedPane == .privacy {
                selectedPane = .models
            }
        }
    }
}
