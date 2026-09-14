import SwiftUI

/// SET-1 (`RSI/DelegateSettingsBacklog.md`) — the real `Settings` scene root, replacing the
/// 460×560 unresizable sheet `SettingsView` used to be. A `TabView` is the macOS-14-budget way
/// to switch panes (`Tab`/`.sidebarAdaptable` are 15.0+ and out of budget — see backlog §3.1).
///
/// Three tabs so far: Models, Providers, Tools (`.privacy` has no tab yet — SET-5). Content is
/// each old `SettingsView` section moved verbatim into its own pane — no copy or grouping
/// changes (that's SET-3+). SET-2 adds the persisted selection: `selectedPane` is
/// `SettingsPane`'s own `String` raw value under `@AppStorage("settingsPane")`, the same key
/// every `SettingsOpener` writes to before it opens the window.
struct SettingsRootView: View {
    @AppStorage("settingsPane") private var selectedPane: SettingsPane = .models

    var body: some View {
        TabView(selection: $selectedPane) {
            ModelsSettingsView()
                .tabItem { Label(SettingsPane.models.title, systemImage: SettingsPane.models.systemImage) }
                .tag(SettingsPane.models)
            ProvidersSettingsView()
                .tabItem { Label(SettingsPane.providers.title, systemImage: SettingsPane.providers.systemImage) }
                .tag(SettingsPane.providers)
            ToolsSettingsView()
                .tabItem { Label(SettingsPane.agentTools.title, systemImage: SettingsPane.agentTools.systemImage) }
                .tag(SettingsPane.agentTools)
        }
    }
}
