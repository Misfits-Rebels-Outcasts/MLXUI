import SwiftUI

/// SET-1 (`RSI/DelegateSettingsBacklog.md`) — the real `Settings` scene root, replacing the
/// 460×560 unresizable sheet `SettingsView` used to be. A `TabView` is the macOS-14-budget way
/// to switch panes (`Tab`/`.sidebarAdaptable` are 15.0+ and out of budget — see backlog §3.1).
///
/// Three tabs this phase: Models, Providers, Tools. Their content is each old `SettingsView`
/// section moved verbatim into its own pane — no copy or grouping changes (that's SET-3+).
/// Tab selection isn't yet persisted (`@AppStorage`) or driven by `SettingsPane.allCases` —
/// the Tools tab has no `SettingsPane` case until SET-2 adds `.agentTools`, so a case-driven
/// loop can't cover all three yet. SET-2 wires selection and the deep link together.
struct SettingsRootView: View {
    var body: some View {
        TabView {
            ModelsSettingsView()
                .tabItem { Label(SettingsPane.models.title, systemImage: SettingsPane.models.systemImage) }
            ProvidersSettingsView()
                .tabItem { Label(SettingsPane.providers.title, systemImage: SettingsPane.providers.systemImage) }
            ToolsSettingsView()
                .tabItem { Label("Tools", systemImage: "wrench.and.screwdriver") }
        }
    }
}
