import SwiftUI

/// SET-2 (`RSI/DelegateSettingsBacklog.md`, D3) — every "Open Settings…" button in the app
/// goes through this one view, so a deep link naming a pane and a plain "open Settings"
/// button can't drift into two different techniques the way the sheet-era code did.
///
/// Setting `selectedPane` in a `simultaneousGesture` and letting `SettingsLink` itself open
/// the window is the documented technique for driving `SettingsLink` to a specific tab — the
/// ordering (does the pane get written before the window reads it?) hasn't been proven not to
/// race in this app (backlog §7). If it ever visibly races, the fallback is writing
/// `@AppStorage` from a plain button action and opening the link on the next runloop tick —
/// write that finding here if it's hit, because it will come up again.
struct SettingsOpener<Label: View>: View {
    let pane: SettingsPane
    @ViewBuilder let label: () -> Label

    @AppStorage("settingsPane") private var selectedPane: SettingsPane = .models

    var body: some View {
        SettingsLink(label: label)
            .simultaneousGesture(TapGesture().onEnded {
                selectedPane = pane
            })
    }
}
