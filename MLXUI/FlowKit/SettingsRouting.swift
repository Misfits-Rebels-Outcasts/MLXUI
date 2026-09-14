import Foundation

/// SET-2's G2 seam (`RSI/DelegateSettingsBacklog.md`, D3) — the pure decision
/// `FlowListView.setupActionButton` delegates to, so it's testable without a SwiftUI host
/// (fact 18: there are no UI tests in this repo, so a view body can't be gated by G2).
///
/// `.openSettings(pane)` names its pane directly. `.enableAppleIntelligence` has none of its
/// own yet — `SettingsPane.models`'s own doc says a system model's setup would be offered
/// there, so that's where it routes. `.installModel` returns `nil`: it is never produced
/// today (MS-4) and there is no "Install" pane to send it to: the caller falls back to a
/// plain `SettingsLink`.
nonisolated func settingsPane(for action: SetupAction) -> SettingsPane? {
    switch action {
    case .openSettings(let pane):
        return pane
    case .enableAppleIntelligence:
        return .models
    case .installModel:
        return nil
    }
}
