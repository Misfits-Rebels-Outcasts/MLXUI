//
//  ChatGate.swift
//  MLXUI — Model Tools, MT2-2.
//
//  The one question every Chat-page surface asks (the sidebar row, the page, selection resolution):
//  "is Chat in this build?" Nothing reads `AppState.hideChat` directly (the `LocalServerGate` pattern).
//  The per-model bench sheet is the existing Run UI and is not behind this gate. Hiding Chat hides the
//  surfaces only: conversations on disk are never touched by the flag.
//
//  `overrideForTesting` is the test seam; production code never sets it.
//

import Foundation

nonisolated enum ChatGate {
    nonisolated(unsafe) static var overrideForTesting: Bool?

    static var isAvailable: Bool {
        overrideForTesting ?? !AppState.hideChat
    }

    /// A selection that names a surface the gate has closed lands on Home.
    static func resolved(_ item: SidebarItem, available: Bool = ChatGate.isAvailable) -> SidebarItem {
        if case .chat = item, !available { return .home }
        return item
    }
}

extension SidebarItem {
    /// The sidebar's first section: Home, then Chat (second, ruling J) only while the gate says so.
    static func primaryItems(chatAvailable: Bool) -> [SidebarItem] {
        chatAvailable ? [.home, .chat] : [.home]
    }
}

/// Which model the Chat page uses until MT2-4's model menu exists (pure).
nonisolated enum ChatPageModel {
    /// Keep `current` while it is still installed; else the model of the newest conversation that is
    /// still installed; else the first installed chat model by name; else nil (the page shows its empty state).
    static func resolve(current: ModelEntry?, newestConversationModelIDs: [String],
                        installed: [ModelEntry]) -> ModelEntry? {
        if let current, let kept = installed.first(where: { $0.id == current.id }) { return kept }
        for id in newestConversationModelIDs {
            if let match = installed.first(where: { $0.id == id }) { return match }
        }
        return installed.sorted { $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending }.first
    }
}
