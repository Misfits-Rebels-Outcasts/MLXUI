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

}

/// HC-1 (ruling A): which screen the Home item shows. Pure.
nonisolated enum HomeSurface: Equatable, Sendable {
    case chat
    case dashboard

    /// The Chat page while the gate is open; today's `HomeView` while it is closed (ruling C).
    static func resolve(chatAvailable: Bool) -> HomeSurface {
        chatAvailable ? .chat : .dashboard
    }
}

extension SidebarItem {
    /// The sidebar's first section: Home only. HC-1 ruling A supersedes MT ruling J: Chat is no longer a second item,
    /// Home shows the Chat page while the gate is open (`HomeSurface`).
    static func primaryItems(chatAvailable: Bool) -> [SidebarItem] {
        [.home]
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

/// Pure helpers for the unified conversation list (MT2-3).
nonisolated enum ChatConversationList {
    /// `live` (the page's working set) over `disk` (a fresh read): a live conversation replaces the disk copy with
    /// the same id; chats only on disk are added; chats only live (blank, memory-only) are kept. Newest first.
    static func merge(disk: [Conversation], live: [Conversation]) -> [Conversation] {
        let liveIDs = Set(live.map(\.id))
        let merged = live + disk.filter { !liveIDs.contains($0.id) }
        return merged.sorted { $0.updatedAt > $1.updatedAt }
    }
}

/// What a list row says about the model behind a conversation (plain facts only).
nonisolated struct ChatConversationBadge: Equatable, Sendable {
    let name: String
    let installed: Bool

    /// "Qwen3.5 9B", or "Qwen3.5 9B · not installed".
    var text: String { installed ? name : "\(name) · not installed" }

    /// `catalog` maps catalog id to display name; an id the catalog no longer knows shows its id without the
    /// `mlx-community--` prefix.
    static func make(modelID: String, catalog: [String: String], installedIDs: Set<String>) -> ChatConversationBadge {
        let name = catalog[modelID] ?? modelID.replacingOccurrences(of: "mlx-community--", with: "")
        return ChatConversationBadge(name: name, installed: installedIDs.contains(modelID))
    }
}
