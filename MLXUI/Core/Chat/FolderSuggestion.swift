//
//  FolderSuggestion.swift
//  MLXUI — chat workspace, CW-UX-1 (`RSI/DelegateChatWorkspaceBacklog.md`).
//
//  One click from "this folder is available" to "this folder is in this chat".
//
//  The owner's report, and why this exists: he granted `~/Desktop/workfolder` in Settings → Tools,
//  saw it in the "+" menu, asked "What files are in workfolder?" and was told *"No folder is
//  attached to this chat"*. That answer is correct — G1 says a chat's tools reach the intersection
//  of **attached** and **granted**, G2 says a new chat starts empty, and the "+" row was unticked —
//  but a folder listed in "+" reads as one already in use, and the refusal gave him no way to find
//  the missing click.
//
//  So the app now says it in the UI: a hint above the message box, and the same buttons on the
//  refused tool card. G1, G2, Settings' grant-only behaviour and every model-facing string are
//  unchanged. Nothing here is model-facing.
//
//  Pure value type, `nonisolated`, no SwiftUI: the decision is tested, the view only renders it.
//

import Foundation

nonisolated enum FolderSuggestion {

    /// Folders worth offering **in this chat**, in the order the caller already has them.
    ///
    /// Empty when the offer would be noise or wrong:
    /// - a folder is already attached — the chat has what it needs, and the chips say so;
    /// - no file tool is on — attaching would change nothing about what the model can do;
    /// - nothing available — there is no one-click fix, just the Add folder… route.
    ///
    /// `available` is the caller's list (`RunChatView.grantCandidates` in the App Store edition,
    /// `recentFolderCandidates` in Direct); passing it in keeps this function free of the edition
    /// split and of any store.
    static func candidates(
        attached: [String],
        available: [String],
        fileToolsOn: Bool
    ) -> [String] {
        guard attached.isEmpty else { return [] }
        guard fileToolsOn else { return [] }
        return available
    }

    /// Whether any file tool is on. Defined from `ChatEnvironment`'s own sets, so "a file tool" has
    /// one definition in the codebase rather than a second list here.
    static func fileToolsOn(enabledToolNames: Set<String>) -> Bool {
        !enabledToolNames.isDisjoint(with: ChatEnvironment.fileToolNames)
            || !enabledToolNames.isDisjoint(with: ChatEnvironment.writeToolNames)
    }

    /// The most buttons one row shows before it collapses behind "More…".
    static let maxButtons = 3

    /// The row's folders, and whether a "More…" affordance is needed.
    static func row(candidates: [String]) -> (shown: [String], hasMore: Bool) {
        (Array(candidates.prefix(maxButtons)), candidates.count > maxButtons)
    }

    /// Whether a finished tool card should offer the same buttons.
    ///
    /// Exactly `FileScope.noFolderMessage` — compared against the constant, never a copied literal,
    /// so the refusal text and this predicate cannot drift — and only while the chat still has
    /// nothing attached. A card from earlier in the conversation loses its buttons as soon as a
    /// folder is attached, so an old card never offers a fix that has already been made.
    static func showsOnCard(result: String?, attached: [String]) -> Bool {
        guard attached.isEmpty else { return false }
        guard let result else { return false }
        return result == FileScope.noFolderMessage
    }
}
