//
//  ChatModelMenu.swift
//  MLXUI — Model Tools, MT2-4.
//
//  The composer's model menu on the Chat page: installed chat models with plain facts only (R51/R56: no
//  ratings, no badges) and a "Get more chat models…" entry. Pure, so the wording and the target are testable.
//

import Foundation

/// What the menu says about one installed model.
nonisolated struct ChatModelFacts: Equatable, Sendable {
    /// From `ToolCallFormatTable` (the source the Connect sheet uses).
    let canUseTools: Bool
    /// The usable context on this Mac in tokens (`ServeModelFacts`, R53); nil when neither the model nor the Mac says.
    let usableTokens: Int?

    /// Reads the installed `config.json` and weights, exactly as the Connect sheet does. File reads: call it once
    /// per model, not per render.
    static func load(for model: ModelEntry) -> ChatModelFacts {
        // MT3-3: the system model has no files to read; its window comes from the OS through the seam when it can.
        if AppleIntelligenceChat.isApple(model.id) {
            return ChatModelFacts(canUseTools: true,
                                  usableTokens: AppleFoundationAvailability.makeToolChatter()?.contextSize ?? AppleIntelligenceChat.defaultContextTokens)
        }
        let directory = ModelStore.shared.directory(forModelID: model.id)
        let facts = ServeModelFacts.facts(modelID: model.id,
                                          configData: ModelContainerPool.configData(directory: directory),
                                          weightsBytes: MLXChatBackend.weightsBytes(in: directory))
        return ChatModelFacts(canUseTools: facts.supportsTools, usableTokens: facts.usableTokens)
    }
}

nonisolated struct ChatModelMenuItem: Equatable, Identifiable, Sendable {
    let id: String
    let name: String
    /// "Can use tools · 32K tokens of context on this Mac"; empty until the facts are loaded.
    let facts: String
    let isSelected: Bool
    /// MT3-3: a dimmed row (Apple Intelligence not ready) is shown but cannot be chosen.
    var isEnabled = true
    /// Why a dimmed row is dimmed.
    var note: String?
    /// The fix-it action for a dimmed row (`.enableAppleIntelligence`).
    var action: SetupAction?
}

nonisolated enum ChatModelMenu {
    static let getMoreTitle = "Get more chat models…"
    /// The Browse section that holds chat models.
    static let chatSectionID = "chat-text"

    /// Installed chat models only (`runnerKind == .llm`), by name; the selected one is marked.
    /// MT3-3: Apple Intelligence is listed **first** when `apple` is non-nil (ready: chooseable; otherwise dimmed with its reason and
    /// fix-it action); `nil` = the OS is below macOS 26 and the row is absent.
    static func items(installed: [ModelEntry], selectedID: String?, facts: [String: ChatModelFacts],
                      apple: AppleIntelligenceMenuEntry? = nil, appleContextTokens: Int? = nil) -> [ChatModelMenuItem] {
        let appleItems: [ChatModelMenuItem] = apple.map { entry in
            [ChatModelMenuItem(
                id: AppleIntelligenceChat.modelID, name: AppleIntelligenceChat.displayName,
                facts: entry.isEnabled ? factsText(ChatModelFacts(canUseTools: true, usableTokens: appleContextTokens ?? AppleIntelligenceChat.defaultContextTokens)) : "",
                isSelected: AppleIntelligenceChat.modelID == selectedID,
                isEnabled: entry.isEnabled, note: entry.reason, action: entry.action)]
        } ?? []
        return appleItems + installed
            .filter { $0.runnerKind == .llm }
            .sorted { $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending }
            .map { model in
                ChatModelMenuItem(id: model.id, name: model.displayName,
                                  facts: facts[model.id].map(factsText) ?? "", isSelected: model.id == selectedID)
            }
    }

    /// Plain facts: tool use, and the usable context when known.
    static func factsText(_ facts: ChatModelFacts) -> String {
        var parts = [facts.canUseTools ? "Can use tools" : "Can't use tools"]
        if let tokens = facts.usableTokens { parts.append("\(tokenText(tokens)) tokens of context on this Mac") }
        return parts.joined(separator: " · ")
    }

    /// 32768 -> "32K", 131072 -> "128K", 1_000_000 -> "1M", 900 -> "900".
    static func tokenText(_ tokens: Int) -> String {
        if tokens >= 1_000_000 { return "\(Int((Double(tokens) / 1_048_576).rounded()))M" }
        if tokens >= 1024 { return "\(Int((Double(tokens) / 1024).rounded()))K" }
        return "\(tokens)"
    }

    /// Where "Get more chat models…" goes: Browse, on the chat section, when the catalog shows it.
    static func getMoreTarget(visibleSectionIDs: [String]) -> SidebarItem? {
        visibleSectionIDs.contains(chatSectionID) ? .browse(chatSectionID) : nil
    }
}
