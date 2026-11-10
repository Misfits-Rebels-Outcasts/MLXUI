//
//  ChatAppleIntelligence.swift
//  MLXUI — Model Tools, MT3-3 (plan §3.3, ruling C).
//
//  Everything pure that puts Apple Intelligence next to the installed chat models: its identity as a chat "model"
//  (a reserved id, so a conversation records it in the same `modelID` field and the same badge), its model-menu item,
//  the default-router decision, and the banners. **No FoundationModels here** (readiness arrives as `Readiness`, the
//  AFM-1 type); the router itself is `AppleFoundationRouter`.
//

import Foundation

/// Apple Intelligence as the chat sees it.
nonisolated enum AppleIntelligenceChat {
    /// Stored in `Conversation.modelID` and on assistant bubbles, so the existing badge names it. Older builds show it as
    /// "apple-intelligence · not installed", which is true for them.
    static let modelID = "apple-intelligence"
    static let displayName = "Apple Intelligence"
    /// What the model menu and the window meter use until the system model reports its own `contextSize`.
    static let defaultContextTokens = 8_192

    static func isApple(_ id: String?) -> Bool { id == modelID }

    /// A `ModelEntry` so the chat's model plumbing (the composer, header, stamps) needs no second path. It is never in the
    /// catalog, never installed, never downloaded: `ModelRunner.send` branches on `isApple`.
    static let entry = ModelEntry(
        id: modelID, family: "Apple", displayName: displayName, paramSize: "", paramCountB: nil,
        modelType: .llm, source: .mlx, format: "system", platforms: ["macos"], minMacOSVersion: "26.0",
        hfRepo: "", hfModelId: "", ramGB: 0, downloadSizeGB: 0, contextWindow: defaultContextTokens,
        license: nil, licenseUrl: nil, description: nil, summary: nil, descriptionSource: nil, architecture: nil,
        languages: nil, lastUpdated: nil, taskTags: nil, benchmarks: nil, speedTokensPerSec: nil, speedHardware: nil,
        speedEstimated: nil, communityDownloads: nil, communityLikes: nil, variants: nil)
}

/// MT3-FIX-1 — the two plain facts the Chat view prints from a model id: the header subtitle and the per-answer badge.
nonisolated enum ChatModelLabels {
    /// "Running locally · MLX" is true only for an MLX model; the system model runs on this Mac but not on MLX.
    static func headerSubtitle(modelID: String) -> String {
        AppleIntelligenceChat.isApple(modelID) ? "Running on this Mac · Apple Intelligence" : "Running locally · MLX"
    }

    /// The small secondary line under an assistant answer: the model that wrote it (the stamp from MT2-3), named as the
    /// conversation list names it ("Qwen3.5 9B", "Apple Intelligence", "<name> · not installed"). `nil` for a user
    /// message or an answer with no stamp (older conversations).
    static func answerBadge(role: ChatMessage.Role, modelID: String?, catalog: [String: String], installedIDs: Set<String>) -> String? {
        guard role == .assistant, let modelID, !modelID.isEmpty else { return nil }
        var names = catalog
        names[AppleIntelligenceChat.modelID] = AppleIntelligenceChat.displayName
        return ChatConversationBadge.make(modelID: modelID, catalog: names, installedIDs: installedIDs).text
    }
}

/// The Apple Intelligence row of the composer's model menu.
nonisolated struct AppleIntelligenceMenuEntry: Equatable, Sendable {
    let isEnabled: Bool
    /// Why it is dimmed (the readiness sentence); nil when enabled.
    let reason: String?
    /// The fix-it button for a dimmed row: the existing `.enableAppleIntelligence` action (AFM-1).
    let action: SetupAction?

    /// `nil` readiness = the OS is below macOS 26 (or the real system is off): the row is **absent**, not greyed (AFM-1).
    static func make(readiness: Readiness?) -> AppleIntelligenceMenuEntry? {
        guard let readiness else { return nil }
        switch readiness {
        case .ready:
            return AppleIntelligenceMenuEntry(isEnabled: true, reason: nil, action: nil)
        case .needsSetup(let reason, let action):
            return AppleIntelligenceMenuEntry(isEnabled: false, reason: reason, action: action)
        case .unavailable(let reason):
            return AppleIntelligenceMenuEntry(isEnabled: false, reason: reason, action: nil)
        case .needsDownload:
            return AppleIntelligenceMenuEntry(isEnabled: false, reason: "macOS is still downloading the model", action: nil)
        }
    }

    /// The label of the fix-it button.
    static let turnOnTitle = "Turn on…"
}

/// What the page defaults to (ruling C).
nonisolated enum ChatRouterDecision: Equatable, Sendable {
    case appleIntelligence
    /// An installed chat model, by catalog id.
    case installedModel(String)
    /// Nothing is installed and Apple Intelligence is not ready: offer this catalog model (the smallest tool-capable one that fits).
    case recommend(String)
    /// Nothing to offer at all.
    case nothing
}

nonisolated enum ChatDefaultRouter {
    /// 1. Apple Intelligence when ready. 2. The most recent router that still works (`recentRouterIDs`, newest first; Apple only
    /// while ready, a model only while installed). 3. The first installed chat model, by name, whose tool-call format is known.
    /// 3b. Any installed chat model, by name (implementer's call: a model without tools still chats, so it is better than an empty
    /// page). 4. The smallest tool-capable catalog model that fits this Mac. 5. Nothing.
    static func decide(appleReadiness: Readiness?, recentRouterIDs: [String], installed: [ModelEntry],
                       installedToolCapableIDs: Set<String>, catalog: [ModelEntry], totalRAMGB: Double,
                       isRunnable: (ModelEntry) -> Bool = { _ in true }) -> ChatRouterDecision {
        let appleReady = appleReadiness == .ready
        if appleReady { return .appleIntelligence }
        for id in recentRouterIDs {
            if AppleIntelligenceChat.isApple(id) { continue }          // not ready (rule 1 would have taken it)
            if installed.contains(where: { $0.id == id }) { return .installedModel(id) }
        }
        let byName = installed.sorted { $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending }
        if let known = byName.first(where: { installedToolCapableIDs.contains($0.id) }) { return .installedModel(known.id) }
        if let any = byName.first { return .installedModel(any.id) }
        if let pick = recommendation(catalog: catalog, totalRAMGB: totalRAMGB, isRunnable: isRunnable) { return .recommend(pick.id) }
        return .nothing
    }

    /// The smallest (by memory) catalog chat model whose tool-call format is known and that fits `totalRAMGB`.
    static func recommendation(catalog: [ModelEntry], totalRAMGB: Double, isRunnable: (ModelEntry) -> Bool = { _ in true }) -> ModelEntry? {
        catalog
            .filter { $0.runnerKind == .llm && $0.ramGB <= totalRAMGB && CatalogToolSupport.supportsTools(architecture: $0.architecture) && isRunnable($0) }
            .min { ($0.ramGB, $0.id) < ($1.ramGB, $1.id) }
    }
}

/// Tool-call format knowledge for a catalog entry that is not installed (no `config.json` to read): its Hugging Face class name
/// is mapped to the `model_type` that `ToolCallFormatTable` is keyed on.
nonisolated enum CatalogToolSupport {
    /// "Qwen3_5ForConditionalGeneration" -> "qwen3_5", "GptOssForCausalLM" -> "gpt_oss", "LlamaForCausalLM" -> "llama".
    static func modelType(fromArchitecture architecture: String?) -> String? {
        guard var name = architecture, !name.isEmpty else { return nil }
        for suffix in ["ForConditionalGeneration", "ForCausalLM", "LMHeadModel"] where name.hasSuffix(suffix) {
            name = String(name.dropLast(suffix.count))
        }
        var out = ""
        for (i, ch) in name.enumerated() {
            if ch.isUppercase, i > 0, let last = out.last, last.isLowercase { out.append("_") }
            out.append(contentsOf: ch.lowercased())
        }
        return out.isEmpty ? nil : out
    }

    static func supportsTools(architecture: String?) -> Bool {
        guard let type = modelType(fromArchitecture: architecture) else { return false }
        return ToolCallFormatTable.toolSupport(forModelType: type).acceptsTools
    }
}

/// A banner above the composer: one sentence and up to two buttons.
nonisolated struct ChatBanner: Equatable, Sendable {
    enum Button: Equatable, Sendable {
        /// Switch to this installed model and send the unanswered message again.
        case continueWith(modelID: String, name: String)
        case startNewChat
        case openSettings(SettingsPane)
    }
    let text: String
    let buttons: [Button]

    var buttonTitles: [String] {
        buttons.map {
            switch $0 {
            case .continueWith(_, let name): return "Continue with \(name)"
            case .startNewChat: return "Start a new chat"
            case .openSettings: return "Open Settings…"
            }
        }
    }

    /// Plan §3.3. `installed` are the installed chat models; the offer names the first (by name) whose tool format is known, else
    /// the first by name, else no "continue" button.
    static func contextExceeded(limit: Int?, installed: [ModelEntry], installedToolCapableIDs: Set<String>) -> ChatBanner {
        let byName = installed.sorted { $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending }
        let pick = byName.first { installedToolCapableIDs.contains($0.id) } ?? byName.first
        var buttons: [Button] = []
        if let pick { buttons.append(.continueWith(modelID: pick.id, name: pick.displayName)) }
        buttons.append(.startNewChat)
        var text = "This chat is too long for Apple Intelligence"
        if let limit { text += " (its window is \(ChatModelMenu.tokenText(limit)) tokens)" }
        text += pick.map { " — continue with \($0.displayName) or start a new chat." } ?? " — start a new chat."
        return ChatBanner(text: text, buttons: buttons)
    }

    /// Guardrail, language, refusal, rate limit, timeout, assets: the router's own sentence; a new chat clears a refused one.
    static func failure(_ error: AppleFoundationToolError, installed: [ModelEntry]) -> ChatBanner {
        if case .contextWindowExceeded(let limit, _) = error {
            return contextExceeded(limit: limit, installed: installed, installedToolCapableIDs: [])
        }
        var buttons: [Button] = []
        switch error {
        case .assetsUnavailable: buttons = [.openSettings(.models)]
        default: break
        }
        return ChatBanner(text: error.userMessage, buttons: buttons)
    }
}

/// What the runner keeps after a turn ended in a context overflow, so the banner's button can send it again.
nonisolated struct ChatContextNotice: Equatable, Sendable {
    let limit: Int?
    let prompt: String
    let attachments: [URL]
    let conversationID: UUID?
}
