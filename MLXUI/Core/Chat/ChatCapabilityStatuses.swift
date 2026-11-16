import Foundation

/// CD-3 — the one place the Chat page turns the live runtime into per-tool statuses, so the grid and the model menu
/// cannot disagree. It is the router's own call (`ChatToolSet` runs the same `CapabilityResolver.status` per spec).
nonisolated enum ChatCapabilityStatuses {
    static func resolve(runtime: ModelToolRuntime, specs: [ModelToolSpec] = ModelToolSpec.modelTools) -> [String: ModelToolStatus] {
        var next: [String: ModelToolStatus] = [:]
        for spec in specs {
            next[spec.name] = CapabilityResolver.status(
                for: spec, catalog: runtime.catalog, installedModelIDs: runtime.installedModelIDs,
                claimableModelIDs: runtime.claimableModelIDs, totalRAMGB: runtime.totalRAMGB)
        }
        return next
    }
}

/// CD-3 — the model menu's text (ruling F), pure.
nonisolated enum ChatCapabilityMenuText {
    /// The label's name and its secondary part ("6 tools"). No secondary part when the router can't use tools or no tool works.
    static func label(modelName: String, cards: [ChatCapabilityCard], canUseTools: Bool) -> (name: String, summary: String?) {
        (modelName, canUseTools ? ChatCapabilityCards.menuSummary(cards: cards) : nil)
    }

    /// The disabled rows under "Uses these models as tools": "‹title› — ‹status line›". Empty (so the section is omitted)
    /// when the router can't use tools (ruling E) or there are no cards.
    @MainActor static func toolRows(cards: [ChatCapabilityCard], canUseTools: Bool) -> [String] {
        canUseTools ? cards.map { "\($0.title) — \($0.statusLine)" } : []
    }

    static let sectionTitle = "Uses these models as tools"
    static let help = "The model that reads your message and decides which other models to run"
    static let dropCaption = "Chat will pick a model for this"
    static let pagePlaceholder = "Ask, or drop a photo, PDF or recording…"
}
