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

    /// CD-FIX-1: the composer placeholder. The invitation to drop files is true only on the page with a tool-capable model.
    static func placeholder(isPage: Bool, modelName: String, canUseTools: Bool) -> String {
        isPage && canUseTools ? pagePlaceholder : "Message \(modelName)…"
    }

    // MARK: Composer width (CD-FIX-1)

    /// Approximate glyph width at the composer's 13 pt, and the icon + menu chrome around the label text.
    static let charWidth = 7.5
    static let chromeWidth = 34.0
    /// Everything in the composer row that is not the model menu: the + menu, the send button, spacings, padding and the
    /// text field's own minimum (about 20 characters).
    static let composerReserved = 122.0 + 170.0
    static let menuWidthRange = 80.0...360.0

    /// The widest the model menu may be given the composer's width; the text field keeps the rest. `nil` until measured.
    static func menuMaxWidth(containerWidth: Double?) -> Double? {
        guard let containerWidth, containerWidth > 0 else { return nil }
        return min(max(containerWidth - composerReserved, menuWidthRange.lowerBound), menuWidthRange.upperBound)
    }

    /// The text field's own minimum in the composer row: about 20 characters plus its padding.
    static let fieldMinWidth = 170.0
    /// The narrowest the Chat page's composer gets (CD-FIX-3 assumption): the window's 900 pt minimum
    /// (`ContentView`), less the app sidebar (about 220 pt) and the conversations column (210 pt) and its divider,
    /// with the inspector hidden.
    static let minimumComposerWidth = 470.0

    /// What the menu label shows in `maxWidth`: "name · N tools", else the name alone, else the name **cut with "…"** to the
    /// characters that fit. The cut is made here, in the string, because a macOS `Menu` label does not honour a `.frame`
    /// or `lineLimit` inside it (CD-FIX-3). No limit (`nil`) shows everything.
    static func fitLabel(name: String, summary: String?, maxWidth: Double?) -> (name: String, summary: String?, fitsWhole: Bool) {
        guard let maxWidth else { return (name, summary, true) }
        if let summary, chromeWidth + Double((name + " · " + summary).count) * charWidth <= maxWidth { return (name, summary, true) }
        if chromeWidth + Double(name.count) * charWidth <= maxWidth { return (name, nil, true) }
        let room = Int(((maxWidth - chromeWidth) / charWidth).rounded(.down))
        let keep = max(room - 1, 3)
        return (String(name.prefix(keep)) + "…", nil, false)
    }
}

/// CD-FIX-1 — the grid's layout and text, pure.
nonisolated enum ChatGridLayout {
    /// One list for one grid, so a row never ends with a gap before the next block starts.
    static func visibleCards(primary: [ChatCapabilityCard], more: [ChatCapabilityCard], expanded: Bool) -> [ChatCapabilityCard] {
        expanded ? primary + more : primary
    }
}

/// CD-FIX-1 — the grid's point sizes. Every size is a base plus the CP-5 message-text step (`ChatTextSize.delta`), so
/// ⌘= and ⌘- move the grid exactly as far as they move message text.
nonisolated enum ChatGridText {
    enum Role: Sendable, Equatable, CaseIterable { case heading, subheading, cardTitle, status, note, link }

    static func basePoints(_ role: Role) -> Double {
        switch role {
        case .heading: return 15
        case .subheading, .note, .link: return 12
        case .cardTitle: return 13
        case .status: return 11
        }
    }

    static func points(_ role: Role, size: ChatTextSize) -> Double {
        basePoints(role) + size.points(for: .message) - ChatTextSize.Role.message.standardPoints
    }
}
