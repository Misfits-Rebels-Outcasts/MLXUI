import Foundation

/// CD-1 — the pure model behind the Chat page's capability grid (`RSI/DelegateChatDiscoveryBacklog.md`).
/// User-facing card copy lives here, not in `ModelToolSpec.description` (that string is router-facing).
nonisolated struct ChatCapabilityCard: Identifiable, Sendable, Equatable {
    enum State: Sendable, Equatable {
        case ready(modelName: String)
        case needsDownload(modelName: String, gb: Double)
        case cantRun(reason: String)
    }

    /// The tool name (`ModelToolSpec.name`).
    let id: String
    let title: String
    let symbol: String
    let starterPrompt: String
    /// The kind of file the tool works on; `nil` when the tool takes only text.
    let takesFile: Kind?
    let state: State

    /// The status line under the title. Main-actor because it reuses `ModelInstallCard.sizeText`.
    @MainActor var statusLine: String {
        switch state {
        case .ready(let modelName): return "Ready · \(modelName)"
        case .needsDownload(_, let gb): return "\(ModelInstallCard.sizeText(gb)) download on first use"
        case .cantRun(let reason): return reason
        }
    }

    /// Ready or needs a download: the tool will work for the user (a download happens through the in-chat install card).
    var isAvailable: Bool {
        if case .cantRun = state { return false }
        return true
    }
}

nonisolated enum ChatCapabilityCards {
    /// One row of the copy table.
    struct Copy: Sendable, Equatable {
        let tool: String
        let title: String
        let symbol: String
        let starterPrompt: String
        let takesFile: Kind?
    }

    /// The copy table, in display order.
    static let table: [Copy] = [
        Copy(tool: "ocr_image", title: "Read text in an image", symbol: "doc.text.viewfinder",
             starterPrompt: "Read the text in this image.", takesFile: .image),
        Copy(tool: "transcribe_audio", title: "Transcribe a recording", symbol: "waveform",
             starterPrompt: "Transcribe this recording.", takesFile: .audio),
        Copy(tool: "describe_image", title: "Describe an image", symbol: "photo",
             starterPrompt: "What's in this image?", takesFile: .image),
        Copy(tool: "speak", title: "Read text aloud", symbol: "speaker.wave.2",
             starterPrompt: "Read this aloud: ", takesFile: nil),
        Copy(tool: "translate", title: "Translate", symbol: "character.bubble",
             starterPrompt: "Translate into French: ", takesFile: nil),
        Copy(tool: "generate_image", title: "Generate an image", symbol: "paintbrush",
             starterPrompt: "Draw ", takesFile: nil),
        Copy(tool: "upscale_image", title: "Upscale an image", symbol: "arrow.up.left.and.arrow.down.right",
             starterPrompt: "Upscale this image.", takesFile: .image),
        Copy(tool: "segment_image", title: "Cut out the subject", symbol: "scissors",
             starterPrompt: "Cut out the main subject of this image.", takesFile: .image),
        Copy(tool: "estimate_depth", title: "Make a depth map", symbol: "square.3.layers.3d",
             starterPrompt: "Make a depth map of this image.", takesFile: .image),
        Copy(tool: "generate_music", title: "Generate a sound", symbol: "music.note",
             starterPrompt: "Make a sound of ", takesFile: nil),
    ]

    /// How many cards show before the "+N more" disclosure.
    static let primaryCount = 6

    /// Cards in the table's order. A tool with no status or `.none`, or absent from the copy table, is dropped.
    /// `specs` are the tools to consider (callers pass `ModelToolSpec.modelTools`); `statuses` is keyed by tool name.
    static func cards(specs: [ModelToolSpec], statuses: [String: ModelToolStatus]) -> (primary: [ChatCapabilityCard], more: [ChatCapabilityCard]) {
        let names = Set(specs.map(\.name))
        var out: [ChatCapabilityCard] = []
        for row in table where names.contains(row.tool) {
            guard let status = statuses[row.tool], let state = state(for: status) else { continue }
            out.append(ChatCapabilityCard(id: row.tool, title: row.title, symbol: row.symbol,
                                          starterPrompt: row.starterPrompt, takesFile: row.takesFile, state: state))
        }
        return (Array(out.prefix(primaryCount)), Array(out.dropFirst(primaryCount)))
    }

    private static func state(for status: ModelToolStatus) -> ChatCapabilityCard.State? {
        switch status {
        case .ready(let model): return .ready(modelName: model.displayName)
        case .needsDownload(let model, let gb): return .needsDownload(modelName: model.displayName, gb: gb)
        case .cantRun(let reason): return .cantRun(reason: reason)
        case .none: return nil
        }
    }

    /// The line above the grid when the selected chat model can't call tools (ruling E); `nil` when it can.
    static func routerNote(modelName: String, canUseTools: Bool) -> String? {
        canUseTools ? nil : "\(modelName) can't use tools, so it can't run these. Switch to a model that can."
    }

    /// "N tools" over the cards that work (ready + needs download); `nil` at zero.
    static func menuSummary(cards: [ChatCapabilityCard]) -> String? {
        let count = cards.filter(\.isAvailable).count
        switch count {
        case 0: return nil
        case 1: return "1 tool"
        default: return "\(count) tools"
        }
    }
}
