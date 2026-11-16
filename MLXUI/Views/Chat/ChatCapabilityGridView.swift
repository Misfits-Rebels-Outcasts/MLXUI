import SwiftUI

/// CD-2 — how a clicked card changes the composer's draft (ruling D: it fills the composer, never sends).
nonisolated enum ChatCardDraft {
    /// An empty (or whitespace-only) draft becomes the starter prompt; otherwise the starter is appended after what the
    /// user already typed, joined by one space unless the draft already ends in whitespace.
    static func draft(afterPicking starter: String, current: String) -> String {
        guard !current.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return starter }
        if let last = current.last, last.isWhitespace { return current + starter }
        return current + " " + starter
    }
}

/// CD-4 — copy for the "no chat model" screen, which explains the router before one is installed.
nonisolated enum ChatNoModelText {
    static let title = "Chat needs one language model"
    static let body = "It understands your request, then downloads and runs OCR, speech and image models for you as needed."
}

/// CD-2 — the capability grid shown in the empty conversation on the Chat page (`RSI/DelegateChatDiscoveryBacklog.md`).
/// Statuses come from the same `ModelToolRuntime.live` + `CapabilityResolver.status` pair the router uses (`ChatToolSet`).
struct ChatCapabilityGridView: View {
    /// The selected chat model's display name (for the router note).
    let modelName: String
    /// Whether the selected chat model can call tools (ruling E). `false` shows the note and disables the cards.
    let canUseTools: Bool
    /// Per-tool statuses, from the host (`ChatCapabilityStatuses`), shared with the model menu.
    let statuses: [String: ModelToolStatus]
    /// A card was clicked; the host fills the draft and opens the file panel.
    let onPick: (ChatCapabilityCard) -> Void
    /// CD-4: no chat model yet. Just the cards and the "+N more" disclosure: no heading, no router note, nothing clickable.
    var readOnly = false

    @AppStorage(ChatTextSize.messageKey) private var messageSizeRaw = ChatTextSize.standard.rawValue
    @State private var showsMore = false

    private var size: ChatTextSize { ChatTextSize(stored: messageSizeRaw) }

    var body: some View {
        let result = ChatCapabilityCards.cards(specs: ModelToolSpec.modelTools, statuses: statuses)
        VStack(spacing: 14) {
            if !readOnly {
            VStack(spacing: 6) {
                Image(systemName: "sparkles")
                    .font(.system(size: 28))
                    .foregroundStyle(.secondary)
                Text("Chat can run other AI models for you")
                    .font(size.font(.message, standard: .title3).weight(.semibold))
                    .multilineTextAlignment(.center)
                Text("Drop a file or pick one below. Chat downloads the model it needs once, then runs it on this Mac.")
                    .font(size.font(.message, standard: .callout))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            }
            if !readOnly, let note = ChatCapabilityCards.routerNote(modelName: modelName, canUseTools: canUseTools) {
                Text(note)
                    .font(size.font(.message, standard: .callout))
                    .foregroundStyle(.orange)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            grid(result.primary)
            if !result.more.isEmpty {
                if showsMore { grid(result.more) }
                Button(showsMore ? "Show fewer" : "+\(result.more.count) more") {
                    withAnimation { showsMore.toggle() }
                }
                .buttonStyle(.link)
                .font(size.font(.message, standard: .callout))
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.top, readOnly ? 0 : 32)
    }

    private func grid(_ cards: [ChatCapabilityCard]) -> some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 200), spacing: 10, alignment: .top)], spacing: 10) {
            ForEach(cards) { card in cardView(card) }
        }
    }

    private func cardView(_ card: ChatCapabilityCard) -> some View {
        let available = card.isAvailable && canUseTools && !readOnly
        return Button { onPick(card) } label: {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    Image(systemName: card.symbol)
                        .font(.system(size: 16))
                        .foregroundStyle(Color.accentColor)
                        .frame(width: 22)
                    Text(card.title)
                        .font(size.font(.message, standard: .body).weight(.medium))
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                }
                statusRow(card)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(10)
            .background(RoundedRectangle(cornerRadius: 10).fill(.quaternary.opacity(0.5)))
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.separator))
            .contentShape(RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
        .disabled(!available)
        .opacity(card.isAvailable ? (canUseTools && !readOnly ? 1 : 0.55) : 0.5)
        .help(card.isAvailable ? card.starterPrompt.trimmingCharacters(in: .whitespaces) : card.statusLine)
    }

    @ViewBuilder
    private func statusRow(_ card: ChatCapabilityCard) -> some View {
        HStack(spacing: 5) {
            switch card.state {
            case .ready:
                Circle().fill(.green).frame(width: 7, height: 7)
            case .needsDownload:
                Image(systemName: "arrow.down.circle").font(.caption).foregroundStyle(.secondary)
            case .cantRun:
                Image(systemName: "exclamationmark.circle").font(.caption).foregroundStyle(.secondary)
            }
            Text(card.statusLine)
                .font(size.font(.toolResult, standard: .caption))
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
