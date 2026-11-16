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

/// CD-FIX-2 — the grid's column rule, pure: how many columns of at least `minimum` fit in `width`
/// (the same count `LazyVGrid`'s adaptive item gives).
nonisolated enum ChatGridColumns {
    static func count(width: Double, minimum: Double = 200, spacing: Double = 10) -> Int {
        guard width.isFinite, width > 0 else { return 1 }
        return max(1, Int(((width + spacing) / (minimum + spacing)).rounded(.down)))
    }
}

/// CD-FIX-2 — adaptive columns (at least 200 pt, wrapping at narrow widths) where every card in a row is as tall as the
/// tallest in that row. `LazyVGrid` cannot equalise row heights, so this is a small `Layout`; the grid has at most ten cards.
struct EqualRowGridLayout: Layout {
    var minimum: Double = 200
    var spacing: Double = 10

    private func metrics(_ subviews: Subviews, width: CGFloat?) -> (columns: Int, cellWidth: CGFloat, rows: [CGFloat]) {
        let available = Double(width ?? CGFloat(minimum))
        let columns = ChatGridColumns.count(width: available, minimum: minimum, spacing: spacing)
        let cellWidth = (CGFloat(available) - CGFloat(spacing) * CGFloat(columns - 1)) / CGFloat(columns)
        var rows: [CGFloat] = []
        var index = 0
        while index < subviews.count {
            let end = min(index + columns, subviews.count)
            let tallest = subviews[index..<end].map { $0.sizeThatFits(ProposedViewSize(width: cellWidth, height: nil)).height }.max() ?? 0
            rows.append(tallest)
            index = end
        }
        return (columns, cellWidth, rows)
    }

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let m = metrics(subviews, width: proposal.width)
        let height = m.rows.reduce(0, +) + CGFloat(spacing) * CGFloat(max(m.rows.count - 1, 0))
        return CGSize(width: proposal.width ?? CGFloat(minimum), height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let m = metrics(subviews, width: bounds.width)
        var y = bounds.minY
        for (row, rowHeight) in m.rows.enumerated() {
            for column in 0..<m.columns {
                let i = row * m.columns + column
                guard i < subviews.count else { break }
                let x = bounds.minX + CGFloat(column) * (m.cellWidth + CGFloat(spacing))
                subviews[i].place(at: CGPoint(x: x, y: y), anchor: .topLeading,
                                  proposal: ProposedViewSize(width: m.cellWidth, height: rowHeight))
            }
            y += rowHeight + CGFloat(spacing)
        }
    }
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

    /// CD-FIX-1: every grid text follows the message size step (`ChatGridText`).
    private func gridFont(_ role: ChatGridText.Role) -> Font {
        .system(size: ChatGridText.points(role, size: size))
    }

    var body: some View {
        let result = ChatCapabilityCards.cards(specs: ModelToolSpec.modelTools, statuses: statuses)
        VStack(spacing: 14) {
            if !readOnly {
            VStack(spacing: 6) {
                Image(systemName: "sparkles")
                    .font(.system(size: 28))
                    .foregroundStyle(.secondary)
                Text("Chat can run other AI models for you")
                    .font(gridFont(.heading).weight(.semibold))
                    .multilineTextAlignment(.center)
                Text("Drop a file or pick one below. Chat downloads the model it needs once, then runs it on this Mac.")
                    .font(gridFont(.subheading))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            }
            if !readOnly, let note = ChatCapabilityCards.routerNote(modelName: modelName, canUseTools: canUseTools) {
                Text(note)
                    .font(gridFont(.note))
                    .foregroundStyle(.orange)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            grid(ChatGridLayout.visibleCards(primary: result.primary, more: result.more, expanded: showsMore))
            if !result.more.isEmpty {
                Button(showsMore ? "Show fewer" : "+\(result.more.count) more") {
                    withAnimation { showsMore.toggle() }
                }
                .buttonStyle(.link)
                .font(gridFont(.link))
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.top, readOnly ? 0 : 32)
    }

    private func grid(_ cards: [ChatCapabilityCard]) -> some View {
        EqualRowGridLayout {
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
                        .font(gridFont(.cardTitle).weight(.medium))
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                }
                statusRow(card)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
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
                .font(gridFont(.status))
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
