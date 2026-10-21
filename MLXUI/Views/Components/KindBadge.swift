import SwiftUI

/// KR-1 (`RSI/DelegateKindRailBacklog.md`) — how a `Kind` looks, in one table, so the kind rail
/// beside each flow row and the Add Step picker draw the same kind the same way. System tints
/// only (they adapt to light, dark and Increase Contrast), and the kind word is always shown,
/// so colour is never the only signal.
nonisolated struct KindStyle: Sendable {
    let symbol: String
    let tint: Color
    let word: String

    /// Exhaustive on purpose: a new `Kind` must be given a look here before this compiles.
    static func style(for kind: Kind) -> KindStyle {
        switch kind {
        case .text:       return KindStyle(symbol: "text.alignleft", tint: .blue, word: kind.rawValue)
        case .image:      return KindStyle(symbol: "photo", tint: .pink, word: kind.rawValue)
        case .audio:      return KindStyle(symbol: "waveform", tint: .orange, word: kind.rawValue)
        case .video:      return KindStyle(symbol: "film", tint: .purple, word: kind.rawValue)
        case .file:       return KindStyle(symbol: "doc", tint: .gray, word: kind.rawValue)
        case .folder:     return KindStyle(symbol: "folder", tint: .brown, word: kind.rawValue)
        case .table:      return KindStyle(symbol: "tablecells", tint: .green, word: kind.rawValue)
        case .index:      return KindStyle(symbol: "books.vertical", tint: .indigo, word: kind.rawValue)
        case .vector:     return KindStyle(symbol: "point.3.connected.trianglepath.dotted", tint: .teal, word: kind.rawValue)
        case .status:     return KindStyle(symbol: "checkmark.seal", tint: .mint, word: kind.rawValue)
        case .context:    return KindStyle(symbol: "book.closed", tint: .cyan, word: kind.rawValue)
        case .occurrence: return KindStyle(symbol: "bolt", tint: .yellow, word: kind.rawValue)
        case .model:      return KindStyle(symbol: "cpu", tint: .secondary, word: kind.rawValue)
        case .latent:     return KindStyle(symbol: "square.stack.3d.down.right", tint: .red, word: kind.rawValue)
        }
    }
}

/// The text model behind a `ShapeChip`: which kinds to draw and how to join them. Pure, so the
/// per-`Shape` rendering is testable without a view.
nonisolated struct ShapeChipModel: Equatable, Sendable {
    enum Join: Equatable, Sendable { case none, list, bundle, union }

    /// The kinds drawn, in order. Empty for `anything` and an unresolved `same`.
    let kinds: [Kind]
    let join: Join
    /// The word shown when `kinds` is empty (`any`, `same`).
    let fallbackWord: String?

    /// `incoming` is the shape arriving from above, used only for `.sameAsInput`: when it is a
    /// single kind or a list of one, `same` reads as that kind; otherwise it stays "same".
    init(_ shape: Shape, incoming: Shape? = nil) {
        switch shape {
        case .single(let k):    self.init(kinds: [k], join: .none, fallbackWord: nil)
        case .listOf(let k):    self.init(kinds: [k], join: .list, fallbackWord: nil)
        case .tupleOf(let ks):  self.init(kinds: ks, join: .bundle, fallbackWord: nil)
        case .unionOf(let ks):  self.init(kinds: ks, join: .union, fallbackWord: nil)
        case .anyKind:          self.init(kinds: [], join: .none, fallbackWord: "any")
        case .sameAsInput:
            switch incoming {
            case .some(.single(let k)): self.init(kinds: [k], join: .none, fallbackWord: nil)
            case .some(.listOf(let k)): self.init(kinds: [k], join: .list, fallbackWord: nil)
            default:                    self.init(kinds: [], join: .none, fallbackWord: "same")
            }
        }
    }

    private init(kinds: [Kind], join: Join, fallbackWord: String?) {
        self.kinds = kinds
        self.join = join
        self.fallbackWord = fallbackWord
    }

    /// Plain-text reading of the chip, e.g. `text`, `[image]`, `[text, vector]`, `audio|video`, `any`.
    var text: String {
        if kinds.isEmpty { return fallbackWord ?? "" }
        let words = kinds.map(\.rawValue)
        switch join {
        case .none:   return words.joined()
        case .list:   return "[" + words.joined(separator: ", ") + "]"
        case .bundle: return "[" + words.joined(separator: ", ") + "]"
        case .union:  return words.joined(separator: "|")
        }
    }
}

/// One kind: its symbol and word, tinted.
struct KindChip: View {
    let kind: Kind

    var body: some View {
        let style = KindStyle.style(for: kind)
        HStack(spacing: 3) {
            Image(systemName: style.symbol)
            Text(style.word)
        }
        .font(.caption2)
        .foregroundStyle(style.tint)
        .padding(.horizontal, 5)
        .padding(.vertical, 1)
        .background(style.tint.opacity(0.12), in: Capsule())
    }
}

/// A `Shape` as chips: one chip for a single kind, brackets around a list or bundle, `|`
/// between a union's kinds, a plain word for `anything` / `same`.
struct ShapeChip: View {
    let shape: Shape
    var incoming: Shape?

    var body: some View {
        let model = ShapeChipModel(shape, incoming: incoming)
        HStack(spacing: 2) {
            if model.join == .list || model.join == .bundle { bracket("[") }
            ForEach(Array(model.kinds.enumerated()), id: \.offset) { index, kind in
                if index > 0 { separator(model.join) }
                KindChip(kind: kind)
            }
            if model.kinds.isEmpty, let word = model.fallbackWord {
                Text(word)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(.quaternary.opacity(0.5), in: Capsule())
            }
            if model.join == .list || model.join == .bundle { bracket("]") }
        }
        .fixedSize()
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(model.text)
    }

    private func bracket(_ s: String) -> some View {
        Text(s).font(.caption2.monospaced()).foregroundStyle(.secondary)
    }

    @ViewBuilder
    private func separator(_ join: ShapeChipModel.Join) -> some View {
        Text(join == .union ? "|" : ",").font(.caption2.monospaced()).foregroundStyle(.secondary)
    }
}

/// The kind rail: `accepts → gives` for one row, drawn at the trailing edge of the row, outside
/// its horizontal scroll. Information plus two small affordances: hovering a "from N" chip
/// highlights row N's output chip, and clicking it selects row N. It never edits a ref
/// (`inputMenu(for:slot:)` stays the place for that), and the rest of the rail has no gesture, so
/// the row's own selection and context menu are untouched.
struct KindRail: View {
    let accepts: Shape
    let gives: Shape
    /// KR-2: where the input comes from. Empty (the Add Step picker) draws just the accepts chip.
    var inputs: [FlowEditorModel.RailInput] = []
    /// KR-2: the editor's sentence when it is about this row's input — red outline + tooltip.
    var mismatch: String?
    /// KR-2: another row's "from N" chip is hovered and names this row.
    var outputHighlighted = false
    var onHoverSource: (UUID?) -> Void = { _ in }
    var onSelectSource: (UUID) -> Void = { _ in }
    /// KR-3: the row's result, shown after the output chip once it has one; clicking it opens the
    /// inspector's Output tab (`onSelectOutput`) — the viewer that already exists.
    var preview: RailPreview?
    var thumbnail: CGImage?
    var onSelectOutput: () -> Void = {}

    var body: some View {
        HStack(spacing: 4) {
            inputSide
                .padding(.horizontal, mismatch == nil ? 0 : 3)
                .overlay {
                    if mismatch != nil {
                        RoundedRectangle(cornerRadius: 6).stroke(Color.red, lineWidth: 1)
                    }
                }
                .help(mismatch ?? "")
            Image(systemName: "arrow.right").font(.caption2).foregroundStyle(.secondary)
            ShapeChip(shape: gives, incoming: accepts)
                .padding(2)
                .overlay {
                    if outputHighlighted {
                        Capsule().stroke(Color.accentColor, lineWidth: 1.5)
                    }
                }
            if let preview {
                ResultPreviewChip(preview: preview, thumbnail: thumbnail, onSelect: onSelectOutput)
            }
        }
        .fixedSize()
        .padding(.top, 2)
        .help("\(accepts.signatureText) → \(gives.signatureText)")
    }

    @ViewBuilder
    private var inputSide: some View {
        let sources = inputs.map(\.source)
        let rowInputs = inputs.filter(\.source.isRowLike)
        HStack(spacing: 3) {
            ForEach(Array(rowInputs.enumerated()), id: \.offset) { _, input in
                SourceChip(input: input, onHover: onHoverSource, onSelect: onSelectSource)
            }
            if let special = sources.first(where: isSpecial) {
                switch special {
                case .own(let word):
                    Text("own \(word)")
                        .font(.caption2)
                        .padding(.horizontal, 5).padding(.vertical, 1)
                        .background(Color.accentColor.opacity(0.25), in: Capsule())
                        .help("This row uses the value typed in its own settings.")
                case .missing:
                    Text("no input")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 5).padding(.vertical, 1)
                        .overlay(Capsule().stroke(style: StrokeStyle(lineWidth: 1, dash: [3, 2])).foregroundStyle(.secondary))
                case .above, .row, .path, .blockInput:
                    EmptyView()
                }
            } else {
                ShapeChip(shape: accepts)
            }
        }
    }

    private func isSpecial(_ source: RailInputSource) -> Bool {
        switch source {
        case .own, .missing: return true
        case .above, .row, .path, .blockInput: return false
        }
    }
}

/// A "from N" chip: hover highlights row N's output chip, click selects row N.
private struct SourceChip: View {
    let input: FlowEditorModel.RailInput
    let onHover: (UUID?) -> Void
    let onSelect: (UUID) -> Void

    var body: some View {
        Text(input.source.label)
            .font(.caption2)
            .foregroundStyle(Color.accentColor)
            .padding(.horizontal, 5).padding(.vertical, 1)
            .background(Color.accentColor.opacity(0.12), in: Capsule())
            .contentShape(Capsule())
            .onHover { inside in if let target = input.target { onHover(inside ? target : nil) } }
            .onTapGesture { if let target = input.target { onSelect(target) } }
            .help(input.tooltip ?? "")
    }
}

extension RailInputSource {
    /// A "from …" chip (as opposed to the own / missing / above states).
    var isRowLike: Bool {
        switch self {
        case .row, .path, .blockInput: return true
        case .above, .own, .missing:   return false
        }
    }
}

/// KR-3: the result preview beside the output chip.
private struct ResultPreviewChip: View {
    let preview: RailPreview
    let thumbnail: CGImage?
    let onSelect: () -> Void

    var body: some View {
        HStack(spacing: 3) {
            switch preview {
            case .image:
                if let thumbnail {
                    Image(decorative: thumbnail, scale: 2)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 16, height: 16)
                        .clipShape(RoundedRectangle(cornerRadius: 3))
                } else {
                    Image(systemName: "photo").font(.caption2).foregroundStyle(.secondary)
                }
            default:
                Text(preview.label ?? "")
                    .font(.caption2)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .frame(maxWidth: 150, alignment: .leading)
            }
        }
        .padding(.horizontal, 5).padding(.vertical, 1)
        .background(.quaternary.opacity(0.5), in: Capsule())
        .contentShape(Capsule())
        .onTapGesture(perform: onSelect)
        .help(preview.tooltip ?? "Show this result in the Output tab")
    }
}
