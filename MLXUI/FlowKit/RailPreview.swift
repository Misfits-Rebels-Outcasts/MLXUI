import Foundation
import AVFoundation

/// KR-3 (`RSI/DelegateKindRailBacklog.md`) — what the output chip shows once a row has a
/// result in the current run: a pure reading of a finished `Asset`. Only two kinds need to touch
/// a file, and only through `RailPreviewProbe` (cheap header/metadata reads, injectable so the
/// model stays testable): never a run, a model load or a download, and never a decode just to
/// draw something.
nonisolated enum RailPreview: Equatable, Sendable {
    /// A picture: the view decodes a thumbnail from the file (off the main actor, cached).
    case image(URL)
    /// An audio clip's length in seconds. (Samples are never held in memory by an `Asset`, so
    /// there is no waveform; the duration is read from the file header.)
    case audio(duration: Double)
    /// The first ~24 characters of a text result, on one line.
    case text(String)
    /// A table's size.
    case table(rows: Int, columns: Int)
    /// A status result: the row's ✓ (the sentence stays in the tooltip).
    case status(String)
    /// A list or tuple result: the count, never several previews.
    case count(Int, Kind)

    /// The chip's wording, or nil for an image (drawn as a thumbnail).
    var label: String? {
        switch self {
        case .image:                       return nil
        case .audio(let seconds):          return Self.clock(seconds)
        case .text(let s):                 return s
        case .table(let r, let c):         return "\(r) × \(c)"
        case .status:                      return "✓"
        case .count(let n, let kind):      return "\(n) \(Self.plural(kind))"
        }
    }

    /// The hover text.
    var tooltip: String? {
        switch self {
        case .status(let sentence): return sentence.isEmpty ? nil : sentence
        default:                    return nil
        }
    }

    static let textLimit = 24

    private static func clock(_ seconds: Double) -> String {
        let total = Int(seconds.rounded())
        return String(format: "%d:%02d", total / 60, total % 60)
    }

    private static func plural(_ kind: Kind) -> String {
        switch kind {
        case .audio:  return "audio clips"
        case .status: return "statuses"
        case .index:  return "indexes"
        default:      return kind.rawValue + "s"
        }
    }
}

/// The file reads a preview may do: header-only, no decode, no model.
nonisolated struct RailPreviewProbe: Sendable {
    var audioDuration: @Sendable (URL) -> Double?
    var tableShape: @Sendable (URL) -> (rows: Int, columns: Int)?

    /// Audio length from the file header; table size from the table file the row already wrote.
    static let live = RailPreviewProbe(
        audioDuration: { url in
            guard let file = try? AVAudioFile(forReading: url), file.processingFormat.sampleRate > 0 else { return nil }
            return Double(file.length) / file.processingFormat.sampleRate
        },
        tableShape: { url in
            guard let table = try? TableTool.readTable(from: url) else { return nil }
            return (table.rows.count, table.columns.count)
        })
}

nonisolated enum RailPreviews {
    /// The preview for a finished result, or nil when there is nothing honest to show.
    static func preview(for asset: Asset, probe: RailPreviewProbe = .live) -> RailPreview? {
        guard let first = asset.items.first else { return nil }
        if asset.items.count > 1 {
            // A list or tuple: the count, when the items share a kind.
            return asset.items.allSatisfy({ $0.kind == first.kind }) ? .count(asset.items.count, first.kind) : nil
        }
        switch first.kind {
        case .text:
            guard let value = first.value else { return nil }
            let line = value.split(whereSeparator: \.isWhitespace).joined(separator: " ")
            guard !line.isEmpty else { return nil }
            return .text(line.count > RailPreview.textLimit ? String(line.prefix(RailPreview.textLimit)) + "…" : line)
        case .image:
            return first.path.map(RailPreview.image)
        case .audio:
            guard let url = first.path, let seconds = probe.audioDuration(url) else { return nil }
            return .audio(duration: seconds)
        case .table:
            guard let url = first.path, let shape = probe.tableShape(url) else { return nil }
            return .table(rows: shape.rows, columns: shape.columns)
        case .status:
            return .status(first.value ?? "")
        default:
            return nil
        }
    }
}
