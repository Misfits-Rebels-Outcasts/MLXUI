import Foundation

/// ER-FIX-4: which viewer the inspector's Output tab uses for a live output item. Pure, so the choice is testable without a view.
/// Only `.file` (a PDF) and `.table` are new; every other kind keeps the viewer it had.
nonisolated enum InspectorViewer: Equatable, Sendable {
    case audio, text, status, image
    /// A PDF file output: an inline Quick Look preview.
    case quickLook(URL)
    /// A table output, rendered as Table to Text's markdown.
    case tableText
    case none

    static func choose(for item: Item) -> InspectorViewer {
        switch item.kind {
        case .audio: return .audio
        case .text: return .text
        case .status: return .status
        case .image: return .image
        case .file:
            if let path = item.path, path.pathExtension.lowercased() == "pdf" { return .quickLook(path) }
            return .none
        case .table: return item.path == nil ? .none : .tableText
        default: return .none
        }
    }

    /// The table as text: Table to Text's own renderer (`TableTool.tableToText`, markdown), not a second one.
    static func tableText(for item: Item) -> String? {
        guard let text = try? TableTool.tableToText(settings: nil, from: item).items.first?.value else { return nil }
        return text
    }
}
