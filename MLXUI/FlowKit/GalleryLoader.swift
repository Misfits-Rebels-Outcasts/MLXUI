import Foundation

/// Metadata for one bundled gallery flow — a trimmed version of
/// `catflow-mlx/gallery/_metadata.json`'s entries (the fields CFM-R1-4 names).
nonisolated struct GalleryFlowMetadata: Codable, Identifiable, Hashable, Sendable {
    /// The gallery number (1, 8, 21, …).
    var number: Int
    /// Friendly title, e.g. "Spoken Summary".
    var title: String
    /// The `.cat` filename, e.g. "01-SpokenSummary.cat".
    var filename: String
    /// Category, e.g. "Voice & Meetings".
    var category: String
    /// One-line description.
    var description: String

    var id: Int { number }

    /// The flow id — the filename minus the `.cat` extension (e.g. `01-SpokenSummary`).
    var flowID: String {
        (filename as NSString).deletingPathExtension
    }
}

/// Loads the bundled gallery: `_metadata.json` for the list, and a `FlowDocument` + the raw
/// `.cat` text on demand.
///
/// **Resource layout note.** The Xcode `PBXFileSystemSynchronizedRootGroup` flattens
/// `MLXUI/Resources/Gallery/**` into the app bundle's `Contents/Resources/` root (no
/// `Gallery/` subdirectory survives — verified on the built `.app`). The gallery's
/// filenames are globally unique (`01-SpokenSummary…`, `08-PolicyDiff…`, `21-PhotoWebPrep…`),
/// so lookups by name work fine without a subdirectory.
nonisolated enum GalleryLoader {
    /// The three (or however many ship) gallery flows, in gallery order.
    static func loadMetadata() -> [GalleryFlowMetadata] {
        guard let url = Bundle.main.url(forResource: "_metadata", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode(GalleryMetadataRoot.self, from: data) else {
            return []
        }
        return decoded.entries
    }

    /// Load the parsed `FlowDocument` for a flow id.
    static func loadDocument(flowID: String) throws -> FlowDocument {
        guard let url = Bundle.main.url(forResource: flowID, withExtension: "parse.json") else {
            throw GalleryError.missingResource(flowID, kind: "parse.json")
        }
        return try JSONDecoder().decode(FlowDocument.self, from: Data(contentsOf: url))
    }

    /// The raw `.cat` file text, verbatim — what the user reads in the disclosure.
    static func rawCatText(flowID: String) throws -> String {
        guard let url = Bundle.main.url(forResource: flowID, withExtension: "cat") else {
            throw GalleryError.missingResource(flowID, kind: ".cat")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// The bundled input assets for a flow: `(flat source name, relative destination)` pairs.
    /// Sources are the flat filenames in `Contents/Resources/` (the synchronized root group
    /// flattens the `Gallery/` tree); destinations are the paths the flow's `.cat` expects
    /// (e.g. `vacation/photo-01.png` for `21-PhotoWebPrep`'s `Read Images vacation/`).
    /// Copied from `catflow-mlx/gallery_fixtures/`.
    static func bundledAssets(flowID: String) -> [(source: String, destination: String)] {
        switch flowID {
        case "01-SpokenSummary":
            return [("memo.m4a", "memo.m4a")]
        case "08-PolicyDiff":
            return [("policy-2025.md", "policy-2025.md"), ("policy-2026.md", "policy-2026.md")]
        case "21-PhotoWebPrep":
            return [("logo.png", "logo.png"),
                    ("photo-01.png", "vacation/photo-01.png"),
                    ("photo-02.png", "vacation/photo-02.png"),
                    ("photo-03.png", "vacation/photo-03.png")]
        default:
            return []
        }
    }

    /// The app bundle's `Contents/Resources/` directory, where the flattened gallery
    /// resources live (or `nil` if the bundle is malformed).
    static var resourcesDirectory: URL? {
        Bundle.main.resourceURL
    }

    private struct GalleryMetadataRoot: Decodable {
        var entries: [GalleryFlowMetadata]
    }
}

/// Gallery-loading failures. Error voice: one plain sentence implying the fix.
nonisolated enum GalleryError: Error, CustomStringConvertible, Equatable {
    case missingResource(String, kind: String)

    var description: String {
        switch self {
        case .missingResource(let flowID, let kind):
            return "The gallery flow '\(flowID)' is missing its \(kind) — reinstall the app to restore it."
        }
    }
}
