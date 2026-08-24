import Foundation

/// CFM-R12-1 — the "My Workflows" shelf: every flow the user has saved (or Duplicate & Edit
/// produced) lives in `flows/<uuid>/` and nothing in the app ever enumerated that directory
/// — a saved flow was unreachable after navigating away. This store scans the flow folder.
///
/// A bundled flow's *working* directory (`flows/<gallery-id>/`, created by
/// `FlowWorkspace.prepare`) must **never** appear under My Workflows — it is a scratch copy
/// of the bundle's assets, not the user's work — so the scan skips any folder whose id
/// matches a bundled flow id. A user flow is a `<uuid>/` folder containing exactly one
/// `.cat` or `.catpipeline`.
///
/// A broken file **lists with an error** rather than vanishing: `Entry.parseIssue` is set
/// when the file doesn't parse, the badge marks it, and opening it shows the sentence.
nonisolated enum UserFlowStore {

    /// One user flow on the shelf.
    struct Entry: Hashable, Identifiable {
        /// The folder name (a uuid, or whatever `save()`'s flowID was).
        let flowID: String
        /// The `.cat`/`.catpipeline` file.
        let url: URL
        /// The display title — the filename stem, which is all `save()` persists today.
        let title: String
        /// Last modification of the flow folder (the shelf sorts newest first).
        let modifiedAt: Date
        /// The parse-failure sentence, or nil when the file parses. The badge marks it and
        /// the detail shows it — a broken file is never silently dropped.
        let parseIssue: String?

        var id: String { flowID }
    }

    /// Scan the flow folder for user flows, newest first. `bundledFlowIDs` are the gallery
    /// flow ids whose working directories must be excluded.
    static func scan(workspace: FlowWorkspace, bundledFlowIDs: Set<String>) -> [Entry] {
        let fm = FileManager.default
        guard let dirs = try? fm.contentsOfDirectory(
            at: workspace.root, includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]) else { return [] }
        var out: [Entry] = []
        for dir in dirs where dir.hasDirectoryPath && !bundledFlowIDs.contains(dir.lastPathComponent) {
            guard let files = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else { continue }
            guard let file = files.first(where: { isFlowFile($0) }) else { continue }
            let modified = (try? dir.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                ?? .distantPast
            let issue = parseIssue(of: file)
            out.append(Entry(flowID: dir.lastPathComponent, url: file,
                             title: file.deletingPathExtension().lastPathComponent,
                             modifiedAt: modified, parseIssue: issue))
        }
        return out.sorted { $0.modifiedAt > $1.modifiedAt }
    }

    /// Parse + resolve a user flow's document from disk.
    static func loadDocument(entry: Entry) throws -> FlowDocument {
        let text = try String(contentsOf: entry.url, encoding: .utf8)
        return try CatParser.parse(text)
    }

    // MARK: - Helpers

    private static func isFlowFile(_ url: URL) -> Bool {
        let ext = url.pathExtension.lowercased()
        return ext == "cat" || ext == "catpipeline"
    }

    /// The parse-failure sentence, or nil when the file parses cleanly.
    private static func parseIssue(of url: URL) -> String? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            return "This file isn't readable — it may be empty or not UTF-8 text."
        }
        do {
            _ = try CatParser.parse(text)
            return nil
        } catch {
            return "'\(url.lastPathComponent)' isn't a valid CAT Flow: \(error)"
        }
    }
}
