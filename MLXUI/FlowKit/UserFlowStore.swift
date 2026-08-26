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

    /// Delete a user flow's folder (and everything in it) from disk. Only the My Workflows
    /// shelf / a user flow's list may call this — a bundled gallery flow's working directory
    /// is scratch, and this store never lists it, so it can't reach here by accident.
    static func remove(flowID: String, workspace: FlowWorkspace) throws {
        let dir = workspace.directory(for: flowID)
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir.path) else {
            throw UserFlowStoreError.notFound(flowID)
        }
        try fm.removeItem(at: dir)
    }

    /// CFM — import a flow folder the user picked (a `.cat`/`.catpipeline` plus its fixtures
    /// — audio, `.txt`, subfolders, everything) into the user's flows folder. The whole folder
    /// is copied verbatim under a **fresh id** so the shelf never collides with an existing
    /// folder; the display title still comes from the `.cat` file's stem, so the name the user
    /// sees is the imported flow's own. Returns the new shelf entry.
    static func importFlow(from sourceDir: URL, workspace: FlowWorkspace,
                           newID: String = UUID().uuidString) throws -> Entry {
        let fm = FileManager.default
        let items = try fm.contentsOfDirectory(at: sourceDir, includingPropertiesForKeys: nil)
        let flowFiles = items.filter { isFlowFile($0) }
        guard flowFiles.count == 1 else {
            throw UserFlowStoreError.importNeedsOneFlowFile(count: flowFiles.count)
        }
        let destDir = workspace.directory(for: newID)
        guard !fm.fileExists(atPath: destDir.path) else {
            throw UserFlowStoreError.importDestinationExists
        }
        try fm.createDirectory(at: destDir, withIntermediateDirectories: true)
        // Copy every immediate child — `copyItem` recurses into subfolders, so fixtures
        // arranged in nested directories come along intact.
        for item in items {
            try fm.copyItem(at: item, to: destDir.appendingPathComponent(item.lastPathComponent))
        }
        let flowFile = flowFiles[0]
        return Entry(flowID: newID,
                     url: destDir.appendingPathComponent(flowFile.lastPathComponent),
                     title: flowFile.deletingPathExtension().lastPathComponent,
                     modifiedAt: .now,
                     parseIssue: parseIssue(of: flowFile))
    }

    /// CFM — export a user flow's whole folder (its `.cat` plus fixtures like audio and
    /// `.txt`) into `destinationParent` under the flow's display title (sanitized to a valid
    /// folder name), returning the created folder. The exported folder round-trips through
    /// `importFlow` verbatim.
    static func export(flowID: String, name: String, to destinationParent: URL,
                       workspace: FlowWorkspace) throws -> URL {
        let source = workspace.directory(for: flowID)
        let fm = FileManager.default
        guard fm.fileExists(atPath: source.path) else {
            throw UserFlowStoreError.notFound(flowID)
        }
        let dest = destinationParent
            .appendingPathComponent(FlowEditorModel.sanitizedFileName(name), isDirectory: true)
        guard !fm.fileExists(atPath: dest.path) else {
            throw UserFlowStoreError.exportDestinationExists(dest.lastPathComponent)
        }
        try fm.createDirectory(at: dest, withIntermediateDirectories: true)
        for item in try fm.contentsOfDirectory(at: source, includingPropertiesForKeys: nil) {
            try fm.copyItem(at: item, to: dest.appendingPathComponent(item.lastPathComponent))
        }
        return dest
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

/// UserFlowStore failures. Error voice: one plain sentence implying the fix.
nonisolated enum UserFlowStoreError: Error, CustomStringConvertible, Equatable {
    case notFound(String)
    case importNeedsOneFlowFile(count: Int)
    case importDestinationExists
    case exportDestinationExists(String)

    var description: String {
        switch self {
        case .notFound(let flowID):
            return "The flow's folder isn't in your flows directory anymore — nothing to remove."
        case .importNeedsOneFlowFile(let count):
            return "A flow folder needs exactly one .cat or .catpipeline file — this one has \(count)."
        case .importDestinationExists:
            return "A flow with that id already exists in your flows folder — try again."
        case .exportDestinationExists(let name):
            return "A folder named '\(name)' already exists where you're exporting — pick another folder or rename it."
        }
    }
}
