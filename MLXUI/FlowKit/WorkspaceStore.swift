import Foundation

/// CFM-R17-2 — a **workspace**: a directory under `workspaces/` holding **one or more** flow
/// files side by side (so `uses: X = ./X.cat` resolves — E114) plus whatever they share: a
/// `docs/` folder, a built `library.index/`, a `session.ctx`. The additive half of the R17
/// model — `flows/` and `UserFlowStore`'s **exactly-one** rule are not touched; a workspace
/// is a second root beside them with the opposite cardinality.
///
/// Like `UserFlowStore`, a broken flow file **lists with an error** (`FlowFile.parseIssue`)
/// rather than vanishing — a workspace with one unparseable `.cat` still shows its others.
nonisolated enum WorkspaceStore {

    /// One flow file inside a workspace.
    struct FlowFile: Hashable, Identifiable {
        /// The `.cat`/`.catpipeline` file.
        let url: URL
        /// Display title — the filename stem.
        let title: String
        /// The parse-failure sentence, or nil when the file parses. The badge marks it and
        /// the detail shows it — a broken file is never silently dropped.
        let parseIssue: String?

        var id: String { url.lastPathComponent }
    }

    /// One workspace on the shelf.
    struct Workspace: Hashable, Identifiable {
        /// The folder name under `workspaces/`.
        let workspaceID: String
        /// The workspace directory itself — every flow in it resolves its `.cat` paths
        /// against this one directory.
        let url: URL
        /// Display name. There is no `workspace.json` (that is deliberately out of scope —
        /// CFM-R17-4), so the name is the folder name; workspace creation (CFM-R17-3) and
        /// `importWorkspace` are what put a readable name on the folder.
        let title: String
        /// Last modification of the workspace folder (the shelf sorts newest first).
        let modifiedAt: Date
        /// The flow files in it, sorted by name — may be **empty** (KW-1-2, Q2, owner ruling
        /// 2026-09-16): a directory with no `.cat` but some other content (an index, `docs/`,
        /// anything left behind) still lists, with no flows, so Remove can reach it. A
        /// directory with nothing in it at all is not a workspace and `scan` skips it.
        let flows: [FlowFile]

        var id: String { workspaceID }
    }

    /// Scan the workspaces root for workspaces, newest first. **Unlike `UserFlowStore.scan`,
    /// this does not exclude any ids.** A bundled workspace (CFM-R17-5/-6) materialises into a
    /// real, editable directory the user runs and can Remove/Restore (CFM-R17-FIX-1), so it
    /// lists like every other workspace — there is nothing to hide, and `AppState` calls this
    /// with no exclusion set on purpose.
    static func scan(workspace: FlowWorkspace) -> [Workspace] {
        let fm = FileManager.default
        guard let dirs = try? fm.contentsOfDirectory(
            at: workspace.root, includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]) else { return [] }
        var out: [Workspace] = []
        for dir in dirs where dir.hasDirectoryPath {
            guard let contents = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else { continue }
            let flowFiles = contents.filter { isFlowFile($0) }
                .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
            // KW-1-2 (Q2, owner ruling 2026-09-16): a directory with no `.cat` still lists as
            // a workspace with no flows when it holds anything else (an index, `docs/`, any
            // leftover) — otherwise that data is stranded with no UI that can reach it. A
            // directory with nothing in it at all is still not a workspace.
            guard !flowFiles.isEmpty || !contents.isEmpty else { continue }
            let modified = (try? dir.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                ?? .distantPast
            out.append(Workspace(
                workspaceID: dir.lastPathComponent,
                url: dir,
                title: dir.lastPathComponent,
                modifiedAt: modified,
                flows: flowFiles.map { FlowFile(url: $0,
                                                title: $0.deletingPathExtension().lastPathComponent,
                                                parseIssue: parseIssue(of: $0)) }))
        }
        return out.sorted { $0.modifiedAt > $1.modifiedAt }
    }

    /// Parse a workspace flow file's document from disk.
    static func loadDocument(flow: FlowFile) throws -> FlowDocument {
        let text = try String(contentsOf: flow.url, encoding: .utf8)
        return try CatParser.parse(text)
    }

    /// A plain sentence for the delete confirmation. "'X' and its files" understates a
    /// workspace, so this names what is actually going — N flows, and any index directories
    /// (CFM-R17-3 wires it into the dialog).
    ///
    /// CFM-R17-FIX-1: `bundled` picks the wording for a workspace that ships in the app.
    /// Removing one is a tombstone, not a permanent delete — so the sentence must **not**
    /// claim irreversibility, but it must still say that the index the user built and any
    /// documents they added here are lost (only the shipped original comes back on Restore).
    static func deletionSummary(_ workspace: Workspace, bundled: Bool = false) -> String {
        // KW-1-2 (Q2): a flows-less workspace (no `.cat`, listed only because it holds an
        // index or other files) has nothing to call "0 flows" about — name what's actually
        // there instead.
        var parts: [String] = workspace.flows.isEmpty ? [] :
            [workspace.flows.count == 1 ? "1 flow" : "\(workspace.flows.count) flows"]
        let indexes = indexDirectoryCount(in: workspace.url)
        if indexes == 1 { parts.append("1 index") }
        else if indexes > 1 { parts.append("\(indexes) indexes") }
        // CFM-R17-FIX-8: `.trash` grows one full copy of the previous index per Rebuild and is
        // hidden from Shared Files by `.skipsHiddenFiles` — name it here so a Build/Rebuild
        // loop's disk cost is visible at the one point the folder is about to go.
        if let trash = directorySizePhrase(workspace.url.appendingPathComponent(".trash", isDirectory: true)) {
            parts.append("\(trash) of earlier versions in .trash")
        }
        if parts.isEmpty { parts.append("its files") }
        let inventory = parts.joined(separator: " and ")
        if bundled {
            return "'\(workspace.title)' — \(inventory) — will be removed. Any documents you added and any index you built here will be lost; you can bring the original back with “Restore bundled workspaces” in AI Workflows."
        }
        return "'\(workspace.title)' — \(inventory) — will be deleted from your workspaces folder. This can't be undone."
    }

    /// Delete a workspace's folder (and everything in it) from disk.
    static func remove(workspaceID: String, workspace: FlowWorkspace) throws {
        let dir = workspace.directory(for: workspaceID)
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir.path) else {
            throw WorkspaceStoreError.notFound(workspaceID)
        }
        try fm.removeItem(at: dir)
    }

    /// Import a workspace folder the user picked — every `.cat`/`.catpipeline` in it (one or
    /// more) plus the files they share (`docs/`, a built index, everything) — into the
    /// workspaces root. The whole folder is copied verbatim.
    ///
    /// **Divergence from `UserFlowStore.importFlow` (deliberate):** a user flow gets a fresh
    /// UUID id and recovers its display name from the single `.cat` stem. A workspace has no
    /// single stem, so `newID` defaults to the **source folder's sanitized name** — the name
    /// is preserved. A collision is an error; the caller retries with an explicit id, exactly
    /// as `importFlow` does.
    static func importWorkspace(from sourceDir: URL, workspace: FlowWorkspace,
                                newID: String? = nil) throws -> Workspace {
        let fm = FileManager.default
        let items = try fm.contentsOfDirectory(at: sourceDir, includingPropertiesForKeys: nil)
        let flowFiles = items.filter { isFlowFile($0) }
        guard !flowFiles.isEmpty else {
            throw WorkspaceStoreError.importNeedsAFlowFile
        }
        let id = newID ?? FlowEditorModel.sanitizedFileName(sourceDir.lastPathComponent)
        let destDir = workspace.directory(for: id)
        guard !fm.fileExists(atPath: destDir.path) else {
            throw WorkspaceStoreError.importDestinationExists(id)
        }
        try fm.createDirectory(at: destDir, withIntermediateDirectories: true)
        // `copyItem` recurses, so nested `docs/` / index directories come along intact.
        for item in items {
            try fm.copyItem(at: item, to: destDir.appendingPathComponent(item.lastPathComponent))
        }
        let sortedFlows = flowFiles
            .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
        return Workspace(
            workspaceID: id, url: destDir, title: id, modifiedAt: .now,
            flows: sortedFlows.map { file in
                let copied = destDir.appendingPathComponent(file.lastPathComponent)
                return FlowFile(url: copied,
                                title: copied.deletingPathExtension().lastPathComponent,
                                parseIssue: parseIssue(of: copied))
            })
    }

    /// Export a workspace's whole folder into `destinationParent` under `name` (sanitized to
    /// a valid folder name), returning the created folder. The exported folder round-trips
    /// through `importWorkspace` verbatim.
    static func export(workspaceID: String, name: String, to destinationParent: URL,
                       workspace: FlowWorkspace) throws -> URL {
        let source = workspace.directory(for: workspaceID)
        let fm = FileManager.default
        guard fm.fileExists(atPath: source.path) else {
            throw WorkspaceStoreError.notFound(workspaceID)
        }
        let dest = destinationParent
            .appendingPathComponent(FlowEditorModel.sanitizedFileName(name), isDirectory: true)
        guard !fm.fileExists(atPath: dest.path) else {
            throw WorkspaceStoreError.exportDestinationExists(dest.lastPathComponent)
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

    /// The parse-failure sentence, or nil when the file parses cleanly. Same voice as
    /// `UserFlowStore`.
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

    /// A short "12.4 MB" phrase for `dir`'s total size on disk, or nil when it doesn't exist
    /// or is empty. Used only for the delete-confirmation sentence (CFM-R17-FIX-8).
    private static func directorySizePhrase(_ dir: URL) -> String? {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue else { return nil }
        guard let en = fm.enumerator(at: dir, includingPropertiesForKeys: [.totalFileAllocatedSizeKey, .fileAllocatedSizeKey]) else { return nil }
        var bytes = 0
        for case let url as URL in en {
            let v = try? url.resourceValues(forKeys: [.totalFileAllocatedSizeKey, .fileAllocatedSizeKey])
            bytes += v?.totalFileAllocatedSize ?? v?.fileAllocatedSize ?? 0
        }
        guard bytes > 0 else { return nil }
        let f = ByteCountFormatter()
        f.allowedUnits = [.useKB, .useMB, .useGB]
        f.countStyle = .file
        return f.string(fromByteCount: Int64(bytes))
    }

    /// Immediate subdirectories of `dir` that are index directories (they carry a
    /// `manifest.json`, per `IndexFormat`). Used only for the delete-confirmation sentence.
    private static func indexDirectoryCount(in dir: URL) -> Int {
        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else { return 0 }
        return contents.filter {
            $0.hasDirectoryPath
                && fm.fileExists(atPath: $0.appendingPathComponent("manifest.json").path)
        }.count
    }
}

/// WorkspaceStore failures. Error voice: one plain sentence implying the fix.
nonisolated enum WorkspaceStoreError: Error, CustomStringConvertible, Equatable {
    case notFound(String)
    case importNeedsAFlowFile
    case importDestinationExists(String)
    case exportDestinationExists(String)

    var description: String {
        switch self {
        case .notFound:
            return "The workspace's folder isn't in your workspaces directory anymore — nothing to remove."
        case .importNeedsAFlowFile:
            return "A workspace needs at least one .cat or .catpipeline file — this folder has none."
        case .importDestinationExists(let id):
            return "A workspace named '\(id)' already exists in your workspaces folder — rename it or try again."
        case .exportDestinationExists(let name):
            return "A folder named '\(name)' already exists where you're exporting — pick another folder or rename it."
        }
    }
}
