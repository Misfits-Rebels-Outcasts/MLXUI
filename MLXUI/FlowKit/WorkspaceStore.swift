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

    /// KW-2-FIX-1 (Q6, owner ruling 2026-09-16 — "just warn them"): the sibling flows whose
    /// `uses:` block names `target`, resolved through `FlowWorkspace.resolve` — the same
    /// authority `UsesResolver` uses — so a caller is caught regardless of which relative-path
    /// spelling it wrote. Read-only: this never edits a `.cat` the user didn't open, only
    /// answers "who calls this" before a rename or delete proceeds.
    static func callers(of target: URL, in workspace: Workspace, ws: FlowWorkspace) -> [String] {
        let targetResolved = target.resolvingSymlinksInPath()
        var names: [String] = []
        for flow in workspace.flows where flow.url != target {
            guard let doc = try? loadDocument(flow: flow) else { continue }
            for rawPath in doc.uses.values {
                guard let resolved = try? ws.resolve(rawPath, flowID: workspace.workspaceID) else { continue }
                if resolved.resolvingSymlinksInPath() == targetResolved {
                    names.append(flow.url.lastPathComponent)
                    break
                }
            }
        }
        return names.sorted { $0.localizedStandardCompare($1) == .orderedAscending }
    }

    /// KW-2-FIX-1: the sentence appended to a rename/delete confirmation when `callers` (from
    /// `callers(of:in:ws:)`) is non-empty — named, so the user decides with the same
    /// information a run-time failure would otherwise surface later, in a file they never
    /// touched. `nil` when nothing calls the flow, so an unaffected flow's confirmation is
    /// unchanged.
    static func usesWarning(callers: [String]) -> String? {
        guard !callers.isEmpty else { return nil }
        let names = callers.map { "'\($0)'" }.joined(separator: " and ")
        let verb = callers.count == 1 ? "calls" : "call"
        return "\(names) \(verb) this flow — this will break that call."
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

    /// KW-2-1: the first unused `<stem>.cat` name in a workspace directory — `Flow.cat`,
    /// `Flow-2.cat`, `Flow-3.cat`, … Purely a name choice; never routes through `save()`'s
    /// sibling-clearing path, so adding a flow can never overwrite an existing one.
    static func firstFreeFlowName(stem: String, in dir: URL) -> String {
        let fm = FileManager.default
        var candidate = "\(stem).cat"
        var n = 2
        while fm.fileExists(atPath: dir.appendingPathComponent(candidate).path) {
            candidate = "\(stem)-\(n).cat"
            n += 1
        }
        return candidate
    }

    /// KW-2-2 (Q3, owner ruling 2026-09-16 — "allow with a warning"): the confirmation
    /// sentence for deleting one flow — names the file, not the workspace. Deleting the last
    /// flow left is allowed, but the sentence says plainly what stays behind (an index, other
    /// files), since `KW-1-2` is what makes that state reachable and cleanable afterward.
    static func flowDeletionSummary(fileName: String, isLastFlow: Bool, workspace: Workspace) -> String {
        guard isLastFlow else {
            return "'\(fileName)' will be deleted from this workspace."
        }
        var parts: [String] = []
        let indexes = indexDirectoryCount(in: workspace.url)
        if indexes == 1 { parts.append("its index") }
        else if indexes > 1 { parts.append("its \(indexes) indexes") }
        if let trash = directorySizePhrase(workspace.url.appendingPathComponent(".trash", isDirectory: true)) {
            parts.append("\(trash) of earlier versions in .trash")
        }
        let stays = parts.isEmpty ? "its other files" : parts.joined(separator: " and ")
        return "'\(fileName)' will be deleted, leaving this workspace with no flows — \(stays) stays until you Remove the whole workspace."
    }

    /// KW-2-2: delete one flow file from a workspace — the file-level counterpart to `remove`
    /// (whole directory). `UserFlowStore.remove` is keyed on `flows/<flowID>` and deletes a
    /// **whole folder** (its own exactly-one rule), the wrong granularity here, so this is its
    /// own function rather than bending that one. Refuses if `file` isn't actually inside
    /// `directory` — never follows a path outside the workspace it was asked to touch.
    static func removeFlow(file url: URL, from directory: URL) throws {
        guard url.deletingLastPathComponent().standardizedFileURL == directory.standardizedFileURL else {
            throw WorkspaceStoreError.flowNotFound(url.lastPathComponent)
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw WorkspaceStoreError.flowNotFound(url.lastPathComponent)
        }
        try FileManager.default.removeItem(at: url)
    }

    /// KW-2-2: rename one flow file in place — `Rename…` as its own action rather than
    /// inferred from the editor's name field. Refuses onto an existing *different* sibling's
    /// name, the same rule `KW-1-FIX-2` gives `FlowEditorModel.save()`, so this can never
    /// destroy another flow; a case-only rename (same file, re-cased) is allowed. KW-2-FIX-2:
    /// the common (non-case-only) path is already proven collision-free by the guard above it
    /// — exactly what an atomic `moveItem` is for, and unlike read-remove-write it can never
    /// lose the flow if the write half fails. Read-then-remove-then-write is kept for the
    /// case-only branch only, where a case-insensitive volume's own move semantics are
    /// unreliable about actually updating the casing. KW-2-FIX-3: refuses a source outside
    /// `directory` (the same containment rule `removeFlow` already has) and a stem that's
    /// empty, escapes the directory, or would write a leading-dot (hidden) file — a hidden
    /// flow is `scan`'s `.skipsHiddenFiles` dropping it from the page while it sits on disk,
    /// `KW-1-2`'s orphan state reached by a typo in the rename box.
    static func renameFlow(file url: URL, toStem newStem: String, in directory: URL) throws -> URL {
        guard url.deletingLastPathComponent().standardizedFileURL == directory.standardizedFileURL else {
            throw WorkspaceStoreError.flowNotFound(url.lastPathComponent)
        }
        guard !newStem.isEmpty, !newStem.hasPrefix("."), !newStem.contains("/") else {
            throw WorkspaceStoreError.invalidFlowName(newStem)
        }
        let newURL = directory.appendingPathComponent("\(newStem).\(url.pathExtension)")
        guard newURL.lastPathComponent != url.lastPathComponent else { return url }   // no-op
        let fm = FileManager.default
        let caseOnly = newURL.lastPathComponent.compare(url.lastPathComponent, options: .caseInsensitive) == .orderedSame
        if !caseOnly, fm.fileExists(atPath: newURL.path) {
            throw WorkspaceStoreError.flowNameTaken(newURL.lastPathComponent)
        }
        if caseOnly {
            let data = try Data(contentsOf: url)
            try fm.removeItem(at: url)
            try data.write(to: newURL)
        } else {
            try fm.moveItem(at: url, to: newURL)
        }
        return newURL
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
    /// KW-2-2: the flow named no longer exists at the expected path (deleted elsewhere, or a
    /// path outside the workspace directory was passed in).
    case flowNotFound(String)
    /// KW-2-2: a rename's target name already belongs to a different flow in the workspace.
    case flowNameTaken(String)
    /// KW-2-FIX-3: a rename's stem is empty, escapes the directory, or would write a
    /// leading-dot (hidden) file — `scan`'s `.skipsHiddenFiles` would drop it from the shelf.
    case invalidFlowName(String)
    /// KW-3-1: a `.catpipeline` flow was asked to copy into a workspace — still an open R17
    /// decision (`.catpipeline` inside a workspace), deliberately out of scope for this phase.
    case catpipelineNotSupportedInWorkspace(String)

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
        case .flowNotFound(let name):
            return "'\(name)' isn't in this workspace anymore — nothing to remove."
        case .flowNameTaken(let name):
            return "'\(name)' already exists in this workspace — pick a different name."
        case .invalidFlowName(let name):
            return name.isEmpty
                ? "A flow needs a name."
                : "'\(name)' isn't a valid flow name — it can't start with a dot or contain '/'."
        case .catpipelineNotSupportedInWorkspace(let title):
            return "'\(title)' is a .catpipeline flow — copying one into a workspace isn't supported yet."
        }
    }
}
