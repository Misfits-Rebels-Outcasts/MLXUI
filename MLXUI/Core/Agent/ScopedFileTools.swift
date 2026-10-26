//
//  ScopedFileTools.swift
//  MLXUI — agentic chat tools, slice AG5 (scoped files).
//
//  File tools for the **sandboxed** edition, scoped to folders the user has granted via the
//  open panel (`FolderGrants`) plus the app container. Registered from
//  `ModelRunner.defaultTools()` only when not `DIRECT_BUILD` — the direct edition ships its own
//  arbitrary-filesystem counterparts (`Apps/Direct/FileTools.swift`) under the same tool names,
//  so the model-facing API is identical across editions.
//
//  Every path goes through `PathPolicy.resolve` with `roots:` = the granted folders + container
//  (escape + denylist guards, symlink-resolved), then I/O runs inside the covering grant's
//  security scope (`FolderGrants.withAccess`). `write_file` and `edit_file` are side-effecting
//  ⇒ approval-gated and disabled by default, mirroring the direct edition.
//

import Foundation
import MLXLMCommon

nonisolated enum ScopedFileTools {
    static func all() -> [any AgentTool] {
        [
            ScopedReadFileTool(),
            ScopedWriteFileTool(),
            ScopedEditFileTool(),
            ScopedListDirectoryTool(),
            ScopedSearchFilesTool(),
        ]
    }

    /// `write_file` and `edit_file` start unticked (visible, but the user opts in),
    /// matching `DirectTools`.
    static let disabledByDefault: Set<String> = ["write_file", "edit_file"]

    /// CW-1-FIX-1 W3 — the one `path` description both editions use. It lives here (shared code)
    /// so `Apps/Direct/FileTools.swift` cannot drift from its copies of the literal.
    static let relativePathDescription =
        "File path. Relative paths start in the working folder (e.g. notes/todo.md);"
        + " full paths and ~/ also work."

    /// The same sentence when the parameter may be left out.
    static let optionalPathDescription =
        "Optional; omit for the working folder. " + ScopedFileTools.relativePathDescription

    /// The base a relative path resolves against: the conversation's working folder, or `nil`.
    static func workingFolderBase() -> URL? {
        guard let folder = FileScope.current?.workingFolder else { return nil }
        return URL(fileURLWithPath: folder, isDirectory: true)
    }

    /// CW-FIX-2: the model-facing text for a relative path that does not exist, or `nil` when this
    /// is not that case (absolute path, no working folder, or the path exists). Shared by both
    /// editions; call it inside the security scope so "exists" is accurate.
    static func notFoundMessage(_ url: URL, askedPath: String?) -> String? {
        guard let asked = askedPath, !asked.isEmpty,
              !(asked as NSString).expandingTildeInPath.hasPrefix("/"),
              let scope = FileScope.current, let working = scope.workingFolder,
              !FileManager.default.fileExists(atPath: url.path)
        else { return nil }
        return "Error: there is no \"\(asked)\" in the working folder \(FileScope.displayName(for: working))"
            + " (\(working)). To list the working folder itself, call list_directory with no path."
            + " This chat can use: \(FileScope.folderList(scope.folders))."
    }

    /// An explicit scope for one tool call. Production passes nothing (`nil`) and the tool reads
    /// the live grant store; a unit test passes temp roots so it never touches `~/Library`.
    ///
    /// `roots` is the policy roots **and** an extra gate in `withAccess` — never a substitute for
    /// the covering grant, which is what re-opens a granted folder's security scope after a
    /// relaunch (CW-1-FIX-1 F1).
    struct ScopeSelection {
        let container: URL
        let roots: [URL]
    }

    /// One resolved call: the path vetoed by the policy, the grants for the security scope, and
    /// the roots both used.
    struct Access {
        let url: URL
        let grants: [FolderGrant]
        let roots: [URL]
    }

    /// The container and roots a tool call should use: the injected test scope, or the live
    /// grant store through `FileScope.current`. Production always takes the grant-store branch
    /// (`injected` is nil, and nothing in the app can set it).
    static func selection(_ injected: ScopeSelection?) -> ScopeSelection {
        injected ?? ScopeSelection(
            container: ModelStore.shared.baseDirectory,
            roots: FileScope.roots(container: ModelStore.shared.baseDirectory))
    }

    /// Shared front half of every tool: find the conversation's scope, apply the path policy
    /// against it — then hand the same roots to `withAccess` so the security-scope half cannot
    /// disagree with the policy half. `container` and `roots` are injectable for tests.
    ///
    /// A missing `path` means the working folder, which is what `list_directory` /
    /// `search_files` with no argument mean (D4).
    static func access(
        _ path: String?,
        container: URL = ModelStore.shared.baseDirectory,
        roots overrideRoots: [URL]? = nil
    ) -> Access? {
        let scope = FileScope.current
        let grants = FolderGrants.load()
        let roots = FileScope.roots(
            grants: grants, scope: scope, container: container, overrides: overrideRoots)
        let asked = (path?.isEmpty ?? true) ? workingFolderBase()?.path : path
        switch PathPolicy.resolve(asked, roots: roots, base: workingFolderBase()) {
        case .allowed(let url): return Access(url: url, grants: grants, roots: roots)
        case .rejected: return nil
        }
    }

    /// The model-facing rejection for a path the scope does not permit. Names only the folders in
    /// scope — never a grant this conversation cannot use, and never one it is not attached to.
    /// Resolution is repeated here rather than kept, so the message cannot drift from `access`.
    static func rejection(
        for path: String?,
        container: URL = ModelStore.shared.baseDirectory,
        roots overrideRoots: [URL]? = nil
    ) -> String {
        if let scope = FileScope.current {
            let grants = FolderGrants.load()
            let roots = FileScope.roots(
                grants: grants, scope: scope, container: container, overrides: overrideRoots)
            let base = workingFolderBase()
            guard let asked = (path?.isEmpty ?? true) ? base?.path : path else {
                // No path and no working folder. With folders in scope this is not "nothing is
                // attached", it is "say which folder" (F2).
                return scope.folders.isEmpty
                    ? FileScope.noFolderMessage
                    : "\(FileScope.noPathMessage) It can use: \(FileScope.folderList(scope.folders))."
            }
            guard case .rejected(let message) = PathPolicy.resolve(asked, roots: roots, base: base)
            else {
                return "Error: unexpected."        // callers only ask after `access` returned nil
            }
            guard !scope.folders.isEmpty else { return FileScope.noFolderMessage }
            return "\(message) This chat can use: \(FileScope.folderList(scope.folders))."
                + " The user can add a folder to this chat with the + button next to the message box."
        }
        let grants = FolderGrants.load()
        let roots = FileScope.roots(
            grants: grants, scope: nil, container: container, overrides: overrideRoots)
        switch PathPolicy.resolve(path, roots: roots) {
        case .allowed:
            return "Error: unexpected."            // callers only ask after `access` returned nil
        case .rejected(let message):
            return grants.isEmpty
                ? "\(message) No folders are granted yet. \(FolderGrants.grantHint)"
                : "\(message) \(FolderGrants.grantHint)"
        }
    }
}

// MARK: - Read

nonisolated struct ScopedReadFileTool: AgentTool {
    /// Test seam: when set, the policy and the security scope use these roots instead of the
    /// live grant store. Production passes nothing.
    var injected: ScopedFileTools.ScopeSelection?

    /// Matches the direct edition's cap: plenty for source files, short of a context blowout.
    static let maxBytes = 256 * 1024

    let name = "read_file"
    let toolDescription = """
        Read a UTF-8 text file from a folder the user has granted access to. Returns the file's \
        contents, truncated at 256 KB.
        """
    var parameters: [ToolParameter] {
        [
            .required("path", type: .string, description: ScopedFileTools.relativePathDescription)
        ]
    }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        let path = arguments.string("path")
        let selection = ScopedFileTools.selection(injected)
        guard let access = ScopedFileTools.access(
            path, container: selection.container, roots: selection.roots)
        else {
            return ScopedFileTools.rejection(
                for: path, container: selection.container, roots: selection.roots)
        }
        return FolderGrants.withAccess(
            to: access.url, grants: access.grants,
            container: selection.container, roots: access.roots
        ) { url in
            if let missing = ScopedFileTools.notFoundMessage(url, askedPath: path) { return missing }
            let data = try Data(contentsOf: url)
            let clipped = data.prefix(Self.maxBytes)
            guard let text = String(data: clipped, encoding: .utf8) else {
                return "Error: \(url.lastPathComponent) is not UTF-8 text (\(data.count) bytes)."
            }
            return clipped.count < data.count ? text + "\n… (truncated)" : text
        }
    }
}

// MARK: - Write

nonisolated struct ScopedWriteFileTool: AgentTool {
    /// Test seam: when set, the policy and the security scope use these roots instead of the
    /// live grant store. Production passes nothing.
    var injected: ScopedFileTools.ScopeSelection?

    let name = "write_file"
    let toolDescription = """
        Write UTF-8 text to a file inside a folder the user has granted access to, creating or \
        overwriting it. The user must approve each write.
        """
    var parameters: [ToolParameter] {
        [
            .required("path", type: .string, description: ScopedFileTools.relativePathDescription),
            .required("contents", type: .string, description: "Text to write."),
        ]
    }
    var requiresApproval: Bool { true }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let contents = arguments.string("contents") else {
            return "Error: 'contents' is required."
        }
        let path = arguments.string("path")
        let selection = ScopedFileTools.selection(injected)
        guard let access = ScopedFileTools.access(
            path, container: selection.container, roots: selection.roots)
        else {
            return ScopedFileTools.rejection(
                for: path, container: selection.container, roots: selection.roots)
        }
        return FolderGrants.withAccess(
            to: access.url, grants: access.grants,
            container: selection.container, roots: access.roots
        ) { url in
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            // Atomic, matching the direct edition: a half-written file is worse than no file.
            try contents.write(to: url, atomically: true, encoding: .utf8)
            return "Wrote \(contents.utf8.count) bytes to \(url.path)."
        }
    }
}

// MARK: - Edit

nonisolated struct ScopedEditFileTool: AgentTool {
    /// Test seam: when set, the policy and the security scope use these roots instead of the
    /// live grant store. Production passes nothing.
    var injected: ScopedFileTools.ScopeSelection?

    let name = "edit_file"
    let toolDescription = """
        Replace exact text in an existing UTF-8 file inside a folder the user has granted \
        access to. old_string must match the file exactly, including spaces and indentation, \
        and must appear once unless replace_all is true. Copy it from read_file. Use this \
        instead of write_file to change part of a file. The user must approve each edit.
        """
    var parameters: [ToolParameter] {
        [
            .required("path", type: .string, description: ScopedFileTools.relativePathDescription),
            .required(
                "old_string", type: .string,
                description: "Exact text to replace, copied from read_file."),
            .required(
                "new_string", type: .string,
                description: "Text to put in its place; \"\" deletes it."),
            .optional(
                "replace_all", type: .bool,
                description: "Replace every occurrence instead of requiring a unique one."),
        ]
    }
    var requiresApproval: Bool { true }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let old = arguments.string("old_string") else {
            return "Error: 'old_string' is required."
        }
        guard let new = arguments.string("new_string") else {
            return "Error: 'new_string' is required."
        }
        let path = arguments.string("path")
        let selection = ScopedFileTools.selection(injected)
        guard let access = ScopedFileTools.access(
            path, container: selection.container, roots: selection.roots)
        else {
            return ScopedFileTools.rejection(
                for: path, container: selection.container, roots: selection.roots)
        }
        return FolderGrants.withAccess(
            to: access.url, grants: access.grants,
            container: selection.container, roots: access.roots
        ) { url in
            if let missing = ScopedFileTools.notFoundMessage(url, askedPath: path) { return missing }
            return TextEdit.editFile(
                at: url,
                old: old,
                new: new,
                replaceAll: arguments.bool("replace_all") ?? false)
        }
    }
}

// MARK: - List

nonisolated struct ScopedListDirectoryTool: AgentTool {
    /// Test seam: when set, the policy and the security scope use these roots instead of the
    /// live grant store. Production passes nothing.
    var injected: ScopedFileTools.ScopeSelection?

    static let maxEntries = 200

    let name = "list_directory"
    let toolDescription = "List the files and folders in a directory the user has granted access to."
    var parameters: [ToolParameter] {
        [
            .optional(
                "path", type: .string,
                description: ScopedFileTools.optionalPathDescription)
        ]
    }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        let path = arguments.string("path")
        let selection = ScopedFileTools.selection(injected)
        guard let access = ScopedFileTools.access(
            path, container: selection.container, roots: selection.roots)
        else {
            return ScopedFileTools.rejection(
                for: path, container: selection.container, roots: selection.roots)
        }
        return FolderGrants.withAccess(
            to: access.url, grants: access.grants,
            container: selection.container, roots: access.roots
        ) { url in
            if let missing = ScopedFileTools.notFoundMessage(url, askedPath: path) { return missing }
            let names = try FileManager.default.contentsOfDirectory(atPath: url.path).sorted()
            guard !names.isEmpty else { return "(empty directory)" }
            let shown = names.prefix(Self.maxEntries).map { name -> String in
                var isDirectory: ObjCBool = false
                _ = FileManager.default.fileExists(
                    atPath: url.appendingPathComponent(name).path,
                    isDirectory: &isDirectory)
                return isDirectory.boolValue ? "\(name)/" : name
            }
            let suffix = names.count > Self.maxEntries
                ? "\n… \(names.count - Self.maxEntries) more"
                : ""
            return shown.joined(separator: "\n") + suffix
        }
    }
}

// MARK: - Search

nonisolated struct ScopedSearchFilesTool: AgentTool {
    /// Test seam: when set, the policy and the security scope use these roots instead of the
    /// live grant store. Production passes nothing.
    var injected: ScopedFileTools.ScopeSelection?

    static let maxResults = 100

    let name = "search_files"
    let toolDescription = """
        Find files by name inside a folder the user has granted access to. Searches the folder \
        recursively for file names containing the query (case-insensitive).
        """
    var parameters: [ToolParameter] {
        [
            .optional(
                "path", type: .string,
                description: ScopedFileTools.optionalPathDescription),
            .required("query", type: .string, description: "Text the file name must contain."),
        ]
    }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        let query = (arguments.string("query") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return "Error: 'query' is required." }
        let path = arguments.string("path")
        let selection = ScopedFileTools.selection(injected)
        guard let access = ScopedFileTools.access(
            path, container: selection.container, roots: selection.roots)
        else {
            return ScopedFileTools.rejection(
                for: path, container: selection.container, roots: selection.roots)
        }
        return FolderGrants.withAccess(
            to: access.url, grants: access.grants,
            container: selection.container, roots: access.roots
        ) { url in
            if let missing = ScopedFileTools.notFoundMessage(url, askedPath: path) { return missing }
            return Self.format(matches: Self.search(root: url, query: query), query: query)
        }
    }

    /// Recursive name search under `root` (hidden files skipped), relative paths, capped at
    /// `maxResults` + 1 so the formatter can flag the overflow. Testable against a temp dir.
    static func search(root: URL, query: String) -> [String] {
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles])
        else { return [] }
        var matches: [String] = []
        let rootPath = root.standardizedFileURL.path
        for case let item as URL in enumerator {
            guard item.lastPathComponent.localizedCaseInsensitiveContains(query) else { continue }
            let full = item.standardizedFileURL.path
            matches.append(full.hasPrefix("\(rootPath)/")
                ? String(full.dropFirst(rootPath.count + 1))
                : full)
            if matches.count > maxResults { break }
        }
        return matches
    }

    /// Model-facing rendering. Pure, so it is unit-testable.
    static func format(matches: [String], query: String) -> String {
        guard !matches.isEmpty else { return "No file names containing \"\(query)\"." }
        let overflowed = matches.count > maxResults
        let shown = matches.prefix(maxResults).sorted().joined(separator: "\n")
        return overflowed ? shown + "\n… more matches not shown" : shown
    }
}
