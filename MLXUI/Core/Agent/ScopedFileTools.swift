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
//  security scope (`FolderGrants.withAccess`). `write_file` is side-effecting ⇒ approval-gated
//  and disabled by default, mirroring the direct edition.
//

import Foundation
import MLXLMCommon

nonisolated enum ScopedFileTools {
    static func all() -> [any AgentTool] {
        [
            ScopedReadFileTool(),
            ScopedWriteFileTool(),
            ScopedListDirectoryTool(),
            ScopedSearchFilesTool(),
        ]
    }

    /// `write_file` starts unticked (visible, but the user opts in), matching `DirectTools`.
    static let disabledByDefault: Set<String> = ["write_file"]

    /// Shared front half of every tool: load grants, apply the path policy against them.
    static func resolve(_ path: String?) -> (url: URL, grants: [FolderGrant])? {
        let grants = FolderGrants.load()
        switch PathPolicy.resolve(path, roots: FolderGrants.rootURLs(grants: grants)) {
        case .allowed(let url): return (url, grants)
        case .rejected: return nil
        }
    }

    /// The model-facing rejection for a path outside every grant.
    static func rejection(for path: String?) -> String {
        let grants = FolderGrants.load()
        switch PathPolicy.resolve(path, roots: FolderGrants.rootURLs(grants: grants)) {
        case .allowed:
            return "Error: unexpected."  // callers only ask after resolve() returned nil
        case .rejected(let message):
            return grants.isEmpty
                ? "\(message) No folders are granted yet. \(FolderGrants.grantHint)"
                : "\(message) \(FolderGrants.grantHint)"
        }
    }
}

// MARK: - Read

nonisolated struct ScopedReadFileTool: AgentTool {
    /// Matches the direct edition's cap: plenty for source files, short of a context blowout.
    static let maxBytes = 256 * 1024

    let name = "read_file"
    let toolDescription = """
        Read a UTF-8 text file from a folder the user has granted access to. Returns the file's \
        contents, truncated at 256 KB.
        """
    var parameters: [ToolParameter] {
        [
            .required("path", type: .string, description: "Absolute path to the file.")
        ]
    }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        let path = arguments.string("path")
        guard let (url, grants) = ScopedFileTools.resolve(path) else {
            return ScopedFileTools.rejection(for: path)
        }
        return FolderGrants.withAccess(to: url, grants: grants) { url in
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
    let name = "write_file"
    let toolDescription = """
        Write UTF-8 text to a file inside a folder the user has granted access to, creating or \
        overwriting it. The user must approve each write.
        """
    var parameters: [ToolParameter] {
        [
            .required("path", type: .string, description: "Absolute path to write."),
            .required("contents", type: .string, description: "Text to write."),
        ]
    }
    var requiresApproval: Bool { true }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let contents = arguments.string("contents") else {
            return "Error: 'contents' is required."
        }
        let path = arguments.string("path")
        guard let (url, grants) = ScopedFileTools.resolve(path) else {
            return ScopedFileTools.rejection(for: path)
        }
        return FolderGrants.withAccess(to: url, grants: grants) { url in
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            // Atomic, matching the direct edition: a half-written file is worse than no file.
            try contents.write(to: url, atomically: true, encoding: .utf8)
            return "Wrote \(contents.utf8.count) bytes to \(url.path)."
        }
    }
}

// MARK: - List

nonisolated struct ScopedListDirectoryTool: AgentTool {
    static let maxEntries = 200

    let name = "list_directory"
    let toolDescription = "List the files and folders in a directory the user has granted access to."
    var parameters: [ToolParameter] {
        [
            .required("path", type: .string, description: "Absolute path to the directory.")
        ]
    }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        let path = arguments.string("path")
        guard let (url, grants) = ScopedFileTools.resolve(path) else {
            return ScopedFileTools.rejection(for: path)
        }
        return FolderGrants.withAccess(to: url, grants: grants) { url in
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
    static let maxResults = 100

    let name = "search_files"
    let toolDescription = """
        Find files by name inside a folder the user has granted access to. Searches the folder \
        recursively for file names containing the query (case-insensitive).
        """
    var parameters: [ToolParameter] {
        [
            .required("path", type: .string, description: "Absolute path of the directory to search."),
            .required("query", type: .string, description: "Text the file name must contain."),
        ]
    }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        let query = (arguments.string("query") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return "Error: 'query' is required." }
        let path = arguments.string("path")
        guard let (url, grants) = ScopedFileTools.resolve(path) else {
            return ScopedFileTools.rejection(for: path)
        }
        return FolderGrants.withAccess(to: url, grants: grants) { url in
            Self.format(matches: Self.search(root: url, query: query), query: query)
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
