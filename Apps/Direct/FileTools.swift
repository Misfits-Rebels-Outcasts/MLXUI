//
//  FileTools.swift
//  MLXUI-Direct target ONLY.
//
//  Arbitrary-filesystem tools for the un-sandboxed edition. The path policy they
//  enforce lives in shared code (`MLXUI/Core/Agent/PathPolicy.swift`) so the
//  existing test target can cover it — see PathPolicyTests.
//
//  Reminder: no sandbox does not mean no gatekeeping. macOS still applies TCC to
//  ~/Desktop, ~/Documents, ~/Downloads and removable volumes, so the first read
//  of one of those raises a consent prompt using the purpose strings in
//  Config/Direct.xcconfig. A denial surfaces here as a plain "permission denied"
//  error — worth wording tool errors so the user knows to check
//  System Settings > Privacy & Security.
//

import Foundation
import MLXLMCommon

/// The base a relative path resolves against: the conversation's working folder, or `nil`.
/// In Direct this is guidance and a sensible default, not a boundary — `run_shell` can reach
/// anything, and the UI says so (G4).
nonisolated func workingFolderBase() -> URL? {
    guard let folder = FileScope.current?.workingFolder else { return nil }
    return URL(fileURLWithPath: folder, isDirectory: true)
}

/// The roots a Direct file tool checks against: the policy's default home root, or the scope's
/// working folder when the conversation has one, so a mounted volume or a folder outside home
/// still works exactly as before when no scope is bound.
nonisolated func fileToolRoots() -> [URL] {
    guard let base = workingFolderBase() else { return PathPolicy.defaultRoots }
    return PathPolicy.defaultRoots + [base]
}

// MARK: - Read

nonisolated struct ReadFileTool: AgentTool {
    /// Enough for source files; short of blowing a small model's context window.
    static let maxBytes = 256 * 1024

    let name = "read_file"
    let toolDescription = """
        Read a UTF-8 text file from the user's Mac. Returns the file's contents, \
        truncated at 256 KB.
        """
    var parameters: [ToolParameter] {
        [
            .required(
                "path", type: .string,
                description: "Path relative to the working folder, or absolute (or starting with ~).")
        ]
    }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        switch PathPolicy.resolve(arguments.string("path"), roots: fileToolRoots(), base: workingFolderBase()) {
        case .rejected(let message):
            return message
        case .allowed(let url):
            do {
                let data = try Data(contentsOf: url)
                let clipped = data.prefix(Self.maxBytes)
                guard let text = String(data: clipped, encoding: .utf8) else {
                    return "Error: \(url.lastPathComponent) is not UTF-8 text (\(data.count) bytes)."
                }
                return clipped.count < data.count ? text + "\n… (truncated)" : text
            } catch {
                return "Error reading \(url.path): \(error.localizedDescription)"
            }
        }
    }
}

// MARK: - Write

nonisolated struct WriteFileTool: AgentTool {
    let name = "write_file"
    let toolDescription = """
        Write UTF-8 text to a file on the user's Mac, creating or overwriting it. \
        The user must approve each write.
        """
    var parameters: [ToolParameter] {
        [
            .required(
                "path", type: .string,
                description: "Path relative to the working folder, or absolute (or starting with ~)."),
            .required("contents", type: .string, description: "Text to write."),
        ]
    }
    var requiresApproval: Bool { true }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let contents = arguments.string("contents") else {
            return "Error: 'contents' is required."
        }
        switch PathPolicy.resolve(arguments.string("path"), roots: fileToolRoots(), base: workingFolderBase()) {
        case .rejected(let message):
            return message
        case .allowed(let url):
            do {
                try FileManager.default.createDirectory(
                    at: url.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                // Atomic: a half-written file is worse than no file, and the
                // model may be rewriting something the user cares about.
                try contents.write(to: url, atomically: true, encoding: .utf8)
                let bytes = contents.utf8.count
                return "Wrote \(bytes) bytes to \(url.path)."
            } catch {
                return "Error writing \(url.path): \(error.localizedDescription)"
            }
        }
    }
}

// MARK: - Edit

/// Exact-substring edit of an existing file. The matching + writing logic is EF-1's
/// `TextEdit` (shared code, so both editions behave identically); this is the direct
/// edition's path policy around it.
nonisolated struct EditFileTool: AgentTool {
    let name = "edit_file"
    let toolDescription = """
        Replace exact text in an existing UTF-8 file on the user's Mac. old_string must match \
        the file exactly, including spaces and indentation, and must appear once unless \
        replace_all is true. Copy it from read_file. Use this instead of write_file to change \
        part of a file. The user must approve each edit.
        """
    var parameters: [ToolParameter] {
        [
            .required(
                "path", type: .string,
                description: "Path relative to the working folder, or absolute (or starting with ~)."),
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
        switch PathPolicy.resolve(arguments.string("path"), roots: fileToolRoots(), base: workingFolderBase()) {
        case .rejected(let message):
            return message
        case .allowed(let url):
            // `editFile` never throws: every refusal comes back as a model-facing string.
            return TextEdit.editFile(
                at: url,
                old: old,
                new: new,
                replaceAll: arguments.bool("replace_all") ?? false)
        }
    }
}

// MARK: - List

nonisolated struct ListDirectoryTool: AgentTool {
    static let maxEntries = 200

    let name = "list_directory"
    let toolDescription = "List the files and folders in a directory on the user's Mac."
    var parameters: [ToolParameter] {
        [
            .optional(
                "path", type: .string,
                description: "Optional — omit for the working folder. "
                    + "Path relative to the working folder, or absolute (or starting with ~).")
        ]
    }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        // `path` is optional (CW-1 D4): without it, list the conversation's working folder — or
        // explain how to get one. In Direct the working folder is guidance, not a boundary.
        let argument = arguments.string("path")
        if argument?.isEmpty ?? true {
            guard let base = workingFolderBase() else { return FileScope.noFolderMessage }
            return list(base, path: FileScope.current?.workingFolder ?? base.path)
        }
        switch PathPolicy.resolve(argument, roots: fileToolRoots(), base: workingFolderBase()) {
        case .rejected(let message):
            return message
        case .allowed(let url):
            return list(url, path: url.path)
        }
    }

    private func list(_ url: URL, path: String) -> String {
        do {
            let names = try FileManager.default.contentsOfDirectory(atPath: url.path).sorted()
            guard !names.isEmpty else { return "(empty directory)" }
            let shown = names.prefix(Self.maxEntries).map { name -> String in
                var isDirectory: ObjCBool = false
                _ = FileManager.default.fileExists(
                    atPath: url.appendingPathComponent(name).path,
                    isDirectory: &isDirectory
                )
                return isDirectory.boolValue ? "\(name)/" : name
            }
            let suffix = names.count > Self.maxEntries
                ? "\n… \(names.count - Self.maxEntries) more"
                : ""
            return shown.joined(separator: "\n") + suffix
        } catch {
            return "Error listing \(path): \(error.localizedDescription)"
        }
    }
}
