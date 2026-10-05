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

/// The Direct edition's file scope. No bookmarks are involved: the working folder is a sensible
/// default and a hint, **not** a boundary (`run_shell` can reach anything). A separate enum rather
/// than free functions so it does not sit in the global namespace beside
/// `ScopedFileTools.workingFolderBase()`.
nonisolated enum DirectFileScope {

    /// The folder relative paths resolve against: the conversation's working folder, or `nil`.
    static func workingFolderBase() -> URL? {
        guard let folder = FileScope.current?.workingFolder else { return nil }
        return URL(fileURLWithPath: folder, isDirectory: true)
    }

    /// The roots the policy checks against: home **plus every folder attached to this chat**
    /// (CW-2 ruling 5). An attached folder on an external volume must be readable by its full path,
    /// which a working-folder-only list would refuse. In Direct this is guidance, not a boundary —
    /// `run_shell` can reach anything, and the composer says so (G4).
    static func toolRoots() -> [URL] {
        var roots = PathPolicy.defaultRoots
        var seen = Set(roots.map(\.standardizedFileURL.path))
        for folder in FileScope.current?.folders ?? [] {
            let url = URL(fileURLWithPath: folder, isDirectory: true)
            guard seen.insert(url.standardizedFileURL.path).inserted else { continue }
            roots.append(url)
        }
        return roots
    }

    /// Refuse a path that is off the denylist even when the user attached it. `run_shell` has
    /// always done this through `allowedRoots`; a folder attached by hand must not be a way around
    /// it (CW-1-FIX-1 nit).
    static func denial(for url: URL) -> String? {
        let refused = PathPolicy.resolve(url.path, roots: [])
        guard case .rejected = refused else { return nil }   // not listed: the user may attach it
        return "Error: \(url.path) is not a path these tools may use."
    }

    /// Told when a tool needs a path and there is no working folder to default to. Direct can
    /// still reach home, so this is "give one", never "nothing is attached" (F2).
    static let noPathMessage = "Give a path, for example ~/Documents."
}

/// The one `path` description both editions use (W3). Re-exported so this file names a single
/// string rather than four copies of the literal.
nonisolated let filePathParameterDescription = ScopedFileTools.relativePathDescription
nonisolated let optionalFilePathParameterDescription = ScopedFileTools.optionalPathDescription

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
        [.required("path", type: .string, description: filePathParameterDescription)]
    }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        switch PathPolicy.resolve(
            arguments.string("path"),
            roots: DirectFileScope.toolRoots(),
            base: DirectFileScope.workingFolderBase()
        ) {
        case .rejected(let message):
            return message
        case .allowed(let url):
            if let denial = DirectFileScope.denial(for: url) { return denial }
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
            .required("path", type: .string, description: filePathParameterDescription),
            .required("contents", type: .string, description: "Text to write."),
        ]
    }
    var requiresApproval: Bool { true }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let contents = arguments.string("contents") else {
            return "Error: 'contents' is required."
        }
        switch PathPolicy.resolve(
            arguments.string("path"),
            roots: DirectFileScope.toolRoots(),
            base: DirectFileScope.workingFolderBase()
        ) {
        case .rejected(let message):
            return message
        case .allowed(let url):
            if let denial = DirectFileScope.denial(for: url) { return denial }
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
            .required("path", type: .string, description: filePathParameterDescription),
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
        switch PathPolicy.resolve(
            arguments.string("path"),
            roots: DirectFileScope.toolRoots(),
            base: DirectFileScope.workingFolderBase()
        ) {
        case .rejected(let message):
            return message
        case .allowed(let url):
            if let denial = DirectFileScope.denial(for: url) { return denial }
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
        [.optional("path", type: .string, description: optionalFilePathParameterDescription)]
    }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        // `path` is optional (CW-1 D4): without it, list the conversation's working folder. With no
        // working folder either, ask for a path — Direct can still reach home, so this is never
        // "nothing is attached" (F2).
        let argument = arguments.string("path")
        if argument?.isEmpty ?? true {
            guard let base = DirectFileScope.workingFolderBase() else {
                return DirectFileScope.noPathMessage
            }
            if let denial = DirectFileScope.denial(for: base) { return denial }
            return list(base, path: FileScope.current?.workingFolder ?? base.path)
        }
        switch PathPolicy.resolve(
            argument, roots: DirectFileScope.toolRoots(), base: DirectFileScope.workingFolderBase()
        ) {
        case .rejected(let message):
            return message
        case .allowed(let url):
            if let denial = DirectFileScope.denial(for: url) { return denial }
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
