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
        [.required("path", type: .string, description: "Absolute path to the file.")]
    }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        switch PathPolicy.resolve(arguments.string("path")) {
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
            .required("path", type: .string, description: "Absolute path to write."),
            .required("contents", type: .string, description: "Text to write."),
        ]
    }
    var requiresApproval: Bool { true }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let contents = arguments.string("contents") else {
            return "Error: 'contents' is required."
        }
        switch PathPolicy.resolve(arguments.string("path")) {
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

// MARK: - List

nonisolated struct ListDirectoryTool: AgentTool {
    static let maxEntries = 200

    let name = "list_directory"
    let toolDescription = "List the files and folders in a directory on the user's Mac."
    var parameters: [ToolParameter] {
        [.required("path", type: .string, description: "Absolute path to the directory.")]
    }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        switch PathPolicy.resolve(arguments.string("path")) {
        case .rejected(let message):
            return message
        case .allowed(let url):
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
                return "Error listing \(url.path): \(error.localizedDescription)"
            }
        }
    }
}
