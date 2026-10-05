//
//  ApprovalPreview.swift
//  MLXUI — shared code (ships in BOTH editions).
//
//  The pure formatter behind the approval sheet (EF-3, D10). Until EF-3 the sheet showed only
//  the tool's name and description, so approving `run_shell` or `write_file` was blind — the
//  AG1 gap. This turns the pending call's arguments into labelled, style-tagged sections the
//  sheet renders as a before/after view.
//
//  Deliberately has no SwiftUI, no file I/O and no actor: `MLXUITests` calls it directly, and
//  the sheet stays display-only. Nothing here decides anything — `AgentSession` and the
//  per-tool policies are untouched by EF-3.
//

import Foundation
import MLXLMCommon

nonisolated enum ApprovalPreview {

    /// How a section's body reads: a plain fact, a removed block (red), an added block (green),
    /// or a code block (monospaced, no tint).
    enum Style: Equatable, Sendable {
        case plain, removed, added, code
    }

    /// One labelled block of the sheet. `Equatable` so the formatter is unit-testable.
    struct Section: Equatable, Sendable {
        let title: String
        let body: String
        let style: Style
    }

    /// Bodies longer than this are clipped with an explicit notice — never silently.
    static let maxBodyBytes = 64 * 1_024

    /// The sections for one pending tool call, in the order the sheet renders them.
    static func sections(toolName: String, arguments: [String: JSONValue]) -> [Section] {
        switch toolName {
        case "edit_file": return editFileSections(arguments)
        case "write_file": return writeFileSections(arguments)
        case "run_shell": return runShellSections(arguments)
        default: return argumentSections(arguments)
        }
    }

    // MARK: - Per-tool layouts

    private static func editFileSections(_ arguments: [String: JSONValue]) -> [Section] {
        var sections: [Section] = []
        if let path = arguments.string("path") {
            sections.append(Section(title: "Path", body: path, style: .plain))
        }
        if let old = arguments.string("old_string") {
            sections.append(Section(title: "Replace", body: old, style: .removed))
        }
        if let new = arguments.string("new_string") {
            sections.append(Section(title: "With", body: new, style: .added))
        }
        if arguments.bool("replace_all") == true {
            sections.append(Section(
                title: "All occurrences",
                body: "Every match in the file will be replaced.",
                style: .plain))
        }
        return sections.map(clipped)
    }

    /// `write_file` has **no** "will overwrite" line on purpose: knowing whether the file exists
    /// needs file I/O, and in the App Store edition that needs the covering grant's security
    /// scope. The sheet does no I/O (the backlog's trap), so the line is left out — see the EF-3
    /// journal, where that is recorded as the implementer's call the backlog allowed.
    private static func writeFileSections(_ arguments: [String: JSONValue]) -> [Section] {
        var sections: [Section] = []
        if let path = arguments.string("path") {
            sections.append(Section(title: "Path", body: path, style: .plain))
        }
        if let contents = arguments.string("contents") {
            sections.append(Section(title: "Contents", body: contents, style: .code))
        }
        return sections.map(clipped)
    }

    private static func runShellSections(_ arguments: [String: JSONValue]) -> [Section] {
        var sections: [Section] = []
        if let command = arguments.string("command") {
            sections.append(Section(title: "Command", body: command, style: .code))
        }
        if let directory = arguments.string("working_directory") {
            sections.append(Section(title: "Working directory", body: directory, style: .plain))
        }
        if let timeout = arguments.int("timeout_seconds") {
            sections.append(Section(title: "Timeout", body: "\(timeout) seconds", style: .plain))
        }
        return sections.map(clipped)
    }

    /// Anything else: one code section per argument, keys sorted, values as compact JSON.
    private static func argumentSections(_ arguments: [String: JSONValue]) -> [Section] {
        arguments.keys.sorted().compactMap { key in
            guard let value = arguments[key] else { return nil }
            return clipped(Section(title: key, body: compactJSON(value), style: .code))
        }
    }

    // MARK: - Helpers

    /// Compact, key-sorted JSON for one argument value (so the same call always previews the
    /// same way, whatever order the dictionary happens to enumerate in).
    static func compactJSON(_ value: JSONValue) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(value),
              let text = String(data: data, encoding: .utf8) else { return "null" }
        return text
    }

    /// Clip on a character boundary and say how many characters are missing.
    static func clipped(_ section: Section) -> Section {
        guard section.body.utf8.count > maxBodyBytes else { return section }
        var kept = ""
        var bytes = 0
        var characters = 0
        for character in section.body {
            let width = String(character).utf8.count
            if bytes + width > maxBodyBytes { break }
            kept.append(character)
            bytes += width
            characters += 1
        }
        let dropped = section.body.count - characters
        return Section(
            title: section.title,
            body: kept + "\n… (\(dropped) more characters not shown)",
            style: section.style)
    }
}
