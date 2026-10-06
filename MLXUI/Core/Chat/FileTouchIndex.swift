//
//  FileTouchIndex.swift
//  MLXUI — chat workspace, CW-5 (`RSI/DelegateChatWorkspaceBacklog.md`, C14).
//
//  Which files this chat touched, derived from the items it already persists. **Not** from
//  `AgentAuditLog`: that log is in-memory only by design (privacy), so it would be empty after a
//  relaunch, and the inspector is precisely the place a user looks after one.
//
//  The result is a pure value: group the `ToolActivity` cards by resolved path, decide what each
//  one did, and keep the order the chat did it in. A path read and later edited appears once, as
//  edited, with both turns.
//

import Foundation

nonisolated enum FileTouchIndex {

    /// What a tool did to a path.
    enum Kind: Sendable, Equatable, Codable {
        /// `write_file` that did not answer with an error.
        case created
        /// `edit_file` that ran.
        case edited
        /// `read_file` / `list_directory` / `search_files`.
        case read
        /// A write or an edit the user refused (or that failed).
        case declined

        var isChange: Bool {
            switch self {
            case .created, .edited: return true
            case .read, .declined: return false
            }
        }

        /// The row's grouping title.
        var label: String {
            switch self {
            case .created: return "New"
            case .edited: return "Edited"
            case .read: return "Read"
            case .declined: return "Declined"
            }
        }
    }

    /// One row.
    struct Entry: Sendable, Equatable, Identifiable {
        /// The resolved path — absolute, whatever the model passed.
        let path: String
        /// What the chat did to it, latest wins.
        /// The latest thing the chat did to it — a read then an edit shows as edited.
        var kind: Kind
        /// Every tool card that touched it, oldest first, so the preview can show the history and
        /// CW-6 can offer the newest edit's diff.
        var activities: [ToolActivity]

        var id: String { path }
        var name: String { URL(fileURLWithPath: path).lastPathComponent }
        var isEmptyName: Bool { name.isEmpty }
        /// The turn of the latest touch (1-based), for "with the turn that did it".
        var lastTurn: Int { activities.count }
    }

    /// The whole picture, grouped the way the sidebar renders it.
    struct Evaluation: Sendable, Equatable {
        var entries: [Entry] = []

        /// New · Edited, in first-touch order.
        var changes: [Entry] { entries.filter { $0.kind.isChange } }
        /// Read-only touches, collapsed in the UI as "N files read".
        var reads: [Entry] { entries.filter { $0.kind == .read } }
        /// Refused edits and writes.
        var declined: [Entry] { entries.filter { $0.kind == .declined } }

        func entry(for path: String) -> Entry? { entries.first { $0.path == path } }

        /// The edits of one path, newest first — what CW-6 renders as red/green diffs.
        func editActivities(for path: String) -> [ToolActivity] {
            (entry(for: path)?.activities ?? [])
                .filter { $0.name == "edit_file" && isFinished($0) }
                .reversed()
        }

        private func isFinished(_ activity: ToolActivity) -> Bool {
            if case .finished = activity.status { return true }
            return false
        }
    }

    /// Tool names and the kind they imply.
    private static let readTools: Set<String> = ["read_file", "list_directory", "search_files"]
    private static let writeTools: Set<String> = ["write_file", "edit_file"]

    /// Build the index. `workingFolder` resolves relative arguments; `nil` means an absolute path
    /// was required, so a relative argument is kept as it was written.
    ///
    /// Known limitation (C14, written here on purpose): a relative path is resolved against the
    /// **current** working folder, not the one in force when the call ran. The conversation stores
    /// attachments but not the working folder per turn, so a chat whose working folder changed
    /// mid-conversation can attribute an old relative path to the new folder.
    static func entries(
        items: [TranscriptItem],
        workingFolder: String? = nil
    ) -> Evaluation {
        var order: [String] = []
        var byPath: [String: Entry] = [:]

        for item in items {
            guard case .tool(let activity) = item else { continue }
            guard readTools.contains(activity.name) || writeTools.contains(activity.name)
            else { continue }
            guard let raw = activity.arguments.string("path"), !raw.isEmpty else { continue }
            // Only terminal cards: an in-flight one has no outcome yet.
            guard isTerminal(activity.status) else { continue }

            let path = resolve(raw, workingFolder: workingFolder)
            let kind = kind(of: activity)

            if var existing = byPath[path] {
                existing.kind = kind
                existing.activities.append(activity)
                byPath[path] = existing
            } else {
                order.append(path)
                byPath[path] = Entry(path: path, kind: kind, activities: [activity])
            }
        }
        return Evaluation(entries: order.compactMap { byPath[$0] })
    }

    /// A tool card that has reached an outcome. `.running` is deliberately excluded: nothing has
    /// happened yet.
    static func isTerminal(_ status: ToolActivity.Status) -> Bool {
        switch status {
        case .running: return false
        case .finished, .failed, .denied, .unknownTool, .budgetExceeded: return true
        }
    }

    /// What one card did.
    static func kind(of activity: ToolActivity) -> Kind {
        switch activity.status {
        case .denied:
            // The user's refusal is the fact worth showing for a write or an edit.
            return .declined
        case .finished:
            if activity.name == "write_file" { return writeOutcome(activity) }
            if activity.name == "edit_file" { return .edited }
            return .read
        case .failed, .unknownTool, .budgetExceeded:
            return writeTools.contains(activity.name) ? .declined : .read
        case .running:
            return writeTools.contains(activity.name) ? .declined : .read
        }
    }

    /// Both editions answer a **failed** write with a string that contains "Error" (the App Store
    /// tool's `withAccess` wrapper, Direct's `catch`, and EF-2's `editFile` all do), and no success
    /// path mentions the word. That is the whole rule — no invented sentinel.
    ///
    /// The result does not say whether the file already existed (the tools write either way), so
    /// there is no "overwritten" distinction: the row is **New**, and the preview shows what was
    /// written. Worth a tool wording change if the reviewer wants the distinction — it would need
    /// the tool to report it, not this index to guess it.
    static func writeOutcome(_ activity: ToolActivity) -> Kind {
        guard let result = activity.result, !result.contains("Error") else { return .declined }
        return .created
    }

    /// Resolve a tool argument the way the tools do: `~` first, then relative against the working
    /// folder. No policy check — this is a record of what was *attempted*, and a refused path is
    /// still worth showing.
    static func resolve(_ raw: String, workingFolder: String?) -> String {
        let expanded = (raw as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded).standardizedFileURL.path
        }
        guard let workingFolder else { return expanded }
        return URL(fileURLWithPath: workingFolder, isDirectory: true)
            .appendingPathComponent(expanded)
            .standardizedFileURL
            .path
    }
}
