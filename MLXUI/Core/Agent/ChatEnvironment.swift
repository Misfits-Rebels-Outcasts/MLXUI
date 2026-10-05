//
//  ChatEnvironment.swift
//  MLXUI — chat workspace, CW-1 (`RSI/DelegateChatWorkspaceBacklog.md`, C3; wording W1/W2 at CW-A).
//
//  The short block prepended to the chat's system prompt on every turn: which folders this
//  conversation can use, which is the working folder, and whether writing is on. Built fresh each
//  turn so a revoke or a change of grants takes effect immediately (`ModelRunner.send`).
//
//  Deterministic on purpose — folders in stored order, no dates, no timestamps — so the prompt
//  prefix is byte-identical from turn to turn. Text only; the Local Server path never calls this
//  (C4).
//

import Foundation

nonisolated enum ChatEnvironment {

    /// Tool names that mean "this run can touch files".
    static let fileToolNames: Set<String> = ["read_file", "list_directory", "search_files"]
    /// ...and the ones that mean "it can change them too".
    static let writeToolNames: Set<String> = ["write_file", "edit_file"]
    /// Direct's shell can reach anywhere, so it is enough to justify the home-hint on its own.
    static let shellToolName = "run_shell"

    /// The environment block, or `nil` when it would only waste tokens: no file tool, no write tool
    /// and no shell, whatever the folders — so plain chat stays plain (C3).
    ///
    /// - Parameters:
    ///   - scope: the conversation's folder scope (CW-1: every grant; Direct: no folders).
    ///   - edition: which guidance fits when there are no folders — the "ask for one" line, or
    ///     Direct's home-folder line.
    ///   - fileToolsEnabled: every tool name the user has switched on (file, write and shell).
    ///   - writeEnabled: whether `write_file` / `edit_file` are on.
    ///   - homePath: the user's home directory (Direct guidance only), or `nil`.
    ///   - attachmentsNote: CW-3's one line per by-reference attachment; always `nil` for now.
    static func instructions(
        scope: FileScope,
        edition: FileScope.Edition,
        fileToolsEnabled: Set<String>,
        writeEnabled: Bool,
        homePath: String? = nil,
        attachmentsNote: String? = nil
    ) -> String? {
        let fileToolsOn = !fileToolsEnabled.isDisjoint(with: fileToolNames)
        let canWrite = writeEnabled && !fileToolsEnabled.isDisjoint(with: writeToolNames)
        let shellOn = fileToolsEnabled.contains(shellToolName)

        var lines: [String] = []
        guard folderLines(
            scope, fileToolsOn: fileToolsOn, canWrite: canWrite, shellOn: shellOn,
            edition: edition, homePath: homePath, into: &lines)
        else { return nil }
        if let attachmentsNote, !attachmentsNote.isEmpty { lines.append(attachmentsNote) }
        return lines.filter { !$0.isEmpty }.joined(separator: "\n")
    }

    /// Fills in the folder lines. Returns false when this turn has nothing to say at all.
    private static func folderLines(
        _ scope: FileScope,
        fileToolsOn: Bool,
        canWrite: Bool,
        shellOn: Bool,
        edition: FileScope.Edition,
        homePath: String?,
        into lines: inout [String]
    ) -> Bool {
        guard !scope.folders.isEmpty else {
            // No folders: the only thing worth saying is where to look. Direct's shell and file
            // tools can both reach home, so the hint needs one of them on (F3).
            switch edition {
            case .appStore:
                guard fileToolsOn else { return false }
                lines.append(FileScope.noFolderMessage)
                return true
            case .direct:
                guard fileToolsOn || canWrite || shellOn, let homePath else { return false }
                lines.append("No folder is attached to this chat. The user's home folder is \(homePath)."
                    + " Use full paths under it, or paths starting with ~/.")
                return true
            }
        }
        // Folders exist. With no tool at all there is nothing the model could do with one, so stay
        // silent rather than spend tokens on a dead list.
        guard fileToolsOn || canWrite || shellOn else { return false }
        lines.append("Folders you can use in this chat:")
        for folder in scope.folders {
            let working = folder == scope.workingFolder ? " (working folder)" : ""
            lines.append("- \(FileScope.displayName(for: folder)): \(folder)\(working)")
        }
        lines.append(scope.workingFolder != nil
            ? "Relative paths start in the working folder, e.g. notes/todo.md."
            : "There is no working folder, so use full paths.")
        lines.append(canWrite
            ? "You can read and change files in these folders. The user approves each change."
            : "You can read these folders. Writing is off; if asked to change a file, tell the user"
                + " to turn on write_file or edit_file.")
        return true
    }
}
