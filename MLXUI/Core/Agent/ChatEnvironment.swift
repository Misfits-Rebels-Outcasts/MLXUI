//
//  ChatEnvironment.swift
//  MLXUI — chat workspace, CW-1 (`RSI/DelegateChatWorkspaceBacklog.md`, C3).
//
//  The short block prepended to the chat's system prompt on every turn: which folders this
//  conversation can use, which is the working folder, and whether writes are allowed. Built fresh
//  each turn so a revoke or a detach takes effect immediately (`ModelRunner.send`).
//
//  Deterministic on purpose — folders in stored order, no dates, no timestamps — so the prompt
//  prefix is byte-identical from turn to turn and stays cache-friendly. Text only; the Local
//  Server path never calls this (C4).
//

import Foundation

nonisolated enum ChatEnvironment {

    /// Tool names that mean "this run can touch files".
    static let fileToolNames: Set<String> = ["read_file", "list_directory", "search_files"]
    /// ...and the ones that mean "it can change them too".
    static let writeToolNames: Set<String> = ["write_file", "edit_file"]

    /// The environment block, or `nil` when it would only waste tokens: no file tool and no write
    /// tool are on — whatever the folders — so plain chat stays plain (C3).
    ///
    /// - Parameters:
    ///   - scope: the conversation's folder scope (CW-1: every grant; Direct: no folders).
    ///   - edition: which guidance fits when there are no folders — the "ask for one" line, or
    ///     Direct's home-folder line.
    ///   - fileToolsEnabled: file tool names the user has switched on.
    ///   - writeEnabled: whether `write_file` / `edit_file` are switched on.
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

        var lines: [String] = []
        guard folderLines(
            scope, fileToolsOn: fileToolsOn, canWrite: canWrite,
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
        edition: FileScope.Edition,
        homePath: String?,
        into lines: inout [String]
    ) -> Bool {
        guard !scope.folders.isEmpty else {
            // No folders: the only thing worth saying is where to get one.
            switch edition {
            case .appStore:
                guard fileToolsOn else { return false }
                lines.append(FileScope.noFolderMessage)
                return true
            case .direct:
                guard let homePath else { return false }
                lines.append("You have no folder attached. Your home folder is \(homePath) — ask the user for one, or use an absolute path under it.")
                return true
            }
        }
        // Folders exist. With neither a file tool nor a write tool switched on there is nothing
        // the model could do with one, so stay silent rather than spend tokens on a dead list.
        guard fileToolsOn || canWrite else { return false }
        let writeNote = canWrite
            ? "you may write in it"
            : "read only (ask the user to switch on write_file or edit_file before writing)"
        lines.append("Folders you can use in this chat:")
        for folder in scope.folders {
            let isWorking = folder == scope.workingFolder
            lines.append("- \(FileScope.displayName(for: folder)) — \(folder) (\(isWorking ? "working folder; " : "")\(writeNote))")
        }
        if let working = scope.workingFolder {
            lines.append("Relative paths resolve against the working folder (\(working)).")
        } else if scope.folders.count > 1 {
            lines.append("Relative paths do not work until the user picks a working folder; use the full path above.")
        }
        // The "Next:" nudge is about the file tools. With no write tool on, a write is the thing a
        // small model is most likely to attempt and be refused, so say what to do instead.
        
        if !canWrite {
            lines.append(fileToolsOn
                ? "Next: read the folders above; ask the user before writing."
                : "Next: ask the user to switch on read_file, list_directory or search_files.")
        }
        return true
    }
}
