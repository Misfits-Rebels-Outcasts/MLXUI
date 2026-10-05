//
//  FileScope.swift
//  MLXUI — chat workspace, CW-1 (`RSI/DelegateChatWorkspaceBacklog.md`, C1).
//
//  Which folders the file tools may use *for this conversation*, and which one relative paths
//  resolve against. `AgentSession` binds the scope for the duration of one tool dispatch, and the
//  tools read it — so nothing has to be rebuilt per turn (`availableTools` is a `let` the Settings
//  UI enumerates) and the Local Server, flows and tests, which never bind a scope, keep today's
//  behaviour exactly.
//
//  `FileScope.current == nil` means "no scope, use every grant" — that is CW-1's interim rule
//  (C6). CW-2 narrows the App Store edition to the folders the conversation is attached to (G1);
//  in Direct the scope is the default for relative paths, not a boundary.
//

import Foundation

nonisolated struct FileScope: Sendable, Equatable {
    /// Which policy this scope belongs to. Not used for decisions in CW-1 (the interim scope is
    /// built from the caller's grants); kept so CW-2's `forConversation` can branch explicitly.
    enum Edition: Sendable, Equatable { case appStore, direct }

    /// Folder paths in scope, in the order the user attached them. CW-1 passes every grant.
    let folders: [String]
    /// The folder relative paths resolve against, or `nil` for "no working folder".
    let workingFolder: String?

    init(folders: [String], workingFolder: String? = nil) {
        self.folders = folders
        self.workingFolder = workingFolder
    }

    /// The scope bound for the current tool dispatch, when one is bound.
    @TaskLocal static var current: FileScope?

    /// The scope for one conversation (CW-2, C7).
    ///
    /// **App Store:** only attached folders that are **still granted** survive, so revoking a grant
    /// in Settings, or a grant that went stale, drops the folder out of the next turn (G1). The
    /// working folder follows the same filter.
    ///
    /// **Direct:** there are no grants, so the attached paths pass through as they are — the scope
    /// is the default for relative paths and the set of roots the tools may reach, not a boundary.
    static func forConversation(
        attached: [String],
        workingFolder: String?,
        grants: [FolderGrant],
        edition: Edition
    ) -> FileScope {
        switch edition {
        case .appStore:
            let granted = Set(grants.compactMap { FolderGrants.canonicalPath($0.path) })
            let folders = attached.filter { path in
                guard let canonical = FolderGrants.canonicalPath(path) else { return false }
                return granted.contains(canonical)
            }
            let working = folders.first { path in
                guard let canonical = FolderGrants.canonicalPath(path),
                      let wanted = workingFolder.flatMap({ FolderGrants.canonicalPath($0) })
                else { return false }
                return canonical == wanted
            }
            return FileScope(folders: folders, workingFolder: working ?? folders.first)
        case .direct:
            let working = attached.first { $0 == workingFolder } ?? attached.first
            return FileScope(folders: attached, workingFolder: working)
        }
    }

    // MARK: - Roots

    /// Resolve the allowlist roots for this scope: the scope's granted folders, plus the app
    /// container — or the injected overrides a test uses in place of a live grant store.
    static func roots(
        grants: [FolderGrant],
        scope: FileScope?,
        container: URL,
        overrides: [URL]? = nil
    ) -> [URL] {
        if let overrides { return overrides }
        guard let scope else {
            return FolderGrants.rootURLs(grants: grants, container: container)
        }
        return FolderGrants.rootURLs(grants: grants, scope: scope, container: container)
    }

    /// The roots the current dispatch should use, read from the live grant store: every grant
    /// plus the real container when no scope is bound — today's App Store behaviour without an
    /// explicit scope.
    static func roots(container: URL, overrides: [URL]? = nil) -> [URL] {
        roots(grants: FolderGrants.load(), scope: current, container: container, overrides: overrides)
    }

    // MARK: - Model-facing text

    /// "workfolder (/Users/me/workfolder), Invoices (/Users/me/Invoices)" — or "none".
    static func folderList(_ paths: [String]) -> String {
        guard !paths.isEmpty else { return "none" }
        return paths.map { "\(Self.displayName(for: $0)) (\($0))" }.joined(separator: ", ")
    }

    /// The folder's name, i.e. its last path component.
    static func displayName(for path: String) -> String {
        let name = URL(fileURLWithPath: path).lastPathComponent
        return name.isEmpty ? path : name
    }

    /// Told to the model when it asks for a file and no folder is in scope. Short, and it says
    /// what to do next.
    static let noFolderMessage =
        "No folder is attached to this chat. Ask the user to add one with the + button next to the message box."

    /// Told when the tool needs a path, some folders *are* in scope, but none of them is the
    /// working folder — so a relative path has nothing to resolve against. Deliberately never
    /// worded as "no folder is attached": that would be false, and the model would repeat it to
    /// the user (CW-1-FIX-1 F2). The caller appends the folder list.
    static let noPathMessage =
        "Error: give a path. This chat has no working folder."
}
