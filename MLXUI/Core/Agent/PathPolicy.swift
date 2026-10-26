//
//  PathPolicy.swift
//  MLXUI — shared code (ships in BOTH editions).
//
//  Where filesystem-touching agent tools may operate. Only the direct-download
//  edition has tools that consult it (`Apps/Direct/FileTools.swift`,
//  `Apps/Direct/ShellTool.swift`), but the policy itself is pure logic with no
//  reference to `Process` or any file I/O, so it lives here where the existing
//  `MLXUITests` target can reach it. Getting this wrong is the difference
//  between "the model read a file you pointed it at" and "the model read your
//  SSH key", so it should be the best-tested code in the feature.
//
//  Two layers: an allowlist of roots, and a denylist of paths that sit inside
//  those roots but must never be readable by a model.
//

import Foundation

nonisolated enum PathPolicy {

    /// Default allowlist: the user's home directory.
    static var defaultRoots: [URL] {
        [FileManager.default.homeDirectoryForCurrentUser]
    }

    /// Denied paths, relative to the home directory: credentials, keys, message
    /// stores, browser state, shell history.
    static let deniedSubpaths: [String] = [
        "Library/Keychains",
        "Library/Application Support/com.apple.TCC",
        "Library/Cookies",
        "Library/Messages",
        "Library/Mail",
        "Library/Safari",
        ".ssh",
        ".gnupg",
        ".aws",
        ".config/gh",
        ".netrc",
        ".zsh_history",
        ".bash_history",
    ]

    /// True when `url` sits inside an allowed root and outside every denied path.
    ///
    /// Symlinks are resolved before either check. Without that,
    /// `~/link-to-dot-ssh` and `~/../../etc/passwd` both look fine as strings and
    /// both escape the allowlist.
    static func isAllowed(_ url: URL, roots: [URL] = defaultRoots) -> Bool {
        let target = url.standardizedFileURL.resolvingSymlinksInPath().path
        let home = FileManager.default.homeDirectoryForCurrentUser
            .standardizedFileURL.resolvingSymlinksInPath().path

        for denied in deniedSubpaths {
            let full = "\(home)/\(denied)"
            if target == full || target.hasPrefix("\(full)/") { return false }
        }
        for root in roots {
            let base = root.standardizedFileURL.resolvingSymlinksInPath().path
            if target == base || target.hasPrefix("\(base)/") { return true }
        }
        return false
    }

    /// Outcome of `resolve`. Not `Result<URL, String>` — `Result.Failure` must
    /// conform to `Error`, and the rejection here is a model-facing string, not
    /// something worth minting an error type for.
    enum Resolution: Equatable {
        case allowed(URL)
        case rejected(String)
    }

    /// Shared argument handling for the file tools: expand `~`, require an
    /// absolute path, apply the policy.
    ///
    /// `base` (CW-1, `RSI/DelegateChatWorkspaceBacklog.md` C2) is the conversation's working
    /// folder. A relative path is joined to it **first**, then goes through the very same
    /// `isAllowed` check — the denylist and the escape/symlink guards see the joined result, never
    /// the raw string. With `base == nil` the behaviour is byte-identical to before: a relative
    /// path is rejected outright.
    ///
    /// `folders` (CW-FIX-2) are the folders attached to the chat, defaulting to the bound
    /// `FileScope`. They only change **which path is asked for** (see `aliasedFolder`); the result
    /// still goes through `isAllowed` unchanged.
    static func resolve(
        _ path: String?,
        roots: [URL] = defaultRoots,
        base: URL? = nil,
        folders: [String] = FileScope.current?.folders ?? []
    ) -> Resolution {
        guard let path, !path.isEmpty else {
            return .rejected("Error: 'path' is required.")
        }
        let expanded = (path as NSString).expandingTildeInPath
        let url: URL
        if expanded.hasPrefix("/") {
            url = URL(fileURLWithPath: expanded)
        } else if let alias = aliasedFolder(for: expanded, base: base, folders: folders) {
            url = alias
        } else if let base {
            // Join before checking. `..` is collapsed by `isAllowed`'s
            // `standardizedFileURL`, so `../outside` cannot escape that way.
            url = base.appendingPathComponent(expanded)
        } else {
            return .rejected("Error: 'path' must be absolute (got \"\(path)\").")
        }
        guard isAllowed(url, roots: roots) else {
            return .rejected("Error: access to \(url.path) is not permitted.")
        }
        return .allowed(url)
    }

    /// CW-FIX-2: the user and the model both call an attached folder by its chip name, so a relative
    /// path whose first component is that name ("workfolder", "workfolder/notes/a.md") means the
    /// folder, not `<working folder>/workfolder`.
    ///
    /// Returns `nil` — meaning "resolve as before" — when the first component names no attached
    /// folder, names more than one (ambiguous names are not guessed), or `<base>/<path>` really
    /// exists (a real subfolder wins). The alias set is only `folders`, so this cannot reach
    /// anything the chat has not attached, and the caller still applies `isAllowed` (so `..` after
    /// the alias is refused as ever).
    static func aliasedFolder(for relative: String, base: URL?, folders: [String]) -> URL? {
        let parts = relative.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
        guard let first = parts.first else { return nil }
        let matches = Set(folders
            .filter { FileScope.displayName(for: $0) == first }
            .map { URL(fileURLWithPath: $0, isDirectory: true).standardizedFileURL.path })
        guard matches.count == 1, let folder = matches.first else { return nil }
        if let base, FileManager.default.fileExists(atPath: base.appendingPathComponent(relative).path) {
            return nil
        }
        let rest = parts.dropFirst().joined(separator: "/")
        let root = URL(fileURLWithPath: folder, isDirectory: true)
        return rest.isEmpty ? root : root.appendingPathComponent(rest)
    }
}
