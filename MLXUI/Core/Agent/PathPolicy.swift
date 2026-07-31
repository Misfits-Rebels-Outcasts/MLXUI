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
    static func resolve(_ path: String?, roots: [URL] = defaultRoots) -> Resolution {
        guard let path, !path.isEmpty else {
            return .rejected("Error: 'path' is required.")
        }
        let expanded = (path as NSString).expandingTildeInPath
        guard expanded.hasPrefix("/") else {
            return .rejected("Error: 'path' must be absolute (got \"\(path)\").")
        }
        let url = URL(fileURLWithPath: expanded)
        guard isAllowed(url, roots: roots) else {
            return .rejected("Error: access to \(url.path) is not permitted.")
        }
        return .allowed(url)
    }
}
