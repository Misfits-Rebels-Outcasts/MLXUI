//
//  ModelToolPathScope.swift
//  MLXUI — Model Tools, MT4-2 (owner ruling, Boo, 2026-11-09: "in both editions, a model tool's path input
//  resolves only inside the chat's attached folders or granted scope. Anything outside is refused, and the
//  refusal names the attach action.").
//
//  App Store: the roots are the chat's attached, still-granted folders plus the app container, exactly what
//  `read_file` uses (`ScopedFileTools.access`), so a model tool can never reach a file `read_file` cannot.
//  Direct: `read_file` may reach all of home (the shell can reach anything anyway), but a MODEL TOOL's `path`
//  is narrower: only the folders attached to this chat and this chat's own attachments copy. Both then go
//  through `PathPolicy.resolve`, so the denylist, `..`, symlink and attached-folder-alias guards are the same.
//
//  Pure and shared (no I/O), so the Direct policy is tested in the App Store test host too.
//

import Foundation

nonisolated enum ModelToolPathScope {

    /// Direct: the roots a model tool's `path` may resolve in — the folders attached to the chat, its working
    /// folder, and this chat's attachments copy. Empty when no scope is bound (flows, the Local Server).
    /// `assets` is this chat's own `assets/` folder (when a host is bound).
    static func directRoots(scope: FileScope?, assets: URL? = nil) -> [URL] {
        guard let scope else { return [] }
        var paths = scope.folders
        if let working = scope.workingFolder { paths.append(working) }
        if let attachments = scope.attachmentsFolder { paths.append(attachments) }
        if let assets { paths.append(assets.path) }
        var seen = Set<String>()
        var roots: [URL] = []
        for path in paths where !path.isEmpty {
            let url = URL(fileURLWithPath: path, isDirectory: true)
            guard seen.insert(url.standardizedFileURL.path).inserted else { continue }
            roots.append(url)
        }
        return roots
    }

    /// App Store: the roots a model tool's `path` may resolve in. `base` is what `read_file` uses (granted attached folders plus the
    /// app `container`); the container itself is dropped (it holds every model, `installed.json` and every other chat's files), and
    /// only THIS chat's `attachments/` and `assets/` folders, which live inside it, are added back. `read_file` is unchanged.
    static func appStoreRoots(base: [URL], container: URL, scope: FileScope?, assets: URL?) -> [URL] {
        func canonical(_ url: URL) -> String { url.standardizedFileURL.resolvingSymlinksInPath().path }
        let containerPath = canonical(container)
        var roots = base.filter { canonical($0) != containerPath }
        if let attachments = scope?.attachmentsFolder { roots.append(URL(fileURLWithPath: attachments, isDirectory: true)) }
        if let assets { roots.append(assets) }
        return roots
    }

    /// The sentence the router gets for a path the scope refuses. Always names the attach action; names the
    /// folders in scope when there are any. `message` is `PathPolicy`'s rejection ("Error: access to … is not
    /// permitted."), whose "Error: " prefix is dropped (the tool failure adds its own).
    static func refusal(_ message: String, scope: FileScope?) -> String {
        guard let scope, !scope.folders.isEmpty else { return FileScope.noFolderMessage }
        let plain = ModelToolRunner.withoutErrorPrefix(message)
        return "\(plain) This chat can use: \(FileScope.folderList(scope.folders))." + " " + attachAction
    }

    /// What the user can do about a refusal.
    static let attachAction = "The user can add a folder to this chat with the + button next to the message box."
}
