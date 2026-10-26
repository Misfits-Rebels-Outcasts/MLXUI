//
//  FolderGrants.swift
//  MLXUI — agentic chat tools, slice AG5 (scoped files).
//
//  User-granted folder access for the sandboxed edition. The sandbox only lets the app touch a
//  path after the user picks it in an open panel; a **security-scoped bookmark** persists that
//  grant across launches. This type is the bookmark store: add a grant from a panel-picked URL,
//  list/remove grants, and run file I/O inside a grant's security scope.
//
//  The *policy* question ("may the model touch this path?") stays in `PathPolicy` — tools pass
//  `roots: FolderGrants.rootURLs(...)` so the allowlist is exactly the granted folders (plus the
//  app container, which is always writable and needs no scope). The denylist (`~/.ssh`, Keychains,
//  …) still applies inside granted roots as defense in depth.
//
//  Pure parts (Codable model, covering-grant matcher, persistence round-trip) are unit-tested;
//  bookmark resolution + NSOpenPanel are exercised by build + smoke.
//

import Foundation

/// One granted folder: the path it had when granted, plus the security-scoped bookmark that
/// re-authorizes it on later launches.
nonisolated struct FolderGrant: Codable, Sendable, Equatable {
    let path: String
    let bookmark: Data
}

nonisolated enum FolderGrants {



    private static let defaultsKey = "agentFolderGrants"

    // MARK: - Store

    static func load(from defaults: UserDefaults = .standard) -> [FolderGrant] {
        guard let data = defaults.data(forKey: defaultsKey),
              let grants = try? JSONDecoder().decode([FolderGrant].self, from: data)
        else { return [] }
        // CW-FIX-3: one grant per real folder. Older builds compared raw strings, so the same
        // folder under two spellings was stored twice; keep the last one saved and write the
        // cleaned list back once.
        let cleaned = deduplicated(grants)
        if cleaned.count != grants.count { save(cleaned, to: defaults) }
        return cleaned
    }

    /// Collapse grants that name the same folder (by `canonicalPath`), keeping the last one saved
    /// and the order of the survivors. A path with no canonical form is never merged.
    static func deduplicated(_ grants: [FolderGrant]) -> [FolderGrant] {
        var seen = Set<String>()
        var kept: [FolderGrant] = []
        for grant in grants.reversed() {
            if let canonical = canonicalPath(grant.path), !seen.insert(canonical).inserted { continue }
            kept.append(grant)
        }
        return kept.reversed()
    }

    /// True when two paths are the same folder however they are spelled (trailing slash, symlink,
    /// firmlink, `/Volumes/...` versus `/Users/...`).
    static func samePath(_ a: String, _ b: String) -> Bool {
        if a == b { return true }
        guard let x = canonicalPath(a), let y = canonicalPath(b) else { return false }
        return x == y
    }

    /// `list.contains(path)`, compared by `samePath`.
    static func contains(_ list: [String], _ path: String) -> Bool {
        list.contains { samePath($0, path) }
    }

    static func save(_ grants: [FolderGrant], to defaults: UserDefaults = .standard) {
        guard let data = try? JSONEncoder().encode(grants) else { return }
        defaults.set(data, forKey: defaultsKey)
    }

    /// Persist a grant for a folder the user just picked in an open panel (the panel's selection
    /// is what authorizes creating a security-scoped bookmark). Replaces an existing grant for
    /// the same path.
    static func addGrant(for url: URL, to defaults: UserDefaults = .standard) throws {
        let bookmark = try url.bookmarkData(
            options: .withSecurityScope,
            includingResourceValuesForKeys: nil,
            relativeTo: nil)
        var grants = load(from: defaults).filter { !samePath($0.path, url.path) }
        grants.append(FolderGrant(path: url.path, bookmark: bookmark))
        save(grants, to: defaults)
    }

    static func removeGrant(path: String, from defaults: UserDefaults = .standard) {
        save(load(from: defaults).filter { !samePath($0.path, path) }, to: defaults)
    }

    /// Paths of all granted folders (for the menu UI).
    static func grantedPaths(from defaults: UserDefaults = .standard) -> [String] {
        load(from: defaults).map(\.path).sorted()
    }

    // MARK: - Tool-side resolution

    /// The allowlist roots for `PathPolicy.resolve`: every granted folder plus the app container
    /// (always accessible under the sandbox — the agent's scratch space).
    static func rootURLs(grants: [FolderGrant], container: URL = ModelStore.shared.baseDirectory) -> [URL] {
        grants.map { URL(fileURLWithPath: $0.path, isDirectory: true) } + [container]
    }

    /// The allowlist roots for one conversation's `FileScope`: only the granted folders named in
    /// `scope.folders`, plus the app container. A grant the conversation is not attached to is not
    /// a root — CW-1 C6 passes every grant to the interim scope; CW-2 narrows this to the attached
    /// ones (G1).
    ///
    /// `withAccess` is still handed the *full* grant list, but it only looks up the security scope
    /// of a path `PathPolicy` has already allowed, so a larger list there cannot widen access.
    static func rootURLs(
        grants: [FolderGrant],
        scope: FileScope,
        container: URL = ModelStore.shared.baseDirectory
    ) -> [URL] {
        // Compare canonically: `scope.folders` holds whatever the conversation stored, which is
        // not always byte-identical to the grant's path (a trailing "/" is the obvious case), and
        // silently dropping the user's folder because of it would look like "access denied".
        let allowed = Set(scope.folders.compactMap { Self.canonicalPath($0) })
        return grants
            .filter { grant in
                guard let canonical = Self.canonicalPath(grant.path) else { return false }
                return allowed.contains(canonical)
            }
            .map { URL(fileURLWithPath: $0.path, isDirectory: true) } + [container]
    }

    /// The path in the form `PathPolicy.isAllowed` compares in: standardized, with symlinks
    /// resolved. `URL.path` already drops a trailing slash, even for a directory URL.
    static func canonicalPath(_ path: String) -> String? {
        guard !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath().path
    }

    /// The grant covering `path` (longest matching root wins, so a grant for a subfolder beats
    /// one for its parent). Pure, so it is unit-testable.
    static func grant(covering path: String, in grants: [FolderGrant]) -> FolderGrant? {
        let target = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath().path
        return grants
            .filter { grant in
                let base = URL(fileURLWithPath: grant.path)
                    .standardizedFileURL.resolvingSymlinksInPath().path
                return target == base || target.hasPrefix("\(base)/")
            }
            .max { $0.path.count < $1.path.count }
    }

    /// Run `body` while holding security-scoped access to the grant covering `url`.
    ///
    /// `roots`, when given, is an **extra gate** — a path under none of them is refused. It is
    /// never a way past the grant lookup: a path inside a granted folder still resolves its
    /// bookmark and calls `startAccessingSecurityScopedResource`, or a grant would work only in
    /// the launch where the open panel handed out its implicit extension (CW-1-FIX-1 F1).
    ///
    /// After the gate, the original order is unchanged: the app container needs no scope, every
    /// other path needs the covering grant's bookmark. Returns the model-facing error string when
    /// the gate refuses, no grant covers the path, or the bookmark no longer resolves.
    static func withAccess(
        to url: URL,
        grants: [FolderGrant],
        container: URL = ModelStore.shared.baseDirectory,
        roots: [URL]? = nil,
        body: (URL) throws -> String
    ) -> String {
        let target = url.standardizedFileURL.resolvingSymlinksInPath().path
        let containerPath = container.standardizedFileURL.resolvingSymlinksInPath().path
        if let roots {
            let permitted = roots.contains { root in
                let base = root.standardizedFileURL.resolvingSymlinksInPath().path
                return target == base || target.hasPrefix("\(base)/")
            }
            guard permitted else {
                return "Error: no granted folder covers \(url.path)."
            }
        }
        do {
            if target == containerPath || target.hasPrefix("\(containerPath)/") {
                return try body(url)  // app container: no security scope needed
            }
            guard let grant = grant(covering: url.path, in: grants) else {
                return "Error: no granted folder covers \(url.path)."
            }
            var isStale = false
            guard let scoped = try? URL(
                resolvingBookmarkData: grant.bookmark,
                options: .withSecurityScope,
                relativeTo: nil,
                bookmarkDataIsStale: &isStale)
            else {
                return staleBookmarkMessage(for: grant.path)
            }
            let accessed = scoped.startAccessingSecurityScopedResource()
            defer { if accessed { scoped.stopAccessingSecurityScopedResource() } }
            return try body(url)
        } catch {
            return "Error: \(error.localizedDescription)"
        }
    }

    /// The refusal when a grant's bookmark no longer resolves (the folder moved or was deleted).
    /// Model-facing: it says where the user can fix it, and the road to the wrench menu is gone
    /// (CW-1-FIX-1 N2 — the same stale instruction as `grantHint`).
    static func staleBookmarkMessage(for path: String) -> String {
        "Error: access to \(path) could not be restored;"
            + " the folder may have moved. The user can grant it again in Settings > Tools,"
            + " or add it to this chat with the + button."
    }

    /// The hint appended to rejections so the model can tell the user how to widen access.
    ///
    /// SET-3 moved folder grants out of the chat's wrench menu into Settings → Tools, so the old
    /// wording sent the model to a menu that no longer lists them (CW-1 D4).
    static let grantHint =
        "The user can grant folder access in Settings, Tools, or with the + button in the chat."
}


// MARK: - Scope helpers for readers that are not `String`

#if !DIRECT_BUILD
extension FolderGrants {
    /// The same scope handling as `withAccess`, for a read that returns something other than a
    /// model-facing string — CW-4's folder listing and CW-5's preview both need it.
    /// `nil` when the path is covered by neither the container nor a grant, or the bookmark no
    /// longer resolves.
    static func withScopedRead<T>(
        to url: URL,
        grants: [FolderGrant],
        container: URL,
        body: (URL) throws -> T
    ) -> T? {
        let target = url.standardizedFileURL.resolvingSymlinksInPath().path
        let containerPath = container.standardizedFileURL.resolvingSymlinksInPath().path
        do {
            if target == containerPath || target.hasPrefix("\(containerPath)/") {
                return try body(url)
            }
            guard let grant = grant(covering: url.path, in: grants) else { return nil }
            var isStale = false
            guard let scoped = try? URL(
                resolvingBookmarkData: grant.bookmark,
                options: .withSecurityScope,
                relativeTo: nil,
                bookmarkDataIsStale: &isStale)
            else { return nil }
            let accessed = scoped.startAccessingSecurityScopedResource()
            defer { if accessed { scoped.stopAccessingSecurityScopedResource() } }
            return try body(url)
        } catch {
            return nil
        }
    }
}
#endif
