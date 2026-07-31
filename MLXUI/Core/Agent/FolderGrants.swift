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
        return grants
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
        var grants = load(from: defaults).filter { $0.path != url.path }
        grants.append(FolderGrant(path: url.path, bookmark: bookmark))
        save(grants, to: defaults)
    }

    static func removeGrant(path: String, from defaults: UserDefaults = .standard) {
        save(load(from: defaults).filter { $0.path != path }, to: defaults)
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

    /// Run `body` while holding security-scoped access to the grant covering `url`. Paths under
    /// the app container need no scope and run directly. Returns the model-facing error string
    /// when no grant covers the path or the bookmark no longer resolves (folder moved/deleted).
    static func withAccess(
        to url: URL,
        grants: [FolderGrant],
        container: URL = ModelStore.shared.baseDirectory,
        body: (URL) throws -> String
    ) -> String {
        let target = url.standardizedFileURL.resolvingSymlinksInPath().path
        let containerPath = container.standardizedFileURL.resolvingSymlinksInPath().path
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
                return "Error: access to \(grant.path) could not be restored — "
                    + "the folder may have moved. Re-grant it from the tools menu."
            }
            let accessed = scoped.startAccessingSecurityScopedResource()
            defer { if accessed { scoped.stopAccessingSecurityScopedResource() } }
            return try body(url)
        } catch {
            return "Error: \(error.localizedDescription)"
        }
    }

    /// The hint appended to rejections so the model can tell the user how to widen access.
    static let grantHint =
        "The user can grant folder access from the tools (wrench) menu in the chat window."
}
