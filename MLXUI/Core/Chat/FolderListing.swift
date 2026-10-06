//
//  FolderListing.swift
//  MLXUI — chat workspace, CW-4 (`RSI/DelegateChatWorkspaceBacklog.md`).
//
//  One level of a folder, for the inspector's disclosure tree. Deliberately one level at a time:
//  the sidebar expands lazily, so nothing here enumerates a tree recursively.
//
//  `children` is pure apart from `stat`, so it is unit-tested on a temp directory. The view calls
//  `listing(for:injected:)`, which wraps the read in the covering grant's security scope in the
//  sandboxed edition (`FolderGrants.withAccess`) — a folder the user granted is not readable
//  outside that scope, and a folder they did **not** grant must return the refusal string rather
//  than an empty list.
//

import Foundation

nonisolated enum FolderListing {

    /// One row in the tree.
    struct Entry: Sendable, Equatable, Identifiable {
        let name: String
        let path: String
        let isDirectory: Bool
        /// Drives the disclosure triangle; a symlink to a directory is deliberately not expandable.
        var isExpandable: Bool { isDirectory }

        var id: String { path }
    }

    /// What one level produced.
    struct Result: Sendable, Equatable {
        let entries: [Entry]
        /// How many children exist in total, so the view can say "… N more".
        let total: Int
        /// A model/user-facing reason the level could not be read, if it could not.
        let error: String?

        static let empty = Result(entries: [], total: 0, error: nil)

        var hidden: Int { max(0, total - entries.count) }
    }

    /// Cap on rows for one level (the contract: ≤ 200 entries per level with "… N more").
    static let maxEntries = 200

    /// Read one level. Symlinks are **listed** but never followed, so a link loop cannot be walked
    /// and a link cannot smuggle a path outside the granted root into the tree.
    static func children(
        of directory: URL,
        limit: Int = maxEntries,
        includeHidden: Bool = false
    ) -> Result {
        let keys: [URLResourceKey] = [.isDirectoryKey, .isSymbolicLinkKey, .nameKey]
        guard let urls = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: keys,
            options: includeHidden ? [] : [.skipsHiddenFiles]
        ) else {
            return Result(entries: [], total: 0, error: "This folder could not be read.")
        }

        let sorted = urls.sorted { left, right in
            let leftDirectory = isDirectory(left)
            let rightDirectory = isDirectory(right)
            if leftDirectory != rightDirectory { return leftDirectory }   // folders first
            return left.lastPathComponent.localizedStandardCompare(right.lastPathComponent)
                == .orderedAscending
        }

        let entries = sorted.prefix(max(0, limit)).map { url -> Entry in
            let symbolic = (try? url.resourceValues(forKeys: [.isSymbolicLinkKey]))?
                .isSymbolicLink ?? false
            return Entry(
                name: url.lastPathComponent,
                path: url.path,
                isDirectory: isDirectory(url) && !symbolic)
        }
        return Result(entries: Array(entries), total: sorted.count, error: nil)
    }

    private static func isDirectory(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
    }

    /// The read the view performs: in the App Store edition the directory read runs inside the
    /// covering grant's security scope, so a granted folder is readable after a relaunch and a
    /// folder the user did **not** grant comes back as a refusal.
    ///
    /// `injected` is nil in the app; a test passes one so a temp root stands in for a grant. The
    /// parameter exists in release too — it is a value, not a `#if DEBUG` seam — but nothing in the
    /// app ever passes one.
    /// Read **at most `limit` bytes** of a file through the same scope handling as a listing.
    /// `nil` means the read was refused or failed — the preview says so rather than showing
    /// nothing. Bounded on purpose: a folder the model listed can hold a 300 MB log, and the
    /// preview pane must not read it to show 64 KB.
    static func scopedData(
        at url: URL,
        limit: Int,
        injected: ((URL, Int) -> Data?)? = nil
    ) -> Data? {
        if let injected { return injected(url, limit) }
        return readBounded(url, limit: limit)
    }

    /// The actual bounded read: `FileHandle` so only `limit + 1` bytes are ever pulled in.
    nonisolated static func readBounded(_ url: URL, limit: Int) -> Data? {
        #if DIRECT_BUILD
        return boundedRead(url, limit: limit)
        #else
        return FolderGrants.withScopedRead(
            to: url,
            grants: FolderGrants.load(),
            container: ModelStore.shared.baseDirectory
        ) { boundedRead($0, limit: limit) } ?? nil
        #endif
    }

    /// `FileHandle.read(upToCount:)` (macOS 10.15+) returns at most `count` bytes and throws at
    /// EOF, which is exactly one bounded read. Used unchanged in both editions.
    nonisolated static func boundedRead(_ url: URL, limit: Int) -> Data? {
        guard limit > 0, let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        return try? handle.read(upToCount: limit)
    }

    static func listing(
        for directory: URL,
        limit: Int = maxEntries,
        injected: ((URL) -> Result)? = nil
    ) -> Result {
        if let injected { return injected(directory) }
        #if DIRECT_BUILD
        return children(of: directory, limit: limit)
        #else
        // The model-facing refusal text is generated by `withAccess`; this only needs to know
        // whether the read happened, so an error string is mapped to the tree's own message.
        let container = ModelStore.shared.baseDirectory
        let grants = FolderGrants.load()
        let outcome = FolderGrants.withScopedRead(
            to: directory, grants: grants, container: container
        ) { children(of: $0, limit: limit) }
        guard let outcome else {
            return Result(entries: [], total: 0, error: "This folder is not attached to this chat.")
        }
        return outcome
        #endif
    }
}
