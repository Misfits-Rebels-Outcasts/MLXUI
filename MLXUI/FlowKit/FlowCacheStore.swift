import Foundation

/// The content-addressed cache store — the Swift mirror of `core/cache.py::CacheStore`
/// (CFM-R3-2), rooted under `ModelStore.shared.flowsDirectory/cache/` with `entries/` +
/// `blobs/` trees.
///
/// Layout (mirrors the Python):
///   `cache/entries/<key[:2]>/<key>/manifest.json` — one row's cached `Asset`
///   `cache/blobs/<hash[:2]>/<hash>` — the file-backed bytes, shared across entries
///
/// **Cross-runtime cache sharing with the Python is an explicit non-goal** — keys will not
/// match while model ids diverge (CFM-R2-2 substitution), and pretending otherwise produces
/// silently wrong answers. This store is app-local and keyed by the *substituted* id.
///
/// LRU: every `get`/`put` touches the entry's mtime; `evictIfNeeded` removes the least
/// recently used entries when the store exceeds a byte budget. `clear` drops everything.
nonisolated struct FlowCacheStore: Sendable {
    let root: URL

    init(root: URL) {
        self.root = root
    }

    /// The app-wide store, rooted at `flows/cache/`.
    static let shared = FlowCacheStore(root: ModelStore.shared.flowsDirectory
        .appendingPathComponent("cache", isDirectory: true))

    var entriesDirectory: URL { root.appendingPathComponent("entries", isDirectory: true) }
    var blobsDirectory: URL { root.appendingPathComponent("blobs", isDirectory: true) }

    // MARK: - Paths

    private func entryDirectory(forKey key: String) -> URL {
        entriesDirectory
            .appendingPathComponent(String(key.prefix(2)), isDirectory: true)
            .appendingPathComponent(key, isDirectory: true)
    }

    private func blobPath(forHash hash: String) -> URL {
        blobsDirectory
            .appendingPathComponent(String(hash.prefix(2)), isDirectory: true)
            .appendingPathComponent(hash, isDirectory: false)
    }

    // MARK: - Get / put

    /// Look up `key`; file-backed items are materialized into `blobDirectory` as a fresh
    /// copy named after the requesting row's `path`, so the caller's workspace stays
    /// self-contained even if the cache is cleared later. `nil` on a miss.
    func get(key: String, blobDirectory: URL, path: String) throws -> Asset? {
        let manifestURL = entryDirectory(forKey: key).appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else { return nil }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))

        try? FileManager.default.createDirectory(at: blobDirectory, withIntermediateDirectories: true)
        var items: [Item] = []
        for (i, entry) in manifest.items.enumerated() {
            if let value = entry.value {
                items.append(Item(kind: entry.kindValue, value: value, path: nil, sourceText: nil))
                continue
            }
            guard let hash = entry.blob else { continue }
            let src = blobPath(forHash: hash)
            let dest = blobDirectory
                .appendingPathComponent("\(path).\(i).\(entry.kind).\(String(hash.prefix(16))).bin")
            if !FileManager.default.fileExists(atPath: dest.path) {
                try FileManager.default.copyItem(at: src, to: dest)
            }
            items.append(Item(kind: entry.kindValue, value: nil, path: dest, sourceText: nil))
        }
        touch(key: key)
        return Asset(items: items)
    }

    /// Store `asset` under `key` (inline values inline, file-backed items by content hash).
    func put(key: String, asset: Asset) throws {
        let dir = entryDirectory(forKey: key)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var entries: [ManifestItem] = []
        for item in asset.items {
            if let value = item.value {
                entries.append(ManifestItem(kind: item.kind, value: value, blob: nil))
                continue
            }
            guard let path = item.path else { continue }
            let hash = try CacheKey.itemContentHash(item)
            let blob = blobPath(forHash: hash)
            if !FileManager.default.fileExists(atPath: blob.path) {
                try FileManager.default.createDirectory(at: blob.deletingLastPathComponent(),
                                                        withIntermediateDirectories: true)
                try FileManager.default.copyItem(at: path, to: blob)
            }
            entries.append(ManifestItem(kind: item.kind, value: nil, blob: hash))
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(Manifest(items: entries))
        try data.write(to: dir.appendingPathComponent("manifest.json"))
        touch(key: key)
    }

    /// Touch an entry's mtime so LRU eviction sees it as recently used.
    private func touch(key: String) {
        let dir = entryDirectory(forKey: key)
        try? FileManager.default.setAttributes([.modificationDate: Date()], ofItemAtPath: dir.path)
    }

    // MARK: - Stats / eviction / clear

    /// Number of cached entries (manifest.json files).
    var entryCount: Int {
        guard FileManager.default.fileExists(atPath: entriesDirectory.path) else { return 0 }
        let enumerator = FileManager.default.enumerator(at: entriesDirectory,
                                                        includingPropertiesForKeys: nil)
        return (enumerator?.allObjects as? [URL])?.filter { $0.lastPathComponent == "manifest.json" }.count ?? 0
    }

    /// Total bytes in the store (manifests + blobs).
    var totalBytes: Int64 {
        var sum: Int64 = 0
        for dir in [entriesDirectory, blobsDirectory] where FileManager.default.fileExists(atPath: dir.path) {
            guard let enumerator = FileManager.default.enumerator(at: dir,
                                                                  includingPropertiesForKeys: [.fileSizeKey]) else { continue }
            for case let url as URL in enumerator {
                let values = try? url.resourceValues(forKeys: [.fileSizeKey])
                sum += Int64(values?.fileSize ?? 0)
            }
        }
        return sum
    }

    /// Evict least-recently-used entries until the store is under `byteBudget`.
    func evictIfNeeded(byteBudget: Int64) throws {
        guard totalBytes > byteBudget else { return }
        let fm = FileManager.default
        guard fm.fileExists(atPath: entriesDirectory.path) else { return }
        let dirs = try fm.contentsOfDirectory(at: entriesDirectory, includingPropertiesForKeys: [.contentModificationDateKey])
        for shard in dirs {
            let entries = try fm.contentsOfDirectory(at: shard, includingPropertiesForKeys: [.contentModificationDateKey])
            for entry in entries.sorted(by: {
                (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate ?? .distantPast)
                    ?? .distantPast
                    <
                (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate ?? .distantPast)
                    ?? .distantPast
            }) {
                try? fm.removeItem(at: entry)
                if totalBytes <= byteBudget { return }
            }
        }
        gcBlobs()
    }

    /// Remove unreferenced blobs (blobs not named by any remaining manifest).
    func gcBlobs() {
        let fm = FileManager.default
        var referenced = Set<String>()
        if let enumerator = fm.enumerator(at: entriesDirectory, includingPropertiesForKeys: nil) {
            for case let url as URL in enumerator where url.lastPathComponent == "manifest.json" {
                if let data = try? Data(contentsOf: url),
                   let manifest = try? JSONDecoder().decode(Manifest.self, from: data) {
                    for item in manifest.items { if let b = item.blob { referenced.insert(b) } }
                }
            }
        }
        guard fm.fileExists(atPath: blobsDirectory.path) else { return }
        for shard in (try? fm.contentsOfDirectory(at: blobsDirectory, includingPropertiesForKeys: nil)) ?? [] {
            for blob in (try? fm.contentsOfDirectory(at: shard, includingPropertiesForKeys: nil)) ?? [] {
                if !referenced.contains(blob.lastPathComponent) {
                    try? fm.removeItem(at: blob)
                }
            }
        }
    }

    /// Drop everything in the store (entries + blobs). Always safe — clearing only costs
    /// recomputation.
    func clear() throws {
        let fm = FileManager.default
        for dir in [entriesDirectory, blobsDirectory] where fm.fileExists(atPath: dir.path) {
            try fm.removeItem(at: dir)
        }
    }
}

// MARK: - Manifest DTO

extension FlowCacheStore {
    /// One cached `Asset`'s manifest — inline values stored directly, file-backed items by
    /// content hash. `kind` is the raw kind string, mirroring the Python's
    /// `{"kind": item.kind.value, ...}` shape.
    struct Manifest: Codable, Sendable {
        var items: [ManifestItem]
    }

    struct ManifestItem: Codable, Sendable {
        var kind: String
        var value: String?
        var blob: String?

        init(kind: Kind, value: String?, blob: String?) {
            self.kind = kind.rawValue
            self.value = value
            self.blob = blob
        }

        var kindValue: Kind { Kind(rawValue: kind) ?? .file }
    }
}

// MARK: - ModelStore location

extension ModelStore {
    /// The flow cache directory: `flows/cache/`.
    var flowCacheDirectory: URL {
        flowsDirectory.appendingPathComponent("cache", isDirectory: true)
    }
}
