import Foundation
import CryptoKit

/// One typed payload flowing between rows. Ported from `catflow-mlx/src/catflow/core/assets.py`
/// (`Item`, `Asset`). Exactly one of `value` / `path` is typically set, per kind — inline
/// for text/status/context, file-backed for everything else — but the port keeps both
/// optional (no `__post_init__` invariant) so decode can be tolerant; the invariant is the
/// tools'/executor's job, exactly as in Python where `core/` never touches the filesystem.
nonisolated struct Item: Sendable {
    var kind: Kind
    var value: String?
    var path: URL?
    var sourceText: String?

    /// Stable content hash — the right half of the cache key (CFM-R3-1). Inline items hash
    /// their bytes; file-backed items hash the file's contents. SHA-256, hex-encoded.
    /// `sourceText` is deliberately excluded (it is provenance, not content).
    func contentHash() throws -> String {
        var hasher = SHA256()
        hasher.update(data: Data(kind.rawValue.utf8))
        hasher.update(data: Data([0x1F]))
        if let value {
            hasher.update(data: Data(value.utf8))
        } else if let path {
            let data = try Data(contentsOf: path, options: [.mappedIfSafe])
            hasher.update(data: data)
        } else {
            hasher.update(data: Data())
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

/// What one row produces: `items` of length 1 is a `Single`-shaped result, length > 1 is a
/// `ListOf`-shaped result. Cardinality is carried by the list itself, not a separate tag —
/// mirrors `core/kinds.py`'s `Single`/`ListOf`.
nonisolated struct Asset: Sendable {
    var items: [Item]

    /// The list of kinds, one per item.
    var shape: [Kind] { items.map(\.kind) }

    /// Ordered combine of the items' content hashes (empty asset → SHA-256 of empty).
    func contentHash() throws -> String {
        var hasher = SHA256()
        for item in items {
            hasher.update(data: Data(try item.contentHash().utf8))
            hasher.update(data: Data([0x1F]))
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
