import Foundation
import CryptoKit

/// The cache key — the Swift port of `catflow-mlx/src/catflow/core/cache.py::cache_key`
/// (CFM-R3-1). Same ingredients, same field order, same `\x1f` join, same canonical settings
/// serialization. Partitioned by realism (`mock` / `real`).
///
/// **SPEC-Q divergence (model substitution):** the `identity-model` ingredient is the
/// **substituted** `browser.json` id that actually loads (CFM-R2-2), never the manifest's
/// pinned id — the same divergence the Python side records in `SPEC_QUESTIONS.md`.
///
/// **Cross-runtime cache sharing is an explicit non-goal** (CFM-R3-2): keys will not match
/// while model ids diverge, and pretending otherwise produces silently wrong answers. The
/// golden test pins keys against Python-produced values anyway, so a drift in the *recipe*
/// (ordering, join, canonical form) is still caught.
nonisolated enum CacheKey {

    // MARK: - NEVER_CACHE (the always-skip write tasks)

    /// Ported verbatim from `cache.py::NEVER_CACHE`. Rows whose real effect is a write to a
    /// fixed destination (or a one-shot human default, or a staged outbox entry) are never
    /// cached — a cache hit would silently skip the real write on a second run.
    static let neverCache: Set<String> = [
        "Save Text", "Save Audio", "Save Image", "Save Video", "Save Images", "Store Index",
        "Save Context", "Store Write", "Ask Human", "Human Input",
        "Stage Send", "Stage Post", "Download File", "Improvise",
    ]

    // MARK: - Canonical settings (whitespace-insensitive)

    /// `canonical_settings` — any run of whitespace collapses to one space, so only the
    /// settings' actual words change the key.
    static func canonicalSettings(_ settings: String?) -> String {
        guard let settings, !settings.isEmpty else { return "" }
        return settings.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    // MARK: - Content hashing

    /// `_item_content_hash` — inline items hash their bytes; file-backed items hash the
    /// file's contents (a directory hashes path+bytes of its sorted file tree).
    static func itemContentHash(_ item: Item) throws -> String {
        if let value = item.value {
            return SHA256.hash(data: Data(value.utf8)).hex
        }
        guard let path = item.path else {
            return SHA256.hash(data: Data()).hex   // unreachable on a valid asset
        }
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: path.path, isDirectory: &isDirectory),
           isDirectory.boolValue {
            return try directoryHash(path)
        }
        return SHA256.hash(data: try Data(contentsOf: path)).hex
    }

    /// `_dir_hash` — a directory hashes each file's relative path + bytes, sorted.
    static func directoryHash(_ directory: URL) throws -> String {
        var hasher = SHA256()
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: directory,
                                             includingPropertiesForKeys: [.isRegularFileKey],
                                             options: [.skipsHiddenFiles]) else {
            return SHA256.hash(data: Data()).hex
        }
        let files = enumerator.compactMap { $0 as? URL }.sorted { $0.path < $1.path }
        for file in files {
            var isDirectory: ObjCBool = false
            guard fm.fileExists(atPath: file.path, isDirectory: &isDirectory),
                  !isDirectory.boolValue else { continue }
            let rel = file.path.replacingOccurrences(of: directory.path + "/", with: "")
            hasher.update(data: Data(rel.utf8))
            hasher.update(data: try Data(contentsOf: file))
        }
        return hasher.finalize().hex
    }

    /// `asset_input_hash` — `kind:content-hash` per item, `|`-joined, then SHA-256.
    static func assetInputHash(_ asset: Asset) throws -> String {
        let parts = try asset.items.map { "\($0.kind.rawValue):\(try itemContentHash($0))" }
        return SHA256.hash(data: Data(parts.joined(separator: "|").utf8)).hex
    }

    // MARK: - The key

    /// `cache_key` — the base five ingredients in Python order, then the conditional
    /// `frame_version`. (`context_version`/`transcript_version`/`used_flow_content`/
    /// `resolved_source_hash` are nil for the linear subset — no `· ctx` rows, no `Think`,
    /// no `uses:`, and every row has a gathered input.)
    static func cacheKey(
        task: String,
        model: String?,
        settings: String?,
        inputs: [Asset],
        realism: String,
        frameVersion: String? = nil
    ) throws -> String {
        let inputsHash = SHA256.hash(
            data: Data(try inputs.map { try assetInputHash($0) }.joined(separator: "|").utf8)
        ).hex
        // identity-model is the substituted id passed in; identity-settings is canonical.
        let identityModel = model ?? ""
        let identitySettings = canonicalSettings(settings)
        var parts = [task, identityModel, identitySettings, inputsHash, realism]
        if let frameVersion {
            parts.append(frameVersion)
        }
        let basis = parts.joined(separator: "\u{1F}")
        return SHA256.hash(data: Data(basis.utf8)).hex
    }

    /// `frame_version(task)` — the frame file's own content hash for a `RefKind.FRAME` task
    /// (stands in for a version number frame files don't carry), `nil` otherwise.
    static func frameVersion(task: String, bundle: Bundle = .main) -> String? {
        guard let desc = TaskCatalog.get(task), desc.refKind == .frame else { return nil }
        let frameName = desc.refName
            .replacingOccurrences(of: "frames/", with: "")
            .replacingOccurrences(of: ".frame.txt", with: "")
        guard let url = bundle.url(forResource: frameName, withExtension: "frame.txt"),
              let data = try? Data(contentsOf: url) else { return nil }
        return SHA256.hash(data: data).hex
    }
}

extension SHA256.Digest {
    var hex: String { map { String(format: "%02x", $0) }.joined() }
}
