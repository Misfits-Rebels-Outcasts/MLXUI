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

    /// `journal_version` — a `· ctx` row's cache-key ingredient (P3-MB-05, Spec §9.2/§13.1):
    /// `(count, sha256(canonical serialization))`, the Python's `context.py::journal_version`.
    static func journalVersion(_ entries: [(label: String, content: String)]) -> String {
        let canon = entries.map { "\($0.label)\u{1E}\(nulSafe($0.content))" }.joined(separator: "\u{1F}")
        let digest = SHA256.hash(data: Data(canon.utf8)).hex
        return "\(entries.count):\(digest)"
    }

    /// `transcript_version` — a transcript-keeping decider's (`Think`'s) cache-key ingredient
    /// (P2-M13-01 follow-up), same `(count, sha256)` shape as `journal_version`.
    static func transcriptVersion(_ entries: [FlowInterpreter.TranscriptEntry]) -> String {
        let canon = entries.map { "\($0.tool)\u{1E}\(nulSafe($0.input))\u{1E}\(nulSafe($0.observation))" }
            .joined(separator: "\u{1F}")
        let digest = SHA256.hash(data: Data(canon.utf8)).hex
        return "\(entries.count):\(digest)"
    }

    /// Control characters (`\x1e`, `\x1f`) can't legally appear in the joined text; a lone
    /// `\x1f` in the input would fake a spurious ingredient. Sanitize defensively.
    private static func nulSafe(_ s: String) -> String {
        s.replacingOccurrences(of: "\u{1F}", with: " ").replacingOccurrences(of: "\u{1E}", with: " ")
    }

    /// `cache_key` — the base five ingredients in Python order, then the conditional
    /// `frame_version`, `context_version`, `transcript_version`, `used_flow_content`, then
    /// `resolved_source_hash` (B5) — the exact `cache.py:446-459` order. The last three fold
    /// in only when given, so a row that carries none keeps a byte-identical key (FIX-12).
    static func cacheKey(
        task: String,
        model: String?,
        settings: String?,
        inputs: [Asset],
        realism: String,
        frameVersion: String? = nil,
        resolvedSourceHash: String? = nil,
        contextVersion: String? = nil,
        transcriptVersion: String? = nil,
        usedFlowContent: String? = nil
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
        if let contextVersion {
            parts.append(contextVersion)
        }
        if let transcriptVersion {
            parts.append(transcriptVersion)
        }
        if let usedFlowContent {
            parts.append(SHA256.hash(data: Data(usedFlowContent.utf8)).hex)
        }
        if let resolvedSourceHash {
            parts.append(resolvedSourceHash)
        }
        let basis = parts.joined(separator: "\u{1F}")
        return SHA256.hash(data: Data(basis.utf8)).hex
    }

    // MARK: - resolved_source_hash (the local-source ingredient, B5)

    /// `cache.py::_LOCAL_SOURCE_TASKS` — the Read family whose row-1 key must fold in the
    /// **content** of the source file it reads, so editing `policy-2025.md` in the flow
    /// folder invalidates the row-1 cache (and two flows reading same-named files with
    /// different contents never collide). Ported from `cache.py:100-109, 241-263, 456-459`.
    static let localSourceTasks: Set<String> = [
        "Read Text", "Read Audio", "Read Image", "Read Video",
        "Read Images", "Read Files", "Read PDF", "Read CSV", "Read JSON", "Read Index",
    ]

    /// The content hash of the source file a local-source row reads, resolved against the
    /// flow's working directory. `nil` when the task isn't a local-source read or the file
    /// doesn't exist yet (a missing source fails the row, so nothing gets cached anyway).
    static func resolvedSourceHash(task: String, settings: String?,
                                   flowID: String, workspace: FlowWorkspace) throws -> String? {
        guard localSourceTasks.contains(task),
              let path = FlowSettings(settings).pathValue() else { return nil }
        let url = try workspace.resolve(path, flowID: flowID)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            return nil
        }
        if isDirectory.boolValue {
            return try directoryHash(url)
        }
        return SHA256.hash(data: try Data(contentsOf: url)).hex
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

nonisolated extension SHA256.Digest {
    var hex: String { map { String(format: "%02x", $0) }.joined() }
}
