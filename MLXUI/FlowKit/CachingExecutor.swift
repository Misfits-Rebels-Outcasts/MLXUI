import Foundation

/// Wraps any `FlowExecutor` with the content-addressed cache (CFM-R3-2), mirroring
/// `core/cache.py::CachingExecutor` for the linear subset: applies the `_with_seed`
/// substitution, decides skip-vs-key per row, checks the store, and serves a hit instead of
/// running the inner executor. A hit is reported via `lastCacheHit` so the runner can emit
/// `.cacheHit` (the Python's duck-typed `last_hit` convention). A reference type so the hit
/// flag's mutation is visible to the runner through the `FlowExecutor` existential.
final class CachingExecutor: FlowExecutor, @unchecked Sendable {
    let inner: any FlowExecutor
    let store: FlowCacheStore
    /// `"mock"` or `"real"` — partitions the cache by whether this row's own execution ran
    /// the real computation or a mock fingerprint (mock/real keys never mix).
    let cacheTier: String
    /// Flat `browser.json` entries, for resolving a model row's display name to the
    /// **substituted** id (the cache-key identity-model ingredient).
    let catalog: [ModelEntry]
    /// The deterministic run-level seed (flow-text-derived) for `_with_seed`.
    let runSeed: UInt32
    /// Resolves local-source rows' file paths for `resolved_source_hash` (B5).
    let workspace: FlowWorkspace
    let flowID: String

    private let hitLock = NSLock()
    private var _lastCacheHit = false
    var lastCacheHit: Bool {
        hitLock.withLock { _lastCacheHit }
    }

    init(inner: any FlowExecutor, store: FlowCacheStore, cacheTier: String,
         catalog: [ModelEntry], runSeed: UInt32, workspace: FlowWorkspace, flowID: String) {
        self.inner = inner
        self.store = store
        self.cacheTier = cacheTier
        self.catalog = catalog
        self.runSeed = runSeed
        self.workspace = workspace
        self.flowID = flowID
    }

    func execute(path: String, row: Row, inputs: [Asset],
                 transcript: [FlowInterpreter.TranscriptEntry]?,
                 context: [(label: String, content: String)]?,
                 usedFlowContent: String?) async throws -> Asset {
        hitLock.withLock { _lastCacheHit = false }

        // _with_seed: substitute a concrete seed into a stochastic row's settings first, so
        // the derived number is what the cache key and engine call both see.
        let seededRow = try withSeed(path: path, row: row)

        // NEVER_CACHE (writes) always skip — a hit would silently skip the real write.
        guard !CacheKey.neverCache.contains(seededRow.task ?? "") else {
            return try await inner.execute(path: path, row: seededRow, inputs: inputs,
                                           transcript: transcript, context: context,
                                           usedFlowContent: usedFlowContent)
        }

        let key = try cacheKey(for: seededRow, inputs: inputs,
                               transcript: transcript, context: context,
                               usedFlowContent: usedFlowContent)
        let blobDir = store.root.appendingPathComponent("materialized", isDirectory: true)
        if let cached = try store.get(key: key, blobDirectory: blobDir, path: path) {
            hitLock.withLock { _lastCacheHit = true }
            return cached
        }
        let asset = try await inner.execute(path: path, row: seededRow, inputs: inputs,
                                            transcript: transcript, context: context,
                                            usedFlowContent: usedFlowContent)
        try? store.put(key: key, asset: asset)
        try? store.evictIfNeeded(byteBudget: 1_000_000_000)   // 1 GB soft budget
        return asset
    }

    // MARK: - Key

    private func cacheKey(for row: Row, inputs: [Asset],
                          transcript: [FlowInterpreter.TranscriptEntry]?,
                          context: [(label: String, content: String)]?,
                          usedFlowContent: String?) throws -> String {
        // The substituted id (CFM-R2-2) is the identity-model ingredient — the same id the
        // real executor actually loads. Unresolvable rows fall back to the display name
        // (never a crash; mock rows aren't resolved by design).
        var modelID: String? = row.model
        if let display = row.model {
            if case .runnable(let entry, _, _) = CatalogBridge.resolve(display, catalog: catalog) {
                modelID = entry.hfModelId
            }
        }
        // B5: a local-source row with no gathered input (row 1 of every flow) folds the
        // source file's content hash into the key, so editing the file invalidates the row.
        let sourceHash = inputs.isEmpty
            ? try? CacheKey.resolvedSourceHash(task: row.task ?? "", settings: row.settings,
                                               flowID: flowID, workspace: workspace)
            : nil
        return try CacheKey.cacheKey(
            task: row.task ?? "",
            model: modelID,
            settings: row.settings,
            inputs: inputs,
            realism: cacheTier,
            frameVersion: CacheKey.frameVersion(task: row.task ?? ""),
            resolvedSourceHash: sourceHash,
            contextVersion: context.map { CacheKey.journalVersion($0) },
            transcriptVersion: transcript.map { CacheKey.transcriptVersion($0) },
            usedFlowContent: usedFlowContent
        )
    }

    // MARK: - _with_seed

    /// The `_with_seed` substitution: only a row whose serving manifest declares a `seed`
    /// setting is stochastic (Speak today). Requires a resolvable model — otherwise the row
    /// is returned unchanged.
    private func withSeed(path: String, row: Row) throws -> Row {
        guard let display = row.model else { return row }
        guard let entry = CatalogBridge.entry(for: display) else { return row }
        guard let manifest = CuratedManifest.load(manifestFile: entry.manifestFile),
              manifest.settings["seed"] != nil else { return row }
        let identity = FlowSeed.rowIdentity(path: path, row: row,
                                            modelID: entry.pinnedID)
        guard let newSettings = FlowSeed.resolveSeedSettings(
            settings: row.settings, runSeed: runSeed,
            activationIndex: FlowSeed.activationIndex(execPath: path), identity: identity) else {
            return row
        }
        return Row(id: row.id, task: row.task, blockKind: row.blockKind, blockName: row.blockName,
                   model: row.model, settings: newSettings, refs: row.refs, chainBreak: row.chainBreak,
                   children: row.children, clause: row.clause, tags: row.tags,
                   visitsLeq: row.visitsLeq, onBudget: row.onBudget, declaredSignature: row.declaredSignature)
    }
}
