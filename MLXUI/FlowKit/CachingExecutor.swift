import Foundation

/// Wraps any `FlowExecutor` with the content-addressed cache (CFM-R3-2), mirroring
/// `core/cache.py::CachingExecutor` for the linear subset: applies the `_with_seed`
/// substitution, decides skip-vs-key per row, checks the store, and serves a hit instead of
/// running the inner executor. A hit is reported via `lastCacheHit` so the runner can emit
/// `.cacheHit` (the Python's duck-typed `last_hit` convention). A reference type so the hit
/// flag's mutation is visible to the runner through the `FlowExecutor` existential.
///
/// **CACHE-SIGNALS-1.** Every `last*` signal `FlowInterpreter` reads right after `execute`
/// (`lastTag`, `lastTimeoutFlag`, `lastProviderDeciderFlag`, `lastStaged`,
/// `lastDeciderDetail` — `lastCacheHit` is this wrapper's own signal, not one it forwards)
/// used to fall through to `FlowExecutor`'s protocol-default `nil`, because this class never
/// overrode them: a real run always wraps `RealExecutor` in this class
/// (`AppFlowExecutorFactory.cachingContext`), so a decider row's fired tag never reached
/// `advance()`'s `.decide` clause — the interpreter silently ended the block right after the
/// decider, on *every* real run, hit or miss (root cause of the `77-TicketRouter` smoke-97
/// "Classify completes, nothing downstream ever fires" report; not Laya-specific — any
/// decider-routed flow run through the real UI was affected). `cache.py:655-983` is the
/// reference for the fix below, cited at each divergent line:
/// - **On a miss, or the `_skip_cache` bypass** (`cache.py:939-950`, `:966-975`): copy every
///   `last*` from `inner` via its own protocol getter (Swift has no `getattr(..., default)`,
///   but every conformer now must implement each property explicitly — see `FlowRunner.swift`
///   — so there is no missing-attribute case to default around).
/// - **On a hit** (`cache.py:960-965`): every signal resets to "nothing ran" (`nil`/`false`)
///   **except** `lastTag`, recovered from `store.getTag(key:)` — `cache.py:669-676`: "a real
///   decider's fired tag is part of what a cache hit must reproduce, not just its payload."
///   No other signal has a reference-side recovery path (`Ask Human`/`Human Input`/
///   `Stage Send`/`Stage Post` are all `NEVER_CACHE`, so they never reach the hit branch to
///   begin with; `lastDeciderDetail` has no Python analogue and no cache-side storage for it).
/// - **The realism-gated decider rule** (`cache.py:901-912`, `_skip_cache`): a decider skips
///   caching entirely unless its own realism is `"real"`. `AppFlowExecutorFactory.cachingContext`
///   always passes `cacheTier: "real"`, so this is currently a no-op for the shipped app, but
///   ported anyway rather than assumed away — a future `"mock"`/`"hybrid"` caller must not
///   silently start caching a decider's visit-count-driven tag.
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

    private let signalLock = NSLock()
    private var _lastCacheHit = false
    private var _lastTag: String?
    private var _lastTimeoutFlag: (code: String, message: String)?
    private var _lastProviderDeciderFlag: (code: String, message: String)?
    private var _lastStaged: (id: String, kind: String, summary: String)?
    private var _lastDeciderDetail: (
        tag: String, confidence: Double, probabilities: [(label: String, probability: Double)],
        expectedLevel: Double?, stateTruncated: Bool
    )?

    var lastCacheHit: Bool { signalLock.withLock { _lastCacheHit } }
    var lastTag: String? { signalLock.withLock { _lastTag } }
    var lastTimeoutFlag: (code: String, message: String)? { signalLock.withLock { _lastTimeoutFlag } }
    var lastProviderDeciderFlag: (code: String, message: String)? { signalLock.withLock { _lastProviderDeciderFlag } }
    var lastStaged: (id: String, kind: String, summary: String)? { signalLock.withLock { _lastStaged } }
    var lastDeciderDetail: (
        tag: String, confidence: Double, probabilities: [(label: String, probability: Double)],
        expectedLevel: Double?, stateTruncated: Bool
    )? { signalLock.withLock { _lastDeciderDetail } }

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

    /// A decider skips the cache lookup/store entirely unless this row's own realism is
    /// genuinely `"real"` (`cache.py:901-912`) — under `"mock"`/`"hybrid"` a decider's fired
    /// tag is driven by visit count, not `(task, model, settings, inputs)` alone, so caching
    /// it would silently replay a stale tag.
    private func skipCache(_ row: Row) -> Bool {
        guard let task = row.task else { return false }
        if CacheKey.neverCache.contains(task) { return true }
        if TaskCatalog.deciderTasks[task] != nil, cacheTier != "real" { return true }
        return false
    }

    func execute(path: String, row: Row, inputs: [Asset],
                 transcript: [FlowInterpreter.TranscriptEntry]?,
                 context: [(label: String, content: String)]?,
                 usedFlowContent: String?) async throws -> Asset {
        // Reset every signal before this activation runs, so a stale value from a previous
        // row (or a previous item of the same row, inside an `<each>`) can never leak forward.
        signalLock.withLock {
            _lastCacheHit = false
            _lastTag = nil
            _lastTimeoutFlag = nil
            _lastProviderDeciderFlag = nil
            _lastStaged = nil
            _lastDeciderDetail = nil
        }

        // _with_seed: substitute a concrete seed into a stochastic row's settings first, so
        // the derived number is what the cache key and engine call both see.
        let seededRow = try withSeed(path: path, row: row)

        // NEVER_CACHE (writes) and the realism-gated decider rule always skip the cache —
        // a hit would silently skip the real write, or replay a stale decider tag.
        guard !skipCache(seededRow) else {
            let asset = try await inner.execute(path: path, row: seededRow, inputs: inputs,
                                                transcript: transcript, context: context,
                                                usedFlowContent: usedFlowContent)
            copySignals(from: inner)
            return asset
        }

        let key = try cacheKey(for: seededRow, inputs: inputs,
                               transcript: transcript, context: context,
                               usedFlowContent: usedFlowContent)
        let blobDir = store.root.appendingPathComponent("materialized", isDirectory: true)
        if let cached = try store.get(key: key, blobDirectory: blobDir, path: path) {
            // cache.py:960-965 — every signal stays reset to "nothing ran" except `lastTag`,
            // recovered from the entry itself: a decider's fired tag is part of what a cache
            // hit must reproduce, not just its payload.
            signalLock.withLock {
                _lastCacheHit = true
                _lastTag = store.getTag(key: key)
            }
            return cached
        }
        let asset = try await inner.execute(path: path, row: seededRow, inputs: inputs,
                                            transcript: transcript, context: context,
                                            usedFlowContent: usedFlowContent)
        copySignals(from: inner)
        try? store.put(key: key, asset: asset, tag: lastTag)
        try? store.evictIfNeeded(byteBudget: 1_000_000_000)   // 1 GB soft budget
        return asset
    }

    /// Copy every per-execution status signal from `inner` after it actually ran — the miss
    /// and `skipCache` bypass paths both take this (`cache.py:945-950`, `:970-975`).
    private func copySignals(from inner: any FlowExecutor) {
        signalLock.withLock {
            _lastTag = inner.lastTag
            _lastTimeoutFlag = inner.lastTimeoutFlag
            _lastProviderDeciderFlag = inner.lastProviderDeciderFlag
            _lastStaged = inner.lastStaged
            _lastDeciderDetail = inner.lastDeciderDetail
        }
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
            if case .runnable(let slot, _, _) = CatalogBridge.resolve(display, catalog: catalog),
               let entry = slot.modelEntry {
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
