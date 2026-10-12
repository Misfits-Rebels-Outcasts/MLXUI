import Foundation
import MLX

/// CFM-R11-2 — the RAM-budgeted engine cache.
///
/// Each model row today builds a **fresh** stage from the registry and releases it at the
/// row boundary (`RealExecutor`'s "sequential load/release"), so a flow like
/// `01-SpokenSummary` reloads Whisper, then Qwen3, then Kokoro — the reload-per-row tax
/// R11-1 measured, and the reason MLX's freed-buffer cache grew to **12.3 GB** (journal
/// `2026-118`). This cache keeps a model's stage (the loaded engine) warm across rows within
/// a byte budget, keyed by `(model id, config)` so a different voice/language/maxTokens
/// build their own stage. An LRU evicts the least-recently-used engine first; each entry's
/// footprint is the catalog's `ramGB` (the same figure the preflight's largest-single-row
/// rule uses).
///
/// **Budget policy** (implementer's call, pending owner confirmation — the owner approved building R11-2 but did not pick the budget): a flat fraction of system RAM —
/// `max(4 GB, totalRAM × 0.6)` — which caps the retained pool that ran to 12.3 GB. The
/// same budget sets MLX's own `Memory.cacheLimit`, so freed buffers from evicted engines
/// don't pile up either. The preflight's largest-single-row rule stays the gate that every
/// engine individually fits RAM; the cache is bounded independently.
///
/// **S1-1b:** the byte accounting now lives in a shared `MemoryBudget` that `ModelContainerPool`
/// also reserves from — one budget for the process. An `EngineCache` built with just
/// `budgetBytes:` gets a *private* budget, which behaves exactly as this cache always did
/// (evict own LRU until it fits); `EngineCache.shared` uses `MemoryBudget.shared`.
nonisolated final class EngineCache: @unchecked Sendable, MemoryBudgetParticipant {
    /// Builds a stage for an installed model (the app's registry path).
    typealias Builder = @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage

    private struct Entry {
        let key: Key
        let model: ModelEntry
        let stage: any PipelineStage
        let bytes: Int64
        let reservation: MemoryBudget.Reservation
    }

    private struct Key: Hashable {
        let id: String
        let config: StageConfig
    }

    private let lock = NSLock()
    private var entries: [Key: Entry] = [:]
    /// Most-recently-used first (index 0). Eviction pops from the end.
    private var recency: [Key] = []
    private let budget: MemoryBudget
    /// S1-1 (T5): when `false`, an LLM stage counts **0** bytes here — `ModelContainerPool` owns
    /// LLM weights, and an LLM stage is now just a closure. Default `true` keeps this cache's
    /// standalone budget policy (and its tests) unchanged; only `shared` opts out.
    private let countsLLMWeights: Bool

    /// `lock`/`unlock` are `noasync` — the SDK's push toward async-safe scoped locking.
    /// Wrapping the critical section in this synchronous helper keeps the lock/unlock pair
    /// out of `stage`'s async frame, which is otherwise held only across synchronous work.
    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }

    /// The app-wide cache. Its budget is fixed from system RAM at first access; the builder
    /// is passed per call (the registry isn't available at static-init time).
    static let shared = EngineCache(budget: .shared, countsLLMWeights: false)

    init(budget: MemoryBudget, countsLLMWeights: Bool = true) {
        self.budget = budget
        self.countsLLMWeights = countsLLMWeights
        Memory.cacheLimit = Int(budget.capacityBytes)
        budget.register(self)
    }

    /// A cache with its own private budget (tests; behaves as the pre-S1-1b cache did).
    convenience init(budgetBytes: Int64, countsLLMWeights: Bool = true) {
        self.init(budget: MemoryBudget(capacityBytes: budgetBytes), countsLLMWeights: countsLLMWeights)
    }

    /// The default budget: `max(4 GB, total RAM × 0.6)`.
    static func defaultBudget(totalRAMGB: Double) -> Int64 {
        Int64(max(4, totalRAMGB * 0.6) * 1_073_741_824)
    }

    /// The engine for `model`/`config`: the cached instance when resident, else built via
    /// `builder` and cached. The footprint is reserved from the shared `MemoryBudget` first —
    /// evicting this cache's LRU engines, then the other cache's, until it fits.
    ///
    /// **Not leased** — see `lease(for:…)`. A caller that keeps using the stage after this returns (a flow row)
    /// must lease it, or a later load may evict the engine's reservation while its weights are still in use.
    func stage(for model: ModelEntry, config: StageConfig,
               builder: Builder) async throws -> any PipelineStage {
        try await acquire(for: model, config: config, maxWait: nil, leased: false, holder: nil, booksWeights: true,
                          builder: builder).stage
    }

    /// SC-2a: `stage(for:)`, leased. The engine is **in use until `hold` is released** (or dropped): LRU eviction
    /// skips it, an explicit `clear()` dooms it (its bytes stay booked until the release), and a build that can
    /// only fit by evicting it waits — at most `maxWait`, then `MemoryBudget.WaitError`.
    func lease(for model: ModelEntry, config: StageConfig, maxWait: Duration = MemoryBudget.defaultMaxWait,
               holder: MemoryBudget.HolderKind? = .flow, booksWeights: Bool = true,
               builder: Builder) async throws -> (stage: any PipelineStage, hold: MemoryBudget.Hold) {
        let got = try await acquire(for: model, config: config, maxWait: maxWait, leased: true, holder: holder,
                                    booksWeights: booksWeights, builder: builder)
        return (got.stage, got.hold ?? .inert)
    }

    /// The stage for a flow row: leased, with the lease riding on the returned stage and returned when the row lets
    /// it go (its last use). A stage that books no bytes here (an LLM stage — its weights are leased in the pool by
    /// `LLMEngine`) comes back unwrapped.
    func leasedStage(for model: ModelEntry, config: StageConfig, maxWait: Duration = MemoryBudget.defaultMaxWait,
                     holder: MemoryBudget.HolderKind? = .flow, booksWeights: Bool = true,
                     builder: Builder) async throws -> any PipelineStage {
        let (stage, hold) = try await lease(for: model, config: config, maxWait: maxWait, holder: holder,
                                            booksWeights: booksWeights, builder: builder)
        return hold.isInert ? stage : LeasedStage(inner: stage, hold: hold)
    }

    private func acquire(for model: ModelEntry, config: StageConfig, maxWait: Duration?, leased: Bool,
                         holder: MemoryBudget.HolderKind?, booksWeights: Bool,
                         builder: Builder) async throws -> (stage: any PipelineStage, hold: MemoryBudget.Hold?) {
        let key = Key(id: model.id, config: config)
        if let cached = withLock({ () -> (stage: any PipelineStage, hold: MemoryBudget.Hold?)? in
            guard let entry = entries[key] else { return nil }
            touchLocked(key)
            return (entry.stage, leased ? budget.hold(entry.reservation, label: model.id, holder: holder) : nil)
        }) {
            return cached
        }

        // Make room *before* building (peak memory must not briefly hold both).
        let bytes = booksWeights ? footprint(of: model) : 0
        let reservation: MemoryBudget.Reservation
        if let maxWait {
            reservation = try await budget.reserveBounded(bytes: bytes, for: self, label: model.id, maxWait: maxWait)
        } else {
            reservation = await budget.reserve(bytes: bytes, for: self, label: model.id)
        }

        let stage: any PipelineStage
        do {
            stage = try await builder(model, config)
        } catch {
            budget.release(reservation)
            throw error
        }

        let (result, hold, wasDuplicate) = withLock { () -> (any PipelineStage, MemoryBudget.Hold?, Bool) in
            // Two concurrent callers for one key both build; keep the first, drop the second.
            if let existing = entries[key] {
                touchLocked(key)
                return (existing.stage, leased ? budget.hold(existing.reservation, label: model.id, holder: holder) : nil, true)
            }
            entries[key] = Entry(key: key, model: model, stage: stage, bytes: bytes, reservation: reservation)
            touchLocked(key)
            // held before it is committed (evictable), in the same critical section
            return (stage, leased ? budget.hold(reservation, label: model.id, holder: holder) : nil, false)
        }
        if wasDuplicate { budget.release(reservation) } else { budget.commit(reservation) }
        return (result, hold)
    }

    /// `MemoryBudgetParticipant`: evict the least-recently-used engine that holds memory **and is not in use**
    /// (SC-2a).
    ///
    /// The ledger release is in the **same critical section** as the removal (MB-1): released afterwards, a
    /// concurrent requester could ask this cache in the gap, hear "nothing to evict", see the bytes still booked
    /// and nothing in flight, and take the "proceed over budget" exit. (Lock order is this cache's → the budget's;
    /// the budget never calls a participant while holding its own lock.)
    func evictLeastRecentlyUsed() async -> Bool {
        withLock {
            for key in recency.reversed() {
                guard let entry = entries[key], entry.bytes > 0 else { continue }
                guard budget.releaseIfUnheld(entry.reservation) else { continue }   // in use: not ours to take
                entries.removeValue(forKey: key)
                recency.removeAll { $0 == key }
                return true
            }
            return false
        }
    }

    /// Total cached footprint, in bytes (the catalog's `ramGB` figures, not measured).
    var totalCachedBytes: Int64 {
        withLock { totalBytesLocked() }
    }

    /// The number of warm engines currently held.
    var count: Int {
        withLock { entries.count }
    }

    /// Drop every engine (e.g. a "Clear Cache" action) — the next row reloads.
    func clear() {
        withLock {
            let held = entries.values.map(\.reservation)
            entries.removeAll()
            recency.removeAll()
            held.forEach { budget.releaseOrDefer($0) }   // same critical section as the removal (see evictLeastRecentlyUsed); an engine in use is doomed
        }
    }

    // MARK: - Lock-held helpers

    private func footprint(of model: ModelEntry) -> Int64 {
        if !countsLLMWeights && model.runnerKind == .llm { return 0 }
        return Int64(model.ramGB * 1_073_741_824)
    }

    private func totalBytesLocked() -> Int64 {
        entries.values.reduce(0) { $0 + $1.bytes }
    }

    private func touchLocked(_ key: Key) {
        recency.removeAll { $0 == key }
        recency.insert(key, at: 0)
    }
}

/// SC-2a — a cached stage with its lease riding on it. The lease is returned when the wrapper is dropped, i.e. when
/// the flow row that asked for the stage is done with it (the hold's deinit; `release()` is explicit and idempotent).
nonisolated struct LeasedStage: PipelineStage {
    let inner: any PipelineStage
    let hold: MemoryBudget.Hold

    var id: String { inner.id }
    var name: String { inner.name }
    var accepts: MediaKind { inner.accepts }
    var produces: MediaKind { inner.produces }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try await inner.run(input, progress: progress)
    }
}

// MARK: - SC-2b — Run windows, and the engines the cache doesn't own

/// Marks a `ModelSDK` whose stage is a thin closure over an engine that **books its own weights** in the shared
/// budget when it loads (the VLM pool, the Laya and CLM lease caches). The cache must then book those stages at
/// 0 — as it already does for an LLM stage — or the same weights would be charged twice.
protocol StageBooksOwnWeights {}

/// SC-2b — a stage handed to a **Run window** (the registry-driven sheet for a non-LLM model). It leases the
/// model's engine from `EngineCache` (`leasedStage`'s lease, held here) and returns it when the window closes
/// (`release()`), on the rebuild that replaces it (`rebuilding`), or when the value is dropped (the hold's
/// deinit — the safety net). A Run window and a flow row for the same model and config share one engine: both
/// go through the same cache key.
///
/// The lease is taken once, lazily and memoised: the sheet starts it as soon as it opens (`prepare()`), and a
/// run that arrives meanwhile joins it. A refusal (the wait ran out) is not swallowed: the next `run` throws the
/// `MemoryBudget.WaitError`, whose sentence the Run view's existing error bar already shows.
nonisolated final class LeasingStage: PipelineStage, @unchecked Sendable {
    let model: ModelEntry
    let config: StageConfig
    private let cache: EngineCache
    private let probe: any PipelineStage
    private let booksWeights: Bool
    private let maxWait: Duration
    private let lock = NSLock()
    private var acquiring: Task<any PipelineStage, Error>?
    private var hold: MemoryBudget.Hold?

    /// `probe` is the stage the SDK built for `model`/`config` (cheap: engines load on first run). It gives this
    /// stage its identity now, and is the builder's result if the cache doesn't already hold one.
    init(model: ModelEntry, config: StageConfig, probe: any PipelineStage, booksWeights: Bool = true,
         cache: EngineCache = .shared, maxWait: Duration = MemoryBudget.defaultMaxWait) {
        self.model = model
        self.config = config
        self.probe = probe
        self.booksWeights = booksWeights
        self.cache = cache
        self.maxWait = maxWait
    }

    var id: String { probe.id }
    var name: String { probe.name }
    var accepts: MediaKind { probe.accepts }
    var produces: MediaKind { probe.produces }

    /// Reserve and lease the engine now. Returns the refusal, if any; `run` will raise the same.
    @discardableResult
    func prepare() async -> Error? {
        do { _ = try await engine(); return nil } catch { return error }
    }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try await engine().run(input, progress: progress)
    }

    /// Return the lease (idempotent). A later `run` leases again.
    func release() {
        lock.lock()
        let held = hold
        hold = nil
        acquiring = nil
        lock.unlock()
        held?.release()
    }

    /// Whether this stage currently holds its engine.
    var isLeased: Bool { lock.lock(); defer { lock.unlock() }; return hold != nil }

    private func engine() async throws -> any PipelineStage {
        let task: Task<any PipelineStage, Error> = {
            lock.lock(); defer { lock.unlock() }
            if let acquiring { return acquiring }
            let probe = self.probe, model = self.model, config = self.config, cache = self.cache
            let books = booksWeights, wait = maxWait
            let task = Task<any PipelineStage, Error> { [weak self] in
                let (stage, hold) = try await cache.lease(for: model, config: config, maxWait: wait,
                                                          holder: .runWindow, booksWeights: books) { _, _ in probe }
                guard let self else { hold.release(); return stage }
                self.lock.lock(); self.hold = hold; self.lock.unlock()
                return stage
            }
            acquiring = task
            return task
        }()
        do { return try await task.value } catch {
            lock.lock(); if acquiring == task { acquiring = nil }; lock.unlock()   // a later run may succeed
            throw error
        }
    }

    /// A Run view's per-run rebuild (OCR prompt, Whisper language): the replacement leases its own engine
    /// (config is part of the cache key); the stage it replaces returns its lease first. When `stage` is not a
    /// `LeasingStage` (a test, a non-sheet caller) this is the SDK's plain `makeStage`.
    static func rebuilding(_ stage: any PipelineStage, model: ModelEntry, sdk: (any ModelSDK)?,
                           config: StageConfig) -> (any PipelineStage)? {
        guard let sdk, let fresh = try? sdk.makeStage(for: model, config: config) else { return nil }
        guard let old = stage as? LeasingStage else { return fresh }
        old.release()
        return LeasingStage(model: model, config: config, probe: fresh, booksWeights: old.booksWeights,
                            cache: old.cache, maxWait: old.maxWait)
    }
}

/// SC-2b — a per-directory cache of loaded engines on the shared budget, for the engines that are not LLM
/// containers (Laya, CLM). Real bytes (the safetensors under the directory × the measured load factor), a lease
/// per use, held entries skipped by eviction — the same `ResidentPool` the LLM weights live in, so every load is
/// on one ledger.
actor EngineLeaseCache<Engine: Sendable> {
    typealias Lease = ResidentPool<Engine>.Lease
    private let pool: ResidentPool<Engine>
    private let load: @Sendable (URL) async throws -> Engine
    nonisolated let budget: MemoryBudget

    init(budget: MemoryBudget = .shared, load: @escaping @Sendable (URL) async throws -> Engine) {
        self.budget = budget
        self.pool = ResidentPool(budget: budget)
        self.load = load
    }

    /// What loading `directory` books: every safetensors file under it (an engine may keep its encoder and heads in
    /// subdirectories) × `ModelWeightCharge.loadFactor`; the directory's whole size when it has none.
    nonisolated static func charge(directory: URL) -> Int64 {
        var onDisk: Int64 = 0
        if let walker = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.fileSizeKey]) {
            for case let url as URL in walker where url.pathExtension == "safetensors" {
                onDisk += Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            }
        }
        return onDisk > 0 ? ModelWeightCharge.charge(onDiskBytes: onDisk) : Int64(ModelStore.directorySize(at: directory))
    }

    /// The engine, **leased**: in use until `release()`. Eviction skips it; a load that can only fit by evicting it
    /// waits (`maxWait`) and then throws `MemoryBudget.WaitError`.
    func lease(for directory: URL, holder: MemoryBudget.HolderKind? = nil,
               maxWait: Duration = MemoryBudget.defaultMaxWait) async throws -> Lease {
        let load = self.load
        return try await pool.lease(for: directory.path, bytes: Self.charge(directory: directory),
                                    maxWait: maxWait, holder: holder) { try await load(directory) }
    }

    /// Unleased (tests, callers that only need the load to have happened).
    func engine(for directory: URL) async throws -> Engine {
        let load = self.load
        return try await pool.value(for: directory.path, bytes: Self.charge(directory: directory)) { try await load(directory) }
    }

    var residentKeys: [String] { get async { await pool.residentKeys } }
    func evict(directory: URL) async { await pool.evict(key: directory.path) }
    func clear() async { await pool.clear() }
}
