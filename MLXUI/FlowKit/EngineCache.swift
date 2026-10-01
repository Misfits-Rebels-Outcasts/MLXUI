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
    func stage(for model: ModelEntry, config: StageConfig,
               builder: Builder) async throws -> any PipelineStage {
        let key = Key(id: model.id, config: config)
        if let cached = withLock({ () -> (any PipelineStage)? in
            guard let entry = entries[key] else { return nil }
            touchLocked(key)
            return entry.stage
        }) {
            return cached
        }

        // Make room *before* building (peak memory must not briefly hold both).
        let bytes = footprint(of: model)
        let reservation = await budget.reserve(bytes: bytes, for: self, label: model.id)

        let stage: any PipelineStage
        do {
            stage = try await builder(model, config)
        } catch {
            budget.release(reservation)
            throw error
        }

        let (result, wasDuplicate) = withLock { () -> (any PipelineStage, Bool) in
            // Two concurrent callers for one key both build; keep the first, drop the second.
            if let existing = entries[key] {
                touchLocked(key)
                return (existing.stage, true)
            }
            entries[key] = Entry(key: key, model: model, stage: stage, bytes: bytes, reservation: reservation)
            touchLocked(key)
            return (stage, false)
        }
        if wasDuplicate { budget.release(reservation) } else { budget.commit(reservation) }
        return result
    }

    /// `MemoryBudgetParticipant`: evict the least-recently-used engine that holds memory.
    func evictLeastRecentlyUsed() async -> Bool {
        let evicted: MemoryBudget.Reservation? = withLock {
            guard let key = recency.last(where: { (entries[$0]?.bytes ?? 0) > 0 }),
                  let entry = entries.removeValue(forKey: key) else { return nil }
            recency.removeAll { $0 == key }
            return entry.reservation
        }
        guard let evicted else { return false }
        budget.release(evicted)
        return true
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
        let reservations: [MemoryBudget.Reservation] = withLock {
            let held = entries.values.map(\.reservation)
            entries.removeAll()
            recency.removeAll()
            return held
        }
        reservations.forEach { budget.release($0) }
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
