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
final class EngineCache: @unchecked Sendable {
    /// Builds a stage for an installed model (the app's registry path).
    typealias Builder = @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage

    private struct Entry {
        let key: Key
        let model: ModelEntry
        let stage: any PipelineStage
    }

    private struct Key: Hashable {
        let id: String
        let config: StageConfig
    }

    private let lock = NSLock()
    private var entries: [Key: Entry] = [:]
    /// Most-recently-used first (index 0). Eviction pops from the end.
    private var recency: [Key] = []
    private let budgetBytes: Int64

    /// The app-wide cache. Its budget is fixed from system RAM at first access; the builder
    /// is passed per call (the registry isn't available at static-init time).
    static let shared = EngineCache(budgetBytes: EngineCache.defaultBudget(totalRAMGB: SystemInfo.detect().totalRAMGB))

    init(budgetBytes: Int64) {
        self.budgetBytes = budgetBytes
        Memory.cacheLimit = Int(budgetBytes)
    }

    /// The default budget: `max(4 GB, total RAM × 0.6)`.
    static func defaultBudget(totalRAMGB: Double) -> Int64 {
        Int64(max(4, totalRAMGB * 0.6) * 1_073_741_824)
    }

    /// The engine for `model`/`config`: the cached instance when resident, else built via
    /// `builder` and cached (evicting least-recently-used engines until the budget fits).
    func stage(for model: ModelEntry, config: StageConfig,
               builder: Builder) async throws -> any PipelineStage {
        let key = Key(id: model.id, config: config)
        lock.lock()
        if let entry = entries[key] {
            touchLocked(key)
            lock.unlock()
            return entry.stage
        }
        // Make room for this model's footprint before building. Evict immediately so the
        // running total reflects each removal (deferring them over-evicts).
        let newBytes = Int64(model.ramGB * 1_073_741_824)
        while totalBytesLocked() + newBytes > budgetBytes, let oldest = recency.last {
            recency.removeLast()
            entries[oldest] = nil
        }
        lock.unlock()

        let stage = try await builder(model, config)

        lock.lock()
        entries[key] = Entry(key: key, model: model, stage: stage)
        touchLocked(key)
        lock.unlock()
        return stage
    }

    /// Total cached footprint, in bytes (the catalog's `ramGB` figures, not measured).
    var totalCachedBytes: Int64 {
        lock.lock(); defer { lock.unlock() }
        return totalBytesLocked()
    }

    /// The number of warm engines currently held.
    var count: Int {
        lock.lock(); defer { lock.unlock() }
        return entries.count
    }

    /// Drop every engine (e.g. a "Clear Cache" action) — the next row reloads.
    func clear() {
        lock.lock(); defer { lock.unlock() }
        entries.removeAll()
        recency.removeAll()
    }

    // MARK: - Lock-held helpers

    private func totalBytesLocked() -> Int64 {
        entries.values.reduce(0) { $0 + Int64($1.model.ramGB * 1_073_741_824) }
    }

    private func touchLocked(_ key: Key) {
        recency.removeAll { $0 == key }
        recency.insert(key, at: 0)
    }
}
