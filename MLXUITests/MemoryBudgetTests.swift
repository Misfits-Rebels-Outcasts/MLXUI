import Testing
import Foundation
@testable import MLXUI

/// S1-1b — one shared `MemoryBudget` for `ModelContainerPool` (here its generic core,
/// `ResidentPool`) and `EngineCache`: own LRU first, then the other cache's LRU, never over
/// budget under concurrency, and the combined counter the flow screens show.
struct MemoryBudgetTests {

    private static let gb = Int64(1_073_741_824)

    private final class Box: Sendable { let name: String; init(_ n: String) { name = n } }
    private struct Boom: Error {}

    private nonisolated struct NoopStage: PipelineStage {
        let id = "noop", name = "Noop"
        var accepts: MediaKind { .text }
        var produces: MediaKind { .text }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media { input }
    }

    /// Highest `usedBytes` seen from inside loaders/builders (i.e. at the moments a reservation
    /// has just been granted).
    private actor Peak {
        private(set) var value: Int64 = 0
        func note(_ v: Int64) { value = max(value, v) }
    }

    private func engine(_ id: String, _ gb: Double, type: ModelType = .asr) -> ModelEntry {
        makeEntry(id: id, modelType: type, ramGB: gb)
    }

    private func put(_ cache: EngineCache, _ model: ModelEntry, peak: Peak? = nil,
                     budget: MemoryBudget? = nil, delayMS: Int = 0) async throws {
        _ = try await cache.stage(for: model, config: .default) { _, _ in
            if let peak, let budget { await peak.note(budget.usedBytes) }
            if delayMS > 0 { try await Task.sleep(for: .milliseconds(delayMS)) }
            return NoopStage()
        }
    }

    private func load(_ pool: ResidentPool<Box>, _ key: String, _ gb: Int64, peak: Peak? = nil,
                      budget: MemoryBudget? = nil, delayMS: Int = 0) async throws {
        _ = try await pool.value(for: key, bytes: gb * Self.gb) {
            if let peak, let budget { await peak.note(budget.usedBytes) }
            if delayMS > 0 { try await Task.sleep(for: .milliseconds(delayMS)) }
            return Box(key)
        }
    }

    // MARK: - The ruling

    @Test func aPoolAndCacheMixNeverExceedsTheSharedBudget() async throws {
        let budget = MemoryBudget(capacityBytes: 10 * Self.gb)
        let pool = ResidentPool<Box>(budget: budget)
        let cache = EngineCache(budget: budget)
        let peak = Peak()
        // 40 concurrent requests of 3–5 GB, alternating pool/cache, overlapping builds.
        try await withThrowingTaskGroup(of: Void.self) { group in
            for i in 0..<40 {
                let size = Int64(3 + (i * 7) % 3)               // 3, 4, 5, 3, …
                if i % 2 == 0 {
                    group.addTask { try await self.load(pool, "p\(i % 6)", size, peak: peak, budget: budget, delayMS: 5) }
                } else {
                    group.addTask { try await self.put(cache, self.engine("c\(i % 6)", Double(size)), peak: peak, budget: budget, delayMS: 5) }
                }
            }
            try await group.waitForAll()
        }
        #expect(await peak.value <= 10 * Self.gb)
        #expect(budget.usedBytes <= 10 * Self.gb)
        // the ledger agrees with what each side actually holds
        #expect(budget.usedBytes == (await pool.totalBytes) + cache.totalCachedBytes)
    }

    @Test func aPoolRequestMakesTheEngineCacheEvict() async throws {
        let budget = MemoryBudget(capacityBytes: 8 * Self.gb)
        let pool = ResidentPool<Box>(budget: budget)
        let cache = EngineCache(budget: budget)
        try await put(cache, engine("x", 5))
        try await load(pool, "llm", 6)
        #expect(cache.count == 0)
        #expect(await pool.residentKeys == ["llm"])
        #expect(budget.usedBytes == 6 * Self.gb)
    }

    @Test func anEngineCacheRequestMakesThePoolEvict() async throws {
        let budget = MemoryBudget(capacityBytes: 8 * Self.gb)
        let pool = ResidentPool<Box>(budget: budget)
        let cache = EngineCache(budget: budget)
        try await load(pool, "llm", 5)
        try await put(cache, engine("x", 6))
        #expect(await pool.count == 0)
        #expect(cache.count == 1)
        #expect(budget.usedBytes == 6 * Self.gb)
    }

    @Test func theRequestersOwnLRUGoesBeforeTheOtherCaches() async throws {
        let budget = MemoryBudget(capacityBytes: 10 * Self.gb)
        let pool = ResidentPool<Box>(budget: budget)
        let cache = EngineCache(budget: budget)
        try await load(pool, "a", 3)        // pool's own, oldest
        try await put(cache, engine("x", 3))
        try await load(pool, "b", 5)        // 3+3+5 = 11 > 10 → own LRU "a" goes; the cache is spared
        #expect(await pool.residentKeys == ["b"])
        #expect(cache.count == 1)
    }

    @Test func theOtherCacheIsAskedLeastRecentlyUsedFirstAndOnlyAsMuchAsNeeded() async throws {
        let budget = MemoryBudget(capacityBytes: 8 * Self.gb)
        let pool = ResidentPool<Box>(budget: budget)
        let cache = EngineCache(budget: budget)
        try await put(cache, engine("old", 5))
        try await put(cache, engine("new", 2))
        try await load(pool, "llm", 4)      // pool empty; 7+4 = 11 > 8 → evict "old" (LRU) → 2+4 = 6 fits
        #expect(cache.count == 1)
        #expect(cache.totalCachedBytes == 2 * Self.gb)   // "new" survived
        #expect(budget.usedBytes == 6 * Self.gb)
    }

    @Test func bothEmptyProceedsOverBudgetAsATodayLikeFallback() async throws {
        let budget = MemoryBudget(capacityBytes: 4 * Self.gb)
        let pool = ResidentPool<Box>(budget: budget)
        try await load(pool, "huge", 9)
        #expect(await pool.residentKeys == ["huge"])
        #expect(budget.usedBytes == 9 * Self.gb)         // over budget, loaded anyway (and logged)
    }

    @Test func aRequestWaitsForAnInFlightLoadInsteadOfOvershooting() async throws {
        // 10 GB. A 6 GB load is in flight (not evictable); a 6 GB cache build arrives. Nothing is
        // evictable yet, so it must wait for the load to land, then evict it — never 12 GB.
        let budget = MemoryBudget(capacityBytes: 10 * Self.gb)
        let pool = ResidentPool<Box>(budget: budget)
        let cache = EngineCache(budget: budget)
        let peak = Peak()
        async let slow: Void = load(pool, "slow", 6, peak: peak, budget: budget, delayMS: 150)
        try await Task.sleep(for: .milliseconds(30))
        try await put(cache, engine("x", 6), peak: peak, budget: budget)
        try await slow
        #expect(await peak.value <= 10 * Self.gb)
        #expect(await pool.count == 0)
        #expect(cache.count == 1)
        #expect(budget.usedBytes == 6 * Self.gb)
    }

    // MARK: - Bookkeeping

    @Test func failedLoadsAndBuildsReleaseTheirReservation() async throws {
        let budget = MemoryBudget(capacityBytes: 10 * Self.gb)
        let pool = ResidentPool<Box>(budget: budget)
        let cache = EngineCache(budget: budget)
        await #expect(throws: Boom.self) {
            _ = try await pool.value(for: "p", bytes: 4 * Self.gb) { throw Boom() }
        }
        await #expect(throws: Boom.self) {
            _ = try await cache.stage(for: self.engine("c", 4), config: .default) { _, _ in throw Boom() }
        }
        #expect(budget.usedBytes == 0)
        #expect(budget.heldCount == 0)
    }

    @Test func evictAndClearReleaseTheirReservations() async throws {
        let budget = MemoryBudget(capacityBytes: 20 * Self.gb)
        let pool = ResidentPool<Box>(budget: budget)
        let cache = EngineCache(budget: budget)
        try await load(pool, "a", 3)
        try await load(pool, "b", 3)
        try await put(cache, engine("x", 4))
        #expect(budget.usedBytes == 10 * Self.gb)
        await pool.evict(key: "a")
        #expect(budget.usedBytes == 7 * Self.gb)
        await pool.clear()
        cache.clear()
        #expect(budget.usedBytes == 0)
    }

    @Test func zeroByteLLMStagesAreNotEvictedForRoom() async throws {
        let budget = MemoryBudget(capacityBytes: 6 * Self.gb)
        let pool = ResidentPool<Box>(budget: budget)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)   // the app's policy
        try await put(cache, engine("llm-stage", 8, type: .llm))           // counts 0 bytes
        try await put(cache, engine("asr", 4))
        try await load(pool, "p", 5)         // needs room: the ASR goes, the free LLM stage stays
        #expect(cache.count == 1)
        #expect(cache.totalCachedBytes == 0)
        #expect(budget.usedBytes == 5 * Self.gb)
    }

    @Test func theCombinedCounterIsPoolPlusCache() async throws {
        let budget = MemoryBudget(capacityBytes: 20 * Self.gb)
        let pool = ResidentPool<Box>(budget: budget)
        let cache = EngineCache(budget: budget)
        try await load(pool, "llm", 3)
        try await put(cache, engine("whisper", 4))
        #expect(budget.usedBytes == 7 * Self.gb)
        #expect(budget.heldCount == 2)
    }

    @Test func aPrivateBudgetBehavesLikeTheOldStandaloneCache() async throws {
        let cache = EngineCache(budgetBytes: 10 * Self.gb)
        try await put(cache, engine("m1", 6, type: .llm))
        try await put(cache, engine("m2", 6, type: .llm))
        #expect(cache.count == 1)               // 6 + 6 > 10 → m1 evicted, as before
        #expect(cache.totalCachedBytes == 6 * Self.gb)
    }
}
