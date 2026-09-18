import Testing
import Foundation
@testable import MLXUI

/// CFM-R11-2 — the RAM-budgeted engine cache: same (model, config) reuses the warm engine,
/// different configs are distinct engines, and the budget evicts the least-recently-used
/// one first. The footprint numbers are the catalog's `ramGB` (the preflight's figure).
struct CatFlowEngineCacheTests {

    private nonisolated struct CountingStage: PipelineStage {
        let id = "counting"
        let name = "Counting"
        var accepts: MediaKind { .text }
        var produces: MediaKind { .text }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
            try require(input, .text)
            progress(1.0)
            return input
        }
    }

    private static let bigBudget = Int64(20 * 1_073_741_824)

    @Test func sameKeyReusesTheWarmEngine() async throws {
        var builds = 0
        let cache = EngineCache(budgetBytes: Self.bigBudget)
        let model = makeEntry(id: "m1", ramGB: 6)
        let a = try await cache.stage(for: model, config: .default) { _, _ in
            builds += 1
            return CountingStage()
        }
        let b = try await cache.stage(for: model, config: .default) { _, _ in
            builds += 1
            return CountingStage()
        }
        #expect(builds == 1)          // built once, reused once
        #expect(a.id == b.id)
    }

    @Test func differentConfigIsADistinctEngine() async throws {
        var builds = 0
        let cache = EngineCache(budgetBytes: Self.bigBudget)
        let model = makeEntry(id: "m1", ramGB: 6)
        _ = try await cache.stage(for: model, config: StageConfig(voice: "af_heart")) { _, _ in
            builds += 1
            return CountingStage()
        }
        _ = try await cache.stage(for: model, config: StageConfig(voice: "bm_george")) { _, _ in
            builds += 1
            return CountingStage()
        }
        #expect(builds == 2)
        #expect(cache.count == 2)
    }

    @Test func budgetEvictsTheOverflowEngine() async throws {
        // 10 GB budget, two 6 GB models can't both fit.
        let cache = EngineCache(budgetBytes: Int64(10 * 1_073_741_824))
        let m1 = makeEntry(id: "m1", ramGB: 6)
        let m2 = makeEntry(id: "m2", ramGB: 6)
        var builds: [String] = []
        let builder = { (model: ModelEntry, _: StageConfig) in
            builds.append(model.id)
            return CountingStage()
        }
        _ = try await cache.stage(for: m1, config: .default, builder: builder)
        _ = try await cache.stage(for: m2, config: .default, builder: builder)
        #expect(cache.count == 1)
        #expect(builds == ["m1", "m2"])
        // Re-requesting m1 rebuilds — m2 evicted it.
        _ = try await cache.stage(for: m1, config: .default, builder: builder)
        #expect(builds == ["m1", "m2", "m1"])
    }

    @Test func touchingKeepsTheWarmEngine() async throws {
        // 12 GB budget fits two 6 GB engines; a third evicts the least-recently-used.
        let cache = EngineCache(budgetBytes: Int64(12 * 1_073_741_824))
        let m1 = makeEntry(id: "m1", ramGB: 6)
        let m2 = makeEntry(id: "m2", ramGB: 6)
        var builds: [String] = []
        let builder = { (model: ModelEntry, _: StageConfig) in
            builds.append(model.id)
            return CountingStage()
        }
        _ = try await cache.stage(for: m1, config: .default, builder: builder)
        _ = try await cache.stage(for: m2, config: .default, builder: builder)
        _ = try await cache.stage(for: m2, config: .default, builder: builder)   // touch m2
        let m3 = makeEntry(id: "m3", ramGB: 6)
        _ = try await cache.stage(for: m3, config: .default, builder: builder)   // evicts m1
        #expect(cache.count == 2)
        // m1 was evicted: requesting it again rebuilds.
        _ = try await cache.stage(for: m1, config: .default, builder: builder)
        #expect(builds == ["m1", "m2", "m3", "m1"])
    }

    @Test func defaultBudgetIsAFractionOfRAMWithAFloor() {
        let gb = Double(1_073_741_824)
        #expect(EngineCache.defaultBudget(totalRAMGB: 16) == Int64(16 * 0.6 * gb))
        // Floor of 4 GB for tiny-RAM machines.
        #expect(EngineCache.defaultBudget(totalRAMGB: 2) == Int64(4 * gb))
    }
}
