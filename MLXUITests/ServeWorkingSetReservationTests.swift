import Testing
import Foundation
@testable import MLXUI

/// S3-8B — a generation's working set (KV, prefill buffers) is reserved *for real*: other resident models are
/// evicted for it before the prefill, and a request that still cannot fit is refused rather than run over budget.
struct ServeWorkingSetReservationTests {

    private static let gb = Int64(1_073_741_824)
    private final class Box: Sendable { let name: String; init(_ n: String) { name = n } }

    private func load(_ pool: ResidentPool<Box>, _ key: String, _ gb: Int64) async throws {
        _ = try await pool.value(for: key, bytes: gb * Self.gb) { Box(key) }
    }

    @Test func otherResidentsAreEvictedUntilTheWorkingSetFits() async throws {
        let budget = MemoryBudget(capacityBytes: 10 * Self.gb)
        let pool = ResidentPool<Box>(budget: budget)
        try await load(pool, "other", 5)
        try await load(pool, "running", 5)                  // MRU: the model that is generating
        #expect(budget.usedBytes == 10 * Self.gb)           // both fit as weights; no KV room at all

        let reservation = await pool.reserveWorkingSet(totalNeedBytes: 8 * Self.gb, protecting: "running")
        #expect(reservation != nil)
        #expect(await pool.residentKeys == ["running"])     // the other model was evicted for the KV
        #expect(budget.usedBytes == 8 * Self.gb)
    }

    @Test func theRunningModelIsNeverEvictedForItsOwnWorkingSet() async throws {
        let budget = MemoryBudget(capacityBytes: 10 * Self.gb)
        let pool = ResidentPool<Box>(budget: budget)
        try await load(pool, "other", 3)
        try await load(pool, "running", 6)

        // 6 GB of weights + 5 GB of KV cannot fit in 10 GB however much is evicted → refuse (nil), and the
        // running model is still resident.
        let reservation = await pool.reserveWorkingSet(totalNeedBytes: 11 * Self.gb, protecting: "running")
        #expect(reservation == nil)
        #expect(await pool.residentKeys == ["running"])
        #expect(budget.usedBytes == 6 * Self.gb)
    }

    @Test func aWorkingSetThatAlreadyFitsEvictsNothing() async throws {
        let budget = MemoryBudget(capacityBytes: 10 * Self.gb)
        let pool = ResidentPool<Box>(budget: budget)
        try await load(pool, "other", 3)
        try await load(pool, "running", 4)
        let reservation = await pool.reserveWorkingSet(totalNeedBytes: 6 * Self.gb, protecting: "running")
        #expect(reservation != nil)
        #expect(Set(await pool.residentKeys) == ["other", "running"])
    }

    @Test func releasingTheReservationGivesTheBytesBack() async throws {
        let budget = MemoryBudget(capacityBytes: 10 * Self.gb)
        let pool = ResidentPool<Box>(budget: budget)
        try await load(pool, "running", 4)
        let reservation = await pool.reserveWorkingSet(totalNeedBytes: 6 * Self.gb, protecting: "running")
        #expect(budget.usedBytes == 6 * Self.gb)
        if let reservation { budget.release(reservation) }
        #expect(budget.usedBytes == 4 * Self.gb)
    }

    /// A prompt-cache snapshot (granted by `tryReserve`) is neither counted against a working set nor evicted for
    /// it — the snapshot is what the request is about to reuse (journal `2026-452`: evicting it cost 25 s → 250 s).
    private final class FakeSnapshots: MemoryBudgetParticipant, @unchecked Sendable {
        var evictions = 0
        var evictionPriority: Int { MemoryBudget.snapshotPriority }
        func evictLeastRecentlyUsed() async -> Bool { evictions += 1; return false }
    }

    @Test func snapshotsAreNeitherCountedNorEvictedForAWorkingSet() async throws {
        let budget = MemoryBudget(capacityBytes: 10 * Self.gb)
        let pool = ResidentPool<Box>(budget: budget)
        let snapshots = FakeSnapshots()
        budget.register(snapshots)
        try await load(pool, "running", 5)
        let snapshot = budget.tryReserve(bytes: 4 * Self.gb)                // 9 GB used, 1 GB free
        #expect(snapshot != nil)
        if let snapshot { budget.commit(snapshot) }                         // as the store does, at once

        // 8 GB total need = 3 GB beyond the running model's 5: only 1 GB is free, but 4 of the 9 are a snapshot.
        let reservation = await pool.reserveWorkingSet(totalNeedBytes: 8 * Self.gb, protecting: "running")
        #expect(reservation != nil)
        #expect(snapshots.evictions == 0)
        #expect(await pool.residentKeys == ["running"])
    }

    @Test func aSnapshotDoesNotLetAnImpossibleRequestThrough() async throws {
        let budget = MemoryBudget(capacityBytes: 10 * Self.gb)
        let pool = ResidentPool<Box>(budget: budget)
        try await load(pool, "other", 3)
        try await load(pool, "running", 5)
        if let snapshot = budget.tryReserve(bytes: 2 * Self.gb) { budget.commit(snapshot) }
        // 5 running + 6 needed > 10 even with the other model gone and the snapshot ignored.
        #expect(await pool.reserveWorkingSet(totalNeedBytes: 11 * Self.gb, protecting: "running") == nil)
    }
}
