import Testing
import Foundation
@testable import MLXUI

// S3.1c — a model's most recent system-boundary snapshot is protected from displacement by other snapshots.

struct PromptCacheProtectionTests {
    private struct Snap: Sendable { let n: Int }
    private final class Box: Sendable { let name: String; init(_ n: String) { name = n } }
    private static let mb = Int64(1_048_576)
    private let key = PromptCacheModelKey(directory: "/models/a", configSHA256: "c", parameters: "p", libraryVersion: "l")

    private func store(_ budget: MemoryBudget, capFraction: Double = 0.5, maxPerModel: Int = 4) -> PromptCacheStore<Snap> {
        PromptCacheStore<Snap>(budget: budget, capFraction: capFraction, maxPerModel: maxPerModel, copy: { $0 })
    }
    /// Distinct prefixes of the given length (so none is a duplicate of another).
    private func tokens(_ n: Int, seed: Int = 0) -> [Int] { (0..<n).map { $0 + seed * 100_000 } }

    @discardableResult
    private func put(_ s: PromptCacheStore<Snap>, _ length: Int, mb: Int64, system: Bool = false, seed: Int = 0) async -> Bool {
        await s.insert(key: key, tokens: tokens(length, seed: seed), snapshot: Snap(n: length), bytes: mb * Self.mb,
                       trimmable: false, systemBoundary: system)
    }

    // MARK: - Displacement by other snapshots

    @Test func aLargeSnapshotDoesNotEvictTheProtectedOne() async {
        let s = store(MemoryBudget(capacityBytes: 100 * Self.mb))                  // cap 50 MB
        #expect(await put(s, 1_000, mb: 10, system: true))
        #expect(await put(s, 2_000, mb: 20))
        #expect(await put(s, 3_000, mb: 30))                                       // 60 > 50: the byte cap drops the LRU turn snapshot…
        #expect(await s.protectedLengths(forDirectory: "/models/a") == [1_000])    // …never the protected one
        #expect(await s.storedLengths(for: key).sorted() == [1_000, 3_000])
    }

    @Test func theCountCapDoesNotEvictItEither() async {
        let s = store(MemoryBudget(capacityBytes: 1_000 * Self.mb), maxPerModel: 2)
        await put(s, 1_000, mb: 1, system: true)
        await put(s, 2_000, mb: 1)
        await put(s, 3_000, mb: 1)                                                 // count cap 2: drops 2,000
        #expect(await s.storedLengths(for: key).sorted() == [1_000, 3_000])
    }

    @Test func aSnapshotThatCanOnlyBeStoredByEvictingItIsSkipped() async {
        let s = store(MemoryBudget(capacityBytes: 100 * Self.mb))                  // cap 50 MB
        await put(s, 1_000, mb: 30, system: true)
        #expect(await !put(s, 2_000, mb: 25))                                      // 30 + 25 > 50 and nothing else to drop
        #expect(await s.storedLengths(for: key) == [1_000])
        let none = store(MemoryBudget(capacityBytes: 1_000 * Self.mb), maxPerModel: 1)
        await put(none, 1_000, mb: 1, system: true)
        #expect(await !put(none, 2_000, mb: 1))                                    // count cap 1, only the protected entry
        #expect(await none.storedLengths(for: key) == [1_000])
    }

    // MARK: - The reply trade

    private final class FakeCache { var fed: [Int] = [] }
    private static let bytesPerToken = 1_000.0

    /// 100 MB budget: 60 MB weights, 20 MB working set, a protected 10 MB system snapshot already stored → 10 MB used
    /// by snapshots, 90 in all. A 15 MB turn snapshot misses the room by 5 MB *because* the protected one is not reclaimable.
    private func rig() async -> (budget: MemoryBudget, store: PromptCacheStore<FakeSnapshot>, ws: MemoryBudget.Reservation) {
        let budget = MemoryBudget(capacityBytes: 100 * Self.mb)
        let store = PromptCacheStore<FakeSnapshot>(budget: budget, capFraction: 0.5, copy: { $0 })
        _ = budget.tryReserve(bytes: 60 * Self.mb)
        let ws = budget.tryReserve(bytes: 20 * Self.mb)!
        let ok = await store.insert(key: key, tokens: Array(0..<500), snapshot: FakeSnapshot(), bytes: 10 * Self.mb,
                                    trimmable: false, systemBoundary: true)
        #expect(ok)
        return (budget, store, ws)
    }
    private struct FakeSnapshot: Sendable {}

    private func plan(_ rig: (budget: MemoryBudget, store: PromptCacheStore<FakeSnapshot>, ws: MemoryBudget.Reservation),
                      replyLimit: Int?) async throws -> (PromptPlan<FakeCache>, ReplyTrade?) {
        let tokens = Array(1000..<2000)
        let conversation = [ServeMessage(role: .system, text: "s"), ServeMessage(role: .user, text: "u")]
        let budget = rig.budget, ws = rig.ws
        let trade = replyLimit.map { limit in
            ReplyTrade(replyLimit: limit, bytesPerToken: Self.bytesPerToken) { budget.shrink(ws, by: $0) }
        }
        let result = try await PromptCheckpointing.plan(
            store: rig.store, key: key, messages: conversation, tokens: tokens,
            render: { probe in
                // one checkpoint only: the turn boundary at 996 (no system probe is asked for with a system-less answer)
                guard probe.count == conversation.count + 2 else { throw CancellationError() }
                return Array(tokens[0..<996]) + [-1, -2]
            },
            newCache: { FakeCache() }, isTrimmable: { _ in false }, restore: { _ in nil },
            prefill: { from, to, cache in cache.fed += tokens[from..<to] },
            snapshot: { cache in (FakeSnapshot(), 15 * Self.mb, false) },
            cacheBytes: { _ in 15 * Self.mb },
            replyTrade: trade)
        return (result, trade)
    }

    @Test func aTurnSnapshotIsStoredWhenAReplyTradeMakesRoom() async throws {
        let r = await rig()
        let (result, trade) = try await plan(r, replyLimit: 32_000)
        #expect(result.outcome.storedAt == [996])
        #expect((trade?.tradedTokens ?? 0) > 0)
        #expect(await r.store.protectedLengths(forDirectory: "/models/a") == [500])      // still there
        #expect(await r.store.count == 2)
        #expect(r.budget.usedBytes <= r.budget.capacityBytes)
    }

    @Test func aTurnSnapshotIsSkippedWhenNoTradeIsPossible() async throws {
        for limit in [nil, 8_192, 10_000] as [Int?] {                                     // no trade object / floor
            let r = await rig()
            let (result, trade) = try await plan(r, replyLimit: limit)
            #expect(result.outcome.storedAt.isEmpty, "limit \(String(describing: limit))")
            #expect((trade?.tradedTokens ?? 0) == 0)
            #expect(await r.store.protectedLengths(forDirectory: "/models/a") == [500])  // never displaced
            #expect(await r.store.count == 1)
        }
    }

    // MARK: - Weights still win

    @Test func memoryPressureFromWeightsEvictsUnprotectedFirstThenTheProtectedOne() async throws {
        let budget = MemoryBudget(capacityBytes: 100 * Self.mb)
        let s = store(budget)
        await put(s, 1_000, mb: 10, system: true)
        await put(s, 2_000, mb: 20)                                                // 30 MB of snapshots, 70 MB free
        let pool = ResidentPool<Box>(budget: budget)
        _ = try await pool.value(for: "w1", bytes: 75 * Self.mb) { Box("w1") }     // needs 5 MB more: one eviction
        #expect(await s.storedLengths(for: key) == [1_000])                        // the turn snapshot went, the protected stayed
        _ = try await pool.value(for: "w2", bytes: 20 * Self.mb) { Box("w2") }     // pool 75 + 20 = 95: the protected one must go too
        #expect(await s.count == 0)
    }

    // MARK: - The protection moves

    @Test func aNewerSystemSnapshotMovesTheProtection() async {
        let s = store(MemoryBudget(capacityBytes: 100 * Self.mb))
        await put(s, 1_000, mb: 10, system: true)
        await put(s, 1_200, mb: 10, system: true, seed: 1)                         // the system prompt changed
        #expect(await s.protectedLengths(forDirectory: "/models/a") == [1_200])
        await put(s, 3_000, mb: 25)                                                // 45 MB: fits…
        await put(s, 4_000, mb: 25)                                                // …this one drops LRU unprotected, i.e. the old system one first
        let lengths = await s.storedLengths(for: key).sorted()
        #expect(lengths.contains(1_200) && !lengths.contains(1_000))
    }

    @Test func aNewSystemSnapshotMayReplaceTheOldProtectedOneEvenAtTheCap() async {
        let s = store(MemoryBudget(capacityBytes: 100 * Self.mb))                  // cap 50 MB
        await put(s, 1_000, mb: 30, system: true)
        #expect(await put(s, 1_200, mb: 30, system: true, seed: 1))                // 30 + 30 > 50: the old one is reclaimable by its successor
        #expect(await s.protectedLengths(forDirectory: "/models/a") == [1_200])
        #expect(await s.count == 1)
    }

    @Test func protectionIsPerModel() async {
        let s = store(MemoryBudget(capacityBytes: 1_000 * Self.mb), maxPerModel: 8)
        let other = PromptCacheModelKey(directory: "/models/b", configSHA256: "c", parameters: "p", libraryVersion: "l")
        await put(s, 1_000, mb: 1, system: true)
        _ = await s.insert(key: other, tokens: tokens(900, seed: 7), snapshot: Snap(n: 900), bytes: Self.mb,
                           trimmable: false, systemBoundary: true)
        #expect(await s.protectedLengths(forDirectory: "/models/a") == [1_000])
        #expect(await s.protectedLengths(forDirectory: "/models/b") == [900])
    }
}
