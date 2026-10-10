import Testing
import Foundation
@testable import MLXUI

// S3-5c — a snapshot the store would refuse is never copied.

struct SnapshotPrecheckTests {
    private final class FakeCache { var fed: [Int] = []; var bytes: Int64 = 0 }
    private struct FakeSnapshot: Sendable { let covered: [Int] }
    private let key = PromptCacheModelKey(directory: "/models/a", configSHA256: "c", parameters: "p", libraryVersion: "l")
    private static let mb = Int64(1_048_576)

    /// A store whose cap is 25 MB (a quarter of a 100 MB budget).
    private func store() -> PromptCacheStore<FakeSnapshot> {
        PromptCacheStore<FakeSnapshot>(budget: MemoryBudget(capacityBytes: 100 * Self.mb), copy: { $0 })
    }

    private final class Counter: @unchecked Sendable { var snapshots = 0 }

    private func plan(_ store: PromptCacheStore<FakeSnapshot>, cacheMB: Int64?, counter: Counter) async throws -> PromptPlan<FakeCache> {
        let tokens = Array(1000..<2000)
        let conversation = [ServeMessage(role: .system, text: "s"), ServeMessage(role: .user, text: "u")]
        return try await PromptCheckpointing.plan(
            store: store, key: key, messages: conversation, tokens: tokens,
            render: { probe in
                guard probe.count == conversation.count + 2 else { throw CancellationError() }
                return Array(tokens[0..<996]) + [-1, -2]
            },
            newCache: { FakeCache() }, isTrimmable: { _ in false },
            restore: { _ in nil },
            prefill: { from, to, cache in cache.fed += tokens[from..<to] },
            snapshot: { cache in counter.snapshots += 1; return (FakeSnapshot(covered: cache.fed), cacheMB.map { $0 * Self.mb } ?? Self.mb, false) },
            cacheBytes: cacheMB.map { mb in { _ in mb * Self.mb } })
    }

    /// 50 MB against a 25 MB cap: the prefill still happens, the copy never does, nothing is stored.
    @Test func aSnapshotOverTheCapIsNeverCopied() async throws {
        let counter = Counter()
        let s = store()
        let result = try await plan(s, cacheMB: 50, counter: counter)
        #expect(counter.snapshots == 0)
        #expect(result.outcome.storedAt.isEmpty)
        #expect(await s.count == 0)
        #expect((result.cache?.fed.count ?? 0) == 996)                           // prefilled to the turn boundary regardless
        #expect((result.cache?.fed ?? []) + result.tailTokens == Array(1000..<2000))
    }

    @Test func aSnapshotThatFitsIsCopiedAndStored() async throws {
        let counter = Counter()
        let s = store()
        let result = try await plan(s, cacheMB: 10, counter: counter)
        #expect(counter.snapshots == 1 && result.outcome.storedAt == [996])
        #expect(await s.count == 1)
    }

    @Test func withoutAPrecheckTheOldBehaviourIsKept() async throws {
        let counter = Counter()
        let s = store()
        _ = try await plan(s, cacheMB: nil, counter: counter)                    // no `cacheBytes`: the copy is made
        #expect(counter.snapshots == 1)
    }

    @Test func wouldAcceptIsTheSameTestInsertAppliesFirst() async {
        let budget = MemoryBudget(capacityBytes: 100 * Self.mb)
        let s = PromptCacheStore<FakeSnapshot>(budget: budget, copy: { $0 })
        #expect(await !s.wouldAccept(bytes: 0))
        #expect(await s.wouldAccept(bytes: 25 * Self.mb))
        #expect(await !s.wouldAccept(bytes: 25 * Self.mb + 1))                    // over the cap
        // over the most room there could ever be: weights hold the budget
        let held = budget.tryReserve(bytes: 90 * Self.mb)
        #expect(held != nil)
        #expect(await !s.wouldAccept(bytes: 20 * Self.mb))                        // only 10 MB free and the store holds nothing
        // …and `insert` agrees
        #expect(await !s.insert(key: key, tokens: [1], snapshot: FakeSnapshot(covered: [1]), bytes: 20 * Self.mb, trimmable: false))
    }
}
