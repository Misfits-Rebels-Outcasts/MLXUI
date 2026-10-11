import Testing
import Foundation
@testable import MLXUI

/// MB-1 — the check-then-wait race in `MemoryBudget.reserve` / `reserveWorkingSet`.
///
/// A requester that found nothing evictable (the only other entry was still *pending*) used to
/// look at `pending` afterwards; if that entry had committed in between, `pending` was empty and
/// the requester proceeded **over budget** (reserve) or refused (working set) although room was one
/// eviction away. These tests inject the commit at the exact suspension point — inside the other
/// participant's eviction call — so the interleaving is deterministic.
struct MemoryBudgetRaceTests {

    private static let gb = Int64(1_073_741_824)

    /// A participant whose eviction is scripted by the test.
    private final class Scripted: MemoryBudgetParticipant, @unchecked Sendable {
        private let lock = NSLock()
        private var calls = 0
        private let script: @Sendable (Int) -> Bool
        init(_ script: @escaping @Sendable (Int) -> Bool) { self.script = script }
        var callCount: Int { lock.lock(); defer { lock.unlock() }; return calls }
        func evictLeastRecentlyUsed() async -> Bool {
            lock.lock(); calls += 1; let n = calls; lock.unlock()
            return script(n)
        }
    }

    /// 10 GB budget: `a` (6 GB, pending, owned by the returned `holder`) + `b` (4 GB, committed,
    /// evictable by nobody). The holder's first eviction call answers "nothing" **and commits `a`
    /// while doing so**; its second releases `a`.
    private func raceFixture() async -> (budget: MemoryBudget, holder: Scripted, requester: Scripted) {
        let budget = MemoryBudget(capacityBytes: 10 * Self.gb)
        let holderBox = LockedBox<MemoryBudget.Reservation?>(nil)
        let holder = Scripted { n in
            guard let a = holderBox.value else { return false }
            if n == 1 { budget.commit(a); return false }     // answered "nothing" — then a settles
            budget.release(a); return true
        }
        let requester = Scripted { _ in false }
        budget.register(holder)
        budget.register(requester)
        let a = await budget.reserve(bytes: 6 * Self.gb, for: holder, label: "a")
        holderBox.value = a
        let b = await budget.reserve(bytes: 4 * Self.gb, for: requester, label: "b")
        budget.commit(b)
        return (budget, holder, requester)
    }

    @Test func aReservationThatSeesAnEntryCommitMidEvictionRetriesInsteadOfGoingOverBudget() async {
        let (budget, holder, requester) = await raceFixture()
        let r = await budget.reserve(bytes: 5 * Self.gb, for: requester, label: "r")
        #expect(budget.usedBytes <= 10 * Self.gb, "granted over budget: \(budget.usedBytes) B used")
        #expect(budget.usedBytes == 9 * Self.gb)          // b (4) + r (5); a was evicted on the retry
        #expect(holder.callCount == 2)
        budget.release(r)
    }

    @Test func aWorkingSetRequestThatSeesAnEntryCommitMidEvictionRetriesInsteadOfRefusing() async {
        let (budget, holder, requester) = await raceFixture()
        let r = await budget.reserveWorkingSet(bytes: 5 * Self.gb, for: requester, protecting: 4 * Self.gb,
                                               evictOwn: { false })
        #expect(r != nil, "refused although one eviction made room")
        #expect(budget.usedBytes <= 10 * Self.gb)
        #expect(holder.callCount == 2)
        if let r { budget.release(r) }
    }

    @Test func aRequestStillWaitsWhenSomethingIsGenuinelyInFlight() async {
        // The fix must not turn "wait for an in-flight load" into a spin or an over-budget grant.
        let budget = MemoryBudget(capacityBytes: 10 * Self.gb)
        let idle = Scripted { _ in false }
        budget.register(idle)
        let a = await budget.reserve(bytes: 8 * Self.gb, for: idle, label: "a")   // pending
        let waiter = Task { await budget.reserve(bytes: 5 * Self.gb, for: idle, label: "r") }
        try? await Task.sleep(for: .milliseconds(50))
        #expect(budget.usedBytes == 8 * Self.gb)                                   // still waiting
        budget.release(a)                                                          // load failed
        let r = await waiter.value
        #expect(budget.usedBytes == 5 * Self.gb)
        budget.release(r)
    }

    /// The gap that was in `EngineCache.evictLeastRecentlyUsed`: entry removed, ledger released a moment later. With
    /// nothing in flight, "nothing to evict" must mean "nothing booked". Concurrent evictors make the gap observable
    /// (not fully deterministic — the window is inside the cache's own critical section — so it repeats).
    @Test func anEvictionThatAnswersNothingToEvictLeavesNothingBooked() async throws {
        let budget = MemoryBudget(capacityBytes: 1000 * Self.gb)
        let cache = EngineCache(budget: budget)
        let violations = LockedBox<Int>(0)
        for round in 0..<300 {
            for i in 0..<4 {
                _ = try await cache.stage(for: makeEntry(id: "r\(round)-\(i)", modelType: .asr, ramGB: 1),
                                          config: .default) { _, _ in RaceNoopStage() }
            }
            await withTaskGroup(of: Void.self) { group in
                for _ in 0..<8 {
                    group.addTask {
                        while await cache.evictLeastRecentlyUsed() {}
                        // nothing left to evict and nothing in flight: the ledger must be empty
                        if budget.usedBytes != 0 { violations.bump() }
                    }
                }
            }
        }
        #expect(violations.value == 0, "\(violations.value) times the cache said 'nothing to evict' while bytes were still booked")
    }
}

private nonisolated struct RaceNoopStage: PipelineStage {
    let id = "noop", name = "Noop"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media { input }
}

private final class LockedBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: T
    init(_ v: T) { stored = v }
    func bump() where T == Int { lock.lock(); stored += 1; lock.unlock() }
    var value: T {
        get { lock.lock(); defer { lock.unlock() }; return stored }
        set { lock.lock(); stored = newValue; lock.unlock() }
    }
}
