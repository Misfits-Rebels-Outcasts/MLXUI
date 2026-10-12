import Testing
import Foundation
@testable import MLXUI

/// SC-2a (gate I) — never free what is still in use. A pool/engine entry that a generation or flow row is using is
/// skipped by LRU eviction (including the S3-8B working-set path), an explicit drop of it is *doomed* (its bytes stay
/// booked until the last holder returns), a load that can only fit by evicting it *waits* for the release — bounded —
/// instead of going over budget, and the lease is returned on cancel and throw.
struct InUseLeaseTests {

    private static let gb = Int64(1_073_741_824)
    private final class Box: Sendable { let name: String; init(_ n: String) { name = n } }
    private struct Boom: Error {}

    /// A budget whose cache clearer is a counting fake (never touches MLX).
    private func makeBudget(_ gbs: Int64, clearer: MLXCacheClearer? = nil) -> MemoryBudget {
        MemoryBudget(capacityBytes: gbs * Self.gb,
                     cacheClearer: clearer ?? MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
    }

    private func load(_ pool: ResidentPool<Box>, _ key: String, _ gb: Int64) async throws {
        _ = try await pool.value(for: key, bytes: gb * Self.gb) { Box(key) }
    }

    // MARK: - LRU skips what is in use

    @Test func lruEvictionSkipsAHeldPoolEntry() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let a = try await pool.lease(for: "a", bytes: 6 * Self.gb) { Box("a") }       // oldest, in use
        try await load(pool, "b", 4)
        // c needs 4 GB: only b is evictable (a is held, though older)
        _ = try await pool.lease(for: "c", bytes: 4 * Self.gb, maxWait: .seconds(2)) { Box("c") }
        #expect(Set(await pool.residentKeys) == ["a", "c"])
        #expect(budget.usedBytes == 10 * Self.gb)
        a.release()
    }

    @Test func aHeldEntryIsEvictableAgainOnceReleased() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let a = try await pool.lease(for: "a", bytes: 6 * Self.gb) { Box("a") }
        a.release()
        try await load(pool, "b", 6)
        #expect(await pool.residentKeys == ["b"])
        #expect(budget.usedBytes == 6 * Self.gb)
    }

    @Test func leaseReleaseIsIdempotent() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let first = try await pool.lease(for: "a", bytes: 6 * Self.gb) { Box("a") }
        let second = try await pool.lease(for: "a", bytes: 6 * Self.gb) { Box("a") }   // two holders of one entry
        first.release(); first.release()
        #expect(budget.inUseCount == 1)                    // the second still holds it
        second.release()
        #expect(budget.inUseCount == 0)
    }

    // MARK: - working set (S3-8B / F15)

    @Test func aWorkingSetNeverEvictsAModelAFlowRowIsStillUsing() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let flow = try await pool.lease(for: "flow-model", bytes: 5 * Self.gb) { Box("flow") }      // a flow row, generating
        let running = try await pool.lease(for: "running", bytes: 4 * Self.gb) { Box("run") }       // the server request's model
        // the request's total need (8) beside the running 4 would need the flow model's room — it is in use
        let r = await pool.reserveWorkingSet(totalNeedBytes: 8 * Self.gb, protecting: "running", maxWait: .milliseconds(150))
        #expect(r == nil)                                   // today's refusal, after the bounded wait
        #expect(Set(await pool.residentKeys) == ["flow-model", "running"])
        #expect(budget.usedBytes == 9 * Self.gb)
        flow.release(); running.release()
    }

    @Test func aWorkingSetWaitsForAHeldModelAndThenProceeds() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let flow = try await pool.lease(for: "flow-model", bytes: 5 * Self.gb) { Box("flow") }
        let running = try await pool.lease(for: "running", bytes: 4 * Self.gb) { Box("run") }
        async let reserved = pool.reserveWorkingSet(totalNeedBytes: 8 * Self.gb, protecting: "running", maxWait: .seconds(5))
        try await Task.sleep(for: .milliseconds(100))
        #expect(await pool.residentKeys.contains("flow-model"))   // still there while the row generates
        flow.release()                                            // the row ends
        let r = await reserved
        #expect(r != nil)
        #expect(await pool.residentKeys == ["running"])           // now it was an ordinary eviction
        running.release()
    }

    @Test func aWorkingSetDoesNotWaitForTheModelItIsRunning() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let running = try await pool.lease(for: "running", bytes: 6 * Self.gb) { Box("run") }        // held by this very request
        // 6 + 5 KV > 10 and nothing else exists: refuse promptly, not after the bound
        let start = ContinuousClock.now
        let r = await pool.reserveWorkingSet(totalNeedBytes: 11 * Self.gb, protecting: "running", maxWait: .seconds(30))
        #expect(r == nil)
        #expect(ContinuousClock.now - start < .seconds(5))
        running.release()
    }

    // MARK: - explicit evict while held = doomed

    @Test func explicitEvictWhileHeldKeepsItsReservationUntilTheLastRelease() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let a = try await pool.lease(for: "a", bytes: 6 * Self.gb) { Box("a") }
        #expect(await pool.evict(key: "a"))
        #expect(await pool.residentKeys.isEmpty)                  // gone from the pool now…
        #expect(budget.usedBytes == 6 * Self.gb)                  // …but the weights are still in memory and booked
        a.release()
        #expect(budget.usedBytes == 0)                            // freed on the last return
    }

    @Test func clearWhileHeldIsDoomedToo() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let a = try await pool.lease(for: "a", bytes: 6 * Self.gb) { Box("a") }
        try await load(pool, "b", 3)
        await pool.clear()
        #expect(budget.usedBytes == 6 * Self.gb)                  // b freed, a doomed
        a.release()
        #expect(budget.usedBytes == 0)
    }

    @Test func aDoomedEntryStillBlocksALoadUntilReleased() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let a = try await pool.lease(for: "a", bytes: 6 * Self.gb) { Box("a") }
        await pool.evict(key: "a")
        async let b = pool.lease(for: "b", bytes: 6 * Self.gb, maxWait: .seconds(5)) { Box("b") }
        try await Task.sleep(for: .milliseconds(100))
        #expect(budget.usedBytes == 6 * Self.gb)                  // b has not been booked on top of a
        a.release()
        let lease = try await b
        #expect(budget.usedBytes == 6 * Self.gb)
        lease.release()
    }

    // MARK: - a second load waits, then proceeds; the bound surfaces

    @Test func aSecondLoadWaitsForTheHolderThenProceeds() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let a = try await pool.lease(for: "a", bytes: 6 * Self.gb) { Box("a") }
        async let b = pool.lease(for: "b", bytes: 6 * Self.gb, maxWait: .seconds(5)) { Box("b") }
        try await Task.sleep(for: .milliseconds(100))
        #expect(budget.usedBytes <= 10 * Self.gb)                 // it did not go over budget to avoid waiting
        #expect(await pool.residentKeys == ["a"])
        a.release()
        let lease = try await b
        #expect(lease.value.name == "b")
        #expect(await pool.residentKeys == ["b"])
        #expect(budget.usedBytes <= 10 * Self.gb)
        lease.release()
    }

    @Test func theBoundExpiringSurfacesToTheCaller() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let a = try await pool.lease(for: "a", bytes: 6 * Self.gb) { Box("a") }
        await #expect(throws: MemoryBudget.WaitError.self) {
            _ = try await pool.lease(for: "b", bytes: 6 * Self.gb, maxWait: .milliseconds(150)) { Box("b") }
        }
        #expect(budget.usedBytes == 6 * Self.gb)                  // nothing was booked for the refused load
        #expect(await pool.residentKeys == ["a"])
        a.release()
        // and the pool is healthy afterwards
        let b = try await pool.lease(for: "b", bytes: 6 * Self.gb, maxWait: .seconds(1)) { Box("b") }
        b.release()
    }

    @Test func aCancelledWaiterStopsWaiting() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let a = try await pool.lease(for: "a", bytes: 6 * Self.gb) { Box("a") }
        let waiter = Task {
            try await pool.lease(for: "b", bytes: 6 * Self.gb, maxWait: .seconds(60)) { Box("b") }
        }
        try await Task.sleep(for: .milliseconds(100))
        let start = ContinuousClock.now
        waiter.cancel()
        _ = await waiter.result
        #expect(ContinuousClock.now - start < .seconds(5))        // not the 60 s bound
        a.release()
    }

    @Test func aSingleModelLargerThanTheWholeBudgetStillLoadsWhenNothingIsHeld() async throws {
        let budget = makeBudget(4)
        let pool = ResidentPool<Box>(budget: budget)
        try await load(pool, "small", 2)                           // resident, not in use
        let big = try await pool.lease(for: "big", bytes: 9 * Self.gb, maxWait: .seconds(1)) { Box("big") }
        #expect(await pool.residentKeys == ["big"])
        big.release()
    }

    @Test func aModelLargerThanTheBudgetWaitsForAHolderFirst() async throws {
        let budget = makeBudget(4)
        let pool = ResidentPool<Box>(budget: budget)
        let small = try await pool.lease(for: "small", bytes: 2 * Self.gb) { Box("small") }
        async let big = pool.lease(for: "big", bytes: 9 * Self.gb, maxWait: .seconds(5)) { Box("big") }
        try await Task.sleep(for: .milliseconds(100))
        #expect(budget.usedBytes == 2 * Self.gb)                   // not stacked on top of the model in use
        small.release()
        let lease = try await big
        #expect(await pool.residentKeys == ["big"])
        lease.release()
    }

    // MARK: - the lease is returned on cancel and throw

    @Test func aThrowingLoadLeavesNothingHeld() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        await #expect(throws: Boom.self) {
            _ = try await pool.lease(for: "a", bytes: 6 * Self.gb) { throw Boom() }
        }
        #expect(budget.usedBytes == 0)
        #expect(budget.inUseCount == 0)
    }

    @Test func aHolderThatThrowsStillReturnsItsLease() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        func generate() async throws {
            let lease = try await pool.lease(for: "a", bytes: 6 * Self.gb) { Box("a") }
            defer { lease.release() }
            throw Boom()
        }
        await #expect(throws: Boom.self) { try await generate() }
        #expect(budget.inUseCount == 0)
        try await load(pool, "b", 6)
        #expect(await pool.residentKeys == ["b"])
    }

    @Test func aCancelledHolderStillReturnsItsLease() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let started = Cell<Bool>(false)
        let task = Task {
            let lease = try await pool.lease(for: "a", bytes: 6 * Self.gb) { Box("a") }
            defer { lease.release() }
            started.value = true
            try await Task.sleep(for: .seconds(60))
        }
        while !started.value { try await Task.sleep(for: .milliseconds(10)) }
        #expect(budget.inUseCount == 1)
        task.cancel()
        _ = await task.result
        #expect(budget.inUseCount == 0)
    }

    @Test func aDroppedLeaseReturnsItselfThroughTheDeinit() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        do {
            let lease = try await pool.lease(for: "a", bytes: 6 * Self.gb) { Box("a") }
            #expect(budget.inUseCount == 1)
            _ = lease.value
        }
        #expect(budget.inUseCount == 0)
    }

    // MARK: - callers that joined a load in flight are held too

    @Test func callersJoiningALoadInFlightAreAllHeldAndEvictingMidLoadDoomsIt() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let release = Cell<Bool>(false)
        async let first = pool.lease(for: "a", bytes: 6 * Self.gb) {
            while !release.value { try await Task.sleep(for: .milliseconds(10)) }
            return Box("a")
        }
        try await Task.sleep(for: .milliseconds(50))
        async let second = pool.lease(for: "a", bytes: 6 * Self.gb) { Box("never") }
        try await Task.sleep(for: .milliseconds(50))
        release.value = true
        let (l1, l2) = try await (first, second)
        #expect(l1.value === l2.value)
        #expect(budget.inUseCount == 1)
        l1.release()
        #expect(await pool.residentKeys == ["a"])                  // second still holds it
        l2.release()
        #expect(budget.inUseCount == 0)
    }

    // MARK: - EngineCache

    private nonisolated struct Stub: PipelineStage {
        let id: String
        var name: String { id }
        var accepts: MediaKind { .text }
        var produces: MediaKind { .text }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media { input }
    }

    @Test func engineCacheLruSkipsAHeldEngine() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: true)
        let m1 = makeEntry(id: "m1", ramGB: 6), m2 = makeEntry(id: "m2", ramGB: 4), m3 = makeEntry(id: "m3", ramGB: 4)
        let held = try await cache.lease(for: m1, config: .default) { m, _ in Stub(id: m.id) }
        _ = try await cache.stage(for: m2, config: .default) { m, _ in Stub(id: m.id) }
        _ = try await cache.lease(for: m3, config: .default, maxWait: .seconds(2)) { m, _ in Stub(id: m.id) }
        #expect(cache.count == 2)                                   // m2 went, m1 (older, but in use) stayed
        #expect(budget.usedBytes == 10 * Self.gb)
        held.hold.release()
    }

    @Test func engineCacheClearWhileHeldIsDoomedAndAWaitingBuildProceedsAfterRelease() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: true)
        let m1 = makeEntry(id: "m1", ramGB: 6), m2 = makeEntry(id: "m2", ramGB: 6)
        let held = try await cache.lease(for: m1, config: .default) { m, _ in Stub(id: m.id) }
        cache.clear()
        #expect(cache.count == 0)
        #expect(budget.usedBytes == 6 * Self.gb)
        async let second = cache.lease(for: m2, config: .default, maxWait: .seconds(5)) { m, _ in Stub(id: m.id) }
        try await Task.sleep(for: .milliseconds(100))
        #expect(budget.usedBytes == 6 * Self.gb)
        held.hold.release()
        let lease = try await second
        #expect(cache.count == 1)
        #expect(budget.usedBytes == 6 * Self.gb)
        lease.hold.release()
    }

    @Test func engineCacheBoundExpiryThrowsAndALeasedStageReturnsItsLeaseWhenDropped() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: true)
        let m1 = makeEntry(id: "m1", ramGB: 6), m2 = makeEntry(id: "m2", ramGB: 6)
        var row: (any PipelineStage)? = try await cache.leasedStage(for: m1, config: .default) { m, _ in Stub(id: m.id) }
        #expect(row is LeasedStage)
        await #expect(throws: MemoryBudget.WaitError.self) {
            _ = try await cache.lease(for: m2, config: .default, maxWait: .milliseconds(150)) { m, _ in Stub(id: m.id) }
        }
        #expect(budget.inUseCount == 1)
        row = nil                                                   // the flow row is done with the stage
        #expect(budget.inUseCount == 0)
        let second = try await cache.lease(for: m2, config: .default, maxWait: .seconds(1)) { m, _ in Stub(id: m.id) }
        second.hold.release()
    }

    @Test func aZeroByteLlmStageIsNotWrappedOrHeld() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)       // the shared cache's mode
        let llm = makeEntry(id: "llm", modelType: .llm, ramGB: 6)
        let stage = try await cache.leasedStage(for: llm, config: .default) { m, _ in Stub(id: m.id) }
        #expect(!(stage is LeasedStage))
        #expect(budget.inUseCount == 0)
    }

    // MARK: - prompt-cache step 0 still evicts snapshots first

    /// A stand-in for the prompt-cache store: one snapshot, priority below every weights holder (PC-1).
    private final class SnapshotStore: MemoryBudgetParticipant, @unchecked Sendable {
        let budget: MemoryBudget
        var reservation: MemoryBudget.Reservation?
        var evictions = 0
        init(_ budget: MemoryBudget) { self.budget = budget; budget.register(self) }
        var evictionPriority: Int { MemoryBudget.snapshotPriority }
        func evictLeastRecentlyUsed() async -> Bool {
            guard let r = reservation else { return false }
            reservation = nil; evictions += 1
            budget.release(r)
            return true
        }
    }

    @Test func snapshotsStillGoBeforeAnyWeightsAndAHeldModelStays() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let snapshots = SnapshotStore(budget)
        let a = try await pool.lease(for: "a", bytes: 5 * Self.gb) { Box("a") }
        let idle = try await pool.lease(for: "idle", bytes: 2 * Self.gb) { Box("idle") }
        idle.release()                                                // resident but not in use
        snapshots.reservation = budget.tryReserve(bytes: 3 * Self.gb)
        #expect(snapshots.reservation != nil)
        // 4 GB more needs 4 GB freed: the snapshot (3) goes first — before even the idle weights — then the idle model
        let b = try await pool.lease(for: "b", bytes: 4 * Self.gb, maxWait: .seconds(1)) { Box("b") }
        #expect(snapshots.evictions == 1)
        #expect(Set(await pool.residentKeys) == ["a", "b"])           // the held `a` is untouched
        a.release(); b.release()
    }

    // MARK: - the cache clearer (step 4)

    @Test func weightsReallyDroppedRequestOneClearAndCoalesceABurst() async throws {
        let cleared = Cell<Int>(0)
        let clearer = MLXCacheClearer(capActive: { false }, clear: { cleared.value += 1 }, delay: .milliseconds(100))
        let budget = makeBudget(10, clearer: clearer)
        let pool = ResidentPool<Box>(budget: budget)
        try await load(pool, "a", 4)
        try await load(pool, "b", 4)
        try await load(pool, "c", 4)                                // evicts a (and later b): two drops in a burst
        try await Task.sleep(for: .milliseconds(400))
        #expect(clearer.requestCount >= 1)
        #expect(cleared.value == 1)                                 // once, not per eviction
    }

    @Test func aLastHolderReturningFromADoomedEntryRequestsAClear() async throws {
        let clearer = MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(50))
        let budget = makeBudget(10, clearer: clearer)
        let pool = ResidentPool<Box>(budget: budget)
        let a = try await pool.lease(for: "a", bytes: 6 * Self.gb) { Box("a") }
        await pool.evict(key: "a")
        #expect(clearer.requestCount == 0)                          // still in use: the weights are not gone yet
        a.release()
        #expect(clearer.requestCount == 1)
    }

    @Test func theCacheIsNotClearedWhileAServerGenerationHoldsItsScopedLimit() async throws {
        let capActive = Cell<Bool>(true)                            // a server generation's cap is in force
        let cleared = Cell<Int>(0)
        let clearer = MLXCacheClearer(capActive: { capActive.value }, clear: { cleared.value += 1 },
                                      delay: .milliseconds(20), retryEvery: .milliseconds(30), giveUpAfter: .seconds(30))
        clearer.requestClear()
        try await Task.sleep(for: .milliseconds(250))
        #expect(cleared.value == 0)                                 // deferred, not dropped, not fought
        capActive.value = false                                     // the generation ended
        try await Task.sleep(for: .milliseconds(250))
        #expect(cleared.value == 1)
    }

    // MARK: - SC-2a-b (4): ServeMLXMemory.isCapActive is read-only and mirrors the scope

    @Test func serveMLXMemoryReportsWhenAGenerationCapIsActiveWithoutChangingItsBehaviour() {
        let limit = Cell<Int>(8 << 30)
        let writes = Cell<[Int]>([])
        let memory = ServeMLXMemory(read: { limit.value }, write: { limit.value = $0; writes.value.append($0) })
        #expect(!memory.isCapActive)
        memory.begin()
        #expect(memory.isCapActive)
        #expect(limit.value == ServeMLXMemory.generationCacheLimit)
        memory.begin()                                              // nested
        memory.end()
        #expect(memory.isCapActive)
        memory.end()
        #expect(!memory.isCapActive)
        #expect(limit.value == 8 << 30)                             // restored exactly as before
        #expect(writes.value == [ServeMLXMemory.generationCacheLimit, 8 << 30])
        _ = memory.isCapActive
        #expect(writes.value.count == 2)                            // reading it writes nothing
    }

    // MARK: - SC-2a-b (1): why a working set returned nothing

    @Test func aWorkingSetThatCanNeverFitIsTooLargeNotBusy() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let running = try await pool.lease(for: "running", bytes: 6 * Self.gb) { Box("run") }
        // 6 GB of weights + 5 GB of KV in a 10 GB budget: nothing else exists to wait for
        let outcome = await pool.reserveWorkingSetReporting(totalNeedBytes: 11 * Self.gb, protecting: "running",
                                                            maxWait: .seconds(30))
        guard case .tooLarge = outcome else { Issue.record("expected .tooLarge, got \(outcome)"); return }
        running.release()
    }

    @Test func aWorkingSetBlockedOnlyByModelsInUseIsBusyAndNamesThem() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let flow = try await pool.lease(for: "/models/mlx-community--Qwen3-8B-4bit", bytes: 5 * Self.gb, holder: .flow) { Box("f") }
        let running = try await pool.lease(for: "running", bytes: 4 * Self.gb, holder: .serverRequest) { Box("r") }
        let outcome = await pool.reserveWorkingSetReporting(totalNeedBytes: 8 * Self.gb, protecting: "running",
                                                            maxWait: .milliseconds(150))
        guard case .busy(let blockers) = outcome else { Issue.record("expected .busy, got \(outcome)"); return }
        #expect(blockers == [.init(model: "Qwen3-8B-4bit", holder: .flow)])        // never the model the request runs
        #expect(await pool.residentKeys.count == 2)
        flow.release(); running.release()
    }

    @Test func theSameWorkingSetIsTooLargeWhenNothingIsHeldAndItCannotFit() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        try await load(pool, "idle", 5)                                              // resident, not in use
        let running = try await pool.lease(for: "running", bytes: 4 * Self.gb) { Box("r") }
        let outcome = await pool.reserveWorkingSetReporting(totalNeedBytes: 12 * Self.gb, protecting: "running",
                                                            maxWait: .seconds(5))
        guard case .tooLarge = outcome else { Issue.record("expected .tooLarge, got \(outcome)"); return }
        #expect(await pool.residentKeys == ["running"])                              // the idle one was evicted trying
        running.release()
    }

    // MARK: - SC-2a-b (3): the sentence names the model and, if known, the holder

    @Test func aRefusedLoadNamesTheModelInUseAndWhoHoldsIt() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let chat = try await pool.lease(for: "/Users/x/models/mlx-community--Qwen3-8B-4bit", bytes: 6 * Self.gb, holder: .chat) { Box("a") }
        do {
            _ = try await pool.lease(for: "b", bytes: 6 * Self.gb, maxWait: .milliseconds(100)) { Box("b") }
            Issue.record("expected a WaitError")
        } catch let error as MemoryBudget.WaitError {
            #expect(error.description == "Waiting for memory: Qwen3-8B-4bit is in use by Chat. Try again when it finishes.")
            #expect(error.errorDescription == error.description)
        }
        chat.release()
    }

    @Test func theSentenceFollowsTheHolderKindAndOmitsItWhenUnknown() {
        typealias B = MemoryBudget.Blocker
        #expect(MemoryBudget.waitSentence([B(model: "Qwen3-8B-4bit", holder: .flow)])
                == "Waiting for memory: Qwen3-8B-4bit is in use by a flow. Try again when it finishes.")
        #expect(MemoryBudget.waitSentence([B(model: "Qwen3-8B-4bit", holder: .serverRequest)])
                == "Waiting for memory: Qwen3-8B-4bit is in use by a server request. Try again when it finishes.")
        #expect(MemoryBudget.waitSentence([B(model: "Qwen3-8B-4bit", holder: nil)])
                == "Waiting for memory: Qwen3-8B-4bit is in use. Try again when it finishes.")
        #expect(MemoryBudget.waitSentence([]).hasPrefix("Waiting for memory: another model"))
        #expect(MemoryBudget.displayName(forLabel: "/a/b/mlx-community--Qwen3-8B-4bit") == "Qwen3-8B-4bit")
        #expect(MemoryBudget.displayName(forLabel: "some-catalog-id") == "some-catalog-id")
    }

    @Test func aFlowRowThatWaitsOutTheBoundGetsTheSentenceNotAWrappedEngineFailure() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: true)
        let m1 = makeEntry(id: "whisper-small", ramGB: 6), m2 = makeEntry(id: "m2", ramGB: 6)
        let held = try await cache.lease(for: m1, config: .default, holder: .chat) { m, _ in Stub(id: m.id) }
        do {
            _ = try await cache.leasedStage(for: m2, config: .default, maxWait: .milliseconds(100)) { m, _ in Stub(id: m.id) }
            Issue.record("expected a WaitError")
        } catch let error as MemoryBudget.WaitError {
            #expect(error.description == "Waiting for memory: whisper-small is in use by Chat. Try again when it finishes.")
        }
        held.hold.release()
    }

    @Test func aLoadJoiningAnInFlightLoadKeepsItsOwnHolderKind() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let go = Cell<Bool>(false)
        async let first = pool.lease(for: "/m/mlx-community--M-4bit", bytes: 6 * Self.gb, holder: .flow) {
            while !go.value { try await Task.sleep(for: .milliseconds(10)) }
            return Box("m")
        }
        try await Task.sleep(for: .milliseconds(50))
        async let second = pool.lease(for: "/m/mlx-community--M-4bit", bytes: 6 * Self.gb, holder: .chat) { Box("never") }
        try await Task.sleep(for: .milliseconds(50))
        go.value = true
        let (l1, l2) = try await (first, second)
        do {
            _ = try await pool.lease(for: "other", bytes: 6 * Self.gb, maxWait: .milliseconds(100)) { Box("o") }
            Issue.record("expected a WaitError")
        } catch let error as MemoryBudget.WaitError {
            #expect(Set(error.blockers.map(\.holder)) == [.flow, .chat])
        }
        l1.release(); l2.release()
    }
}

private final class Cell<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: T
    init(_ v: T) { stored = v }
    var value: T {
        get { lock.lock(); defer { lock.unlock() }; return stored }
        set { lock.lock(); stored = newValue; lock.unlock() }
    }
}
