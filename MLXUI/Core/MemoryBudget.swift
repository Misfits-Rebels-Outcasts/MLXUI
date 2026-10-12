import Foundation
import MLX
import os

/// S1-1b — the **one memory budget** `ModelContainerPool` and `EngineCache` both reserve from
/// (owner ruling 2026-10-01, `RSI/DelegateServeBacklog.md` §S1-1 review rulings).
///
/// Each cache *reserves* an entry's bytes before loading/building it, *commits* the reservation
/// when the entry becomes resident, and *releases* it when the entry is evicted, cleared, or its
/// load fails. The ledger is the single source of "how much is held".
///
/// **When a reservation doesn't fit** (the ruling, in order):
/// 0. (PC-1, owner ruling A1 2026-10-08) ask every participant of **lower `evictionPriority` than
///    the requester** to evict its LRU entry first — the prompt cache's snapshots (priority −1)
///    go before any weights, including the requester's own LRU;
/// 1. evict from the **requester's own** LRU, one entry at a time;
/// 2. then ask the **other** participant(s) to evict their least-recently-used entry;
/// 3. if neither has anything evictable but other loads are still **in flight** (reserved, not yet
///    resident — not evictable), *wait* for one to commit or release, then retry;
/// 4. if both are empty and nothing is in flight, **proceed over budget and log** (as when a single
///    model exceeds the budget today).
/// Step 3 is the implementer's call, pending owner confirmation: without it two concurrent loads
/// could both overshoot, and the ruling's "never exceeds" could not hold under concurrency.
///
/// **SC-2a (gate I, owner 2026-10-07): never free what is still in use.** An entry whose container/stage a
/// generation or flow row is still using carries a *hold* (`hold(_:)`, returned with `Hold.release()` or by the
/// `Hold`'s deinit). Eviction skips held entries (`releaseIfUnheld`); an explicit drop of a held one is *doomed*
/// (`releaseOrDefer`) — it leaves its cache now but its bytes stay booked until the last holder returns. Step 4
/// above then applies only when **nothing is held**: when held entries are all that stand in the way, a requester
/// that passed a bound (`reserveBounded`, `reserveWorkingSet(maxWait:)`) *waits* for a release, and when the bound
/// expires it is refused instead of going over budget. A requester that passed no bound (`reserve`) behaves as it
/// always did.
///
/// The check-and-add is atomic (one lock section), so concurrent requesters cannot both claim
/// the same free room. Entries that hold **0 bytes** (an LLM stage in `EngineCache.shared`,
/// whose weights live in the pool) are never evicted for room — evicting them frees nothing.
nonisolated protocol MemoryBudgetParticipant: AnyObject, Sendable {
    /// Evict the least-recently-used entry that actually holds memory, releasing its
    /// reservation. Returns `false` when there is nothing to evict.
    func evictLeastRecentlyUsed() async -> Bool

    /// PC-1: who is dropped first when a reservation doesn't fit. Higher survives longer. Weights
    /// (`ResidentPool`, `EngineCache`) are 0 (the default); the prompt-cache store is
    /// `MemoryBudget.snapshotPriority`. A requester evicts every strictly lower participant before
    /// it touches its own entries.
    var evictionPriority: Int { get }
}

nonisolated extension MemoryBudgetParticipant {
    var evictionPriority: Int { 0 }
}

nonisolated final class MemoryBudget: @unchecked Sendable {
    struct Reservation: Hashable, Sendable {
        let id: UUID
    }

    /// The app-wide budget: `max(4 GB, total RAM × 0.6)` (the figure `EngineCache` always used).
    static let shared = MemoryBudget(
        capacityBytes: EngineCache.defaultBudget(totalRAMGB: SystemInfo.detect().totalRAMGB))

    /// Eviction priority of prompt-cache snapshots: below every weights holder (PC-1, ruling A1).
    static let snapshotPriority = -1

    let capacityBytes: Int64

    private struct Weak {
        weak var participant: (any MemoryBudgetParticipant)?
    }

    private let lock = NSLock()
    private var ledger: [UUID: Int64] = [:]
    /// Reservations made but not yet resident — not evictable.
    private var pending: Set<UUID> = []
    /// Reservations granted by `tryReserve` (prompt-cache snapshots): opportunistic, ignored by `reserveWorkingSet`.
    private var snapshots: Set<UUID> = []
    private var participants: [Weak] = []
    private var waiters: [UUID: CheckedContinuation<Void, Never>] = [:]
    /// SC-2a: how many holders each committed reservation has right now (absent = none).
    private var holds: [UUID: Int] = [:]
    /// SC-2a: reservations dropped by their cache while held — released from the ledger on the last `unhold`.
    private var doomed: Set<UUID> = []
    /// Bumped by every `commit`/`release` — i.e. whenever something becomes evictable or room appears. A requester
    /// captures it in the same lock section as its failed fit check, so "did anything settle since?" is exact (MB-1).
    private var settled: UInt64 = 0
    private static let log = Logger(subsystem: "com.ai-browser", category: "memory-budget")

    /// What a caller that passes no bound of its own waits for a held entry: the Server screen's "waiting for memory"
    /// window (`RSI/DelegateServerConsoleBacklog.md`, "Who wins", row 7). Implementer's call, pending owner
    /// confirmation, for the figure itself.
    static let defaultMaxWait: Duration = .seconds(60)

    private let cacheClearer: MLXCacheClearer

    init(capacityBytes: Int64, cacheClearer: MLXCacheClearer = .shared) {
        self.capacityBytes = capacityBytes
        self.cacheClearer = cacheClearer
    }

    /// The bounded wait ran out (or the waiting task was cancelled) with only held entries in the way.
    struct WaitError: Error, LocalizedError, CustomStringConvertible {
        let label: String
        var description: String {
            "Waiting for memory — another model is still in use and didn't finish in time. Try again when it's done."
        }
        var errorDescription: String? { description }
    }

    /// Bytes currently held (resident + in flight) across every participant.
    var usedBytes: Int64 { withLock { ledger.values.reduce(0, +) } }
    /// Number of entries holding memory (resident + in flight). Zero-byte entries aren't counted.
    var heldCount: Int { withLock { ledger.count } }

    func register(_ participant: some MemoryBudgetParticipant) {
        withLock {
            participants.removeAll { $0.participant == nil }
            participants.append(Weak(participant: participant))
        }
    }

    /// Reserve `bytes` for `owner`, making room as described above. `label` is a model id (used
    /// only in the over-budget log line — never content). **No bound on waiting for held entries**: this is the
    /// pre-SC-2a contract (over budget when nothing is evictable and nothing is in flight). Callers that can hold
    /// entries use `reserveBounded`.
    func reserve(bytes: Int64, for owner: some MemoryBudgetParticipant, label: String) async -> Reservation {
        await reserveCore(bytes: bytes, for: owner, label: label, deadline: nil).reservation
    }

    /// SC-2a (gate I): `reserve`, but when the only thing between the request and its room is entries **in use**
    /// it waits for one to be returned, for at most `maxWait` (and not past the caller's cancellation), instead of
    /// going over budget. Throws `WaitError` when the bound runs out; the reservation is then not booked.
    /// Still over budget — as in the ruling — for a single model larger than the whole capacity, or when nothing
    /// is held.
    func reserveBounded(bytes: Int64, for owner: some MemoryBudgetParticipant, label: String,
                        maxWait: Duration = MemoryBudget.defaultMaxWait) async throws -> Reservation {
        let outcome = await reserveCore(bytes: bytes, for: owner, label: label,
                                        deadline: ContinuousClock.now.advanced(by: maxWait))
        if outcome.timedOut { throw WaitError(label: label) }
        return outcome.reservation
    }

    private func reserveCore(bytes: Int64, for owner: some MemoryBudgetParticipant, label: String,
                             deadline: ContinuousClock.Instant?) async -> (reservation: Reservation, timedOut: Bool) {
        let reservation = Reservation(id: UUID())
        guard bytes > 0 else { return (reservation, false) }   // holds nothing; commit/release are no-ops

        while true {
            if deadline != nil, Task.isCancelled { return (reservation, true) }
            let (fits, seen) = tryAddCapturingSettled(reservation, bytes: bytes)
            if fits { return (reservation, false) }

            let peers = others(excluding: owner)
            let mine = owner.evictionPriority

            // 0. participants that rank below the requester (prompt-cache snapshots): before any
            //    weights, the requester's own included.
            var freed = false
            for lower in peers.filter({ $0.evictionPriority < mine }).sorted(by: { $0.evictionPriority < $1.evictionPriority }) {
                if await lower.evictLeastRecentlyUsed() { freed = true; break }
            }
            if freed { continue }

            // 1. the requester's own LRU (entries in use are skipped by the participant — SC-2a).
            if await owner.evictLeastRecentlyUsed() { continue }

            // 2. then the other participant(s) of equal or higher priority.
            for other in peers.filter({ $0.evictionPriority >= mine }) {
                if await other.evictLeastRecentlyUsed() { freed = true; break }
            }
            if freed { continue }

            // 3. nothing evictable. If anything settled while we were asking, what we were told is stale (an entry
            //    that was in flight may be evictable now) — ask again; else if other loads are still in flight, or
            //    (SC-2a, bounded callers) entries are in use, wait for one to settle; else (4) both are empty and
            //    nothing is in flight or held: proceed over budget and say so. The verdict and the over-budget grant
            //    are one lock section (MB-1): an `await` between them let a release land in the gap and the grant
            //    went through although the request now fit.
            var overBudget = false
            let verdict = await settleOrFinalize(excluding: [reservation.id], since: seen, deadline: deadline, finalize: {
                ledger[reservation.id] = bytes
                pending.insert(reservation.id)
                overBudget = true
            })
            switch verdict {
            case .retry: continue
            case .timedOut: return (reservation, true)
            case .finalized: break
            }
            if overBudget {
                Self.log.notice("over budget: \(label, privacy: .public) needs \(bytes) B, capacity \(self.capacityBytes) B, nothing left to evict")
            }
            return (reservation, false)
        }
    }

    /// PC-1: reserve `bytes` **only if they fit right now** — evicts nothing, waits for nothing.
    /// The prompt-cache store uses this: a snapshot may never push weights (or anything) out.
    /// The reservation is pending until `commit`/`release`, like `reserve`'s.
    func tryReserve(bytes: Int64) -> Reservation? {
        let reservation = Reservation(id: UUID())
        guard bytes > 0 else { return nil }
        guard tryAdd(reservation, bytes: bytes) else { return nil }
        withLock { _ = snapshots.insert(reservation.id) }
        return reservation
    }

    /// S3-8B — reserve the **working set** of a running generation (KV cache, prefill buffers) on top of the
    /// weights already resident, and make the room *for real*: a model's weights are charged at load, but a
    /// request's KV is not in the ledger, so `ServeMemoryAdmission`'s "every other resident is evictable"
    /// stayed an assumption until something actually evicted them. This does the evicting.
    ///
    /// **Other models' weights are what gets evicted** — the requester's own LRU (through `evictOwn`, which
    /// must skip the model that is running), then the other participants of equal or higher priority. It
    /// never evicts below `protectedBytes` (the running model's own weights) and it **never proceeds over
    /// budget**: when nothing else can be freed it returns `nil` and the caller refuses the request. Loads in
    /// flight are waited for, as in `reserve`.
    ///
    /// **Prompt-cache snapshots are not counted and not evicted.** They are granted with `tryReserve` only
    /// while they fit, they have their own cap, and the one the request is about to reuse is exactly the memory
    /// that makes its prefill cheap: evicting it for the request's own working set trades a 25 s turn for a
    /// 250 s one (journal `2026-452`). Weights still win over snapshots wherever `reserve` runs, as before.
    ///
    /// **SC-2a:** entries in use are not evicted (the participants skip them). When only such entries stand in the
    /// way the request waits for one to be returned — for at most `maxWait` (nil = don't wait: today's immediate
    /// `nil`) — and then returns `nil`, the caller's refusal. `protectingReservation` is the running model's own
    /// entry: this request holds it, so it is never something to wait for.
    func reserveWorkingSet(bytes: Int64, for owner: some MemoryBudgetParticipant, protecting protectedBytes: Int64,
                           protectingReservation: Reservation? = nil, maxWait: Duration? = nil,
                           evictOwn: () async -> Bool) async -> Reservation? {
        let reservation = Reservation(id: UUID())
        guard bytes > 0 else { return reservation }
        let deadline = maxWait.map { ContinuousClock.now.advanced(by: $0) }
        var ignore: Set<UUID> = [reservation.id]
        if let protectingReservation { ignore.insert(protectingReservation.id) }
        while true {
            if deadline != nil, Task.isCancelled { return nil }
            let (added, held, seen) = withLock { () -> (Bool, Int64, UInt64) in
                let held = ledger.filter { !snapshots.contains($0.key) }.values.reduce(0, +)
                guard held + bytes <= capacityBytes else { return (false, held, settled) }
                ledger[reservation.id] = bytes
                pending.insert(reservation.id)
                return (true, held, settled)
            }
            if added { return reservation }
            guard held > protectedBytes else { return nil }             // only the running model is left

            if await evictOwn() { continue }
            let mine = owner.evictionPriority
            var freed = false
            for other in others(excluding: owner).filter({ $0.evictionPriority >= mine }) {
                if await other.evictLeastRecentlyUsed() { freed = true; break }
            }
            if freed { continue }
            var granted = false
            let verdict = await settleOrFinalize(excluding: ignore, since: seen, deadline: deadline, finalize: {
                // last look, in the same lock section as the verdict: it may fit after all
                let held = ledger.filter { !snapshots.contains($0.key) }.values.reduce(0, +)
                guard held + bytes <= capacityBytes else { return }
                ledger[reservation.id] = bytes
                pending.insert(reservation.id)
                granted = true
            })
            if verdict == .retry { continue }
            return granted ? reservation : nil
        }
    }

    /// S3.1b — lower an existing reservation by `bytes`. **Only ever lowers**: `false` (and nothing changes) when
    /// `bytes` is not positive or exceeds what the reservation holds. Used to give back a generation's unused reply
    /// allowance in exchange for a prompt-cache snapshot (`ReplyTrade`).
    func shrink(_ reservation: Reservation, by bytes: Int64) -> Bool {
        let (shrunk, toResume): (Bool, [CheckedContinuation<Void, Never>]) = withLock {
            guard bytes > 0, let held = ledger[reservation.id], bytes <= held else { return (false, []) }
            ledger[reservation.id] = held - bytes
            settled &+= 1                       // room appeared: a requester's earlier "doesn't fit" is stale
            return (true, takeWaitersLocked())
        }
        toResume.forEach { $0.resume() }
        return shrunk
    }

    /// The entry is resident now (and therefore evictable).
    func commit(_ reservation: Reservation) {
        let toResume: [CheckedContinuation<Void, Never>] = withLock {
            pending.remove(reservation.id)
            settled &+= 1
            return takeWaitersLocked()
        }
        toResume.forEach { $0.resume() }
    }

    /// The entry is gone (evicted, cleared, load failed, or never retained).
    func release(_ reservation: Reservation) {
        let toResume: [CheckedContinuation<Void, Never>] = withLock {
            ledger.removeValue(forKey: reservation.id)
            pending.remove(reservation.id)
            snapshots.remove(reservation.id)
            holds.removeValue(forKey: reservation.id)
            doomed.remove(reservation.id)
            settled &+= 1
            return takeWaitersLocked()
        }
        toResume.forEach { $0.resume() }
    }

    // MARK: - Holds (SC-2a)

    /// A holder's claim on a committed reservation: the entry is **in use** until `release()` (or the deinit — the
    /// safety net for a path that forgets, never the plan). Idempotent. A hold on a reservation that books nothing
    /// (a zero-byte entry) is inert.
    nonisolated final class Hold: @unchecked Sendable {
        private let lock = NSLock()
        private var budget: MemoryBudget?
        private let reservation: Reservation?

        fileprivate init(budget: MemoryBudget?, reservation: Reservation?) {
            self.budget = budget
            self.reservation = reservation
        }

        /// A hold that holds nothing.
        static var inert: Hold { Hold(budget: nil, reservation: nil) }
        var isInert: Bool { reservation == nil }

        func release() {
            lock.lock()
            let b = budget
            budget = nil
            lock.unlock()
            if let b, let reservation { b.unhold(reservation) }
        }

        deinit { release() }
    }

    /// Mark the entry behind `reservation` as in use. Called by a cache, in the same critical section that decides
    /// the entry is the one handed out (so it can never be evicted in between).
    func hold(_ reservation: Reservation) -> Hold {
        withLock {
            guard ledger[reservation.id] != nil else { return Hold.inert }
            holds[reservation.id, default: 0] += 1
            return Hold(budget: self, reservation: reservation)
        }
    }

    /// Entries currently in use (committed, booked, not snapshots).
    var inUseCount: Int { withLock { holds.keys.filter { ledger[$0] != nil }.count } }

    private func unhold(_ reservation: Reservation) {
        let (toResume, dropped): ([CheckedContinuation<Void, Never>], Bool) = withLock {
            guard let n = holds[reservation.id] else { return ([], false) }
            var dropped = false
            if n <= 1 {
                holds.removeValue(forKey: reservation.id)
                if doomed.remove(reservation.id) != nil {          // its cache dropped it while it was in use
                    ledger.removeValue(forKey: reservation.id)
                    pending.remove(reservation.id)
                    dropped = true
                }
            } else {
                holds[reservation.id] = n - 1
            }
            settled &+= 1                                           // an entry became evictable / room appeared
            return (takeWaitersLocked(), dropped)
        }
        toResume.forEach { $0.resume() }
        if dropped { weightsDropped() }
    }

    /// Release `reservation` **only if nobody is using the entry** (LRU eviction). `false` — and nothing changes —
    /// when it is held. One lock section, so a cache that removes its entry on `true` cannot race a new holder
    /// (holders are handed out under the cache's own lock, which eviction also holds).
    func releaseIfUnheld(_ reservation: Reservation) -> Bool {
        let toResume: [CheckedContinuation<Void, Never>]? = withLock {
            guard holds[reservation.id] == nil else { return nil }
            ledger.removeValue(forKey: reservation.id)
            pending.remove(reservation.id)
            snapshots.remove(reservation.id)
            settled &+= 1
            return takeWaitersLocked()
        }
        guard let toResume else { return false }
        toResume.forEach { $0.resume() }
        weightsDropped()
        return true
    }

    /// An explicit drop (Serve off, uninstall, maintenance clear). Releases now when nothing uses the entry;
    /// otherwise the entry is *doomed*: the cache forgets it, the bytes stay booked, and the last `Hold.release`
    /// frees them. Returns `true` when the release was deferred.
    @discardableResult
    func releaseOrDefer(_ reservation: Reservation) -> Bool {
        let (deferred, toResume): (Bool, [CheckedContinuation<Void, Never>]) = withLock {
            if holds[reservation.id] != nil, ledger[reservation.id] != nil {
                doomed.insert(reservation.id)
                return (true, [])
            }
            ledger.removeValue(forKey: reservation.id)
            pending.remove(reservation.id)
            snapshots.remove(reservation.id)
            settled &+= 1
            return (false, takeWaitersLocked())
        }
        toResume.forEach { $0.resume() }
        if !deferred { weightsDropped() }
        return deferred
    }

    /// Weights were really dropped (an eviction or a last holder returning): let MLX give its buffer cache back,
    /// once, off the caller's path (see `MLXCacheClearer`).
    func weightsDropped() { cacheClearer.requestClear() }

    // MARK: - Private

    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }

    /// Atomic check-and-add: reserves only if it fits right now.
    private func tryAdd(_ reservation: Reservation, bytes: Int64) -> Bool {
        tryAddCapturingSettled(reservation, bytes: bytes).fits
    }

    /// `tryAdd`, also returning the `settled` count read in the **same** lock section as the fit check.
    private func tryAddCapturingSettled(_ reservation: Reservation, bytes: Int64) -> (fits: Bool, seen: UInt64) {
        withLock {
            let used = ledger.values.reduce(0, +)
            guard used + bytes <= capacityBytes else { return (false, settled) }
            ledger[reservation.id] = bytes
            pending.insert(reservation.id)
            return (true, settled)
        }
    }

    private func others(excluding owner: AnyObject) -> [any MemoryBudgetParticipant] {
        withLock {
            participants.removeAll { $0.participant == nil }
            return participants.compactMap { $0.participant }.filter { $0 !== owner }
        }
    }

    private func takeWaitersLocked() -> [CheckedContinuation<Void, Never>] {
        let all = Array(waiters.values)
        waiters.removeAll()
        return all
    }

    /// Wake one waiter now (its bound expired, or its task was cancelled).
    private func expireWaiter(_ id: UUID) {
        let continuation: CheckedContinuation<Void, Never>? = withLock { waiters.removeValue(forKey: id) }
        continuation?.resume()
    }

    /// Some committed, booked entry other than `ignoring` is in use (SC-2a). Call with `lock` held.
    private func othersHeldLocked(ignoring: Set<UUID>) -> Bool {
        holds.contains { ledger[$0.key] != nil && !ignoring.contains($0.key) }
    }

    private enum Verdict { case retry, finalized, timedOut }

    /// `seen` is the `settled` count the caller read together with its failed fit check. Returns `.retry` when the
    /// caller must go round again: anything has committed or released since (the eviction answers it just got may
    /// be stale), or — after suspending until the next commit/release/unhold — some other reservation was in flight
    /// or (SC-2a, only when `deadline` is set) an entry was in use. Returns `.finalized` after running `finalize`
    /// when there is nothing to wait for, and `.timedOut` when there is but `deadline` has passed (or the waiting
    /// task was cancelled).
    ///
    /// The `settled` comparison, the waiter registration and `finalize` each run in one lock section with the
    /// check that selects them, so a commit cannot slip between "nothing evictable" and "nothing pending" (MB-1),
    /// no wake-up is lost, and the caller's last action cannot act on a stale verdict.
    private func settleOrFinalize(excluding mine: Set<UUID>, since seen: UInt64, deadline: ContinuousClock.Instant?,
                                  finalize: () -> Void) async -> Verdict {
        let waiterID = UUID()
        var verdict = Verdict.finalized
        await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                lock.lock()
                if settled != seen {
                    verdict = .retry
                    lock.unlock()
                    continuation.resume()
                    return
                }
                let inFlight = !pending.subtracting(mine).isEmpty
                let inUse = deadline != nil && othersHeldLocked(ignoring: mine)
                if inFlight || inUse {
                    if let deadline, (ContinuousClock.now >= deadline || Task.isCancelled) {
                        verdict = .timedOut
                        lock.unlock()
                        continuation.resume()
                        return
                    }
                    verdict = .retry
                    waiters[waiterID] = continuation
                    lock.unlock()
                    if let deadline {
                        Task { [weak self] in
                            try? await Task.sleep(until: deadline, clock: .continuous)
                            self?.expireWaiter(waiterID)
                        }
                    }
                } else {
                    finalize()
                    lock.unlock()
                    continuation.resume()
                }
            }
        } onCancel: {
            if deadline != nil { expireWaiter(waiterID) }      // an unbounded (legacy) wait ignores cancellation, as before
        }
        return verdict
    }
}

/// SC-2a step 4 — give MLX's buffer cache back once weights are really gone. Freed weights stay in MLX's cache
/// (`Memory.cacheMemory`) until `Memory.clearCache()`; nothing cleared it on eviction (SC-0: ~4.4 GiB after release).
///
/// A clear is **requested** when a pool/cache drops weights and **runs once, later, off every generation path**: a
/// short delay lets the last reference to the evicted container die first (the lease is returned before the
/// caller's local goes away; clearing earlier frees nothing) and coalesces a burst of evictions into one clear.
/// It never runs while the server's scoped `cacheLimit` is in force (`ServeMLXMemory`, S3-5c: a generation caps
/// the pool at 1 GiB) — it re-asks once a second, for up to `giveUpAfter` — so it can't fight that limit or slow a
/// running generation. Implementer's call on the delay/cadence, pending owner confirmation.
nonisolated final class MLXCacheClearer: @unchecked Sendable {
    static let shared = MLXCacheClearer(
        currentCacheLimit: { Memory.cacheLimit },
        serverGenerationLimit: ServeMLXMemory.generationCacheLimit,
        clear: { Memory.clearCache() })

    private let currentCacheLimit: @Sendable () -> Int
    private let serverGenerationLimit: Int
    private let clear: @Sendable () -> Void
    private let delay: Duration
    private let retryEvery: Duration
    private let giveUpAfter: Duration
    private let lock = NSLock()
    private var scheduled = false
    private var requests = 0
    private var cleared = 0

    init(currentCacheLimit: @escaping @Sendable () -> Int, serverGenerationLimit: Int,
         clear: @escaping @Sendable () -> Void,
         delay: Duration = .milliseconds(300), retryEvery: Duration = .seconds(1), giveUpAfter: Duration = .seconds(60)) {
        self.currentCacheLimit = currentCacheLimit
        self.serverGenerationLimit = serverGenerationLimit
        self.clear = clear
        self.delay = delay
        self.retryEvery = retryEvery
        self.giveUpAfter = giveUpAfter
    }

    /// How many `Memory.clearCache()` calls this instance has made (tests).
    var clearCount: Int { lock.lock(); defer { lock.unlock() }; return cleared }
    var requestCount: Int { lock.lock(); defer { lock.unlock() }; return requests }

    func requestClear() {
        lock.lock()
        requests += 1
        guard !scheduled else { lock.unlock(); return }
        scheduled = true
        lock.unlock()
        Task.detached(priority: .utility) { [self] in
            try? await Task.sleep(for: delay)
            let start = ContinuousClock.now
            while currentCacheLimit() <= serverGenerationLimit, ContinuousClock.now - start < giveUpAfter {
                try? await Task.sleep(for: retryEvery)       // a server generation holds its scoped limit
            }
            lock.lock(); scheduled = false; lock.unlock()   // requests after this point schedule the next clear
            if currentCacheLimit() > serverGenerationLimit {
                clear()
                lock.lock(); cleared += 1; lock.unlock()
            }
        }
    }
}
