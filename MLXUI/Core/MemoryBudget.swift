import Foundation
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
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private static let log = Logger(subsystem: "com.ai-browser", category: "memory-budget")

    init(capacityBytes: Int64) {
        self.capacityBytes = capacityBytes
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
    /// only in the over-budget log line — never content).
    func reserve(bytes: Int64, for owner: some MemoryBudgetParticipant, label: String) async -> Reservation {
        let reservation = Reservation(id: UUID())
        guard bytes > 0 else { return reservation }   // holds nothing; commit/release are no-ops

        while true {
            if tryAdd(reservation, bytes: bytes) { return reservation }

            let peers = others(excluding: owner)
            let mine = owner.evictionPriority

            // 0. participants that rank below the requester (prompt-cache snapshots): before any
            //    weights, the requester's own included.
            var freed = false
            for lower in peers.filter({ $0.evictionPriority < mine }).sorted(by: { $0.evictionPriority < $1.evictionPriority }) {
                if await lower.evictLeastRecentlyUsed() { freed = true; break }
            }
            if freed { continue }

            // 1. the requester's own LRU.
            if await owner.evictLeastRecentlyUsed() { continue }

            // 2. then the other participant(s) of equal or higher priority.
            for other in peers.filter({ $0.evictionPriority >= mine }) {
                if await other.evictLeastRecentlyUsed() { freed = true; break }
            }
            if freed { continue }

            // 3. nothing evictable: if other loads are still in flight, wait for one to settle.
            if await waitForPendingToSettle(excluding: reservation) { continue }

            // 4. both empty, nothing in flight: proceed over budget and say so.
            withLock {
                ledger[reservation.id] = bytes
                pending.insert(reservation.id)
            }
            Self.log.notice("over budget: \(label, privacy: .public) needs \(bytes) B, capacity \(self.capacityBytes) B, nothing left to evict")
            return reservation
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
    func reserveWorkingSet(bytes: Int64, for owner: some MemoryBudgetParticipant, protecting protectedBytes: Int64,
                           evictOwn: () async -> Bool) async -> Reservation? {
        let reservation = Reservation(id: UUID())
        guard bytes > 0 else { return reservation }
        while true {
            let (added, held) = withLock { () -> (Bool, Int64) in
                let held = ledger.filter { !snapshots.contains($0.key) }.values.reduce(0, +)
                guard held + bytes <= capacityBytes else { return (false, held) }
                ledger[reservation.id] = bytes
                pending.insert(reservation.id)
                return (true, held)
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
            if await waitForPendingToSettle(excluding: reservation) { continue }
            return nil
        }
    }

    /// S3.1b — lower an existing reservation by `bytes`. **Only ever lowers**: `false` (and nothing changes) when
    /// `bytes` is not positive or exceeds what the reservation holds. Used to give back a generation's unused reply
    /// allowance in exchange for a prompt-cache snapshot (`ReplyTrade`).
    func shrink(_ reservation: Reservation, by bytes: Int64) -> Bool {
        withLock {
            guard bytes > 0, let held = ledger[reservation.id], bytes <= held else { return false }
            ledger[reservation.id] = held - bytes
            return true
        }
    }

    /// The entry is resident now (and therefore evictable).
    func commit(_ reservation: Reservation) {
        let toResume: [CheckedContinuation<Void, Never>] = withLock {
            pending.remove(reservation.id)
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
            return takeWaitersLocked()
        }
        toResume.forEach { $0.resume() }
    }

    // MARK: - Private

    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }

    /// Atomic check-and-add: reserves only if it fits right now.
    private func tryAdd(_ reservation: Reservation, bytes: Int64) -> Bool {
        withLock {
            let used = ledger.values.reduce(0, +)
            guard used + bytes <= capacityBytes else { return false }
            ledger[reservation.id] = bytes
            pending.insert(reservation.id)
            return true
        }
    }

    private func others(excluding owner: AnyObject) -> [any MemoryBudgetParticipant] {
        withLock {
            participants.removeAll { $0.participant == nil }
            return participants.compactMap { $0.participant }.filter { $0 !== owner }
        }
    }

    private func takeWaitersLocked() -> [CheckedContinuation<Void, Never>] {
        let all = waiters
        waiters.removeAll()
        return all
    }

    /// If some *other* reservation is still in flight, suspend until any commit/release and
    /// return `true` (the caller retries); otherwise return `false` immediately.
    private func waitForPendingToSettle(excluding mine: Reservation) async -> Bool {
        var waited = false
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            lock.lock()
            if pending.subtracting([mine.id]).isEmpty {
                lock.unlock()
                continuation.resume()
            } else {
                waited = true
                waiters.append(continuation)
                lock.unlock()
            }
        }
        return waited
    }
}
