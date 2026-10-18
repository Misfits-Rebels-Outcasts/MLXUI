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

    /// SC-2: the last resort for a requester that may take it — evict the least-recently-used **pinned** entry that is
    /// not in use (a served model kept ready), recording that `requester` pushed it out. `evictLeastRecentlyUsed`
    /// never takes a pinned entry. Participants with no pins have nothing to give.
    func evictPinnedLeastRecentlyUsed(for requester: MemoryBudget.HolderKind?) async -> Bool
}

nonisolated extension MemoryBudgetParticipant {
    var evictionPriority: Int { 0 }
    func evictPinnedLeastRecentlyUsed(for requester: MemoryBudget.HolderKind?) async -> Bool { false }
}

nonisolated final class MemoryBudget: @unchecked Sendable {
    struct Reservation: Hashable, Sendable {
        let id: UUID
    }

    /// The app-wide budget: `max(4 GB, total RAM × 0.6)` (the figure `EngineCache` always used).
    static let shared = MemoryBudget(capacityBytes: sharedCapacity(totalRAMGB: SystemInfo.detect().totalRAMGB))

    /// The capacity `shared` is created with: `max(4 GB, RAM × 0.6)`, as always.
    ///
    /// **SC-2b (owner, 2026-10-12): in a DEBUG build only**, the launch argument `-MLXUIMemoryBudgetGiB <n>` sets it
    /// to `n` GiB, so a contention check can be run on a Mac with plenty of room (SC-6 owns a runtime-adjustable
    /// budget). The argument, its key and this branch are inside `#if DEBUG`: a release build neither contains the
    /// string nor reads `UserDefaults` for it, and only Dedicated server mode (SC-6, `setCapacity`) changes `capacityBytes` afterwards.
    static func sharedCapacity(totalRAMGB: Double, defaults: UserDefaults = .standard) -> Int64 {
        #if DEBUG
        if let gib = debugOverrideGiB(defaults: defaults) { return Int64(gib * 1_073_741_824) }
        #endif
        return EngineCache.defaultBudget(totalRAMGB: totalRAMGB)
    }

    /// Whether this build can honour the override at all (tests assert it matches `DEBUG`).
    static var debugOverrideIsCompiledIn: Bool {
        #if DEBUG
        true
        #else
        false
        #endif
    }

    #if DEBUG
    static let debugOverrideKey = "MLXUIMemoryBudgetGiB"
    /// The override in GiB when the launch argument (or a defaults entry) names a positive number.
    static func debugOverrideGiB(defaults: UserDefaults) -> Double? {
        guard defaults.object(forKey: debugOverrideKey) != nil else { return nil }
        let gib = defaults.double(forKey: debugOverrideKey)
        return gib > 0 ? gib : nil
    }
    #endif

    /// Eviction priority of prompt-cache snapshots: below every weights holder (PC-1, ruling A1).
    static let snapshotPriority = -1

    /// The capacity right now. Mutable since SC-6 (Dedicated server mode raises it and puts it back); read under the
    /// ledger's lock so a check never sees it half-way through a change.
    var capacityBytes: Int64 { withLock { capacity } }
    /// What the budget was created with — the figure Dedicated mode goes back to.
    let baseCapacityBytes: Int64
    private var capacity: Int64

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
    /// SC-2a-b: who holds each reservation (display name + kind), one entry per hold, for the refusal sentence.
    private var holdInfo: [UUID: [Blocker]] = [:]
    /// SC-2a: reservations dropped by their cache while held — released from the ledger on the last `unhold`.
    private var doomed: Set<UUID> = []
    /// SC-2: committed entries their pool keeps ready. Idle, so never "held" — but a requester that may not take a pinned
    /// entry must wait (bounded) for room rather than be granted over budget because nothing was evictable.
    private var pinnedIDs: Set<UUID> = []
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
        self.capacity = capacityBytes
        self.baseCapacityBytes = capacityBytes
        self.cacheClearer = cacheClearer
    }

    /// Who is using memory the requester needs — the model's display name and, when the lease said so, who.
    struct Blocker: Sendable, Equatable {
        let model: String
        let holder: HolderKind?
    }

    /// Who holds a model (the lease's caller), for the sentence a refused waiter shows.
    enum HolderKind: Sendable, Equatable {
        case chat, flow, serverRequest
        /// A model's Run window (ASR, OCR, VLM, Laya, CLM … — SC-2b).
        case runWindow
        var phrase: String {
            switch self {
            case .chat: "Chat"
            case .runWindow: "a Run window"
            case .flow: "a flow"
            case .serverRequest: "a server request"
            }
        }
    }

    /// SC-2 (gate C): may a reservation take a **pinned** idle entry when nothing else makes room? Interactive
    /// requesters (Chat, a Run window, a flow) may, as before pinning existed; the server's own loads, preloads and
    /// working-set reservations may not (until SC-5 turns the interactive case into a question).
    enum PinnedAccess: Sendable, Equatable {
        case never
        case may(requester: HolderKind?)

        /// SC-5a (gate D): what a load by `holder` may take. **Chat, a Run window and a flow never take a pinned model on their
        /// own** — they ask first (`ServedLoadConfirmer`), and "Load anyway" unpins and evicts the models named, so by
        /// the time the load reserves, nothing pinned is in its way; if one still is (pinned in the meantime) the load
        /// waits, bounded, and is refused rather than pushing it out unasked. So does a flow row (SC-5b: it asks, or parks
        /// when unattended). Only a server request may. A load made inside a Run window's run (`RunWindowScope`) is a Run window's, whoever the engine says.
        static func access(for holder: HolderKind?) -> PinnedAccess {
            switch RunWindowScope.holder ?? holder {
            case .chat, .runWindow, .flow: .never
            default: .may(requester: holder)
            }
        }
    }

    /// SC-5a: marks the loads made inside a Run window's run (a task-local, so it reaches an engine that was written
    /// to say `.flow`, such as the VLM's). Set by `LeasingStage.run`; nothing else sets it.
    enum RunWindowScope {
        @TaskLocal static var holder: HolderKind?
    }

    /// SC-5a: one booked entry, as the displacement check sees it.
    struct LedgerItem: Sendable, Equatable {
        let id: UUID
        let bytes: Int64
        let pinned: Bool
        /// Leased, or reserved but not yet resident: not takeable now, but it will be released.
        let inUse: Bool
        let isSnapshot: Bool
    }

    /// SC-5a: the ledger as the displacement check reads it — a copy, taken in one lock section.
    func ledgerItems() -> [LedgerItem] {
        withLock {
            ledger.map { id, bytes in
                LedgerItem(id: id, bytes: bytes, pinned: pinnedIDs.contains(id),
                           inUse: holds[id] != nil || pending.contains(id), isSnapshot: snapshots.contains(id))
            }
        }
    }

    /// The bounded wait ran out (or the waiting task was cancelled) with only held entries in the way.
    /// `description` follows the error voice: it names the model that is in use and, if known, who is using it.
    struct WaitError: Error, LocalizedError, CustomStringConvertible {
        let label: String
        let blockers: [Blocker]
        init(label: String, blockers: [Blocker] = []) {
            self.label = label
            self.blockers = blockers
        }
        var description: String { MemoryBudget.waitSentence(blockers) }
        var errorDescription: String? { description }
    }

    /// "Waiting for memory: Qwen3-8B-4bit is in use by Chat. Try again when it finishes."
    static func waitSentence(_ blockers: [Blocker]) -> String {
        guard let first = blockers.first else {
            return "Waiting for memory: another model is still loading or in use. Try again when it finishes."
        }
        if let holder = first.holder {
            return "Waiting for memory: \(first.model) is in use by \(holder.phrase). Try again when it finishes."
        }
        return "Waiting for memory: \(first.model) is in use. Try again when it finishes."
    }

    /// A model's display name from a ledger label: a pool key is a directory path (`…/mlx-community--Qwen3-8B-4bit`),
    /// an `EngineCache` label is a catalog id.
    static func displayName(forLabel label: String) -> String {
        let last = label.split(separator: "/").last.map(String.init) ?? label
        return last.components(separatedBy: "--").last ?? last
    }

    /// Why `reserveWorkingSet` returned nothing (SC-2a-b).
    enum WorkingSetOutcome: Sendable {
        case granted(Reservation)
        /// It cannot fit even with everything evictable gone: the caller's `400`, as before.
        case tooLarge
        /// Only entries in use (or loads in flight) stood in the way for the whole bound.
        case busy([Blocker])
    }

    // MARK: - Runtime capacity (SC-6, gate G)

    /// Dedicated server mode's share of RAM (gate G, planner's recommendation 0.75 — implementer's call, pending owner
    /// confirmation). The ordinary share is `EngineCache.defaultBudget`'s 0.6.
    static let dedicatedFraction = 0.75

    /// The raised capacity for a Mac with `totalRAMGB`: `max(4 GB, RAM × 0.75)`, never below `base` (a DEBUG override
    /// or a tiny Mac keeps whatever is larger).
    static func dedicatedCapacity(totalRAMGB: Double, base: Int64) -> Int64 {
        max(base, Int64(max(4, totalRAMGB * dedicatedFraction) * 1_073_741_824))
    }

    /// Change the capacity. A **raise** wakes waiters (room appeared). A **lower** leaves what is booked alone — call
    /// `enforceCapacity()` to evict down to it. Returns whether it changed.
    @discardableResult
    func setCapacity(_ bytes: Int64) -> Bool {
        guard bytes > 0 else { return false }
        let (changed, toResume): (Bool, [CheckedContinuation<Void, Never>]) = withLock {
            guard capacity != bytes else { return (false, []) }
            let raised = bytes > capacity
            capacity = bytes
            guard raised else { return (true, []) }
            settled &+= 1
            return (true, takeWaitersLocked())
        }
        toResume.forEach { $0.resume() }
        return changed
    }

    /// After a lower: evict least-recently-used entries — snapshots first, then the participants' weights — until what
    /// is booked fits. Pinned and in-use entries are skipped by the participants, so this may stop short (they free
    /// when released or unpinned; the controller re-pins against the new capacity first).
    func enforceCapacity() async {
        while usedBytes > capacityBytes {
            var freed = false
            for participant in liveParticipants().sorted(by: { $0.evictionPriority < $1.evictionPriority }) {
                if await participant.evictLeastRecentlyUsed() { freed = true; break }
            }
            if !freed { return }
            cacheClearer.clearNow()
        }
    }

    private func liveParticipants() -> [any MemoryBudgetParticipant] {
        withLock { participants.compactMap { $0.participant } }
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
    func reserve(bytes: Int64, for owner: some MemoryBudgetParticipant, label: String,
                 pinned: PinnedAccess = .may(requester: nil)) async -> Reservation {
        await reserveCore(bytes: bytes, for: owner, label: label, deadline: nil, pinned: pinned).reservation
    }

    /// SC-2a (gate I): `reserve`, but when the only thing between the request and its room is entries **in use**
    /// it waits for one to be returned, for at most `maxWait` (and not past the caller's cancellation), instead of
    /// going over budget. Throws `WaitError` when the bound runs out; the reservation is then not booked.
    /// Still over budget — as in the ruling — for a single model larger than the whole capacity, or when nothing
    /// is held.
    func reserveBounded(bytes: Int64, for owner: some MemoryBudgetParticipant, label: String,
                        maxWait: Duration = MemoryBudget.defaultMaxWait,
                        pinned: PinnedAccess = .may(requester: nil)) async throws -> Reservation {
        let outcome = await reserveCore(bytes: bytes, for: owner, label: label,
                                        deadline: ContinuousClock.now.advanced(by: maxWait), pinned: pinned)
        if outcome.timedOut { throw WaitError(label: label, blockers: outcome.blockers) }
        return outcome.reservation
    }

    private func reserveCore(bytes: Int64, for owner: some MemoryBudgetParticipant, label: String,
                             deadline: ContinuousClock.Instant?,
                             pinned: PinnedAccess) async -> (reservation: Reservation, timedOut: Bool, blockers: [Blocker]) {
        let reservation = Reservation(id: UUID())
        guard bytes > 0 else { return (reservation, false, []) }   // holds nothing; commit/release are no-ops

        while true {
            if deadline != nil, Task.isCancelled { return (reservation, true, currentBlockers(ignoring: [reservation.id])) }
            let (fits, seen) = tryAddCapturingSettled(reservation, bytes: bytes)
            if fits { return (reservation, false, []) }

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
            if await owner.evictLeastRecentlyUsed() { cacheClearer.clearNow(); continue }

            // 2. then the other participant(s) of equal or higher priority.
            for other in peers.filter({ $0.evictionPriority >= mine }) {
                if await other.evictLeastRecentlyUsed() { freed = true; cacheClearer.clearNow(); break }
            }
            if freed { continue }

            // 2b. (SC-2) a requester that may: the least-recently-used pinned entry that is not in use.
            if case .may(let requester) = pinned {
                if await owner.evictPinnedLeastRecentlyUsed(for: requester) { cacheClearer.clearNow(); continue }
                for other in peers.filter({ $0.evictionPriority >= mine }) {
                    if await other.evictPinnedLeastRecentlyUsed(for: requester) { freed = true; cacheClearer.clearNow(); break }
                }
                if freed { continue }
            }

            // 3. nothing evictable. If anything settled while we were asking, what we were told is stale (an entry
            //    that was in flight may be evictable now) — ask again; else if other loads are still in flight, or
            //    (SC-2a, bounded callers) entries are in use, wait for one to settle; else (4) both are empty and
            //    nothing is in flight or held: proceed over budget and say so. The verdict and the over-budget grant
            //    are one lock section (MB-1): an `await` between them let a release land in the gap and the grant
            //    went through although the request now fit.
            var overBudget = false
            let (verdict, blockers) = await settleOrFinalize(excluding: [reservation.id], since: seen, deadline: deadline,
                                                           pinnedBlocks: pinned == .never, finalize: {
                ledger[reservation.id] = bytes
                pending.insert(reservation.id)
                overBudget = true
            })
            switch verdict {
            case .retry: continue
            case .timedOut: return (reservation, true, blockers)
            case .finalized: break
            }
            if overBudget {
                Self.log.notice("over budget: \(label, privacy: .public) needs \(bytes) B, capacity \(self.capacityBytes) B, nothing left to evict")
            }
            return (reservation, false, [])
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
                           pinned: PinnedAccess = .never, evictOwnPinned: (() async -> Bool)? = nil,
                           evictOwn: () async -> Bool) async -> Reservation? {
        if case .granted(let reservation) = await reserveWorkingSetReporting(
            bytes: bytes, for: owner, protecting: protectedBytes, protectingReservation: protectingReservation,
            maxWait: maxWait, pinned: pinned, evictOwnPinned: evictOwnPinned, evictOwn: evictOwn) { return reservation }
        return nil
    }

    /// `reserveWorkingSet`, saying **why** when it returns nothing (SC-2a-b): `.tooLarge` — it cannot fit even with
    /// everything evictable gone (the caller's `400`) — or `.busy` — only entries in use, or loads in flight, stood
    /// in the way for the whole bound (a `503`, `Retry-After`), with who was using them.
    func reserveWorkingSetReporting(bytes: Int64, for owner: some MemoryBudgetParticipant,
                                    protecting protectedBytes: Int64, protectingReservation: Reservation? = nil,
                                    maxWait: Duration? = nil, pinned: PinnedAccess = .never,
                                    evictOwnPinned: (() async -> Bool)? = nil,
                                    evictOwn: () async -> Bool) async -> WorkingSetOutcome {
        let reservation = Reservation(id: UUID())
        guard bytes > 0 else { return .granted(reservation) }
        let deadline = maxWait.map { ContinuousClock.now.advanced(by: $0) }
        var ignore: Set<UUID> = [reservation.id]
        if let protectingReservation { ignore.insert(protectingReservation.id) }
        while true {
            if deadline != nil, Task.isCancelled { return .busy(currentBlockers(ignoring: ignore)) }
            let (added, held, seen) = withLock { () -> (Bool, Int64, UInt64) in
                let held = ledger.filter { !snapshots.contains($0.key) }.values.reduce(0, +)
                guard held + bytes <= capacity else { return (false, held, settled) }
                ledger[reservation.id] = bytes
                pending.insert(reservation.id)
                return (true, held, settled)
            }
            if added { return .granted(reservation) }
            guard held > protectedBytes else { return .tooLarge }       // only the running model is left

            if await evictOwn() { cacheClearer.clearNow(); continue }
            let mine = owner.evictionPriority
            var freed = false
            for other in others(excluding: owner).filter({ $0.evictionPriority >= mine }) {
                if await other.evictLeastRecentlyUsed() { freed = true; cacheClearer.clearNow(); break }
            }
            if freed { continue }
            // SC-2: only a requester that may (gate I row 8: a served model that is not kept ready) takes a pinned entry.
            if case .may(let requester) = pinned {
                if let evictOwnPinned, await evictOwnPinned() { cacheClearer.clearNow(); continue }
                for other in others(excluding: owner).filter({ $0.evictionPriority >= mine }) {
                    if await other.evictPinnedLeastRecentlyUsed(for: requester) { freed = true; cacheClearer.clearNow(); break }
                }
                if freed { continue }
            }
            var granted = false
            let (verdict, blockers) = await settleOrFinalize(excluding: ignore, since: seen, deadline: deadline,
                                                           pinnedBlocks: pinned == .never, finalize: {
                // last look, in the same lock section as the verdict: it may fit after all
                let held = ledger.filter { !snapshots.contains($0.key) }.values.reduce(0, +)
                guard held + bytes <= capacity else { return }
                ledger[reservation.id] = bytes
                pending.insert(reservation.id)
                granted = true
            })
            switch verdict {
            case .retry: continue
            case .timedOut: return .busy(blockers)
            case .finalized: return granted ? .granted(reservation) : .tooLarge
            }
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
            holdInfo.removeValue(forKey: reservation.id)
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
        private let info: Blocker?

        fileprivate init(budget: MemoryBudget?, reservation: Reservation?, info: Blocker? = nil) {
            self.budget = budget
            self.reservation = reservation
            self.info = info
        }

        /// A hold that holds nothing.
        static var inert: Hold { Hold(budget: nil, reservation: nil) }
        var isInert: Bool { reservation == nil }

        func release() {
            lock.lock()
            let b = budget
            budget = nil
            lock.unlock()
            if let b, let reservation { b.unhold(reservation, info: info) }
        }

        deinit { release() }
    }

    /// Mark the entry behind `reservation` as in use. Called by a cache, in the same critical section that decides
    /// the entry is the one handed out (so it can never be evicted in between).
    /// `label` is the ledger label (a model key) and `holder` who is using it — both only for the sentence a
    /// refused waiter shows (SC-2a-b); neither changes behaviour.
    func hold(_ reservation: Reservation, label: String = "", holder: HolderKind? = nil) -> Hold {
        withLock {
            guard ledger[reservation.id] != nil else { return Hold.inert }
            holds[reservation.id, default: 0] += 1
            holdInfo[reservation.id, default: []].append(
                Blocker(model: label.isEmpty ? "A model" : Self.displayName(forLabel: label), holder: holder))
            return Hold(budget: self, reservation: reservation, info: holdInfo[reservation.id]?.last)
        }
    }

    /// SC-2: the pool pins / unpins the entry behind `reservation`. Wakes waiters on an unpin (room may now be takeable).
    func setPinned(_ reservation: Reservation, _ on: Bool) {
        let toResume: [CheckedContinuation<Void, Never>] = withLock {
            pinnedIDs = pinnedIDs.filter { ledger[$0] != nil }
            if on {
                guard ledger[reservation.id] != nil else { return [] }
                pinnedIDs.insert(reservation.id)
                return []
            }
            guard pinnedIDs.remove(reservation.id) != nil else { return [] }
            settled &+= 1
            return takeWaitersLocked()
        }
        toResume.forEach { $0.resume() }
    }

    /// Entries currently in use (committed, booked, not snapshots).
    var inUseCount: Int { withLock { holds.keys.filter { ledger[$0] != nil }.count } }

    private func unhold(_ reservation: Reservation, info: Blocker?) {
        let (toResume, dropped): ([CheckedContinuation<Void, Never>], Bool) = withLock {
            guard let n = holds[reservation.id] else { return ([], false) }
            var dropped = false
            if let info, var list = holdInfo[reservation.id], let at = list.firstIndex(of: info) {
                list.remove(at: at)
                if list.isEmpty { holdInfo.removeValue(forKey: reservation.id) } else { holdInfo[reservation.id] = list }
            }
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
            guard used + bytes <= capacity else { return (false, settled) }
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

    /// Who is using the entries other than `ignoring` (committed, booked, held), in a stable order.
    private func blockersLocked(ignoring: Set<UUID>) -> [Blocker] {
        holdInfo.filter { ledger[$0.key] != nil && !ignoring.contains($0.key) }
            .sorted { $0.key.uuidString < $1.key.uuidString }
            .flatMap(\.value)
    }

    private func currentBlockers(ignoring: Set<UUID>) -> [Blocker] { withLock { blockersLocked(ignoring: ignoring) } }

    /// Some committed, booked entry other than `ignoring` is in use (SC-2a). Call with `lock` held.
    private func othersHeldLocked(ignoring: Set<UUID>) -> Bool {
        holds.contains { ledger[$0.key] != nil && !ignoring.contains($0.key) }
    }

    /// Some committed, booked entry other than `ignoring` is pinned (SC-2). Call with `lock` held.
    private func othersPinnedLocked(ignoring: Set<UUID>) -> Bool {
        pinnedIDs.contains { ledger[$0] != nil && !ignoring.contains($0) }
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
                                  pinnedBlocks: Bool = false, finalize: () -> Void) async -> (Verdict, [Blocker]) {
        let waiterID = UUID()
        var verdict = Verdict.finalized
        var blockers: [Blocker] = []
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
                // SC-2: a pinned idle entry that this requester may not take stands in the way like one in use
                let inUse = deadline != nil && (othersHeldLocked(ignoring: mine)
                                                || (pinnedBlocks && othersPinnedLocked(ignoring: mine)))
                if inFlight || inUse {
                    if let deadline, (ContinuousClock.now >= deadline || Task.isCancelled) {
                        verdict = .timedOut
                        blockers = blockersLocked(ignoring: mine)
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
        return (verdict, blockers)
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
        capActive: { ServeMLXMemory.shared.isCapActive },
        clear: { Memory.clearCache() })

    private let capActive: @Sendable () -> Bool
    private let clear: @Sendable () -> Void
    private let delay: Duration
    private let retryEvery: Duration
    private let giveUpAfter: Duration
    private let lock = NSLock()
    private var scheduled = false
    private var requests = 0
    private var cleared = 0

    /// - Parameter capActive: whether a server generation's scoped `cacheLimit` is in force (`ServeMLXMemory.isCapActive`).
    init(capActive: @escaping @Sendable () -> Bool,
         clear: @escaping @Sendable () -> Void,
         delay: Duration = .milliseconds(300), retryEvery: Duration = .seconds(1), giveUpAfter: Duration = .seconds(60)) {
        self.capActive = capActive
        self.clear = clear
        self.delay = delay
        self.retryEvery = retryEvery
        self.giveUpAfter = giveUpAfter
    }

    /// How many `Memory.clearCache()` calls this instance has made (tests).
    var clearCount: Int { lock.lock(); defer { lock.unlock() }; return cleared }
    var requestCount: Int { lock.lock(); defer { lock.unlock() }; return requests }

    /// SC-2c: clear **now**, on the caller's path — used right after the budget evicted an idle entry to make room for a
    /// load that is about to start, so the evicted engine's MLX buffers (5.7 GB for dots.ocr, ≈ 6 GB for a VLM with its
    /// cache) are gone before the next load allocates, not 0.3 s later. Skipped while a server generation holds its
    /// scoped cache limit (`requestClear` is the deferred path for that and for "last holder returned"). The evicted
    /// value is already dropped by the participant when this runs, so there is something to free.
    func clearNow() {
        guard !capActive() else { requestClear(); return }
        clear()
        lock.lock(); cleared += 1; lock.unlock()
    }

    func requestClear() {
        lock.lock()
        requests += 1
        guard !scheduled else { lock.unlock(); return }
        scheduled = true
        lock.unlock()
        Task.detached(priority: .utility) { [self] in
            try? await Task.sleep(for: delay)
            let start = ContinuousClock.now
            while capActive(), ContinuousClock.now - start < giveUpAfter {
                try? await Task.sleep(for: retryEvery)       // a server generation holds its scoped limit
            }
            lock.lock(); scheduled = false; lock.unlock()   // requests after this point schedule the next clear
            if !capActive() {
                clear()
                lock.lock(); cleared += 1; lock.unlock()
            }
        }
    }
}
