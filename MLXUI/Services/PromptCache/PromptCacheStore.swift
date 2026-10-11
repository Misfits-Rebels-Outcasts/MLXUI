import CryptoKit
import Foundation
import MLX
import MLXLMCommon

// PC-1 — the Local Server's prompt cache, in RAM (`RSI/DelegatePromptCacheBacklog.md` §PC-1).
//
// The store keeps **snapshots** of a model's KV cache at prompt checkpoints. A later request whose
// prompt starts with the same tokens takes a *copy* of the snapshot and prefills only the tail.
// Nothing here knows about MLX arrays except `KVSnapshot` at the bottom: the store is generic over
// its snapshot type (the `ResidentPool<Value>` pattern, F8) so the logic is tested with a fake.
//
// Privacy (rule 9): the token ids are the prompt. They live in this actor's RAM only — never in a
// log, a file name or the request log. PC-A writes nothing to disk.

/// What makes two snapshots comparable: the same weights on disk, the same config, the same
/// cache-shaping parameters, the same library. Any difference means "no match".
nonisolated struct PromptCacheModelKey: Hashable, Sendable {
    /// Standardized model directory path — the key `ModelContainerPool` uses (F8).
    let directory: String
    let configSHA256: String
    /// `kvBits`, `maxKVSize`, `prefillStepSize` (all default today), as one string.
    let parameters: String
    let libraryVersion: String

    /// The pinned `mlx-swift-lm` (`Package.resolved`). Bump it with the package.
    static let pinnedLibraryVersion = "mlx-swift-lm 3.31.3"
    static let defaultParameters = "kvBits=nil;maxKVSize=nil;prefillStepSize=512"

    init(directory: String, configSHA256: String, parameters: String = Self.defaultParameters,
         libraryVersion: String = Self.pinnedLibraryVersion) {
        self.directory = directory
        self.configSHA256 = configSHA256
        self.parameters = parameters
        self.libraryVersion = libraryVersion
    }

    /// Reads `config.json` from `directory`; a missing file hashes as "none" (it still keys the dir).
    init(directory: URL, parameters: String = Self.defaultParameters,
         libraryVersion: String = Self.pinnedLibraryVersion) {
        let data = try? Data(contentsOf: directory.appendingPathComponent("config.json"))
        self.init(directory: ModelContainerPool.key(for: directory),
                  configSHA256: data.map { Self.sha256($0) } ?? "none",
                  parameters: parameters, libraryVersion: libraryVersion)
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

/// A reusable prefix handed back by `lookup`.
///
/// **The copy is not made here.** `KVCache.copy()` builds lazy MLX slices; the store is an actor
/// whose executor is not the model container's, and mlx-swift's graph construction is not
/// documented as thread-safe off the thread that evaluates it (PC-1 review, carry-over 3; journal
/// 2026-420). So the hit carries the stored reference and the copy step, and the backend calls
/// `makeCopy()` **inside `container.perform`** — and only there (T4: never hand the model the
/// stored one). The stored snapshot is never mutated, so holding its reference across the actor
/// boundary is safe.
nonisolated struct PromptCacheHit<Snapshot: Sendable>: Sendable {
    let stored: Snapshot
    let copyStep: @Sendable (Snapshot) -> Snapshot
    /// How many prompt tokens the snapshot covers once trimmed: prefill starts here. Always ≤ n − 1.
    let reusedLength: Int
    /// Tokens to trim off the copy first (0 for an exact-prefix hit; > 0 only for a trimmable one).
    let trimBy: Int
    /// The stored snapshot's own length, for logs.
    let storedLength: Int

    /// A fresh copy of the stored snapshot. Call it where the model runs.
    func makeCopy() -> Snapshot { copyStep(stored) }
}

/// S3.1d — which snapshot a full store keeps on an 18 GiB Mac, where only one large snapshot fits. One internal constant
/// (`PromptCacheDefaults.retention`), no setting; both policies keep every safety rule (weights win, snapshots are never
/// reserved beyond the budget, the reply trade's floor is unchanged).
nonisolated enum SnapshotRetention: Sendable {
    /// S3.1b: the newest snapshot wins; the LRU entry — usually the system-boundary one — is displaced first.
    case newestWins
    /// S3.1c: a model's newest system-boundary snapshot is protected from displacement by other snapshots.
    case protectSystem
}

nonisolated enum PromptCacheDefaults {
    /// Chosen by the journal-`2026-456` A/B (median total session time over two valid runs per policy; within 5% → `.newestWins`).
    static let retention: SnapshotRetention = .newestWins
    /// Below this a hit is not worth a copy (implementer's call, pending owner confirmation; PC-0
    /// item 5: a 2K reuse already saves 5.6 s, so 256 tokens is conservative).
    static let minimumReuse = 256
    /// Snapshots kept per model (implementer's call, pending owner confirmation: two checkpoints
    /// per request, E1, times the last two turns).
    static let maxPerModel = 4
    /// Ruling A1: snapshots may hold at most a quarter of the shared budget.
    static let capFraction = 0.25
}

actor PromptCacheStore<Snapshot: Sendable>: MemoryBudgetParticipant {
    private struct Entry {
        let id: UInt64
        let key: PromptCacheModelKey
        /// The prompt prefix this snapshot covers. **Its count is the snapshot's length** (T17:
        /// never read a cache's own `offset` — a hybrid's layer 0 is a `MambaCache`, offset 0).
        let tokens: [Int]
        let tokenDigest: String
        let snapshot: Snapshot
        let bytes: Int64
        let trimmable: Bool
        /// S3.1c: the snapshot of the end of the system content (see `PromptCheckpoints.systemBoundary`).
        let systemBoundary: Bool
        let reservation: MemoryBudget.Reservation
        var lastUsed: UInt64
    }

    private var entries: [Entry] = []
    private var clock: UInt64 = 0
    private var nextID: UInt64 = 0

    private let budget: MemoryBudget
    private let copySnapshot: @Sendable (Snapshot) -> Snapshot
    let capBytes: Int64
    let minimumReuse: Int
    let maxPerModel: Int
    let retention: SnapshotRetention

    /// - Parameter copy: how to duplicate a snapshot (`KVCache.copy()` per layer). Handed to every
    ///   hit as `copyStep`; the backend runs it inside `container.perform`.
    init(budget: MemoryBudget = .shared,
         capFraction: Double = PromptCacheDefaults.capFraction,
         minimumReuse: Int = PromptCacheDefaults.minimumReuse,
         maxPerModel: Int = PromptCacheDefaults.maxPerModel,
         retention: SnapshotRetention = PromptCacheDefaults.retention,
         copy: @escaping @Sendable (Snapshot) -> Snapshot) {
        self.budget = budget
        self.capBytes = Int64(Double(budget.capacityBytes) * capFraction)
        self.minimumReuse = minimumReuse
        self.maxPerModel = maxPerModel
        self.retention = retention
        self.copySnapshot = copy
        budget.register(self)
    }

    // MARK: MemoryBudgetParticipant

    /// Lowest priority: dropped before any weights, on every path (ruling A1, T11).
    nonisolated var evictionPriority: Int { MemoryBudget.snapshotPriority }

    /// Budget pressure (weights, or anything that outranks a snapshot): the unprotected LRU goes first, and when only
    /// protected system-boundary snapshots are left they go too — **weights win** (PC rule 8, S3.1c).
    func evictLeastRecentlyUsed() -> Bool {
        let shielded = protectedIDs(replacing: nil)
        return dropLeastRecentlyUsed(matching: { !shielded.contains($0.id) })
            || dropLeastRecentlyUsed(matching: { _ in true })
    }

    // MARK: Protection (S3.1c)

    /// **A model's most recent system-boundary snapshot is protected from displacement by other snapshots** (count cap,
    /// byte cap, budget room). It is what a compaction — a request that shares only the system prompt and tools with
    /// the conversation before it — reads (journal `2026-455`), and at ~0.6 GB it is small next to the turn snapshots
    /// that otherwise push it out as the conversation grows. One protected entry per model (`key.directory`): a newer
    /// system-boundary snapshot of the same model moves the protection, and the older one is an ordinary entry.
    /// `replacing`: the directory whose protection a system-boundary snapshot being stored is about to take over.
    private func protectedIDs(replacing directory: String?) -> Set<UInt64> {
        guard retention == .protectSystem else { return [] }                 // S3.1d: .newestWins protects nothing
        var newest: [String: Entry] = [:]
        for entry in entries where entry.systemBoundary && entry.key.directory != directory {
            if let seen = newest[entry.key.directory], seen.id > entry.id { continue }
            newest[entry.key.directory] = entry
        }
        return Set(newest.values.map(\.id))
    }

    private func protectedBytes(replacing directory: String?) -> Int64 {
        let shielded = protectedIDs(replacing: directory)
        return entries.filter { shielded.contains($0.id) }.reduce(0) { $0 + $1.bytes }
    }

    /// Lengths of the protected snapshots of `directory` (tests, pane).
    func protectedLengths(forDirectory directory: String) -> [Int] {
        let shielded = protectedIDs(replacing: nil)
        return entries.filter { shielded.contains($0.id) && $0.key.directory == directory }.map { $0.tokens.count }
    }

    // MARK: Lookup

    /// The longest reusable prefix of `prompt` for `key`, or nil.
    ///
    /// - An **exact-prefix** snapshot (its tokens are the first *L* tokens of the prompt) is used
    ///   as is. That works for every cache class, hybrids included.
    /// - A **trimmable** snapshot also matches by longest common prefix; the caller trims the copy.
    /// - At least one tail token is always left (T6): reuse is clamped to `n − 1`, which is only
    ///   possible for a trimmable snapshot.
    func lookup(key: PromptCacheModelKey, prompt: [Int]) -> PromptCacheHit<Snapshot>? {
        let n = prompt.count
        guard n >= 2 else { return nil }
        var best: (index: Int, reuse: Int)?
        for (i, entry) in entries.enumerated() where entry.key == key {
            let common = Self.commonPrefix(entry.tokens, prompt)
            let reuse: Int
            if common == entry.tokens.count {
                if entry.tokens.count <= n - 1 {
                    reuse = entry.tokens.count
                } else if entry.trimmable {
                    reuse = n - 1
                } else {
                    continue          // would leave no tail and can't be trimmed (T6)
                }
            } else if entry.trimmable {
                reuse = min(common, n - 1)
            } else {
                continue              // diverges inside a snapshot that can't be trimmed back
            }
            guard reuse >= minimumReuse else { continue }
            if best == nil || reuse > best!.reuse
                || (reuse == best!.reuse && entry.lastUsed > entries[best!.index].lastUsed) {
                best = (i, reuse)
            }
        }
        guard let best else { return nil }
        clock += 1
        entries[best.index].lastUsed = clock
        let entry = entries[best.index]
        return PromptCacheHit(stored: entry.snapshot, copyStep: copySnapshot, reusedLength: best.reuse,
                              trimBy: entry.tokens.count - best.reuse, storedLength: entry.tokens.count)
    }

    // MARK: Insert

    /// Whether a snapshot of `bytes` could ever be stored: the two early refusals of `insert`, asked **before**
    /// the copy is made (S3-5c, INV-1 step 5). A snapshot over the store's cap, or over the most room there could
    /// ever be (what is free now plus everything this store holds, so the store is never flushed for a snapshot
    /// that can't fit even then — PC-1 review, carry-over 2), is refused by `insert` anyway; building it first
    /// only spends a second KV copy.
    ///
    /// S3.1c: protected snapshots are not reclaimable by this one, so they count against both tests. A system-boundary
    /// snapshot (`systemBoundary`) of `key`'s model takes over that model's protection, so the old one is reclaimable.
    func wouldAccept(bytes: Int64, key: PromptCacheModelKey? = nil, systemBoundary: Bool = false) -> Bool {
        let shielded = protectedBytes(replacing: systemBoundary ? key?.directory : nil)
        guard bytes > 0, bytes <= capBytes - shielded else { return false }
        return bytes <= budget.capacityBytes - budget.usedBytes + totalBytes - shielded
    }

    /// S3.1b — by how many bytes a snapshot of `bytes` misses the room (`wouldAccept`'s second test), or `nil` when
    /// room is not why it is refused (empty, or over the store's own cap) or when it fits. Frees nothing.
    func roomShortfall(bytes: Int64, key: PromptCacheModelKey? = nil, systemBoundary: Bool = false) -> Int64? {
        let shielded = protectedBytes(replacing: systemBoundary ? key?.directory : nil)
        guard bytes > 0, bytes <= capBytes - shielded else { return nil }
        let room = budget.capacityBytes - budget.usedBytes + totalBytes - shielded
        return bytes > room ? bytes - room : nil
    }

    /// Stores `snapshot`, which covers exactly `tokens`. Returns whether it was stored.
    ///
    /// Never fails a request and never evicts weights (rule 8): it makes room by dropping its own
    /// LRU, then asks the budget for room *without* evicting anyone else (`tryReserve`); if that
    /// still doesn't fit the snapshot is simply not kept.
    @discardableResult
    func insert(key: PromptCacheModelKey, tokens: [Int], snapshot: Snapshot, bytes: Int64,
                trimmable: Bool, systemBoundary: Bool = false) -> Bool {
        guard !tokens.isEmpty, wouldAccept(bytes: bytes, key: key, systemBoundary: systemBoundary) else { return false }
        let digest = Self.digest(tokens)
        if let i = entries.firstIndex(where: {
            $0.key == key && $0.tokens.count == tokens.count && $0.tokenDigest == digest && $0.tokens == tokens
        }) {
            clock += 1
            entries[i].lastUsed = clock           // a duplicate refreshes recency, stores nothing
            return false
        }
        // Other snapshots may displace anything but the protected entries (S3.1c); when only those are left the
        // new snapshot is not stored.
        let shielded = protectedIDs(replacing: systemBoundary ? key.directory : nil)
        while entries.filter({ $0.key == key }).count >= maxPerModel {
            guard dropLeastRecentlyUsed(matching: { $0.key == key && !shielded.contains($0.id) }) else { return false }
        }
        while totalBytes + bytes > capBytes {
            guard dropLeastRecentlyUsed(matching: { !shielded.contains($0.id) }) else { return false }
        }
        var reservation = budget.tryReserve(bytes: bytes)
        while reservation == nil {
            guard dropLeastRecentlyUsed(matching: { !shielded.contains($0.id) }) else { return false }
            reservation = budget.tryReserve(bytes: bytes)
        }
        guard let reservation else { return false }
        budget.commit(reservation)
        clock += 1
        nextID += 1
        entries.append(Entry(id: nextID, key: key, tokens: tokens, tokenDigest: digest, snapshot: snapshot,
                             bytes: bytes, trimmable: trimmable, systemBoundary: systemBoundary,
                             reservation: reservation, lastUsed: clock))
        return true
    }

    // MARK: Eviction

    /// Uninstall hook (F11): drop every snapshot of the model in `directory`. Returns how many.
    @discardableResult
    func evict(directory: String) -> Int {
        removeAll { $0.key.directory == directory }
    }

    /// Drop every snapshot of exactly this key (a changed config, a pane "clear").
    @discardableResult
    func evict(key: PromptCacheModelKey) -> Int {
        removeAll { $0.key == key }
    }

    @discardableResult
    func clear() -> Int { removeAll { _ in true } }

    // MARK: Inspection (pane, tests)

    var count: Int { entries.count }
    var totalBytes: Int64 { entries.reduce(0) { $0 + $1.bytes } }
    func count(for key: PromptCacheModelKey) -> Int { entries.filter { $0.key == key }.count }
    func bytes(forDirectory directory: String) -> Int64 {
        entries.filter { $0.key.directory == directory }.reduce(0) { $0 + $1.bytes }
    }
    /// Stored lengths, most recently used first — lengths only, never tokens.
    func storedLengths(for key: PromptCacheModelKey) -> [Int] {
        entries.filter { $0.key == key }.sorted { $0.lastUsed > $1.lastUsed }.map { $0.tokens.count }
    }

    // MARK: Private

    @discardableResult
    private func dropLeastRecentlyUsed(matching: (Entry) -> Bool) -> Bool {
        guard let i = entries.indices.filter({ matching(entries[$0]) }).min(by: { entries[$0].lastUsed < entries[$1].lastUsed })
        else { return false }
        budget.release(entries.remove(at: i).reservation)
        return true
    }

    private func removeAll(where predicate: (Entry) -> Bool) -> Int {
        let gone = entries.filter(predicate)
        entries.removeAll(where: predicate)
        gone.forEach { budget.release($0.reservation) }
        return gone.count
    }

    private static func commonPrefix(_ a: [Int], _ b: [Int]) -> Int {
        var i = 0
        let limit = min(a.count, b.count)
        while i < limit, a[i] == b[i] { i += 1 }
        return i
    }

    /// Dedupe digest only; it is compared in RAM and never leaves it.
    private static func digest(_ tokens: [Int]) -> String {
        var hasher = SHA256()
        tokens.withUnsafeBytes { hasher.update(bufferPointer: $0) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - The real snapshot

/// A snapshot of a model's KV cache: one `KVCache` per layer, evaluated before it was copied (T3).
/// Reference type so the actor can hold it; the store only ever hands out `copied()` values (T4).
nonisolated final class KVSnapshot: @unchecked Sendable {
    let caches: [KVCache]
    init(caches: [KVCache]) { self.caches = caches }

    func copied() -> KVSnapshot { KVSnapshot(caches: caches.map { $0.copy() }) }

    /// Sum of the layers' state arrays — what the budget is charged.
    var bytes: Int64 { Int64(caches.reduce(0) { $0 + $1.state.reduce(0) { $0 + $1.nbytes } }) }
}

/// The app's one store (Local Server only, rule 12). PC-2 wires `MLXChatBackend` to it.
nonisolated enum PromptCacheShared {
    static let store = PromptCacheStore<KVSnapshot>(budget: .shared, copy: { $0.copied() })
}
