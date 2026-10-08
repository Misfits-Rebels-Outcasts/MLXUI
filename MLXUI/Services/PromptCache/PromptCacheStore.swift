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
nonisolated struct PromptCacheHit<Snapshot: Sendable>: Sendable {
    /// A **copy** of the stored snapshot (T4) — the store's own is never handed out.
    let snapshot: Snapshot
    /// How many prompt tokens the snapshot covers once trimmed: prefill starts here. Always ≤ n − 1.
    let reusedLength: Int
    /// Tokens to trim off the copy first (0 for an exact-prefix hit; > 0 only for a trimmable one).
    let trimBy: Int
    /// The stored snapshot's own length, for logs.
    let storedLength: Int
}

nonisolated enum PromptCacheDefaults {
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

    /// - Parameter copy: how to duplicate a snapshot (`KVCache.copy()` per layer). Called on every
    ///   hit, so the stored one stays pristine.
    init(budget: MemoryBudget = .shared,
         capFraction: Double = PromptCacheDefaults.capFraction,
         minimumReuse: Int = PromptCacheDefaults.minimumReuse,
         maxPerModel: Int = PromptCacheDefaults.maxPerModel,
         copy: @escaping @Sendable (Snapshot) -> Snapshot) {
        self.budget = budget
        self.capBytes = Int64(Double(budget.capacityBytes) * capFraction)
        self.minimumReuse = minimumReuse
        self.maxPerModel = maxPerModel
        self.copySnapshot = copy
        budget.register(self)
    }

    // MARK: MemoryBudgetParticipant

    /// Lowest priority: dropped before any weights, on every path (ruling A1, T11).
    nonisolated var evictionPriority: Int { MemoryBudget.snapshotPriority }

    func evictLeastRecentlyUsed() -> Bool { dropLeastRecentlyUsed(matching: { _ in true }) }

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
        return PromptCacheHit(snapshot: copySnapshot(entry.snapshot), reusedLength: best.reuse,
                              trimBy: entry.tokens.count - best.reuse, storedLength: entry.tokens.count)
    }

    // MARK: Insert

    /// Stores `snapshot`, which covers exactly `tokens`. Returns whether it was stored.
    ///
    /// Never fails a request and never evicts weights (rule 8): it makes room by dropping its own
    /// LRU, then asks the budget for room *without* evicting anyone else (`tryReserve`); if that
    /// still doesn't fit the snapshot is simply not kept.
    @discardableResult
    func insert(key: PromptCacheModelKey, tokens: [Int], snapshot: Snapshot, bytes: Int64,
                trimmable: Bool) -> Bool {
        guard bytes > 0, !tokens.isEmpty, bytes <= capBytes else { return false }
        let digest = Self.digest(tokens)
        if let i = entries.firstIndex(where: {
            $0.key == key && $0.tokens.count == tokens.count && $0.tokenDigest == digest && $0.tokens == tokens
        }) {
            clock += 1
            entries[i].lastUsed = clock           // a duplicate refreshes recency, stores nothing
            return false
        }
        while entries.filter({ $0.key == key }).count >= maxPerModel {
            guard dropLeastRecentlyUsed(matching: { $0.key == key }) else { return false }
        }
        while totalBytes + bytes > capBytes {
            guard dropLeastRecentlyUsed(matching: { _ in true }) else { return false }
        }
        var reservation = budget.tryReserve(bytes: bytes)
        while reservation == nil {
            guard dropLeastRecentlyUsed(matching: { _ in true }) else { return false }
            reservation = budget.tryReserve(bytes: bytes)
        }
        guard let reservation else { return false }
        budget.commit(reservation)
        clock += 1
        nextID += 1
        entries.append(Entry(id: nextID, key: key, tokens: tokens, tokenDigest: digest, snapshot: snapshot,
                             bytes: bytes, trimmable: trimmable, reservation: reservation, lastUsed: clock))
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
