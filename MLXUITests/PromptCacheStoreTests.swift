import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

/// PC-1 — the prompt-cache store and its place in the shared `MemoryBudget`. Pure logic: the
/// snapshot is a fake, so no model is needed (the real-model equivalence tests are PC-2's).
struct PromptCacheStoreTests {

    private static let gb = Int64(1_073_741_824)
    private static let mb = Int64(1_048_576)

    /// A snapshot stand-in. `offset` is always 0 — like a hybrid's layer 0 (T17) — so a store that
    /// read the length from the snapshot instead of the tokens would see every entry as empty.
    private nonisolated struct Fake: Sendable, Equatable {
        let name: String
        var copies = 0
        let offset = 0
    }

    private func store(_ budget: MemoryBudget, capFraction: Double = 0.25, minimumReuse: Int = 256,
                       maxPerModel: Int = 4) -> PromptCacheStore<Fake> {
        PromptCacheStore<Fake>(budget: budget, capFraction: capFraction, minimumReuse: minimumReuse,
                               maxPerModel: maxPerModel, copy: { var c = $0; c.copies += 1; return c })
    }

    private func key(_ dir: String = "/models/a", config: String = "c1", parameters: String = "p",
                     library: String = "lib1") -> PromptCacheModelKey {
        PromptCacheModelKey(directory: dir, configSHA256: config, parameters: parameters, libraryVersion: library)
    }

    private func tokens(_ n: Int, from start: Int = 0) -> [Int] { Array(start..<(start + n)) }

    // MARK: - Lookup

    @Test func anExactPrefixIsAHitAndTheCallerGetsACopy() async {
        let s = store(MemoryBudget(capacityBytes: 100 * Self.gb))
        await s.insert(key: key(), tokens: tokens(300), snapshot: Fake(name: "s"), bytes: Self.mb, trimmable: false)
        let hit = await s.lookup(key: key(), prompt: tokens(350))
        #expect(hit?.reusedLength == 300)
        #expect(hit?.trimBy == 0)
        #expect(hit?.snapshot.copies == 1)                       // T4: never the stored one
        #expect(await s.lookup(key: key(), prompt: tokens(350))?.snapshot.copies == 1)   // …and the store's stays pristine
    }

    @Test func oneDifferentTokenInsideANonTrimmableSnapshotIsAMiss() async {
        let s = store(MemoryBudget(capacityBytes: 100 * Self.gb))
        await s.insert(key: key(), tokens: tokens(300), snapshot: Fake(name: "s"), bytes: Self.mb, trimmable: false)
        var prompt = tokens(350)
        prompt[299] = -1                                         // the very last snapshot token differs
        #expect(await s.lookup(key: key(), prompt: prompt) == nil)
        prompt = tokens(350); prompt[0] = -1
        #expect(await s.lookup(key: key(), prompt: prompt) == nil)
    }

    @Test func aTrimmableSnapshotMatchesByCommonPrefixAndSaysHowMuchToTrim() async {
        let s = store(MemoryBudget(capacityBytes: 100 * Self.gb))
        await s.insert(key: key(), tokens: tokens(600), snapshot: Fake(name: "s"), bytes: Self.mb, trimmable: true)
        var prompt = tokens(700)
        prompt[400] = -1                                         // shares the first 400 tokens
        let hit = await s.lookup(key: key(), prompt: prompt)
        #expect(hit?.reusedLength == 400)
        #expect(hit?.trimBy == 200)
        #expect(hit?.storedLength == 600)
    }

    @Test func atLeastOneTailTokenIsAlwaysLeft() async {
        let s = store(MemoryBudget(capacityBytes: 100 * Self.gb))
        let k = key()
        await s.insert(key: k, tokens: tokens(300), snapshot: Fake(name: "exact"), bytes: Self.mb, trimmable: false)
        // the prompt *is* the snapshot: a non-trimmable one can't give back a tail token → miss
        #expect(await s.lookup(key: k, prompt: tokens(300)) == nil)
        #expect(await s.lookup(key: k, prompt: tokens(301))?.reusedLength == 300)   // one tail token is enough

        let t = store(MemoryBudget(capacityBytes: 100 * Self.gb))
        await t.insert(key: k, tokens: tokens(300), snapshot: Fake(name: "trim"), bytes: Self.mb, trimmable: true)
        let hit = await t.lookup(key: k, prompt: tokens(300))                       // trimmable → n − 1
        #expect(hit?.reusedLength == 299)
        #expect(hit?.trimBy == 1)
        // a shorter prompt that is a prefix of a longer trimmable snapshot
        #expect(await t.lookup(key: k, prompt: tokens(280))?.reusedLength == 279)
    }

    @Test func belowTheMinimumReuseIsAMiss() async {
        let s = store(MemoryBudget(capacityBytes: 100 * Self.gb), minimumReuse: 256)
        await s.insert(key: key(), tokens: tokens(255), snapshot: Fake(name: "s"), bytes: Self.mb, trimmable: false)
        #expect(await s.lookup(key: key(), prompt: tokens(400)) == nil)
        let lenient = store(MemoryBudget(capacityBytes: 100 * Self.gb), minimumReuse: 100)
        await lenient.insert(key: key(), tokens: tokens(255), snapshot: Fake(name: "s"), bytes: Self.mb, trimmable: false)
        #expect(await lenient.lookup(key: key(), prompt: tokens(400))?.reusedLength == 255)
    }

    @Test func theLongestReusablePrefixWinsAndTheLengthComesFromTheTokensNotTheSnapshot() async {
        let s = store(MemoryBudget(capacityBytes: 100 * Self.gb))
        await s.insert(key: key(), tokens: tokens(300), snapshot: Fake(name: "S"), bytes: Self.mb, trimmable: false)
        await s.insert(key: key(), tokens: tokens(900), snapshot: Fake(name: "T"), bytes: Self.mb, trimmable: false)
        let hit = await s.lookup(key: key(), prompt: tokens(1000))
        #expect(hit?.snapshot.name == "T")
        #expect(hit?.reusedLength == 900)                        // T17: every Fake has offset 0
        #expect(await s.lookup(key: key(), prompt: tokens(500))?.snapshot.name == "S")
    }

    @Test func aDifferentModelKeyNeverMatches() async {
        let s = store(MemoryBudget(capacityBytes: 100 * Self.gb))
        await s.insert(key: key(), tokens: tokens(300), snapshot: Fake(name: "s"), bytes: Self.mb, trimmable: true)
        let prompt = tokens(400)
        #expect(await s.lookup(key: key(), prompt: prompt) != nil)
        #expect(await s.lookup(key: key("/models/b"), prompt: prompt) == nil)              // another directory
        #expect(await s.lookup(key: key(config: "c2"), prompt: prompt) == nil)             // config.json changed
        #expect(await s.lookup(key: key(parameters: "kvBits=4"), prompt: prompt) == nil)   // cache-shaping parameter
        #expect(await s.lookup(key: key(library: "lib2"), prompt: prompt) == nil)          // library bump
    }

    @Test func theModelKeyHashesTheConfigFile() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("pc1-key-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("{\"a\":1}".utf8).write(to: dir.appendingPathComponent("config.json"))
        let first = PromptCacheModelKey(directory: dir)
        #expect(first == PromptCacheModelKey(directory: dir))
        #expect(first.directory == ModelContainerPool.key(for: dir))
        try Data("{\"a\":2}".utf8).write(to: dir.appendingPathComponent("config.json"))
        #expect(PromptCacheModelKey(directory: dir) != first)
        try FileManager.default.removeItem(at: dir.appendingPathComponent("config.json"))
        #expect(PromptCacheModelKey(directory: dir).configSHA256 == "none")
    }

    // MARK: - Insert

    @Test func aDuplicateIsNotStoredTwice() async {
        let s = store(MemoryBudget(capacityBytes: 100 * Self.gb))
        #expect(await s.insert(key: key(), tokens: tokens(300), snapshot: Fake(name: "a"), bytes: Self.mb, trimmable: false))
        #expect(await s.insert(key: key(), tokens: tokens(300), snapshot: Fake(name: "b"), bytes: Self.mb, trimmable: false) == false)
        #expect(await s.count == 1)
        #expect(await s.totalBytes == Self.mb)
        // same length, different tokens → not a duplicate
        #expect(await s.insert(key: key(), tokens: tokens(300, from: 1), snapshot: Fake(name: "c"), bytes: Self.mb, trimmable: false))
        #expect(await s.count == 2)
    }

    @Test func atMostKSnapshotsPerModelAndTheOldestGoesFirst() async {
        let s = store(MemoryBudget(capacityBytes: 100 * Self.gb), maxPerModel: 2)
        let k = key(), other = key("/models/b")
        await s.insert(key: k, tokens: tokens(300), snapshot: Fake(name: "1"), bytes: Self.mb, trimmable: false)
        await s.insert(key: k, tokens: tokens(400), snapshot: Fake(name: "2"), bytes: Self.mb, trimmable: false)
        await s.insert(key: other, tokens: tokens(300), snapshot: Fake(name: "x"), bytes: Self.mb, trimmable: false)
        await s.insert(key: k, tokens: tokens(500), snapshot: Fake(name: "3"), bytes: Self.mb, trimmable: false)
        #expect(await s.storedLengths(for: k) == [500, 400])         // 300 was the oldest
        #expect(await s.count(for: other) == 1)                      // another model's budget of K is its own
    }

    @Test func evictionIsLRUByBytesUnderTheTwentyFivePercentCap() async {
        let budget = MemoryBudget(capacityBytes: 400 * Self.mb)      // cap = 100 MB
        let s = store(budget)
        #expect(await s.capBytes == 100 * Self.mb)
        let k = key()
        await s.insert(key: k, tokens: tokens(300), snapshot: Fake(name: "a"), bytes: 40 * Self.mb, trimmable: false)
        await s.insert(key: k, tokens: tokens(400), snapshot: Fake(name: "b"), bytes: 40 * Self.mb, trimmable: false)
        _ = await s.lookup(key: k, prompt: tokens(350))              // touches "a"; "b" is now the LRU
        #expect(await s.insert(key: k, tokens: tokens(500), snapshot: Fake(name: "c"), bytes: 40 * Self.mb, trimmable: false))
        #expect(await s.storedLengths(for: k).sorted() == [300, 500])
        #expect(await s.totalBytes == 80 * Self.mb)
        #expect(budget.usedBytes == 80 * Self.mb)                    // the budget's ledger agrees
    }

    @Test func aSnapshotOverTheCapIsRefusedAndTheTotalNeverPassesIt() async {
        let budget = MemoryBudget(capacityBytes: 400 * Self.mb)
        let s = store(budget, maxPerModel: 50)
        #expect(await s.insert(key: key(), tokens: tokens(300), snapshot: Fake(name: "big"), bytes: 101 * Self.mb, trimmable: false) == false)
        #expect(await s.count == 0)
        for i in 0..<20 {
            await s.insert(key: key(), tokens: tokens(300 + i), snapshot: Fake(name: "n\(i)"), bytes: 30 * Self.mb, trimmable: false)
            #expect(await s.totalBytes <= 100 * Self.mb)
            #expect(budget.usedBytes <= 100 * Self.mb)
        }
        #expect(await s.count == 3)
    }

    // MARK: - Memory: snapshots go before weights (ruling A1, T11)

    private nonisolated final class Weights: Sendable { let name: String; init(_ n: String) { name = n } }

    @Test func aWeightsLoadDropsSnapshotsBeforeThePoolsOwnLRU() async throws {
        let budget = MemoryBudget(capacityBytes: 10 * Self.gb)
        let pool = ResidentPool<Weights>(budget: budget)
        let s = store(budget)                                        // cap 2.5 GB
        _ = try await pool.value(for: "A", bytes: 5 * Self.gb) { Weights("A") }
        #expect(await s.insert(key: key(), tokens: tokens(300), snapshot: Fake(name: "s"), bytes: 2 * Self.gb, trimmable: false))
        #expect(budget.usedBytes == 7 * Self.gb)

        // 7 + 5 = 12 > 10. Before PC-1 the pool dropped model A (its own LRU) and the snapshot stayed.
        _ = try await pool.value(for: "B", bytes: 5 * Self.gb) { Weights("B") }
        #expect(await pool.residentKeys.sorted() == ["A", "B"])      // A survived
        #expect(await s.count == 0)                                  // the snapshot went instead
        #expect(budget.usedBytes == 10 * Self.gb)
    }

    @Test func anEngineCacheReservationAlsoDropsSnapshotsBeforeAnyWeights() async throws {
        let budget = MemoryBudget(capacityBytes: 10 * Self.gb)
        let pool = ResidentPool<Weights>(budget: budget)
        let cache = EngineCache(budget: budget)
        let s = store(budget)
        _ = try await pool.value(for: "A", bytes: 5 * Self.gb) { Weights("A") }
        await s.insert(key: key(), tokens: tokens(300), snapshot: Fake(name: "s"), bytes: 2 * Self.gb, trimmable: false)

        // The requester (EngineCache) is a *different* participant from the pool that holds weights:
        // the snapshot still goes first; the pool's A is not asked.
        _ = try await cache.stage(for: makeEntry(id: "e", modelType: .asr, ramGB: 4), config: .default) { _, _ in NoopStage() }
        #expect(await pool.residentKeys == ["A"])
        #expect(await s.count == 0)
    }

    @Test func weightsAreStillEvictedOnceNoSnapshotsRemain() async throws {
        let budget = MemoryBudget(capacityBytes: 10 * Self.gb)
        let pool = ResidentPool<Weights>(budget: budget)
        let s = store(budget)
        _ = try await pool.value(for: "A", bytes: 6 * Self.gb) { Weights("A") }
        await s.insert(key: key(), tokens: tokens(300), snapshot: Fake(name: "s"), bytes: 1 * Self.gb, trimmable: false)
        _ = try await pool.value(for: "B", bytes: 6 * Self.gb) { Weights("B") }     // needs both gone
        #expect(await pool.residentKeys == ["B"])
        #expect(await s.count == 0)
        #expect(budget.usedBytes == 6 * Self.gb)
    }

    @Test func insertingASnapshotNeverEvictsWeights() async throws {
        let budget = MemoryBudget(capacityBytes: 10 * Self.gb)
        let pool = ResidentPool<Weights>(budget: budget)
        let s = store(budget)
        _ = try await pool.value(for: "A", bytes: 5 * Self.gb) { Weights("A") }
        _ = try await pool.value(for: "B", bytes: 4 * Self.gb) { Weights("B") }     // 1 GB free
        #expect(await s.insert(key: key(), tokens: tokens(300), snapshot: Fake(name: "big"), bytes: 2 * Self.gb, trimmable: false) == false)
        #expect(await pool.residentKeys.sorted() == ["A", "B"])
        #expect(await s.insert(key: key(), tokens: tokens(300), snapshot: Fake(name: "small"), bytes: 512 * Self.mb, trimmable: false))
        #expect(await pool.residentKeys.sorted() == ["A", "B"])
        #expect(budget.usedBytes == 9 * Self.gb + 512 * Self.mb)
    }

    @Test func theBudgetsPriorityOrderIsLowestFirstThenOwnThenOthers() async {
        // The ordering itself, with recording participants (no pool, no store).
        let budget = MemoryBudget(capacityBytes: 100)
        let log = OrderLog()
        let low = Recorder(name: "low", priority: -1, log: log)
        let own = Recorder(name: "own", priority: 0, log: log)
        let peer = Recorder(name: "peer", priority: 0, log: log)
        let high = Recorder(name: "high", priority: 5, log: log)
        [low, own, peer, high].forEach { budget.register($0) }
        let blocker = budget.tryReserve(bytes: 100)                  // the budget is full…
        if let blocker { budget.commit(blocker) }                    // …with something resident, nothing in flight
        // A Recorder frees nothing, so `reserve` walks the whole chain once and then proceeds over
        // budget (step 4). The order of asking is what is under test.
        _ = await budget.reserve(bytes: 1, for: own, label: "t")
        #expect(await log.names == ["low", "own", "peer", "high"])
    }

    // MARK: - Uninstall

    @Test func uninstallDropsThatModelsSnapshotsAndFreesTheirBytes() async {
        let budget = MemoryBudget(capacityBytes: 100 * Self.gb)
        let s = store(budget)
        let a = key("/models/a"), b = key("/models/b")
        await s.insert(key: a, tokens: tokens(300), snapshot: Fake(name: "a1"), bytes: Self.mb, trimmable: false)
        await s.insert(key: a, tokens: tokens(400), snapshot: Fake(name: "a2"), bytes: Self.mb, trimmable: false)
        await s.insert(key: b, tokens: tokens(300), snapshot: Fake(name: "b1"), bytes: Self.mb, trimmable: false)
        #expect(await s.evict(directory: "/models/a") == 2)
        #expect(await s.count == 1)
        #expect(await s.lookup(key: a, prompt: tokens(500)) == nil)
        #expect(await s.lookup(key: b, prompt: tokens(500)) != nil)
        #expect(budget.usedBytes == Self.mb)
    }

    @MainActor
    @Test func theControllersUninstallPathEvictsThePromptCacheToo() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let recorded = OrderLog()
        let server = LocalServer(settings: ServeSettings(servedModelIDs: []), defaults: nil, listenPort: 0,
                                 environment: ServeEnvironment(servedModelIDs: { [] }, installed: { InstalledModelIndex(entries: []) },
                                                               appleFoundationReadiness: { nil }, created: 1))
        let controller = LocalServerController(
            server: server, defaults: UserDefaults(suiteName: "pc1-\(UUID().uuidString)")!,
            installed: { InstalledModelIndex(entries: []) }, appleFoundation: { .ready }, catalogLoaded: { true },
            evict: { await recorded.add("weights:\($0)") },
            evictPromptCache: { await recorded.add("snapshots:\($0)") })
        await controller.modelUninstalled("mlx-community/Qwen3-8B-4bit")
        #expect(await recorded.names == ["weights:mlx-community/Qwen3-8B-4bit", "snapshots:mlx-community/Qwen3-8B-4bit"])
    }

    @Test func theDefaultHookEvictsTheRealStoreByDirectory() async {
        let directory = ModelStore.shared.directory(forHFModelID: "mlx-community/PC1-Hook-Test")
        let k = PromptCacheModelKey(directory: ModelContainerPool.key(for: directory), configSHA256: "x")
        let store = PromptCacheShared.store
        _ = await store.insert(key: k, tokens: tokens(300), snapshot: KVSnapshot(caches: [KVCacheSimple()]), bytes: 1024, trimmable: false)
        #expect(await store.count(for: k) == 1)
        await store.evict(directory: ModelContainerPool.key(for: directory))
        #expect(await store.count(for: k) == 0)
    }
}

private actor OrderLog {
    private(set) var names: [String] = []
    func add(_ n: String) { names.append(n) }
}

/// A participant that records that it was asked and (once) pretends to free something.
private final class Recorder: MemoryBudgetParticipant, @unchecked Sendable {
    let name: String
    let evictionPriority: Int
    let log: OrderLog
    init(name: String, priority: Int, log: OrderLog) {
        self.name = name; self.evictionPriority = priority; self.log = log
    }
    func evictLeastRecentlyUsed() async -> Bool {
        await log.add(name)
        return false
    }
}

private nonisolated struct NoopStage: PipelineStage {
    let id = "noop", name = "Noop"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media { input }
}
