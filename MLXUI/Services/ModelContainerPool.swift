import Foundation
import MLXLLM
import MLXLMCommon

/// S1-1 — the **one owner of loaded LLM weights** (`RSI/plan-local-server-2026-09.md` F1).
///
/// Before this, `LLMEngine.generate` called `LLMModelFactory.loadContainer` on *every* call
/// (a "warm" flow row still reloaded 4–8 GB) and the chat sheet kept its own one-slot cache.
/// Now chat (`ModelRunner`), flow rows (`LLMEngine`) and — from S1-3 — the Local Server all
/// ask this pool. It is **not** gated by `LocalServerGate`: it is an internal speed/memory fix.
///
/// **Budget (S1-1b).** The pool and `EngineCache` reserve from **one** `MemoryBudget`
/// (`max(4 GB, RAM × 0.6)`); the pool owns LLM bytes (`EngineCache.shared` counts LLM stages as 0).
/// A reservation that doesn't fit evicts this pool's LRU first, then asks the other cache.
///
/// The LRU / single-flight logic lives in `ResidentPool` (generic) because a real
/// `ModelContainer` cannot be fabricated in a unit test.
actor ResidentPool<Value: Sendable>: MemoryBudgetParticipant {
    private struct Entry {
        let value: Value
        let bytes: Int64
        let reservation: MemoryBudget.Reservation
    }

    private struct Loading {
        let token: UUID
        let task: Task<Value, Error>
    }

    /// A value together with the claim that keeps its weights booked while it is in use (SC-2a). Return it with
    /// `release()` — in a `defer`, so cancel and throw return it too. The hold's deinit is the safety net only.
    struct Lease: Sendable {
        let value: Value
        let hold: MemoryBudget.Hold
        func release() { hold.release() }
    }

    private var entries: [String: Entry] = [:]
    /// Most-recently-used first. Eviction pops from the end.
    private var recency: [String] = []
    private var loading: [String: Loading] = [:]
    /// SC-2a: callers waiting on a load in flight that want a lease, by load token; `finish` takes that many holds
    /// in the same actor turn that makes the entry resident, so nothing can evict it before they pick them up.
    private var leasers: [UUID: [MemoryBudget.HolderKind?]] = [:]
    private var preHolds: [UUID: [(kind: MemoryBudget.HolderKind?, hold: MemoryBudget.Hold)]] = [:]
    /// SC-2a: callers currently awaiting each load (leased or not), and the loads still waiting for *room* (not yet
    /// loading). When the last caller of a load that is still waiting for room is cancelled, the wait is cancelled
    /// with it; once weights are loading nobody's cancel stops them (other callers may have joined).
    private var joined: [UUID: Int] = [:]
    private var waitingForRoom: [UUID: Task<Value, Error>] = [:]
    private let budget: MemoryBudget

    /// Reserve from a shared budget (the app's: `MemoryBudget.shared`).
    init(budget: MemoryBudget) {
        self.budget = budget
        budget.register(self)
    }

    /// A pool with its own private budget (tests; standalone use).
    convenience init(budgetBytes: Int64) {
        self.init(budget: MemoryBudget(capacityBytes: budgetBytes))
    }

    /// The resident value for `key`; else await the load already in flight; else reserve
    /// `bytes` (evicting per the shared-budget rules) and start **one** load. Concurrent callers
    /// for one key share that load. A failed load is not cached and releases its reservation.
    ///
    /// **Not leased:** nothing marks the value in use, so a later load may evict it from under the caller. Use
    /// `lease(for:…)` wherever the value is used for longer than the call (SC-2a); this one is for tests and for
    /// callers that only need the load to have happened.
    func value(for key: String, bytes: Int64,
               load: @escaping @Sendable () async throws -> Value) async throws -> Value {
        try await acquire(for: key, bytes: bytes, maxWait: nil, leased: false, holder: nil, load: load).value
    }

    /// `value(for:)`, leased (SC-2a): the entry is **in use until the lease is released**. LRU eviction skips it, and
    /// a load that can only fit by evicting it waits for the release (for at most `maxWait`, then throws
    /// `MemoryBudget.WaitError`) instead of going over budget.
    func lease(for key: String, bytes: Int64, maxWait: Duration = MemoryBudget.defaultMaxWait,
               holder: MemoryBudget.HolderKind? = nil,
               load: @escaping @Sendable () async throws -> Value) async throws -> Lease {
        let got = try await acquire(for: key, bytes: bytes, maxWait: maxWait, leased: true, holder: holder, load: load)
        return Lease(value: got.value, hold: got.hold ?? .inert)
    }

    private func acquire(for key: String, bytes: Int64, maxWait: Duration?, leased: Bool,
                         holder: MemoryBudget.HolderKind?,
                         load: @escaping @Sendable () async throws -> Value) async throws -> (value: Value, hold: MemoryBudget.Hold?) {
        if let entry = entries[key] {
            touch(key)
            return (entry.value, leased ? budget.hold(entry.reservation, label: key, holder: holder) : nil)
        }
        if let inFlight = loading[key] {
            if leased { leasers[inFlight.token, default: []].append(holder) }
            let value = try await awaitLoad(inFlight.task, token: inFlight.token)
            return (value, leased ? claimHold(inFlight.token, holder: holder) : nil)
        }

        // Install the in-flight marker *before* the first suspension, so a concurrent caller for
        // this key joins this load instead of starting a second one while we wait for room.
        let token = UUID()
        let budget = self.budget
        let task = Task {
            let reservation: MemoryBudget.Reservation
            do {
                if let maxWait {
                    reservation = try await budget.reserveBounded(bytes: bytes, for: self, label: key, maxWait: maxWait)
                } else {
                    reservation = await budget.reserve(bytes: bytes, for: self, label: key)
                }
                self.waitingForRoom.removeValue(forKey: token)
            } catch {
                self.fail(key: key, token: token)
                throw error
            }
            do {
                let value = try await load()
                self.finish(key: key, token: token, value: value, bytes: bytes, reservation: reservation)
                return value
            } catch {
                budget.release(reservation)
                self.fail(key: key, token: token)
                throw error
            }
        }
        loading[key] = Loading(token: token, task: task)
        if maxWait != nil { waitingForRoom[token] = task }
        if leased { leasers[token, default: []].append(holder) }
        let value = try await awaitLoad(task, token: token)
        return (value, leased ? claimHold(token, holder: holder) : nil)
    }

    /// Await a load, and let a cancelled caller stop a wait for room that nobody else is waiting on.
    private func awaitLoad(_ task: Task<Value, Error>, token: UUID) async throws -> Value {
        joined[token, default: 0] += 1
        defer { leaveLoad(token) }
        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            Task { await self.callerCancelled(token) }
        }
    }

    private func callerCancelled(_ token: UUID) {
        guard (joined[token] ?? 0) <= 1, let waiting = waitingForRoom[token] else { return }
        waiting.cancel()
    }

    private func leaveLoad(_ token: UUID) {
        guard let n = joined[token] else { return }
        if n <= 1 { joined.removeValue(forKey: token) } else { joined[token] = n - 1 }
    }

    /// Drop `key` (resident, or an in-flight load's result — the waiters still get their value
    /// but it is not retained). Returns whether anything was dropped. **An entry in use is doomed, not freed**
    /// (SC-2a): it leaves the pool now, its bytes stay booked until the last lease is returned.
    @discardableResult
    func evict(key: String) -> Bool {
        var dropped = false
        if let entry = entries.removeValue(forKey: key) {
            recency.removeAll { $0 == key }
            budget.releaseOrDefer(entry.reservation)
            dropped = true
        }
        if loading.removeValue(forKey: key) != nil { dropped = true }
        return dropped
    }

    /// Drop everything resident, and stop retaining anything still loading. Entries in use are doomed (see `evict`).
    func clear() {
        let reservations = entries.values.map(\.reservation)
        entries.removeAll()
        recency.removeAll()
        loading.removeAll()
        reservations.forEach { budget.releaseOrDefer($0) }
    }

    /// `MemoryBudgetParticipant`: evict the least-recently-used resident entry that holds memory **and is not in
    /// use** (SC-2a).
    func evictLeastRecentlyUsed() -> Bool { evictOldest(except: nil) }

    /// S3-8B: evict the LRU entry that holds memory and is not in use, **except** `key` (the model that is running).
    func evictLeastRecentlyUsed(except key: String) -> Bool { evictOldest(except: key) }

    private func evictOldest(except key: String?) -> Bool {
        for victim in recency.reversed() where victim != key {
            guard let entry = entries[victim], entry.bytes > 0 else { continue }
            guard budget.releaseIfUnheld(entry.reservation) else { continue }   // in use: not ours to take
            entries.removeValue(forKey: victim)
            recency.removeAll { $0 == victim }
            return true
        }
        return false
    }

    /// S3-8B: make room for a running generation that needs `totalNeedBytes` in all (weights + KV), evicting
    /// other residents for it. The running model's own ledger entry already counts toward that (it was charged
    /// at load), so only the **excess** over it is reserved: `others + need ≤ capacity`.
    ///
    /// SC-2a: other residents in use are skipped; if they are all that is in the way the request waits for one
    /// (`maxWait`, nil = don't) and then gets `nil`. The running model's own entry is never waited for.
    func reserveWorkingSet(totalNeedBytes: Int64, protecting key: String,
                           maxWait: Duration? = nil) async -> MemoryBudget.Reservation? {
        if case .granted(let reservation) = await reserveWorkingSetReporting(
            totalNeedBytes: totalNeedBytes, protecting: key, maxWait: maxWait) { return reservation }
        return nil
    }

    /// `reserveWorkingSet`, saying why it returned nothing (SC-2a-b): too large for this Mac, or busy (only models
    /// in use stood in the way for the whole bound).
    func reserveWorkingSetReporting(totalNeedBytes: Int64, protecting key: String,
                                    maxWait: Duration? = nil) async -> MemoryBudget.WorkingSetOutcome {
        let protected = entries[key]?.bytes ?? 0
        let extra = max(0, totalNeedBytes - protected)
        return await budget.reserveWorkingSetReporting(bytes: extra, for: self, protecting: protected,
                                                       protectingReservation: entries[key]?.reservation,
                                                       maxWait: maxWait) { [self] in
            await self.evictLeastRecentlyUsed(except: key)
        }
    }

    /// Resident keys, most-recently-used first.
    var residentKeys: [String] { recency }
    var totalBytes: Int64 { entries.values.reduce(0) { $0 + $1.bytes } }
    var count: Int { entries.count }

    // MARK: - Private

    private func touch(_ key: String) {
        recency.removeAll { $0 == key }
        recency.insert(key, at: 0)
    }

    private func finish(key: String, token: UUID, value: Value, bytes: Int64,
                        reservation: MemoryBudget.Reservation) {
        let waiting = leasers.removeValue(forKey: token) ?? []
        // Only retain if this load is still the current one (not evicted/cleared meanwhile).
        guard loading[key]?.token == token else {
            // Dropped while loading. Callers still get the value; if any of them lease it, the weights stay booked
            // (doomed) until they return it.
            if !waiting.isEmpty && bytes > 0 {
                preHolds[token] = waiting.map { ($0, budget.hold(reservation, label: key, holder: $0)) }
                budget.commit(reservation)
                budget.releaseOrDefer(reservation)
            } else {
                budget.release(reservation)
            }
            return
        }
        loading[key] = nil
        entries[key] = Entry(value: value, bytes: bytes, reservation: reservation)
        touch(key)
        budget.commit(reservation)
        if !waiting.isEmpty { preHolds[token] = waiting.map { ($0, budget.hold(reservation, label: key, holder: $0)) } }
    }

    private func fail(key: String, token: UUID) {
        leasers.removeValue(forKey: token)
        waitingForRoom.removeValue(forKey: token)
        if loading[key]?.token == token { loading[key] = nil }
    }

    /// One of the holds `finish` took for this load's lessees.
    private func claimHold(_ token: UUID, holder: MemoryBudget.HolderKind?) -> MemoryBudget.Hold {
        guard var list = preHolds[token], !list.isEmpty else { return .inert }
        let at = list.firstIndex { $0.kind == holder } ?? list.startIndex
        let hold = list.remove(at: at).hold
        if list.isEmpty { preHolds.removeValue(forKey: token) } else { preHolds[token] = list }
        return hold
    }
}

/// The app-wide pool of loaded MLX LLM containers, keyed by model **directory** (so a caller
/// holding a `ModelEntry` and one holding only a directory — `LLMEngine`, the Summarize tool —
/// resolve to the same container).
actor ModelContainerPool {
    /// Loads a container from an installed model directory. Injectable so tests never need a
    /// real model.
    typealias Loader = @Sendable (_ directory: URL, _ architecture: String?) async throws -> ModelContainer

    static let shared = ModelContainerPool(budget: .shared)

    private let pool: ResidentPool<ModelContainer>
    private let loader: Loader
    /// The budget the pool reserves from (S3-8B: a generation releases its working set back to it).
    nonisolated let budget: MemoryBudget

    /// Reserve from a shared budget (the app's: `MemoryBudget.shared`).
    init(budget: MemoryBudget, loader: @escaping Loader = ModelContainerPool.loadFromDisk) {
        self.pool = ResidentPool(budget: budget)
        self.budget = budget
        self.loader = loader
    }

    /// A pool with its own private budget (tests).
    init(budgetBytes: Int64, loader: @escaping Loader = ModelContainerPool.loadFromDisk) {
        let budget = MemoryBudget(capacityBytes: budgetBytes)
        self.pool = ResidentPool(budget: budget)
        self.budget = budget
        self.loader = loader
    }

    /// The container for an installed catalog model. The catalog's `ramGB` is only the fallback for the ledger
    /// charge (`ModelWeightCharge`, R63).
    func container(for model: ModelEntry, directory: URL) async throws -> ModelContainer {
        try await container(directory: directory,
                            footprintBytes: Int64(model.ramGB * 1_073_741_824),
                            architecture: model.architecture)
    }

    /// The container for a model known only by its directory. **Every** LLM load in the app passes through here, so this
    /// is where the weights are booked: `ModelWeightCharge.bytes` (R63) — the safetensors on disk × the measured load
    /// factor. `footprintBytes` is the catalog figure, used only when the directory has no safetensors to measure;
    /// with neither, the whole directory's size.
    func container(directory: URL, footprintBytes: Int64? = nil,
                   architecture: String? = nil) async throws -> ModelContainer {
        let key = Self.key(for: directory)
        let bytes = ModelWeightCharge.bytes(directory: directory, catalogFallback: footprintBytes)
        let loader = self.loader
        return try await pool.value(for: key, bytes: bytes) {
            try await loader(directory, architecture)
        }
    }

    /// SC-2a: `container(directory:…)`, **leased**. The model is in use until `release()` — call it in a `defer`, so a
    /// cancelled or throwing generation returns it too. Eviction skips a leased model; a load that can only fit by
    /// evicting it waits (bounded by `maxWait`) and then throws `MemoryBudget.WaitError`.
    /// Every caller that uses the container for longer than the call itself uses this one.
    func lease(directory: URL, footprintBytes: Int64? = nil, architecture: String? = nil,
               maxWait: Duration = MemoryBudget.defaultMaxWait,
               holder: MemoryBudget.HolderKind? = nil) async throws -> ContainerLease {
        let key = Self.key(for: directory)
        let bytes = ModelWeightCharge.bytes(directory: directory, catalogFallback: footprintBytes)
        let loader = self.loader
        let leased = try await pool.lease(for: key, bytes: bytes, maxWait: maxWait, holder: holder) {
            try await loader(directory, architecture)
        }
        return ContainerLease(container: leased.value, hold: leased.hold)
    }

    /// `lease(directory:…)` for an installed catalog model.
    func lease(for model: ModelEntry, directory: URL,
               maxWait: Duration = MemoryBudget.defaultMaxWait,
               holder: MemoryBudget.HolderKind? = nil) async throws -> ContainerLease {
        try await lease(directory: directory, footprintBytes: Int64(model.ramGB * 1_073_741_824),
                        architecture: model.architecture, maxWait: maxWait, holder: holder)
    }

    /// `reserveWorkingSet`, saying why it returned nothing (SC-2a-b).
    func reserveWorkingSetReporting(totalNeedBytes: Int64, directory: URL,
                                    maxWait: Duration? = nil) async -> MemoryBudget.WorkingSetOutcome {
        await pool.reserveWorkingSetReporting(totalNeedBytes: totalNeedBytes, protecting: Self.key(for: directory),
                                              maxWait: maxWait)
    }

    /// A container in use (SC-2a).
    struct ContainerLease: Sendable {
        let container: ModelContainer
        let hold: MemoryBudget.Hold
        func release() { hold.release() }
    }

    /// S3-8B: make room for a generation on `directory`'s model that needs `totalNeedBytes` (weights + KV),
    /// evicting other residents for it. `nil` = it cannot fit even with everything else gone — or (SC-2a) only
    /// models in use stood in the way for `maxWait` (nil = don't wait).
    func reserveWorkingSet(totalNeedBytes: Int64, directory: URL,
                           maxWait: Duration? = nil) async -> MemoryBudget.Reservation? {
        await pool.reserveWorkingSet(totalNeedBytes: totalNeedBytes, protecting: Self.key(for: directory),
                                     maxWait: maxWait)
    }

    /// Free one model's weights (e.g. after the user deletes it).
    @discardableResult
    func evict(modelID: String) async -> Bool {
        await pool.evict(key: Self.key(for: ModelStore.shared.directory(forModelID: modelID)))
    }

    /// Free every loaded container.
    func clear() async { await pool.clear() }

    var count: Int { get async { await pool.count } }
    var totalBytes: Int64 { get async { await pool.totalBytes } }

    // MARK: - Loading

    nonisolated static func key(for directory: URL) -> String {
        directory.standardizedFileURL.path
    }

    /// The real loader: `LLMModelFactory` load plus the app's tool-call-format override, if any.
    ///
    /// **The app defers to upstream here.** mlx-swift-lm 3.31.3 already calls
    /// `ToolCallFormat.infer(from:configData:)` inside `LLMModelFactory.loadContainer`
    /// (`LLMModelFactory.swift:543`) when the configuration carries no format, so this overrides
    /// only where `ToolCallFormatTable.override` says upstream is wrong or missing — which, for
    /// every model in the catalog as of S2-1, is nowhere. (S1 installed `qwen3_5` → `.xmlFunction`
    /// here; that duplicated a row upstream already had — readme F3b.)
    ///
    /// This remains **the one place in the app that sets `configuration.toolCallFormat`**: chat
    /// sheet, flow rows and the Local Server all get their container from this pool.
    nonisolated static func loadFromDisk(directory: URL, architecture: String?) async throws -> ModelContainer {
        let container = try await LLMModelFactory.shared.loadContainer(
            from: directory, using: HFTokenizerLoader())

        let configData = configData(directory: directory)
        let modelType = modelTypeString(configData: configData) ?? architecture ?? ""
        if let format = ToolCallFormatTable.override(forModelType: modelType, configData: configData) {
            await container.update { $0.configuration.toolCallFormat = format }
        }
        return container
    }

    /// The checkpoint's `config.json`, read once per load. Two consumers: the `model_type` the
    /// family tables key on, and `ToolCallFormat.infer`'s secondary signals (Llama 3's
    /// `vocab_size` / `rope_scaling`).
    nonisolated static func configData(directory: URL) -> Data? {
        try? Data(contentsOf: directory.appendingPathComponent("config.json"))
    }

    /// The checkpoint's `model_type` (from `config.json`), falling back to the catalog
    /// architecture string. Used to pick the tool-call format and the `system` placement.
    nonisolated static func modelTypeString(directory: URL, architecture: String?) -> String {
        modelTypeString(configData: configData(directory: directory)) ?? architecture ?? ""
    }

    /// `model_type` out of an already-read `config.json`, or `nil` when it isn't there.
    ///
    /// All 12 catalog chat models carry a top-level `model_type`; the nested `text_config` copy is
    /// deliberately not consulted, because upstream's `ToolCallFormat.infer` doesn't consult it
    /// either (e.g. `qwen3_5` at the top level, `qwen3_5_text` under `text_config`).
    nonisolated static func modelTypeString(configData: Data?) -> String? {
        guard let configData,
              let obj = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
              let modelType = obj["model_type"] as? String else { return nil }
        return modelType
    }
}
