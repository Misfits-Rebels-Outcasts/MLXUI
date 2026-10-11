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

    private var entries: [String: Entry] = [:]
    /// Most-recently-used first. Eviction pops from the end.
    private var recency: [String] = []
    private var loading: [String: Loading] = [:]
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
    func value(for key: String, bytes: Int64,
               load: @escaping @Sendable () async throws -> Value) async throws -> Value {
        if let entry = entries[key] {
            touch(key)
            return entry.value
        }
        if let inFlight = loading[key] {
            return try await inFlight.task.value
        }

        // Install the in-flight marker *before* the first suspension, so a concurrent caller for
        // this key joins this load instead of starting a second one while we wait for room.
        let token = UUID()
        let budget = self.budget
        let task = Task {
            let reservation = await budget.reserve(bytes: bytes, for: self, label: key)
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
        return try await task.value
    }

    /// Drop `key` (resident, or an in-flight load's result — the waiters still get their value
    /// but it is not retained). Returns whether anything was dropped.
    @discardableResult
    func evict(key: String) -> Bool {
        var dropped = false
        if let entry = entries.removeValue(forKey: key) {
            recency.removeAll { $0 == key }
            budget.release(entry.reservation)
            dropped = true
        }
        if loading.removeValue(forKey: key) != nil { dropped = true }
        return dropped
    }

    /// Drop everything resident, and stop retaining anything still loading.
    func clear() {
        let reservations = entries.values.map(\.reservation)
        entries.removeAll()
        recency.removeAll()
        loading.removeAll()
        reservations.forEach { budget.release($0) }
    }

    /// `MemoryBudgetParticipant`: evict the least-recently-used resident entry that holds memory.
    func evictLeastRecentlyUsed() -> Bool {
        guard let key = recency.last(where: { (entries[$0]?.bytes ?? 0) > 0 }),
              let entry = entries.removeValue(forKey: key) else { return false }
        recency.removeAll { $0 == key }
        budget.release(entry.reservation)
        return true
    }

    /// S3-8B: evict the LRU entry that holds memory **except** `key` (the model that is running).
    func evictLeastRecentlyUsed(except key: String) -> Bool {
        guard let victim = recency.last(where: { $0 != key && (entries[$0]?.bytes ?? 0) > 0 }),
              let entry = entries.removeValue(forKey: victim) else { return false }
        recency.removeAll { $0 == victim }
        budget.release(entry.reservation)
        return true
    }

    /// S3-8B: make room for a running generation that needs `totalNeedBytes` in all (weights + KV), evicting
    /// other residents for it. The running model's own ledger entry already counts toward that (it was charged
    /// at load), so only the **excess** over it is reserved: `others + need ≤ capacity`.
    func reserveWorkingSet(totalNeedBytes: Int64, protecting key: String) async -> MemoryBudget.Reservation? {
        let protected = entries[key]?.bytes ?? 0
        let extra = max(0, totalNeedBytes - protected)
        return await budget.reserveWorkingSet(bytes: extra, for: self, protecting: protected) { [self] in
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
        // Only retain if this load is still the current one (not evicted/cleared meanwhile).
        guard loading[key]?.token == token else {
            budget.release(reservation)
            return
        }
        loading[key] = nil
        entries[key] = Entry(value: value, bytes: bytes, reservation: reservation)
        touch(key)
        budget.commit(reservation)
    }

    private func fail(key: String, token: UUID) {
        if loading[key]?.token == token { loading[key] = nil }
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

    /// S3-8B: make room for a generation on `directory`'s model that needs `totalNeedBytes` (weights + KV),
    /// evicting other residents for it. `nil` = it cannot fit even with everything else gone.
    func reserveWorkingSet(totalNeedBytes: Int64, directory: URL) async -> MemoryBudget.Reservation? {
        await pool.reserveWorkingSet(totalNeedBytes: totalNeedBytes, protecting: Self.key(for: directory))
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
