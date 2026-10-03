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

    /// Reserve from a shared budget (the app's: `MemoryBudget.shared`).
    init(budget: MemoryBudget, loader: @escaping Loader = ModelContainerPool.loadFromDisk) {
        self.pool = ResidentPool(budget: budget)
        self.loader = loader
    }

    /// A pool with its own private budget (tests).
    init(budgetBytes: Int64, loader: @escaping Loader = ModelContainerPool.loadFromDisk) {
        self.pool = ResidentPool(budgetBytes: budgetBytes)
        self.loader = loader
    }

    /// The container for an installed catalog model. Footprint = the catalog's `ramGB`
    /// (the same figure `EngineCache` uses).
    func container(for model: ModelEntry, directory: URL) async throws -> ModelContainer {
        try await container(directory: directory,
                            footprintBytes: Int64(model.ramGB * 1_073_741_824),
                            architecture: model.architecture)
    }

    /// The container for a model known only by its directory. With no catalog figure,
    /// `footprintBytes == nil` falls back to the on-disk size of the model directory.
    func container(directory: URL, footprintBytes: Int64? = nil,
                   architecture: String? = nil) async throws -> ModelContainer {
        let key = Self.key(for: directory)
        let bytes = footprintBytes ?? Int64(ModelStore.directorySize(at: directory))
        let loader = self.loader
        return try await pool.value(for: key, bytes: bytes) {
            try await loader(directory, architecture)
        }
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

    /// The real loader: `LLMModelFactory` + the tool-call-format wiring that used to live in
    /// `ModelRunner.containerForModel`, now done once, in the pool, for every consumer.
    nonisolated static func loadFromDisk(directory: URL, architecture: String?) async throws -> ModelContainer {
        let container = try await LLMModelFactory.shared.loadContainer(
            from: directory, using: HFTokenizerLoader())

        // Wire the tool-call parser to match the model family. Qwen3.5 emits the XML
        // `<tool_call><function=…>` form (`ToolCallFormat.xmlFunction`); without this the
        // generate path parses tool calls as `.json` and never fires them (AG0 finding).
        if let format = AgentSession.toolCallFormat(
            forModelType: modelTypeString(directory: directory, architecture: architecture)) {
            await container.update { $0.configuration.toolCallFormat = format }
        }
        return container
    }

    /// The checkpoint's `model_type` (from `config.json`), falling back to the catalog
    /// architecture string. Used to pick the tool-call format.
    nonisolated static func modelTypeString(directory: URL, architecture: String?) -> String {
        if let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let modelType = obj["model_type"] as? String {
            return modelType
        }
        return architecture ?? ""
    }
}
