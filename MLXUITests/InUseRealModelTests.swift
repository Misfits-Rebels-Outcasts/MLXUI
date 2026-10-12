import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import MLXUI

/// SC-2a step 5 — the real-model repeat of SC-0's pool and `EngineCache` cases, plus the F15 case (a server request's
/// working set versus a model a flow row is still generating with). **Gated on env**, one case per process so MLX
/// starts clean (pass as `TEST_RUNNER_<NAME>=…`; run the host unsandboxed, as the SC-0 scripts did):
///
///     MLXUI_TEST_SC2A_CASE=pool|ec|ws   MLXUI_TEST_SC2A_A=<dir>   MLXUI_TEST_SC2A_B=<dir>
///     MLXUI_TEST_SC2A_CAPACITY_GIB=7    (pool, ec: A + B must exceed it, B alone must fit; ws: 11)
///     MLXUI_TEST_SC2A_KV_GIB=3          (ws: the request's working set, allocated for real)
///
/// "Real" is `task_info(TASK_VM_INFO).phys_footprint`, the figure the OS acts on. The bound at **every** step is
/// capacity + 0.5 GiB (SC-2a step 5). Before SC-2a the same steps measured +3.2 GiB (pool), and +1.4 GiB (ws) over
/// capacity (journal `2026-4xx-cc-sc2a-in-use`).
@Suite(.serialized)
struct InUseRealModelTests {
    static let env = ProcessInfo.processInfo.environment
    static let only = env["MLXUI_TEST_SC2A_CASE"]
    static let dirA = env["MLXUI_TEST_SC2A_A"]
    static let dirB = env["MLXUI_TEST_SC2A_B"]
    static let capacityGiB = Int64(env["MLXUI_TEST_SC2A_CAPACITY_GIB"] ?? "7") ?? 7
    static let kvGiB = Int64(env["MLXUI_TEST_SC2A_KV_GIB"] ?? "3") ?? 3
    static let slack = Int64(512 << 20)

    private static func phys() -> Int64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int64(info.phys_footprint) : -1
    }
    private static func gib(_ b: Int64) -> String { String(format: "%.2f", Double(b) / 1_073_741_824) }

    private final class Weak<T: AnyObject>: @unchecked Sendable { weak var ref: T?; init(_ r: T) { ref = r } }
    private final class Flag: @unchecked Sendable {
        private let lock = NSLock(); private var v = false
        var isSet: Bool { lock.lock(); defer { lock.unlock() }; return v }
        func set() { lock.lock(); v = true; lock.unlock() }
    }
    private final class Peak: @unchecked Sendable {
        private let lock = NSLock(); private var v: Int64 = 0
        func note(_ x: Int64) { lock.lock(); v = max(v, x); lock.unlock() }
        var value: Int64 { lock.lock(); defer { lock.unlock() }; return v }
    }

    /// Samples `phys_footprint` every 50 ms for the whole test: the bound holds *between* the printed steps too.
    private func startSampler(_ peak: Peak) -> Task<Void, Never> {
        Task.detached { while !Task.isCancelled { peak.note(Self.phys()); try? await Task.sleep(for: .milliseconds(50)) } }
    }

    private func step(_ label: String, _ budget: MemoryBudget, capacity: Int64, _ extra: String = "") {
        let mlx = MLX.Memory.snapshot()
        let p = Self.phys()
        print("SC2A \(label): ledger=\(Self.gib(budget.usedBytes)) in-use=\(budget.inUseCount) phys=\(Self.gib(p)) mlxActive=\(Self.gib(Int64(mlx.activeMemory))) mlxCache=\(Self.gib(Int64(mlx.cacheMemory))) over-capacity=\(Self.gib(p - capacity)) \(extra)")
        #expect(p <= capacity + Self.slack, "\(label): real memory \(Self.gib(p)) GiB exceeds capacity \(Self.gib(capacity)) GiB + 0.5")
        #expect(budget.usedBytes <= capacity, "\(label): ledger over capacity")
    }

    private static func generate(_ container: ModelContainer, tokens: Int = 24) async throws -> Int {
        try await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(chat: [.user("Count from one to twenty in words.")]))
            let stream = try MLXLMCommon.generateTokens(
                input: input, parameters: GenerateParameters(maxTokens: tokens, temperature: 0), context: context)
            var n = 0
            for await g in stream { if case .token = g { n += 1 } }
            return n
        }
    }

    /// Allocate and free `gib` GiB so MLX keeps the freed buffers in its cache.
    @inline(never)
    private static func fillBufferCache(gib: Int) {
        let x = MLXArray.zeros([gib << 28], dtype: .float32)
        let y = x + 1
        MLX.eval(y)
        _ = y[0].item(Float.self)          // wait for the stream: the buffers are idle, then freed on return
    }

    private nonisolated struct HoldingStage: PipelineStage {
        let id: String, name = "Holder"
        var accepts: MediaKind { .text }
        var produces: MediaKind { .text }
        let container: ModelContainer
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media { input }
    }

    // MARK: pool — a generation holds A; a load of B must wait, not overshoot

    @Test(.enabled(if: InUseRealModelTests.only == "pool" && InUseRealModelTests.dirA != nil && InUseRealModelTests.dirB != nil))
    func poolHolderIsNeverFreed() async throws {
        let a = URL(fileURLWithPath: try #require(Self.dirA)), b = URL(fileURLWithPath: try #require(Self.dirB))
        let capacity = Self.capacityGiB << 30
        let budget = MemoryBudget(capacityBytes: capacity)
        let pool = ModelContainerPool(budget: budget)
        let peak = Peak(); let sampler = startSampler(peak)
        step("pool 1 before A", budget, capacity: capacity)

        // `leaseA` is an optional so that releasing it lets the weights die: nothing else in this frame holds A.
        var leaseA: ModelContainerPool.ContainerLease? = try await pool.lease(directory: a)
        _ = try await Self.generate(try #require(leaseA).container)
        let weakA = Weak(try #require(leaseA).container)
        step("pool 2 A leased, generated", budget, capacity: capacity)

        // B can only fit by evicting A, which is in use: it waits for the bound and is refused — never loaded.
        await #expect(throws: MemoryBudget.WaitError.self) {
            _ = try await pool.lease(directory: b, maxWait: .seconds(2))
        }
        step("pool 3 B refused while A in use", budget, capacity: capacity, "A=\(weakA.ref != nil ? "alive" : "gone")")
        #expect(await pool.count == 1)
        _ = try await Self.generate(try #require(leaseA).container)     // A is still resident and usable
        step("pool 3b A generated again", budget, capacity: capacity)

        // …and a waiter that is already waiting proceeds as soon as A is returned.
        async let waiting = pool.lease(directory: b, maxWait: .seconds(120))
        try await Task.sleep(for: .seconds(1))
        step("pool 4 B waiting, A still leased", budget, capacity: capacity)
        leaseA?.release()
        leaseA = nil
        let leaseB = try await waiting
        _ = try await Self.generate(leaseB.container)
        step("pool 5 A returned, B loaded (A evicted)", budget, capacity: capacity, "A=\(weakA.ref != nil ? "alive" : "gone")")
        #expect(weakA.ref == nil, "A's weights are still alive after its lease was returned and B loaded")
        leaseB.release()

        // step 4: with the weights really gone, MLX gives its buffer cache back by itself
        let cacheBefore = MLX.Memory.cacheMemory
        try await Task.sleep(for: .seconds(2))
        let cacheAfter = MLX.Memory.cacheMemory
        step("pool 6 settled (cache \(Self.gib(Int64(cacheBefore))) → \(Self.gib(Int64(cacheAfter))) GiB)", budget, capacity: capacity)
        #expect(cacheAfter < 512 << 20, "MLX buffer cache still \(Self.gib(Int64(cacheAfter))) GiB after the weights were dropped")

        // The janitor on real MLX: fill the buffer cache (~2 GiB of freed buffers), say weights were dropped, and the
        // cache figure falls by itself — off this path, without anyone calling `Memory.clearCache()`.
        Self.fillBufferCache(gib: 1)
        let filled = MLX.Memory.cacheMemory
        step("pool 7 buffer cache filled (\(Self.gib(Int64(filled))) GiB cached)", budget, capacity: capacity)
        #expect(filled > 1 << 30, "the test could not fill MLX's buffer cache")
        budget.weightsDropped()
        try await Task.sleep(for: .seconds(2))
        let drained = MLX.Memory.cacheMemory
        step("pool 8 after weightsDropped (cache \(Self.gib(Int64(filled))) → \(Self.gib(Int64(drained))) GiB)", budget, capacity: capacity)
        #expect(drained < 256 << 20, "MLX's cache figure did not fall: \(Self.gib(Int64(drained))) GiB")
        sampler.cancel()
        print("SC2A pool PEAK phys over the whole test = \(Self.gib(peak.value)) GiB (capacity \(Self.capacityGiB) GiB)")
        #expect(peak.value <= capacity + Self.slack)
    }

    // MARK: EngineCache — a flow row's stage

    @Test(.enabled(if: InUseRealModelTests.only == "ec" && InUseRealModelTests.dirA != nil && InUseRealModelTests.dirB != nil))
    func engineCacheHolderIsNeverFreed() async throws {
        let a = URL(fileURLWithPath: try #require(Self.dirA)), b = URL(fileURLWithPath: try #require(Self.dirB))
        let capacity = Self.capacityGiB << 30
        let budget = MemoryBudget(capacityBytes: capacity)
        let cache = EngineCache(budget: budget, countsLLMWeights: true)
        func entry(_ dir: URL) -> ModelEntry {
            makeEntry(id: dir.lastPathComponent, ramGB: Double(ModelWeightCharge.bytes(directory: dir, catalogFallback: nil)) / 1_073_741_824)
        }
        let builder: EngineCache.Builder = { model, _ in
            let dir = model.id == a.lastPathComponent ? a : b
            return HoldingStage(id: model.id, container: try await ModelContainerPool.loadFromDisk(directory: dir, architecture: nil))
        }
        let peak = Peak(); let sampler = startSampler(peak)
        step("ec 1 before A", budget, capacity: capacity)

        // the flow row's stage; `rowA = nil` is "the row is done with it" — nothing else in this frame holds A
        var rowA: (any PipelineStage)? = try await cache.leasedStage(for: entry(a), config: .default, builder: builder)
        let weakA = try await Self.generateOnRow(try #require(rowA))
        step("ec 2 A leased, generated", budget, capacity: capacity)

        await #expect(throws: MemoryBudget.WaitError.self) {
            _ = try await cache.leasedStage(for: entry(b), config: .default, maxWait: .seconds(2), builder: builder)
        }
        step("ec 3 B refused while A's row runs", budget, capacity: capacity)
        #expect(cache.count == 1)
        _ = try await Self.generateOnRow(try #require(rowA))
        step("ec 3b A generated again", budget, capacity: capacity)

        rowA = nil                                                      // the row ends: its lease is returned
        let leaseB = try await cache.lease(for: entry(b), config: .default, maxWait: .seconds(120), builder: builder)
        step("ec 5 A's row ended, B built (A evicted)", budget, capacity: capacity, "A=\(weakA.ref != nil ? "alive" : "gone")")
        #expect(weakA.ref == nil, "A's weights are still alive after its row ended and B was built")
        leaseB.hold.release()
        let before = MLX.Memory.cacheMemory
        try await Task.sleep(for: .seconds(2))
        let after = MLX.Memory.cacheMemory
        step("ec 6 settled (cache \(Self.gib(Int64(before))) → \(Self.gib(Int64(after))) GiB)", budget, capacity: capacity)
        #expect(after < 512 << 20, "MLX buffer cache still \(Self.gib(Int64(after))) GiB after the weights were dropped")
        sampler.cancel()
        print("SC2A ec PEAK phys over the whole test = \(Self.gib(peak.value)) GiB (capacity \(Self.capacityGiB) GiB)")
        #expect(peak.value <= capacity + Self.slack)
    }

    /// Generate on a row's leased stage, returning only a weak probe of its container (this frame ends on return).
    @inline(never)
    private static func generateOnRow(_ row: any PipelineStage) async throws -> Weak<ModelContainer> {
        let stage = try #require((row as? LeasedStage)?.inner as? HoldingStage)
        _ = try await generate(stage.container)
        return Weak(stage.container)
    }

    // MARK: F15 — a server request's working set versus a flow row still generating

    @Test(.enabled(if: InUseRealModelTests.only == "ws" && InUseRealModelTests.dirA != nil && InUseRealModelTests.dirB != nil))
    func workingSetNeverEvictsAModelAFlowRowIsUsing() async throws {
        let a = URL(fileURLWithPath: try #require(Self.dirA)), b = URL(fileURLWithPath: try #require(Self.dirB))
        let capacity = Self.capacityGiB << 30
        let budget = MemoryBudget(capacityBytes: capacity)
        let pool = ModelContainerPool(budget: budget)
        let peak = Peak(); let sampler = startSampler(peak)
        step("ws 1 before", budget, capacity: capacity)

        // the flow row: leases A, generates, keeps it until told to stop
        let rowDone = Flag(), rowStarted = Flag()
        let row = Task {
            let lease = try await pool.lease(directory: a)
            defer { lease.release() }
            rowStarted.set()
            while !rowDone.isSet {
                _ = try await Self.generate(lease.container, tokens: 8)
                try await Task.sleep(for: .milliseconds(100))
            }
        }
        while !rowStarted.isSet { try await Task.sleep(for: .milliseconds(20)) }
        // the server request: leases B (its model), then asks for the working set
        let request = try await pool.lease(directory: b)
        _ = try await Self.generate(request.container)
        step("ws 2 both resident (flow row on A, request on B)", budget, capacity: capacity)

        let need = ModelWeightCharge.bytes(directory: b, catalogFallback: nil) + (Self.kvGiB << 30)
        let refused = await pool.reserveWorkingSet(totalNeedBytes: need, directory: b, maxWait: .seconds(2))
        #expect(refused == nil, "the working set evicted a model a flow row is still generating with")
        #expect(await pool.count == 2)
        step("ws 3 working set refused, A untouched", budget, capacity: capacity)

        // the row finishes → the same request now gets its room by an ordinary eviction of A
        rowDone.set()
        try await row.value
        let granted = await pool.reserveWorkingSet(totalNeedBytes: need, directory: b, maxWait: .seconds(10))
        #expect(granted != nil)
        #expect(await pool.count == 1)
        try await Task.sleep(for: .seconds(2))                          // A gone, cache given back
        let kv = MLXArray.zeros([Int(Self.kvGiB << 30) / 4], dtype: .float32)
        MLX.eval(kv)                                                    // the request's KV, for real
        step("ws 4 row ended, A evicted, KV allocated", budget, capacity: capacity)
        _ = try await Self.generate(request.container)
        step("ws 5 request generated", budget, capacity: capacity)
        if let granted { budget.release(granted) }
        request.release()
        _ = kv
        sampler.cancel()
        print("SC2A ws PEAK phys over the whole test = \(Self.gib(peak.value)) GiB (capacity \(Self.capacityGiB) GiB)")
        #expect(peak.value <= capacity + Self.slack)
    }
}
