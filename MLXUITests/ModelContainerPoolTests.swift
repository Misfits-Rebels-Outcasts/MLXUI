import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

/// S1-1 — `ModelContainerPool`: one owner of loaded LLM weights. The LRU / single-flight core
/// (`ResidentPool`) is tested with plain values (a real `ModelContainer` can't be fabricated);
/// routing through `LLMEngine` and `ModelRunner` is proven with an injected loader that never
/// loads anything.
struct ModelContainerPoolTests {

    private static let gb = Int64(1_073_741_824)

    private final class Box: Sendable { let name: String; init(_ n: String) { name = n } }

    private actor Counter {
        private(set) var loads: [String] = []
        func record(_ s: String) { loads.append(s) }
        var count: Int { loads.count }
    }

    private struct Sentinel: Error, Equatable {}

    // MARK: - ResidentPool

    @Test func returnsTheSameInstanceTwice() async throws {
        let pool = ResidentPool<Box>(budgetBytes: 20 * Self.gb)
        let counter = Counter()
        let a = try await pool.value(for: "m1", bytes: 6 * Self.gb) { await counter.record("m1"); return Box("m1") }
        let b = try await pool.value(for: "m1", bytes: 6 * Self.gb) { await counter.record("m1"); return Box("m1") }
        #expect(a === b)
        #expect(await counter.count == 1)
    }

    @Test func aSecondModelOverBudgetEvictsTheFirst() async throws {
        let pool = ResidentPool<Box>(budgetBytes: 10 * Self.gb)
        _ = try await pool.value(for: "m1", bytes: 6 * Self.gb) { Box("m1") }
        _ = try await pool.value(for: "m2", bytes: 6 * Self.gb) { Box("m2") }
        #expect(await pool.residentKeys == ["m2"])
        #expect(await pool.totalBytes == 6 * Self.gb)
    }

    @Test func evictionIsLeastRecentlyUsedFirst() async throws {
        let pool = ResidentPool<Box>(budgetBytes: 12 * Self.gb)
        _ = try await pool.value(for: "a", bytes: 4 * Self.gb) { Box("a") }
        _ = try await pool.value(for: "b", bytes: 4 * Self.gb) { Box("b") }
        _ = try await pool.value(for: "c", bytes: 4 * Self.gb) { Box("c") }
        _ = try await pool.value(for: "a", bytes: 4 * Self.gb) { Box("a2") }   // touch a → b is now oldest
        _ = try await pool.value(for: "d", bytes: 4 * Self.gb) { Box("d") }
        #expect(await pool.residentKeys == ["d", "a", "c"])
    }

    @Test func aModelLargerThanTheBudgetStillLoadsAlone() async throws {
        let pool = ResidentPool<Box>(budgetBytes: 4 * Self.gb)
        _ = try await pool.value(for: "small", bytes: 2 * Self.gb) { Box("small") }
        let big = try await pool.value(for: "big", bytes: 9 * Self.gb) { Box("big") }
        #expect(big.name == "big")
        #expect(await pool.residentKeys == ["big"])
    }

    @Test func concurrentLoadsOfOneIdShareOneLoad() async throws {
        let pool = ResidentPool<Box>(budgetBytes: 20 * Self.gb)
        let counter = Counter()
        let results = try await withThrowingTaskGroup(of: Box.self) { group in
            for _ in 0..<8 {
                group.addTask {
                    try await pool.value(for: "m1", bytes: 6 * Self.gb) {
                        await counter.record("m1")
                        try await Task.sleep(for: .milliseconds(80))   // wide window for the others to arrive
                        return Box("m1")
                    }
                }
            }
            var all: [Box] = []
            for try await box in group { all.append(box) }
            return all
        }
        #expect(await counter.count == 1)
        #expect(results.count == 8)
        #expect(results.allSatisfy { $0 === results[0] })
    }

    @Test func aFailedLoadIsNotCachedAndCanBeRetried() async throws {
        let pool = ResidentPool<Box>(budgetBytes: 20 * Self.gb)
        await #expect(throws: Sentinel.self) {
            _ = try await pool.value(for: "m1", bytes: 6 * Self.gb) { throw Sentinel() }
        }
        #expect(await pool.count == 0)
        let ok = try await pool.value(for: "m1", bytes: 6 * Self.gb) { Box("m1") }
        #expect(ok.name == "m1")
    }

    @Test func evictAndClearDropWeights() async throws {
        let pool = ResidentPool<Box>(budgetBytes: 20 * Self.gb)
        _ = try await pool.value(for: "a", bytes: Self.gb) { Box("a") }
        _ = try await pool.value(for: "b", bytes: Self.gb) { Box("b") }
        #expect(await pool.evict(key: "a"))
        #expect(await !pool.evict(key: "a"))
        #expect(await pool.residentKeys == ["b"])
        await pool.clear()
        #expect(await pool.count == 0)
    }

    @Test func aLoadEvictedMidFlightIsNotRetained() async throws {
        let pool = ResidentPool<Box>(budgetBytes: 20 * Self.gb)
        async let loaded = pool.value(for: "m1", bytes: Self.gb) {
            try await Task.sleep(for: .milliseconds(150))
            return Box("m1")
        }
        try await Task.sleep(for: .milliseconds(40))
        await pool.evict(key: "m1")
        let box = try await loaded
        #expect(box.name == "m1")               // the waiter still gets its value…
        #expect(await pool.count == 0)          // …but the pool doesn't keep it
    }

    @Test func inFlightLoadsCountAgainstTheBudget() async throws {
        // 10 GB budget. "r" (5 GB) is resident; "a" (4 GB) is mid-load; "b" (4 GB) then starts.
        // Counting residents alone (5 + 4 = 9 ≤ 10) would keep "r" and end at 13 GB; counting
        // the in-flight "a" (5 + 4 + 4 = 13 > 10) must evict "r" up front.
        let pool = ResidentPool<Box>(budgetBytes: 10 * Self.gb)
        _ = try await pool.value(for: "r", bytes: 5 * Self.gb) { Box("r") }
        async let a = pool.value(for: "a", bytes: 4 * Self.gb) {
            try await Task.sleep(for: .milliseconds(100)); return Box("a") }
        try await Task.sleep(for: .milliseconds(20))
        async let b = pool.value(for: "b", bytes: 4 * Self.gb) { Box("b") }
        _ = try await (a, b)
        #expect(Set(await pool.residentKeys) == ["a", "b"])
        #expect(await pool.totalBytes == 8 * Self.gb)
    }

    // MARK: - Routing seams (no real model)

    private static func tempModelDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("pool-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    @Test func llmEngineLoadsThroughThePool() async throws {
        let dir = try Self.tempModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let seen = Counter()
        let pool = ModelContainerPool(budgetBytes: 20 * Self.gb) { directory, _ in
            await seen.record(directory.path)
            throw Sentinel()
        }
        do {
            _ = try await LLMEngine.generate(prompt: "hi", modelDir: dir, maxTokens: 4,
                                             footprintBytes: Self.gb, pool: pool)
            Issue.record("expected the sentinel loader to fail the call")
        } catch let StageError.engineFailure(_, underlying) {
            #expect(underlying is Sentinel)
        }
        #expect(await seen.loads == [dir.path])
    }

    @MainActor
    @Test func modelRunnerLoadsThroughThePool() async throws {
        let dir = try Self.tempModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let seen = Counter()
        let pool = ModelContainerPool(budgetBytes: 20 * Self.gb) { directory, architecture in
            await seen.record("\(directory.path)|\(architecture ?? "nil")")
            throw Sentinel()
        }
        let runner = ModelRunner(pool: pool)
        let model = makeEntry(id: "chat-m", ramGB: 3)
        await #expect(throws: Sentinel.self) {
            _ = try await runner.containerForModel(model, dir: dir)
        }
        #expect(await seen.count == 1)
        #expect(await seen.loads.first?.hasPrefix(dir.path) == true)
    }

    @Test func entryAndDirectoryCallersShareOneKey() async throws {
        let dir = try Self.tempModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(ModelContainerPool.key(for: dir) == ModelContainerPool.key(for: dir.appendingPathComponent("../\(dir.lastPathComponent)")))
    }

    // MARK: - EngineCache no longer double-counts LLM bytes (T5)

    private nonisolated struct NoopStage: PipelineStage {
        let id = "noop", name = "Noop"
        var accepts: MediaKind { .text }
        var produces: MediaKind { .text }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media { input }
    }

    @Test func sharedPolicyCountsNoBytesForLLMStagesButStillCountsOthers() async throws {
        let cache = EngineCache(budgetBytes: 10 * Self.gb, countsLLMWeights: false)
        let llm1 = makeEntry(id: "l1", modelType: .llm, ramGB: 6)
        let llm2 = makeEntry(id: "l2", modelType: .llm, ramGB: 6)
        let asr = makeEntry(id: "a1", modelType: .asr, ramGB: 6)
        _ = try await cache.stage(for: llm1, config: .default) { _, _ in NoopStage() }
        _ = try await cache.stage(for: llm2, config: .default) { _, _ in NoopStage() }
        #expect(cache.count == 2)                 // two 6 GB LLMs in a 10 GB budget: both stay
        #expect(cache.totalCachedBytes == 0)
        _ = try await cache.stage(for: asr, config: .default) { _, _ in NoopStage() }
        #expect(cache.totalCachedBytes == 6 * Self.gb)   // a non-LLM still counts
    }

    // MARK: - Real model (skipped unless MLXUI_TEST_LLM_DIR points at an installed MLX LLM)

    private static let realModelDir = ProcessInfo.processInfo.environment["MLXUI_TEST_LLM_DIR"]

    @Test(.enabled(if: ModelContainerPoolTests.realModelDir != nil))
    func twoCallsLoadOnceAndAreFasterThanPerCallLoading() async throws {
        let dir = URL(fileURLWithPath: try #require(Self.realModelDir))
        func rssMB() -> Double {
            var info = mach_task_basic_info()
            var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
            let kr = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
                }
            }
            return kr == KERN_SUCCESS ? Double(info.resident_size) / 1_048_576 : -1
        }
        let loads = Counter()
        let pool = ModelContainerPool(budgetBytes: 40 * Self.gb) { d, a in
            await loads.record(d.path)
            return try await ModelContainerPool.loadFromDisk(directory: d, architecture: a)
        }
        // "Before": what LLMEngine did — a fresh load per call (a one-entry pool with budget 0 evicts every time).
        let perCall = ModelContainerPool(budgetBytes: 0) { d, a in
            try await ModelContainerPool.loadFromDisk(directory: d, architecture: a)
        }
        let t0 = Date()
        for _ in 0..<2 { _ = try await LLMEngine.generate(prompt: "Say hi.", modelDir: dir, maxTokens: 8, temperature: 0, pool: perCall) }
        let before = Date().timeIntervalSince(t0)
        let t1 = Date()
        for _ in 0..<2 { _ = try await LLMEngine.generate(prompt: "Say hi.", modelDir: dir, maxTokens: 8, temperature: 0, pool: pool) }
        let after = Date().timeIntervalSince(t1)
        print("S1-1 MEASURE two LLM calls: per-call-load \(String(format: "%.2f", before)) s, pooled \(String(format: "%.2f", after)) s, loads=\(await loads.count), RSS \(Int(rssMB())) MB")
        #expect(await loads.count == 1)
        #expect(after < before)
    }
}
