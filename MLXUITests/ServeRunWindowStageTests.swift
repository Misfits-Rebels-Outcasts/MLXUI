import Testing
import Foundation
@testable import MLXUI

/// SC-5c — every Run window runs through the stage the sheet handed it (a `LeasingStage`: the ask, the lease, the
/// booking). Six views used to build their own and so bypassed all three.
private let gb = Int64(1_073_741_824)

@MainActor
struct ServeRunWindowStageTests {
    private final class Box: Sendable {}

    private func makeBudget(_ gbs: Int64) -> MemoryBudget {
        MemoryBudget(capacityBytes: gbs * gb,
                     cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
    }

    private nonisolated final class Recorder: PipelineStage, @unchecked Sendable {
        let id: String
        var name: String { id }
        var accepts: MediaKind { .text }
        var produces: MediaKind { .text }
        private let lock = NSLock()
        private var _runs = 0
        private var _seen: [(count: Int, used: Int64)] = []
        private let observe: (@Sendable () -> (Int, Int64))?
        init(id: String, observe: (@Sendable () -> (Int, Int64))? = nil) { self.id = id; self.observe = observe }
        var runs: Int { lock.lock(); defer { lock.unlock() }; return _runs }
        /// (cache entries, ledger bytes) as seen from inside each run
        var seen: [(count: Int, used: Int64)] { lock.lock(); defer { lock.unlock() }; return _seen }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
            let snap = observe?()
            lock.lock(); _runs += 1; if let snap { _seen.append((snap.0, snap.1)) }; lock.unlock(); return input
        }
    }

    private func handed(_ model: ModelEntry, cache: EngineCache, confirmer: ServedLoadConfirmer? = nil,
                        books: Bool = true) -> LeasingStage {
        LeasingStage(model: model, config: .default, probe: Recorder(id: model.id), booksWeights: books,
                     cache: cache, maxWait: .milliseconds(300), confirmer: confirmer)
    }

    private func waitForQuestion(_ c: ServedLoadConfirmer) async -> Bool {
        for _ in 0..<500 where c.question == nil { try? await Task.sleep(for: .milliseconds(10)) }
        return c.question != nil
    }

    // MARK: the VLM Run window asks

    @Test func theVLMRunWindowAsksWhenAPinnedModelIsInTheWay() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let budget = makeBudget(10)
        let served = ResidentPool<Box>(budget: budget)
        _ = try await served.value(for: "served", bytes: 8 * gb) { Box() }
        #expect(await served.pin("served"))
        let confirmer = ServedLoadConfirmer(
            check: { bytes, requester in
                ServeMemory.wouldDisplaceServed(loadBytes: bytes, requester: requester,
                                                snapshot: await served.displacementSnapshot(), servedID: { $0 })
            },
            pushOut: { ids, requester in _ = await served.pushOutPinned(keys: ids, for: requester) },
            reloadSeconds: { _ in nil })
        let model = makeEntry(id: "vlm-ask", modelType: .vision, ramGB: 6)
        let cache = EngineCache(budget: budget, countsLLMWeights: true)
        // what ImageQARunView.run() now does: a per-run stage over the handed one
        let stage = LeasingStage.perRun(handed(model, cache: cache, confirmer: confirmer, books: false),
                                        config: StageConfig(prompt: "What is this?")) { _ in Recorder(id: model.id) }
        let run = Task { try await stage.run(.text("x")) { _ in } }
        #expect(await waitForQuestion(confirmer))
        confirmer.respond(false)
        await #expect(throws: CancellationError.self) { _ = try await run.value }
        #expect(await served.isPinned("served"))
    }

    // MARK: the other five reserve and lease

    @Test(arguments: [ModelType.image, .music, .tts])
    func aRunReservesRealBytesAndLeasesThem(kind: ModelType) async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: true)
        let model = makeEntry(id: "run-\(kind.rawValue)", modelType: kind, ramGB: 3)
        let probe = Recorder(id: model.id, observe: { (cache.count, budget.usedBytes) })
        let stage = LeasingStage.perRun(handed(model, cache: cache), config: StageConfig(seed: 1)) { _ in probe }
        _ = try await stage.run(.text("x")) { _ in }
        #expect(probe.runs == 1)                                         // the view's per-run stage is what ran
        #expect(probe.seen.first?.count == 1 && (probe.seen.first?.used ?? 0) >= 3 * gb)   // booked during the run
        #expect(!(stage as? LeasingStage)!.isRunning)                    // lease returned
    }

    @Test func aSeedChangeLoadsNoSecondEngine() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: true)
        let model = makeEntry(id: "seeded", modelType: .image, ramGB: 3)
        let base = handed(model, cache: cache)
        let probes = (0..<3).map { _ in Recorder(id: model.id, observe: { (cache.count, budget.usedBytes) }) }
        for (i, probe) in probes.enumerated() {
            let stage = LeasingStage.perRun(base, config: StageConfig(seed: UInt64(i))) { _ in probe }
            _ = try await stage.run(.text("x")) { _ in }
        }
        #expect(probes.flatMap(\.seen).allSatisfy { $0.count == 1 })     // one engine's worth each time, never a second
        #expect(Set(probes.flatMap(\.seen).map(\.used)).count == 1)
        #expect(probes.map(\.runs) == [1, 1, 1])                          // each run used its own seed's stage
    }

    @Test func aPerRunStageBooksOnlyForTheRun() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: true)
        let model = makeEntry(id: "book-run", modelType: .image, ramGB: 3)
        let base = handed(model, cache: cache)
        let before = budget.usedBytes
        for _ in 0..<2 {                                                 // a second run books again, then gives back again
            let stage = LeasingStage.perRun(base, config: StageConfig(seed: 1)) { _ in Recorder(id: model.id) }
            _ = try await stage.run(.text("x")) { _ in }
            #expect(budget.usedBytes == before && cache.count == 0)
        }
    }

    private nonisolated final class Gate: PipelineStage, @unchecked Sendable {
        let id = "gate"; var name: String { id }
        var accepts: MediaKind { .text }; var produces: MediaKind { .text }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
            try await Task.sleep(for: .seconds(30)); return input
        }
    }

    @Test func aCancelledOrFailedPerRunStageLeavesNothingBooked() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: true)
        let model = makeEntry(id: "book-cancel", modelType: .tts, ramGB: 3)
        let base = handed(model, cache: cache)
        let before = budget.usedBytes
        let running = LeasingStage.perRun(base, config: StageConfig(voice: "v")) { _ in Gate() }
        let task = Task { try await running.run(.text("x")) { _ in } }
        for _ in 0..<500 where budget.usedBytes == before { try? await Task.sleep(for: .milliseconds(10)) }
        #expect(budget.usedBytes > before)                              // booked while running
        task.cancel()
        _ = try? await task.value
        #expect(budget.usedBytes == before && cache.count == 0)
    }

    @Test func notHandedALeasingStageItJustRunsTheViewsStage() async throws {
        let probe = Recorder(id: "x")
        let stage = LeasingStage.perRun(Recorder(id: "handed"), config: .default) { _ in probe }
        _ = try await stage.run(.text("x")) { _ in }
        #expect(probe.runs == 1)
    }

    // MARK: no Run view builds its own stage

    /// Every module is enumerated; every file that is a Run view or a `ModelUI` and constructs a `…Stage(` must go
    /// through a `LeasingStage.perRun` / `.rebuilding` call in the same file. Source-level on purpose: a view's `run()` is private SwiftUI.
    @Test func noRunViewBuildsItsOwnStage() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("MLXUI")
        let fm = FileManager.default
        guard let walker = fm.enumerator(at: root, includingPropertiesForKeys: nil) else {
            Issue.record("cannot read the source tree at \(root.path)"); return
        }
        var offenders: [String] = [], inspected = 0
        let ctor = try NSRegularExpression(pattern: #"(?<![A-Za-z])(?!Leasing|Pipeline)[A-Z][A-Za-z0-9]*Stage\("#)
        for case let url as URL in walker where url.pathExtension == "swift" {
            let path = url.path
            let isRunSurface = path.contains("/Views/Run/") || url.lastPathComponent.hasSuffix("UI.swift")
            guard isRunSurface, !path.contains("/Modules/Chat/") else { continue }
            let text = try String(contentsOf: url, encoding: .utf8)
            inspected += 1
            let hit = ctor.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
            if hit, !text.contains("LeasingStage.perRun(") && !text.contains("LeasingStage.rebuilding(") { offenders.append(url.lastPathComponent) }
        }
        #expect(inspected > 15, "the scan found only \(inspected) Run surfaces")
        #expect(offenders.isEmpty, "Run views building their own stage: \(offenders)")
    }
}
