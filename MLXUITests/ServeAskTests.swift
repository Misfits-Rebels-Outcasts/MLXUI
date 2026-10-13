import Testing
import Foundation
@testable import MLXUI

/// SC-5a (gate D) — ask before pushing out a served model: the pure check's truth table, the confirmer (Load anyway
/// unpins, evicts and loads; Cancel changes nothing), and the surfaces that must go through it — chat and a Run window
/// never push out a pinned model without the question; a flow is unchanged until SC-5b.
private let gb = Int64(1_073_741_824)

private func item(_ bytes: Int64, pinned: Bool = false, inUse: Bool = false, snapshot: Bool = false) -> MemoryBudget.LedgerItem {
    MemoryBudget.LedgerItem(id: UUID(), bytes: bytes * gb, pinned: pinned, inUse: inUse, isSnapshot: snapshot)
}

private func snapshot(capacity: Int64, _ items: [MemoryBudget.LedgerItem], pins: [(String, MemoryBudget.LedgerItem)]) -> DisplacementSnapshot {
    DisplacementSnapshot(capacityBytes: capacity * gb, items: items,
                         pins: pins.map { DisplacementSnapshot.Pin(key: $0.0, item: $0.1.id) })
}

// MARK: - the check

struct ServeAskCheckTests {
    private func displaced(_ load: Int64, capacity: Int64 = 10, _ items: [MemoryBudget.LedgerItem],
                           pins: [(String, MemoryBudget.LedgerItem)] = [], requester: MemoryBudget.HolderKind? = .chat,
                           gate: Bool = true) -> [String] {
        ServeMemory.wouldDisplaceServed(loadBytes: load * gb, requester: requester,
                                        snapshot: snapshot(capacity: capacity, items, pins: pins),
                                        servedID: { $0 }, gateEnabled: gate)
    }

    @Test func fitsAsksNothing() {
        let kept = item(4, pinned: true)
        #expect(displaced(5, [kept], pins: [("kept", kept)]).isEmpty)
    }

    @Test func onlyUnpinnedIdleModelsNeedingToGoAsksNothing() {
        let kept = item(4, pinned: true), idle = item(4), snap = item(1, snapshot: true)
        #expect(displaced(5, [kept, idle, snap], pins: [("kept", kept)]).isEmpty)
    }

    @Test func onePinnedModelWouldGoSoItIsNamed() {
        let kept = item(6, pinned: true)
        #expect(displaced(6, [kept], pins: [("kept", kept)]) == ["kept"])
    }

    @Test func severalPinnedModelsAreNamedLeastRecentlyUsedFirstAndNoMoreThanNeeded() {
        let a = item(3, pinned: true), b = item(3, pinned: true), c = item(3, pinned: true)
        let pins = [("a", a), ("b", b), ("c", c)]                       // a is the least recently used
        #expect(displaced(2, [a, b, c], pins: pins) == ["a"])                   // 9 + 2 - 10 = 1 short: a (3) covers it
        #expect(displaced(5, [a, b, c], pins: pins) == ["a", "b"])              // 4 short: a is not enough, a and b are
        #expect(displaced(8, [a, b, c], pins: pins) == ["a", "b", "c"])         // 7 short: all three
    }

    @Test func aPinnedModelInUseIsNeverNamed() {
        let busy = item(6, pinned: true, inUse: true)
        #expect(displaced(6, [busy], pins: [("busy", busy)]).isEmpty)
    }

    @Test func onlyModelsInUseInTheWayAsksNothing() {
        let busy = item(8, inUse: true)
        let kept = item(1, pinned: true)
        #expect(displaced(4, [busy, kept], pins: [("kept", kept)]).isEmpty)     // in use frees by itself; kept alone isn't enough
    }

    @Test func pushingOutEverythingPinnedStillNotEnoughAsksNothing() {
        let kept = item(2, pinned: true)
        #expect(displaced(11, [kept], pins: [("kept", kept)]).isEmpty)           // can never fit: pushing out would be for nothing
    }

    @Test func gateClosedAsksNothing() {
        let kept = item(6, pinned: true)
        #expect(displaced(6, [kept], pins: [("kept", kept)], gate: false).isEmpty)
    }

    @Test func theServerNeverAsks() {
        let kept = item(6, pinned: true)
        #expect(displaced(6, [kept], pins: [("kept", kept)], requester: .serverRequest).isEmpty)
    }

    @Test func aPinnedKeyThatIsNotAServedModelIsNeverNamed() {
        let kept = item(6, pinned: true)
        let named = ServeMemory.wouldDisplaceServed(loadBytes: 6 * gb, requester: .chat,
                                                    snapshot: snapshot(capacity: 10, [kept], pins: [("kept", kept)]),
                                                    servedID: { _ in nil }, gateEnabled: true)
        #expect(named.isEmpty)
    }
}

// MARK: - the confirmer and the surfaces

private final class Box: Sendable {}

@MainActor
@Suite(.serialized)
struct ServeAskConfirmerTests {
    private func makeBudget(_ gbs: Int64) -> MemoryBudget {
        MemoryBudget(capacityBytes: gbs * gb,
                     cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
    }

    private func pinned(_ key: String, gbs: Int64, in pool: ResidentPool<Box>) async throws {
        _ = try await pool.value(for: key, bytes: gbs * gb) { Box() }
        #expect(await pool.pin(key))
    }

    /// A confirmer over one `ResidentPool` of served models: the real check and the real push-out, the pool's keys as ids.
    private func confirmer(over pool: ResidentPool<Box>, checks: Counter = Counter(), reload: Int? = nil) -> ServedLoadConfirmer {
        ServedLoadConfirmer(
            check: { bytes, requester in
                checks.bump()
                return ServeMemory.wouldDisplaceServed(loadBytes: bytes, requester: requester,
                                                       snapshot: await pool.displacementSnapshot(), servedID: { $0 })
            },
            pushOut: { ids, requester in _ = await pool.pushOutPinned(keys: ids, for: requester) },
            reloadSeconds: { _ in reload })
    }

    final class Counter: @unchecked Sendable {
        private let lock = NSLock(); private var n = 0
        func bump() { lock.lock(); n += 1; lock.unlock() }
        var count: Int { lock.lock(); defer { lock.unlock() }; return n }
    }

    private func waitForQuestion(_ confirmer: ServedLoadConfirmer) async -> Bool {
        for _ in 0..<500 where confirmer.question == nil { try? await Task.sleep(for: .milliseconds(10)) }
        return confirmer.question != nil
    }

    // MARK: sentence

    @Test func theSentenceNamesTheModelsAndTheReloadTime() {
        let one = ServedLoadConfirmer.Question(modelName: "Qwen3-8B", servedIDs: ["mlx-community/Llama-3.2-3B-4bit"], reloadSeconds: 12)
        #expect(one.message == "Loading Qwen3-8B will stop keeping Llama-3.2-3B-4bit ready for connected apps. "
                + "It stays served and reloads on the next request (about 12 s).")
        let two = ServedLoadConfirmer.Question(modelName: "Whisper", servedIDs: ["a/A", "b/B"], reloadSeconds: nil)
        #expect(two.message == "Loading Whisper will stop keeping A and B ready for connected apps. "
                + "They stay served and reload on the next request.")
    }

    // MARK: Load anyway / Cancel

    @Test func loadAnywayUnpinsAndEvictsOnlyTheNamedModelsThenTheLoadSucceeds() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        try await pinned("a", gbs: 4, in: pool)
        try await pinned("b", gbs: 4, in: pool)
        let confirmer = confirmer(over: pool, reload: 9)

        let asked = Task { await confirmer.confirm(modelName: "Chat model", loadBytes: 4 * gb, requester: .chat) }
        #expect(await waitForQuestion(confirmer))
        #expect(confirmer.question?.servedIDs == ["a"])                          // the least recently used, no more than needed
        #expect(confirmer.question?.reloadSeconds == 9)
        #expect(await pool.pinnedKeys == ["a", "b"])                              // nothing happens until the answer
        confirmer.respond(true)
        #expect(await asked.value)
        #expect(confirmer.question == nil)
        #expect(await pool.pinnedKeys == ["b"])
        #expect(await pool.loadState(for: "a") == .notLoaded(reason: "pushed out by Chat"))
        #expect(await pool.loadState(for: "b") == .ready)
        let lease = try await pool.lease(for: "chat", bytes: 4 * gb, maxWait: .milliseconds(300), holder: .chat) { Box() }
        lease.release()
        #expect(Set(await pool.residentKeys) == ["b", "chat"])
    }

    @Test func cancelLoadsNothingAndChangesNoState() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        try await pinned("a", gbs: 6, in: pool)
        let confirmer = confirmer(over: pool)
        let asked = Task { await confirmer.confirm(modelName: "Chat model", loadBytes: 6 * gb, requester: .runWindow) }
        #expect(await waitForQuestion(confirmer))
        confirmer.respond(false)
        #expect(await asked.value == false)
        #expect(confirmer.question == nil)
        #expect(await pool.pinnedKeys == ["a"])
        #expect(await pool.residentKeys == ["a"])
        #expect(await pool.loadState(for: "a") == .ready)
    }

    @Test func dismissingTheSheetOrCancellingTheCallerIsCancel() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        try await pinned("a", gbs: 6, in: pool)
        let confirmer = confirmer(over: pool)
        let asked = Task { await confirmer.confirm(modelName: "M", loadBytes: 6 * gb, requester: .chat) }
        #expect(await waitForQuestion(confirmer))
        asked.cancel()
        #expect(await asked.value == false)
        #expect(await pool.pinnedKeys == ["a"])
    }

    @Test func nothingToAskGoesStraightThrough() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        try await pinned("a", gbs: 4, in: pool)
        let confirmer = confirmer(over: pool)
        #expect(await confirmer.confirm(modelName: "M", loadBytes: 4 * gb, requester: .chat))
        #expect(confirmer.question == nil)
        #expect(await pool.pinnedKeys == ["a"])
    }

    @Test func gateClosedMeansNoCheckAtAll() async throws {
        LocalServerGate.overrideForTesting = false
        defer { LocalServerGate.overrideForTesting = nil }
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        try await pinned("a", gbs: 6, in: pool)
        let checks = Counter()
        let confirmer = confirmer(over: pool, checks: checks)
        #expect(await confirmer.confirm(modelName: "M", loadBytes: 6 * gb, requester: .chat))
        #expect(checks.count == 0 && confirmer.question == nil)
        #expect(await pool.pinnedKeys == ["a"])
    }

    // MARK: Chat and Run windows never push out without asking; a flow is unchanged

    @Test(arguments: [MemoryBudget.HolderKind.chat, .runWindow])
    func chatAndARunWindowDoNotPushOutAPinnedModelOnTheirOwn(holder: MemoryBudget.HolderKind) async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        try await pinned("kept", gbs: 6, in: pool)
        await #expect(throws: MemoryBudget.WaitError.self) {
            _ = try await pool.lease(for: "mine", bytes: 6 * gb, maxWait: .milliseconds(200), holder: holder) { Box() }
        }
        #expect(await pool.isPinned("kept"))
        #expect(await pool.loadState(for: "kept") == .ready)
    }

    @Test func anEngineThatSaysFlowInsideARunWindowRunIsStillARunWindow() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        try await pinned("kept", gbs: 6, in: pool)
        await #expect(throws: MemoryBudget.WaitError.self) {
            try await MemoryBudget.RunWindowScope.$holder.withValue(.runWindow) {
                _ = try await pool.lease(for: "vlm", bytes: 6 * gb, maxWait: .milliseconds(200), holder: .flow) { Box() }
            }
        }
        #expect(await pool.isPinned("kept"))
    }

    @Test func aFlowIsUnchangedUntilSC5b() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        try await pinned("kept", gbs: 6, in: pool)
        let lease = try await pool.lease(for: "row", bytes: 6 * gb, maxWait: .milliseconds(300), holder: .flow) { Box() }
        lease.release()
        #expect(await pool.loadState(for: "kept") == .notLoaded(reason: "pushed out by a flow"))
    }

    // MARK: Run windows

    private nonisolated struct Stub: PipelineStage {
        let id: String
        var name: String { id }
        var accepts: MediaKind { .text }
        var produces: MediaKind { .text }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media { input }
    }

    @Test func aRunWindowRunAsksThenLoadsOrStops() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let budget = makeBudget(10)
        let served = ResidentPool<Box>(budget: budget)
        try await pinned("served", gbs: 6, in: served)
        let cache = EngineCache(budget: budget, countsLLMWeights: true)
        let confirmer = confirmer(over: served)
        let model = makeEntry(id: "ask-tts", modelType: .tts, ramGB: 6)
        let stage = LeasingStage(model: model, config: .default, probe: Stub(id: model.id), cache: cache,
                                 maxWait: .milliseconds(300), confirmer: confirmer)

        // Cancel: the run does not start, nothing was loaded, the served model is untouched
        let cancelled = Task { try await stage.run(.text("x")) { _ in } }
        #expect(await waitForQuestion(confirmer))
        confirmer.respond(false)
        await #expect(throws: CancellationError.self) { _ = try await cancelled.value }
        #expect(cache.count == 0 && !stage.isRunning)
        #expect(await served.isPinned("served") && budget.usedBytes == 6 * gb)

        // Load anyway: the served model is pushed out for this run, and the run happens
        let run = Task { try await stage.run(.text("x")) { _ in } }
        #expect(await waitForQuestion(confirmer))
        confirmer.respond(true)
        _ = try await run.value
        #expect(cache.count == 1)
        #expect(await served.loadState(for: "served") == .notLoaded(reason: "pushed out by a Run window"))
    }

    @Test func aRunWindowWithNoQuestionAnsweredNeverPushesOutSilently() async throws {
        let budget = makeBudget(10)
        let served = ResidentPool<Box>(budget: budget)
        try await pinned("served", gbs: 6, in: served)
        let cache = EngineCache(budget: budget, countsLLMWeights: true)
        let model = makeEntry(id: "silent-tts", modelType: .tts, ramGB: 6)
        let stage = LeasingStage(model: model, config: .default, probe: Stub(id: model.id), cache: cache,
                                 maxWait: .milliseconds(200))                     // no confirmer: the old silent path is gone
        await #expect(throws: MemoryBudget.WaitError.self) { _ = try await stage.run(.text("x")) { _ in } }
        #expect(await served.isPinned("served") && cache.count == 0)
    }

    // MARK: Chat

    private final class OneChunk: ChatStreamingSession, @unchecked Sendable {
        func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
            AsyncThrowingStream { $0.yield("hi"); $0.finish() }
        }
    }

    private final class Calls: @unchecked Sendable {
        private let lock = NSLock(); private var _sessions = 0; private var _pushed: [String] = []
        func session() { lock.lock(); _sessions += 1; lock.unlock() }
        func push(_ ids: [String]) { lock.lock(); _pushed += ids; lock.unlock() }
        var sessions: Int { lock.lock(); defer { lock.unlock() }; return _sessions }
        var pushed: [String] { lock.lock(); defer { lock.unlock() }; return _pushed }
    }

    private func chatRunner(calls: Calls) throws -> (ModelRunner, ServedLoadConfirmer, URL) {
        let store = FileManager.default.temporaryDirectory.appendingPathComponent("sc5a-store-\(UUID().uuidString)")
        let modelDir = FileManager.default.temporaryDirectory.appendingPathComponent("sc5a-model-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        let confirmer = ServedLoadConfirmer(check: { _, _ in ["served"] },
                                            pushOut: { ids, _ in calls.push(ids) },
                                            reloadSeconds: { _ in nil })
        let runner = ModelRunner(loadConfirmer: confirmer)
        let entry = makeEntry(id: "chat-model", displayName: "Chat Model")
        runner.prepare(for: entry, in: store)
        runner.modelDirectoryForTesting = { _ in modelDir }
        runner.streamingSessionForTesting = { calls.session(); return OneChunk() }
        return (runner, confirmer, store)
    }

    @Test func aChatTurnAsksFirstAndCancelStartsNothing() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let calls = Calls()
        let (runner, confirmer, store) = try chatRunner(calls: calls)
        defer { try? FileManager.default.removeItem(at: store) }
        let entry = makeEntry(id: "chat-model", displayName: "Chat Model")

        let turn = Task { await runner.send("hello", for: entry) }
        #expect(await waitForQuestion(confirmer))
        #expect(runner.transcript.items.isEmpty && calls.sessions == 0)         // asked before anything starts
        confirmer.respond(false)
        await turn.value
        #expect(runner.transcript.items.isEmpty && calls.sessions == 0 && calls.pushed.isEmpty)
        #expect(runner.errorMessage == nil && !runner.isRunning)
    }

    @Test func aChatTurnAsksFirstAndLoadAnywayPushesOutThenRuns() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let calls = Calls()
        let (runner, confirmer, store) = try chatRunner(calls: calls)
        defer { try? FileManager.default.removeItem(at: store) }
        let entry = makeEntry(id: "chat-model", displayName: "Chat Model")

        let turn = Task { await runner.send("hello", for: entry) }
        #expect(await waitForQuestion(confirmer))
        confirmer.respond(true)
        await turn.value
        for _ in 0..<500 where runner.isRunning { try await Task.sleep(for: .milliseconds(10)) }
        #expect(calls.pushed == ["served"] && calls.sessions == 1)
        #expect(!runner.transcript.items.isEmpty && runner.errorMessage == nil)
    }
}
