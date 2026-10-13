import Testing
import Foundation
@testable import MLXUI

/// SC-2 (gate C) — a served model that is resident is kept ready: `ResidentPool.pin`, who may push a pinned idle
/// model out (Chat, a Run window, a flow; and the server only for a model that is not itself kept ready), and the
/// controller's pin recompute. Plain values in a `ResidentPool`, since a real `ModelContainer` can't be fabricated.
struct ServePinPoolTests {

    private static let gb = Int64(1_073_741_824)
    private final class Box: Sendable {}

    private func makeBudget(_ gbs: Int64) -> MemoryBudget {
        MemoryBudget(capacityBytes: gbs * Self.gb,
                     cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
    }

    /// `kept` resident and pinned, idle.
    private func pinned(_ key: String, gbs: Int64, in pool: ResidentPool<Box>) async throws {
        _ = try await pool.value(for: key, bytes: gbs * Self.gb) { Box() }
        #expect(await pool.pin(key))
    }

    @Test func onlyAResidentEntryCanBePinned() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        #expect(await pool.pin("a") == false)
        try await pinned("a", gbs: 2, in: pool)
        #expect(await pool.pinnedKeys == ["a"])
        await pool.unpin("a")
        #expect(await pool.pinnedKeys.isEmpty)
    }

    @Test func removalTakesThePinWithIt() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        try await pinned("a", gbs: 2, in: pool)
        await pool.evict(key: "a")                                         // explicit evict still works on a pinned entry
        #expect(await pool.residentKeys.isEmpty)
        _ = try await pool.value(for: "a", bytes: 2 * Self.gb) { Box() }
        #expect(await pool.isPinned("a") == false)                         // a reload is not pinned until pinned again
        try await pinned("b", gbs: 2, in: pool)
        await pool.clear()
        #expect(await pool.pinnedKeys.isEmpty)
    }

    @Test func aPinnedModelSurvivesAServerRequestAndAPreloadForAnotherModel() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        try await pinned("kept", gbs: 5, in: pool)
        _ = try await pool.value(for: "idle", bytes: 4 * Self.gb) { Box() }

        // the order: the unpinned idle model goes first, the pinned one only if that were not enough
        let request = try await pool.lease(for: "other", bytes: 4 * Self.gb, maxWait: .milliseconds(300), holder: .serverRequest) { Box() }
        request.release()
        #expect(await pool.residentKeys.contains("kept"))
        #expect(await pool.residentKeys.contains("idle") == false)

        // a preload that needs the pinned model's room declines instead
        let state = await pool.preload(for: "big", bytes: 7 * Self.gb) { Box() }
        #expect(state == .notLoaded(reason: ResidentPool<Box>.notEnoughMemoryReason))
        #expect(await pool.isPinned("kept"))
        #expect(await pool.loadState(for: "kept") == .ready)
    }

    @Test func aServerRequestForAModelNotKeptReadyMayTakeAnIdlePinnedOne() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        try await pinned("kept", gbs: 6, in: pool)
        let request = try await pool.lease(for: "unkept", bytes: 6 * Self.gb, maxWait: .milliseconds(300), holder: .serverRequest) { Box() }
        request.release()
        #expect(await pool.residentKeys == ["unkept"])
        #expect(await pool.isPinned("kept") == false)
        #expect(await pool.loadState(for: "kept") == .notLoaded(reason: "pushed out by a server request"))
    }

    @Test(arguments: [
        (MemoryBudget.HolderKind.chat, "pushed out by Chat"),
        (MemoryBudget.HolderKind.runWindow, "pushed out by a Run window"),
        (MemoryBudget.HolderKind.flow, "pushed out by a flow"),
    ])
    func chatARunWindowAndAFlowStillPushOutAPinnedIdleModelAndSayWho(holder: MemoryBudget.HolderKind, reason: String) async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        try await pinned("kept", gbs: 6, in: pool)
        let lease = try await pool.lease(for: "mine", bytes: 6 * Self.gb, maxWait: .milliseconds(300), holder: holder) { Box() }
        lease.release()
        #expect(await pool.residentKeys == ["mine"])
        #expect(await pool.isPinned("kept") == false)
        #expect(await pool.loadState(for: "kept") == .notLoaded(reason: reason))
    }

    @Test func pushedOutIsReportedToObserversWithTheReason() async throws {
        final class Seen: @unchecked Sendable {
            let lock = NSLock(); var list: [ModelLoadState] = []
            func add(_ s: ModelLoadState) { lock.lock(); list.append(s); lock.unlock() }
        }
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        try await pinned("kept", gbs: 6, in: pool)
        let seen = Seen()
        await pool.addObserver { key, state in if key == "kept" { seen.add(state) } }
        let lease = try await pool.lease(for: "mine", bytes: 6 * Self.gb, holder: .chat) { Box() }
        lease.release()
        #expect(seen.list == [.notLoaded(reason: "pushed out by Chat")])
    }

    @Test func aPinnedModelInUseIsNeverEvictedByAnyone() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        _ = try await pool.value(for: "kept", bytes: 6 * Self.gb) { Box() }
        let busy = try await pool.lease(for: "kept", bytes: 6 * Self.gb) { Box() }
        await pool.pin("kept")
        await #expect(throws: MemoryBudget.WaitError.self) {
            _ = try await pool.lease(for: "mine", bytes: 6 * Self.gb, maxWait: .milliseconds(200), holder: .chat) { Box() }
        }
        #expect(await pool.residentKeys == ["kept"])
        #expect(await pool.isPinned("kept"))
        busy.release()
    }

    @Test func unpinnedIdleModelsGoBeforeAPinnedOne() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        try await pinned("kept", gbs: 4, in: pool)                         // least recently used, but pinned
        _ = try await pool.value(for: "plain", bytes: 4 * Self.gb) { Box() }
        let lease = try await pool.lease(for: "mine", bytes: 4 * Self.gb, holder: .chat) { Box() }
        lease.release()
        #expect(Set(await pool.residentKeys) == ["kept", "mine"])
        #expect(await pool.isPinned("kept"))
    }

    @Test func anotherPoolsInteractiveLoadReachesAPinnedEntryAcrossTheSharedBudget() async throws {
        let budget = makeBudget(10)
        let a = ResidentPool<Box>(budget: budget)
        let b = ResidentPool<Box>(budget: budget)
        try await pinned("kept", gbs: 6, in: a)
        // the server's own loads in the other pool cannot take it (its model is the pinned kind: here, the same key class)
        let state = await b.preload(for: "p", bytes: 6 * Self.gb) { Box() }
        #expect(state == .notLoaded(reason: ResidentPool<Box>.notEnoughMemoryReason))
        #expect(await a.isPinned("kept"))
        // an interactive one can
        let lease = try await b.lease(for: "mine", bytes: 6 * Self.gb, holder: .flow) { Box() }
        lease.release()
        #expect(await a.residentKeys.isEmpty)
        #expect(await a.loadState(for: "kept") == .notLoaded(reason: "pushed out by a flow"))
    }

    @Test func aPinnedModelsOwnRequestMayTakeAnotherIdlePinnedModelButNeverItself() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        try await pinned("a", gbs: 4, in: pool)
        try await pinned("b", gbs: 4, in: pool)
        // b's request needs 5 GB more than the 2 GB that is free; "a" is the only thing that can give it
        let outcome = await pool.reserveWorkingSetReporting(totalNeedBytes: 9 * Self.gb, protecting: "b")
        guard case .granted = outcome else { Issue.record("not granted: \(outcome)"); return }
        #expect(await pool.residentKeys == ["b"])
        #expect(await pool.isPinned("b"))                                   // never the requester
        #expect(await pool.loadState(for: "a") == .notLoaded(reason: "pushed out by a server request"))
    }

    @Test func aServerRequestTakesTheLeastRecentlyUsedIdleServedModelAndNoMore() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        try await pinned("old", gbs: 4, in: pool)
        try await pinned("new", gbs: 4, in: pool)
        let lease = try await pool.lease(for: "third", bytes: 4 * Self.gb, maxWait: .milliseconds(300), holder: .serverRequest) { Box() }
        lease.release()
        #expect(Set(await pool.residentKeys) == ["new", "third"])
        #expect(await pool.isPinned("new"))
        #expect(await pool.loadState(for: "old") == .notLoaded(reason: "pushed out by a server request"))
    }

    @Test func tooLargeOnlyWhenNothingCanBeFreedAndAServedModelInUseIsNeverTaken() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        try await pinned("a", gbs: 4, in: pool)
        try await pinned("b", gbs: 4, in: pool)
        let busy = try await pool.lease(for: "a", bytes: 4 * Self.gb) { Box() }      // a is in use
        let outcome = await pool.reserveWorkingSetReporting(totalNeedBytes: 9 * Self.gb, protecting: "b",
                                                            maxWait: .milliseconds(200))
        if case .busy = outcome {} else { Issue.record("expected busy, got \(outcome)") }
        #expect(Set(await pool.residentKeys) == ["a", "b"])
        busy.release()
        // and when even the requester alone is the whole budget, it is too large
        let huge = await pool.reserveWorkingSetReporting(totalNeedBytes: 11 * Self.gb, protecting: "b")
        if case .tooLarge = huge {} else { Issue.record("expected tooLarge, got \(huge)") }
    }

    @Test func aPreloadStillNeverTakesAServedModel() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        try await pinned("a", gbs: 6, in: pool)
        let state = await pool.preload(for: "p", bytes: 6 * Self.gb) { Box() }
        #expect(state == .notLoaded(reason: ResidentPool<Box>.notEnoughMemoryReason))
        #expect(await pool.isPinned("a"))
    }

    // MARK: - the real requester (SC-2f items 2 and 3)

    private func sparseModelDir(gib: Int64) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("sc2f-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("model.safetensors")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: UInt64(gib * Self.gb))
        try handle.close()
        return dir
    }

    @Test(arguments: [
        (MemoryBudget.HolderKind.runWindow, "pushed out by a Run window"),
        (MemoryBudget.HolderKind.flow, "pushed out by a flow"),
    ])
    func aVLMCallsWorkingSetPassesItsRealRequester(holder: MemoryBudget.HolderKind, reason: String) async throws {
        struct Boom: Error {}
        let dir = try sparseModelDir(gib: 3)
        defer { try? FileManager.default.removeItem(at: dir) }
        let budget = makeBudget(20)
        let served = ResidentPool<Box>(budget: budget)
        try await pinned("served", gbs: 16, in: served)                     // leaves too little for weights + working set
        let vlm = ModelContainerPool(budget: budget, loader: { _, _ in throw Boom() })
        let working = try await VLMEngine.reserveCallWorkingSet(pool: vlm, modelDir: dir, maxWait: .seconds(1), requester: holder)
        budget.release(working)
        #expect(await served.loadState(for: "served") == .notLoaded(reason: reason))
    }

    private nonisolated struct Stub: PipelineStage {
        let id: String
        var name: String { id }
        var accepts: MediaKind { .text }
        var produces: MediaKind { .text }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media { input }
    }

    @Test(arguments: [
        (MemoryBudget.HolderKind.runWindow, "pushed out by a Run window"),
        (MemoryBudget.HolderKind.flow, "pushed out by a flow"),
    ])
    func anEngineCacheBuildPassesItsRealRequester(holder: MemoryBudget.HolderKind, reason: String) async throws {
        let budget = makeBudget(10)
        let served = ResidentPool<Box>(budget: budget)
        try await pinned("served", gbs: 6, in: served)
        let cache = EngineCache(budget: budget, countsLLMWeights: true)
        let lease = try await cache.lease(for: makeEntry(id: "engine", ramGB: 6), config: .default,
                                          maxWait: .seconds(1), holder: holder) { m, _ in Stub(id: m.id) }
        lease.hold.release()
        #expect(await served.loadState(for: "served") == .notLoaded(reason: reason))
    }

    @Test func theWorkingSetOfAModelNotKeptReadyMayTakeAnIdlePinnedOne() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        try await pinned("a", gbs: 4, in: pool)
        _ = try await pool.value(for: "x", bytes: 4 * Self.gb) { Box() }
        let outcome = await pool.reserveWorkingSetReporting(totalNeedBytes: 9 * Self.gb, protecting: "x")
        guard case .granted = outcome else { Issue.record("not granted: \(outcome)"); return }
        #expect(await pool.loadState(for: "a") == .notLoaded(reason: "pushed out by a server request"))
    }
}

// MARK: - The controller

private let pinA = "mlx-community/Pin-A-4bit"
private let pinB = "mlx-community/Pin-B-4bit"
private let pinC = "mlx-community/Pin-C-4bit"

private final class PinBox: Sendable {}

/// A `ServePreloading` over a `ResidentPool` of plain values, with the pin calls the controller makes.
private struct PinPreloader: ServePreloading {
    let pool: ResidentPool<PinBox>
    func key(_ id: String) -> String { ModelContainerPool.key(for: ModelStore.shared.directory(forHFModelID: id)) }
    func preload(directory: URL, footprintBytes: Int64?) async -> ModelLoadState {
        await pool.preload(for: ModelContainerPool.key(for: directory), bytes: footprintBytes ?? 1) { PinBox() }
    }
    func addLoadObserver(_ observer: @escaping @Sendable (String, ModelLoadState) -> Void) async {
        await pool.addObserver(observer)
    }
    func residentBytes(directory: URL) async -> Int64? { await pool.residentBytes(for: ModelContainerPool.key(for: directory)) }
    func setPinned(directories: [URL]) async { await pool.setPinned(Set(directories.map { ModelContainerPool.key(for: $0) })) }
}

@MainActor
struct ServePinControllerTests {
    private static let gb = Int64(1_073_741_824)

    private func budget(_ gbs: Int64) -> MemoryBudget {
        MemoryBudget(capacityBytes: gbs * Self.gb,
                     cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
    }

    private func installed() -> InstalledModelIndex {
        InstalledModelIndex(entries: [pinA, pinB, pinC].map {
            .init(id: $0, hfModelId: $0, kind: .llm, ramGB: 4, downloadSizeGB: 2.67)
        })
    }

    /// `poolGBs` is the shared budget the models load into; `controllerGBs` is the capacity the controller pins against.
    private func makeController(preloader: PinPreloader, controllerGBs: Int64 = 20) -> LocalServerController {
        let server = LocalServer(settings: ServeSettings(servedModelIDs: []), defaults: nil, listenPort: 0,
                                 environment: ServeEnvironment(servedModelIDs: { [] },
                                                               installed: { InstalledModelIndex(entries: []) },
                                                               appleFoundationReadiness: { nil }, created: 1))
        return LocalServerController(server: server,
                                     defaults: UserDefaults(suiteName: "pin-tests-\(UUID().uuidString)")!,
                                     installed: { self.installed() },
                                     appleFoundation: { .ready }, catalogLoaded: { true },
                                     evict: { _ in }, evictPromptCache: { _ in },
                                     budgetBytes: controllerGBs * Self.gb,
                                     preloadPool: preloader)
    }

    private func until(_ timeout: Duration = .seconds(5), _ condition: () async -> Bool) async -> Bool {
        let end = ContinuousClock.now.advanced(by: timeout)
        while ContinuousClock.now < end {
            if await condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return await condition()
    }

    private func preloader(_ gbs: Int64 = 20) -> PinPreloader { PinPreloader(pool: ResidentPool<PinBox>(budget: budget(gbs))) }

    @Test func aServedModelThatBecomesReadyIsPinned() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let p = preloader()
        let controller = makeController(preloader: p)
        await controller.setServed(pinA, true)
        #expect(await until { await p.pool.pinnedKeys == [p.key(pinA)] })
        #expect(controller.loadState(for: pinA) == .ready)
        await controller.setServed(pinA, false)
        await controller.server.stop()
    }

    @Test func serveOffUnpinsButDoesNotEvict() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let p = preloader()
        let controller = makeController(preloader: p)
        await controller.setServed(pinA, true)
        await controller.setServed(pinB, true)
        #expect(await until { await p.pool.pinnedKeys.count == 2 })
        await controller.setServed(pinA, false)
        #expect(await until { await p.pool.pinnedKeys == [p.key(pinB)] })
        #expect(Set(await p.pool.residentKeys) == [p.key(pinA), p.key(pinB)])      // A stays resident, an ordinary idle entry
        await controller.setServed(pinB, false)
        await controller.server.stop()
    }

    @Test func masterSwitchOffUnpinsButDoesNotEvict() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let p = preloader()
        let controller = makeController(preloader: p)
        await controller.setServed(pinA, true)
        #expect(await until { await p.pool.pinnedKeys.count == 1 })
        await controller.setServerEnabled(false)
        #expect(await p.pool.pinnedKeys.isEmpty)
        #expect(await p.pool.residentKeys == [p.key(pinA)])
        await controller.setServerEnabled(true)                                    // and back on pins it again
        #expect(await until { await p.pool.pinnedKeys == [p.key(pinA)] })
        await controller.setServed(pinA, false)
        await controller.server.stop()
    }

    @Test func gateClosedMeansNoPins() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let p = preloader()
        let controller = makeController(preloader: p)
        await controller.setServed(pinA, true)
        #expect(await until { await p.pool.pinnedKeys.count == 1 })
        LocalServerGate.overrideForTesting = false
        await controller.recomputePins()
        #expect(await p.pool.pinnedKeys.isEmpty)
        LocalServerGate.overrideForTesting = true
        await controller.setServed(pinA, false)
        await controller.server.stop()
    }

    @Test func aServedSetLargerThanTheBudgetPinsTheMostRecentlyServedThatFit() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let p = preloader(30)
        // all three are already in memory (loaded by requests); the controller's capacity fits two of the three 4 GB models
        let controller = makeController(preloader: p, controllerGBs: 9)
        for id in [pinA, pinB, pinC] { _ = try await p.pool.value(for: p.key(id), bytes: 4 * Self.gb) { PinBox() } }
        await controller.setServed(pinA, true)
        await controller.setServed(pinB, true)
        await controller.setServed(pinC, true)                                     // most recent: C, then B, then A
        #expect(await until { await p.pool.pinnedKeys == [p.key(pinB), p.key(pinC)] })
        #expect(controller.loadState(for: pinC) == .ready)
        #expect(controller.loadState(for: pinB) == .ready)
        #expect(controller.loadState(for: pinA) == .notLoaded(reason: "not enough memory to keep ready"))
        for id in [pinA, pinB, pinC] { await controller.setServed(id, false) }
        await controller.server.stop()
    }

    @Test func aDeclinedPreloadIsRetriedWhenARecomputeFindsRoom() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let p = preloader(6)                                                       // the model books 4 GB
        let busy = try await p.pool.lease(for: "someone-elses", bytes: 5 * Self.gb) { PinBox() }
        let controller = makeController(preloader: p)
        await controller.setServed(pinA, true)
        let declined = ModelLoadState.notLoaded(reason: ResidentPool<PinBox>.notEnoughMemoryReason)
        #expect(await until { controller.loadState(for: pinA) == declined })
        busy.release()
        await p.pool.evict(key: "someone-elses")                                   // room appears: the pool says so
        #expect(await until { controller.loadState(for: pinA) == .ready })
        #expect(await until { await p.pool.pinnedKeys == [p.key(pinA)] })          // and it is pinned
        await controller.setServed(pinA, false)
        await controller.server.stop()
    }

    @Test func aPinnedModelSurvivesAnotherServedModelsPreloadAndIsNotRetriedAsPushedOut() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let p = preloader(6)                                                       // room for one 4 GB model
        let controller = makeController(preloader: p)
        await controller.setServed(pinA, true)
        #expect(await until { await p.pool.pinnedKeys == [p.key(pinA)] })
        await controller.setServed(pinB, true)                                     // B's preload must not push out kept-ready A
        let declined = ModelLoadState.notLoaded(reason: ResidentPool<PinBox>.notEnoughMemoryReason)
        #expect(await until { controller.loadState(for: pinB) == declined })
        #expect(controller.loadState(for: pinA) == .ready)
        #expect(await p.pool.residentKeys == [p.key(pinA)])

        // Chat needs the room: A is pushed out, unpinned, with the reason; nothing reloads it behind Chat's back
        let chat = try await p.pool.lease(for: "chat-model", bytes: 4 * Self.gb, holder: .chat) { PinBox() }
        #expect(await until { controller.loadState(for: pinA) == .notLoaded(reason: "pushed out by Chat") })
        #expect(await p.pool.pinnedKeys.isEmpty)
        try? await Task.sleep(for: .milliseconds(200))
        #expect(controller.loadState(for: pinA) == .notLoaded(reason: "pushed out by Chat"))
        #expect(controller.servedIDs.contains(pinA))                               // still served: only the user changes that
        chat.release()
        await controller.setServed(pinA, false)
        await controller.setServed(pinB, false)
        await controller.server.stop()
    }
}
