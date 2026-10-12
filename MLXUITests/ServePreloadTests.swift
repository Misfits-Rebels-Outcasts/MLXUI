import Testing
import Foundation
@testable import MLXUI

/// SC-1 (gate B) — a served model loads when Serve is turned on: the pool's load state and unleased, never-waiting
/// `preload` (plain values in a `ResidentPool`, since a real `ModelContainer` can't be fabricated), and the
/// controller's wiring of it (a fake `ServePreloading` over the same pool).
struct ServePreloadPoolTests {

    private static let gb = Int64(1_073_741_824)
    private final class Box: Sendable { let name: String; init(_ n: String) { name = n } }
    private struct Boom: Error, LocalizedError { var errorDescription: String? { "boom" } }
    private actor Counter {
        private(set) var n = 0
        private(set) var cancelled = 0
        func bump() { n += 1 }
        func sawCancel() { cancelled += 1 }
    }

    private func makeBudget(_ gbs: Int64) -> MemoryBudget {
        MemoryBudget(capacityBytes: gbs * Self.gb,
                     cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
    }

    @Test func stateIsDerivedFromTheMaps() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        #expect(await pool.loadState(for: "a") == .notLoaded(reason: nil))
        _ = try await pool.value(for: "a", bytes: Self.gb) { Box("a") }
        #expect(await pool.loadState(for: "a") == .ready)
        await pool.evict(key: "a")
        #expect(await pool.loadState(for: "a") == .notLoaded(reason: nil))
    }

    @Test func stateIsLoadingWhileTheLoadRuns() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        async let loaded = pool.value(for: "a", bytes: Self.gb) {
            try await Task.sleep(for: .milliseconds(150)); return Box("a")
        }
        try await Task.sleep(for: .milliseconds(50))
        #expect(await pool.loadState(for: "a") == .loading)
        _ = try await loaded
        #expect(await pool.loadState(for: "a") == .ready)
    }

    @Test func preloadLoadsUnleased() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let state = await pool.preload(for: "a", bytes: 6 * Self.gb) { Box("a") }
        #expect(state == .ready)
        #expect(budget.inUseCount == 0)                                   // nothing holds it
        // evictable like any idle model: a bigger load pushes it out
        _ = try await pool.value(for: "b", bytes: 6 * Self.gb) { Box("b") }
        #expect(await pool.residentKeys == ["b"])
        #expect(await pool.loadState(for: "a") == .notLoaded(reason: nil))
    }

    @Test func preloadMayEvictAnIdleModel() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        _ = try await pool.value(for: "old", bytes: 6 * Self.gb) { Box("old") }
        let state = await pool.preload(for: "new", bytes: 6 * Self.gb) { Box("new") }
        #expect(state == .ready)
        #expect(await pool.residentKeys == ["new"])
    }

    @Test func preloadNeverPushesOutAModelInUseAndNeverWaits() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let busy = try await pool.lease(for: "busy", bytes: 6 * Self.gb) { Box("busy") }
        let loads = Counter()
        let start = ContinuousClock.now
        let state = await pool.preload(for: "new", bytes: 6 * Self.gb) { await loads.bump(); return Box("new") }
        #expect(ContinuousClock.now - start < .seconds(1))                // did not wait for the lease (60 s)
        #expect(state == .notLoaded(reason: ResidentPool<Box>.notEnoughMemoryReason))
        #expect(await loads.n == 0)
        #expect(await pool.residentKeys == ["busy"])                      // nothing in use was evicted
        #expect(budget.usedBytes == 6 * Self.gb)
        busy.release()
        // the first request then loads it as today
        _ = try await pool.value(for: "new", bytes: 6 * Self.gb) { Box("new") }
        #expect(await pool.loadState(for: "new") == .ready)
    }

    @Test func preloadDeclinesAModelLargerThanTheBudget() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(4))
        let state = await pool.preload(for: "huge", bytes: 8 * Self.gb) { Box("huge") }
        #expect(state == .notLoaded(reason: ResidentPool<Box>.notEnoughMemoryReason))
        #expect(await pool.count == 0)
    }

    @Test func aRequestDuringAPreloadSharesTheLoad() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        let loads = Counter()
        let load: @Sendable () async throws -> Box = {
            await loads.bump()
            try await Task.sleep(for: .milliseconds(200))
            return Box("a")
        }
        let preloading = Task { await pool.preload(for: "a", bytes: Self.gb, load: load) }
        try await Task.sleep(for: .milliseconds(50))
        let leased = try await pool.lease(for: "a", bytes: Self.gb, load: load)        // the request
        #expect(await preloading.value == .ready)
        #expect(await loads.n == 1)                                                    // one load, not two
        #expect(await pool.count == 1)
        leased.release()
    }

    @Test func aRequestThatJoinedAdoptsThePreloadSoCancellingItKeepsTheLoad() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        let load: @Sendable () async throws -> Box = {
            try await Task.sleep(for: .milliseconds(250)); return Box("a")
        }
        let preloading = Task { await pool.preload(for: "a", bytes: Self.gb, load: load) }
        try await Task.sleep(for: .milliseconds(50))
        let request = Task { try await pool.lease(for: "a", bytes: Self.gb, load: load) }
        try await Task.sleep(for: .milliseconds(50))
        preloading.cancel()                                                             // Serve off, mid-request
        let leased = try await request.value
        #expect(await pool.loadState(for: "a") == .ready)
        leased.release()
    }

    @Test func cancellingAPreloadStopsItsLoadAndKeepsNothing() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let loads = Counter()
        let preloading = Task {
            await pool.preload(for: "a", bytes: 4 * Self.gb) {
                await loads.bump()
                do { try await Task.sleep(for: .seconds(30)) } catch { await loads.sawCancel(); throw error }
                return Box("a")
            }
        }
        try await Task.sleep(for: .milliseconds(100))
        #expect(await pool.loadState(for: "a") == .loading)
        preloading.cancel()
        let state = await preloading.value
        #expect(state == .notLoaded(reason: nil))
        #expect(await loads.cancelled == 1)
        #expect(await pool.count == 0)
        #expect(budget.usedBytes == 0)                                                  // the booking is returned
    }

    @Test func aLoaderThatIgnoresCancellationStillLeavesNothingResident() async throws {
        let budget = makeBudget(10)
        let pool = ResidentPool<Box>(budget: budget)
        let preloading = Task {
            await pool.preload(for: "a", bytes: 4 * Self.gb) {
                // not cooperative: runs to the end whatever happens
                let end = ContinuousClock.now.advanced(by: .milliseconds(200))
                while ContinuousClock.now < end { try? await Task.sleep(for: .milliseconds(10)) }
                return Box("a")
            }
        }
        try await Task.sleep(for: .milliseconds(50))
        preloading.cancel()
        _ = await preloading.value
        #expect(await pool.count == 0)
        #expect(budget.usedBytes == 0)
    }

    @Test func aFailedLoadIsFailedWithItsMessageAndTheNextRequestRetries() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        let state = await pool.preload(for: "a", bytes: Self.gb) { throw Boom() }
        #expect(state == .failed("boom"))
        #expect(await pool.loadState(for: "a") == .failed("boom"))
        let ok = try await pool.value(for: "a", bytes: Self.gb) { Box("a") }             // today's behaviour: retry
        #expect(ok.name == "a")
        #expect(await pool.loadState(for: "a") == .ready)
    }

    @Test func aRequestsLoadFailureIsReportedToo() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        await #expect(throws: Boom.self) { _ = try await pool.value(for: "a", bytes: Self.gb) { throw Boom() } }
        #expect(await pool.loadState(for: "a") == .failed("boom"))
    }

    @Test func observersSeeEveryTransition() async throws {
        let pool = ResidentPool<Box>(budget: makeBudget(10))
        let seen = Seen()
        await pool.addObserver { key, state in seen.add(key, state) }
        _ = await pool.preload(for: "a", bytes: Self.gb) { Box("a") }
        await pool.evict(key: "a")
        #expect(seen.states == [.loading, .ready, .notLoaded(reason: nil)])
    }

    final class Seen: @unchecked Sendable {
        private let lock = NSLock()
        private var list: [ModelLoadState] = []
        func add(_ key: String, _ state: ModelLoadState) { lock.lock(); list.append(state); lock.unlock() }
        var states: [ModelLoadState] { lock.lock(); defer { lock.unlock() }; return list }
    }
}

// MARK: - Controller wiring

private let modelA = "mlx-community/Preload-A-4bit"
private let modelB = "mlx-community/Preload-B-4bit"

private final class FakeBox: Sendable {}

/// A `ServePreloading` over a `ResidentPool` of plain values — the real pool logic, a fake load.
private struct FakePreloader: ServePreloading {
    let pool: ResidentPool<FakeBox>
    let load: @Sendable () async throws -> FakeBox

    func preload(directory: URL, footprintBytes: Int64?) async -> ModelLoadState {
        await pool.preload(for: ModelContainerPool.key(for: directory), bytes: footprintBytes ?? 1, load: load)
    }
    func addLoadObserver(_ observer: @escaping @Sendable (String, ModelLoadState) -> Void) async {
        await pool.addObserver(observer)
    }
    func key(_ id: String) -> String { ModelContainerPool.key(for: ModelStore.shared.directory(forHFModelID: id)) }
}

private actor Calls {
    private(set) var loads = 0
    private(set) var cancelled = 0
    func load() { loads += 1 }
    func cancel() { cancelled += 1 }
}

@MainActor
struct ServePreloadControllerTests {
    private static let gb = Int64(1_073_741_824)

    private func budget(_ gbs: Int64) -> MemoryBudget {
        MemoryBudget(capacityBytes: gbs * Self.gb,
                     cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
    }

    private func installed(_ ids: [String]) -> InstalledModelIndex {
        InstalledModelIndex(entries: ids.map { .init(id: $0, hfModelId: $0, kind: .llm, ramGB: 4, downloadSizeGB: 2.67) })
    }

    private func makeController(served: Set<String> = [], preloader: (any ServePreloading)?) -> LocalServerController {
        let server = LocalServer(settings: ServeSettings(servedModelIDs: served), defaults: nil, listenPort: 0,
                                 environment: ServeEnvironment(servedModelIDs: { [] },
                                                               installed: { InstalledModelIndex(entries: []) },
                                                               appleFoundationReadiness: { nil }, created: 1))
        return LocalServerController(server: server,
                                     defaults: UserDefaults(suiteName: "preload-tests-\(UUID().uuidString)")!,
                                     installed: { self.installed([modelA, modelB]) },
                                     appleFoundation: { .ready }, catalogLoaded: { true },
                                     evict: { _ in }, evictPromptCache: { _ in },
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

    private func fake(_ gbs: Int64 = 20, calls: Calls, slow: Bool = false, fail: Bool = false) -> FakePreloader {
        FakePreloader(pool: ResidentPool<FakeBox>(budget: budget(gbs))) {
            await calls.load()
            if fail { throw NSError(domain: "t", code: 1, userInfo: [NSLocalizedDescriptionKey: "no weights"]) }
            if slow {
                do { try await Task.sleep(for: .seconds(30)) } catch { await calls.cancel(); throw error }
            }
            return FakeBox()
        }
    }

    @Test func serveOnPreloadsTheModel() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let calls = Calls()
        let preloader = fake(calls: calls)
        let controller = makeController(preloader: preloader)
        await controller.setServed(modelA, true)
        #expect(await until { controller.loadState(for: modelA) == .ready })
        #expect(await calls.loads == 1)
        #expect(await preloader.pool.residentKeys == [preloader.key(modelA)])
        await controller.setServed(modelA, false)
        await controller.server.stop()
    }

    @Test func autoStartPreloadsEveryReachableServedModel() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let calls = Calls()
        let controller = makeController(served: [modelA, modelB, ServedModels.appleFoundationID], preloader: fake(calls: calls))
        await controller.autoStart()
        #expect(await until { controller.loadState(for: modelA) == .ready && controller.loadState(for: modelB) == .ready })
        #expect(await calls.loads == 2)                                    // AFM needs nothing
        #expect(controller.loadState(for: ServedModels.appleFoundationID) == .ready)
        await controller.server.stop()
    }

    @Test func serveOffCancelsAPreloadInFlightAndKeepsNothing() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let calls = Calls()
        let preloader = fake(calls: calls, slow: true)
        let controller = makeController(preloader: preloader)
        await controller.setServed(modelA, true)
        #expect(await until { await calls.loads == 1 })
        #expect(await until { controller.loadState(for: modelA) == .loading })
        await controller.setServed(modelA, false)
        #expect(await until { await calls.cancelled == 1 })                // the load itself was cancelled
        #expect(await preloader.pool.count == 0)
        #expect(controller.loadState(for: modelA) == .notLoaded(reason: nil))
        await controller.server.stop()
    }

    @Test func masterSwitchOffCancelsPreloadsToo() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let calls = Calls()
        let controller = makeController(preloader: fake(calls: calls, slow: true))
        await controller.setServed(modelA, true)
        #expect(await until { await calls.loads == 1 })
        await controller.setServerEnabled(false)
        #expect(await until { await calls.cancelled == 1 })
    }

    @Test func aModelThatDoesNotFitStaysNotLoadedWithAReason() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let calls = Calls()
        let preloader = fake(6, calls: calls)                               // a 6 GB Mac budget; the model books 4 GB
        let busy = try await preloader.pool.lease(for: "someone-elses", bytes: 5 * Self.gb) { FakeBox() }
        let controller = makeController(preloader: preloader)
        await controller.setServed(modelA, true)
        #expect(await until {
            controller.loadState(for: modelA) == .notLoaded(reason: ResidentPool<FakeBox>.notEnoughMemoryReason)
        })
        #expect(await calls.loads == 0)
        #expect(await preloader.pool.residentKeys == ["someone-elses"])     // the model in use stayed
        busy.release()
        await controller.setServed(modelA, false)
        await controller.server.stop()
    }

    @Test func gateClosedPreloadsNothing() async {
        LocalServerGate.overrideForTesting = false
        defer { LocalServerGate.overrideForTesting = nil }
        let calls = Calls()
        let controller = makeController(served: [modelA], preloader: fake(calls: calls))
        await controller.autoStart()
        await controller.setServed(modelB, true)
        try? await Task.sleep(for: .milliseconds(200))
        #expect(await calls.loads == 0)
        #expect(controller.loadState(for: modelA) == .notLoaded(reason: nil))
    }

    @Test func aFailedPreloadLeavesTheModelServedWithTheErrorAndTheNextRequestRetries() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let calls = Calls()
        let preloader = fake(calls: calls, fail: true)
        let controller = makeController(preloader: preloader)
        await controller.setServed(modelA, true)
        #expect(await until { controller.loadState(for: modelA) == .failed("no weights") })
        #expect(controller.isServing(modelA))                               // still served
        // the next request loads again (today's path)
        _ = try? await preloader.pool.value(for: preloader.key(modelA), bytes: Self.gb) { FakeBox() }
        #expect(await preloader.pool.loadState(for: preloader.key(modelA)) == .ready)
        #expect(await until { controller.loadState(for: modelA) == .ready })
        await controller.setServed(modelA, false)
        await controller.server.stop()
    }

    @Test func noPreloaderMeansNoPreload() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let controller = makeController(preloader: nil)
        await controller.setServed(modelA, true)
        #expect(controller.loadState(for: modelA) == .notLoaded(reason: nil))
        await controller.setServed(modelA, false)
        await controller.server.stop()
    }
}
