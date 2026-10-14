import Testing
import Foundation
@testable import MLXUI

// SC-4b — the Local Server screen: its view model is derived from the controller (rows, state lines, memory, stats,
// disk), lists only servable models, and the screen's visibility follows `isEnabled`.

private let srvA = "mlx-community/Srv-A-4bit"
private let srvB = "mlx-community/Srv-B-4bit"
private let gib = Int64(1_073_741_824)
private final class SrvBox: Sendable {}

private struct SrvPreloader: ServePreloading {
    let pool: ResidentPool<SrvBox>
    func preload(directory: URL, footprintBytes: Int64?) async -> ModelLoadState {
        await pool.preload(for: ModelContainerPool.key(for: directory), bytes: footprintBytes ?? 1) { SrvBox() }
    }
    func addLoadObserver(_ observer: @escaping @Sendable (String, ModelLoadState) -> Void) async { await pool.addObserver(observer) }
    func residentBytes(directory: URL) async -> Int64? { await pool.residentBytes(for: ModelContainerPool.key(for: directory)) }
    func setPinned(directories: [URL]) async { await pool.setPinned(Set(directories.map { ModelContainerPool.key(for: $0) })) }
}

@Suite(.serialized)
@MainActor
struct ServerScreenTests {
    private func entries(_ items: [(String, RunnerKind, Double)]) -> InstalledModelIndex {
        InstalledModelIndex(entries: items.map { .init(id: $0.0, hfModelId: $0.0, kind: $0.1, ramGB: $0.2, downloadSizeGB: $0.2 / 1.5) })
    }

    private func controller(served: Set<String> = [], installed: InstalledModelIndex? = nil, afm: Readiness? = nil,
                            preloader: (any ServePreloading)? = nil,
                            ledger: @escaping @Sendable () async -> MemoryLedgerSample = { .init(used: 0, poolWeights: 0, snapshots: 0) })
        -> LocalServerController {
        let server = LocalServer(settings: ServeSettings(servedModelIDs: served), defaults: nil, listenPort: 0,
                                 environment: ServeEnvironment(servedModelIDs: { [] }, installed: { InstalledModelIndex(entries: []) },
                                                               appleFoundationReadiness: { nil }, created: 1))
        let index = installed ?? entries([(srvA, .llm, 6), (srvB, .llm, 3)])
        return LocalServerController(server: server, defaults: UserDefaults(suiteName: "screen-\(UUID().uuidString)")!,
                                     installed: { index }, appleFoundation: { afm }, catalogLoaded: { true },
                                     evict: { _ in }, evictPromptCache: { _ in }, budgetBytes: 20 * gib,
                                     preloadPool: preloader, ledger: ledger)
    }

    private func model(_ c: LocalServerController, memory: ServeMemoryReading? = nil, free: Int64? = 100 * gib,
                       context: @escaping (String) -> Int? = { _ in 32768 }) -> ServerScreenModel {
        c.screenModel(memory: memory, freeDiskBytes: free, contextLength: context)
    }

    // MARK: Sidebar

    @Test func theSidebarItemFollowsIsEnabled() {
        defer { LocalServerGate.overrideForTesting = nil; LocalServerGate.optionOverrideForTesting = nil }
        LocalServerGate.overrideForTesting = true
        LocalServerGate.optionOverrideForTesting = false
        let c = controller()
        #expect(!c.surfacesVisible)
        #expect(LocalServerPolicy.resolvedSidebarItem(.server, serving: c.surfacesVisible) == .home)
        LocalServerGate.optionOverrideForTesting = true
        #expect(c.surfacesVisible)
        #expect(LocalServerPolicy.resolvedSidebarItem(.server, serving: c.surfacesVisible) == .server)
        LocalServerGate.overrideForTesting = false                      // hideLocalServer closes it whatever the option says
        #expect(!c.surfacesVisible)
        #expect(LocalServerPolicy.resolvedSidebarItem(.browse("x"), serving: false) == .browse("x"))
    }

    // MARK: Rows

    @Test func onlyServableModelsAreListed() {
        let c = controller(installed: entries([(srvA, .llm, 6), ("mlx-community/whisper", .asr, 2), ("mlx-community/vl", .vision, 5),
                                                ("mlx-community/emb", .embedding, 1), ("mlx-community/flux", .image, 12), (srvB, .llm, 3)]))
        #expect(model(c).rows.map(\.id) == [srvA, srvB])
    }

    @Test func aSharedRepoIsListedOnce() {
        // two catalog cards, one download (e.g. "Qwen3.5-9B" and "Qwen3.5-9B Vision")
        let index = InstalledModelIndex(entries: [
            .init(id: "Srv-A", hfModelId: srvA, kind: .llm, ramGB: 6, downloadSizeGB: 4),
            .init(id: "Srv-A Vision", hfModelId: srvA, kind: .llm, ramGB: 6, downloadSizeGB: 4)])
        #expect(model(controller(installed: index)).rows.count == 1)
    }

    @Test func appleFoundationModelsIsListedWhenTheSystemHasIt() {
        #expect(model(controller(afm: nil)).rows.contains { $0.isAppleFoundation } == false)
        let c = controller(served: [ServedModels.appleFoundationID], afm: .ready)
        let row = model(c).rows.last
        #expect(row?.id == ServedModels.appleFoundationID && row?.stateLine == "Ready")
        let notReady = controller(served: [ServedModels.appleFoundationID], afm: .unavailable(reason: "not supported"))
        #expect(model(notReady).rows.last?.stateLine == ReadinessSentence.text(for: .unavailable(reason: "not supported")))
    }

    @Test func rowsCarryPlainFacts() {
        let c = controller(served: [srvA])
        let row = model(c, context: { $0 == srvA ? 40960 : nil }).rows.first { $0.id == srvA }
        #expect(row?.name == "Srv-A-4bit")
        #expect(row?.isServed == true)
        #expect(row?.sizeText == "4.0 GB")
        #expect(row?.contextText?.contains("40,960") == true || row?.contextText?.contains("40.960") == true || row?.contextText?.contains("40960") == true)
        #expect(model(c, context: { _ in nil }).rows.first?.contextText == nil)
        #expect(model(c).rows.first { $0.id == srvB }?.stateLine == nil)            // not served: no state line
    }

    // MARK: State lines (§UI states)

    @Test func stateLinesFollowTheFourStates() {
        func line(_ s: ModelLoadState, served: Bool = true, on: Bool = true) -> String? {
            ServerScreenModel.stateLine(served: served, serverOn: on, state: s)
        }
        #expect(line(.ready) == "Ready")
        #expect(line(.loading) == "Loading…")
        #expect(line(.notLoaded(reason: nil)) == "Not loaded · loads on next request")
        #expect(line(.notLoaded(reason: "pushed out by Chat")) == "Not loaded · loads on next request · pushed out by Chat")
        #expect(line(.notLoaded(reason: LocalServerController.notKeptReadyReason))?.hasSuffix("not enough memory to keep ready") == true)
        #expect(line(.ready, served: false) == nil)
        #expect(line(.ready, on: false) == "Server is off")
        #expect(line(.failed("/Users/x/secret path"))?.contains("secret") == false)
    }

    // MARK: Memory (ledger bytes, not catalog estimates)

    @Test func theMemoryBarSplitsTheLedgersAndNeverUsesCatalogEstimates() async {
        // the catalog says 6 GB (ramGB); the pool booked 5 GiB for the served model — the bar must say 5.
        let pool = SrvPreloader(pool: ResidentPool<SrvBox>(budget: MemoryBudget(
            capacityBytes: 20 * gib, cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))))
        let url = ModelStore.shared.directory(forHFModelID: srvA)
        #expect(await pool.preload(directory: url, footprintBytes: 5 * gib) == .ready)
        let c = controller(served: [srvA], preloader: pool, ledger: {
            .init(used: 12 * gib, poolWeights: 8 * gib, snapshots: 1 * gib) })
        // make the controller see srvA as ready (it mirrors pool transitions)
        await c.setServed(srvA, true)
        try? await Task.sleep(for: .milliseconds(200))
        defer { Task { await c.server.stop() } }
        let m = await c.memoryReading()
        #expect(m.capacity == 20 * gib)
        #expect(m.servedKeptReady == 5 * gib)
        #expect(m.otherModels == 3 * gib)
        #expect(m.snapshots == 1 * gib)
        #expect(m.engines == 3 * gib)                                     // 12 − 8 − 1
        #expect(m.used == 12 * gib)
    }

    // SC-4e: the running switch going off unpins everything (`applyPins()`), but the weights stay resident and
    // `loadState` keeps reporting `.ready` — so kept-ready must follow the pin, not the load state, or the Memory
    // card still calls stopped-serving weights "kept ready".
    @Test func keptReadyFollowsTheRunningSwitchNotJustLoadState() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let pool = SrvPreloader(pool: ResidentPool<SrvBox>(budget: MemoryBudget(
            capacityBytes: 20 * gib, cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))))
        let url = ModelStore.shared.directory(forHFModelID: srvA)
        #expect(await pool.preload(directory: url, footprintBytes: 5 * gib) == .ready)
        let c = controller(served: [srvA], preloader: pool, ledger: {
            .init(used: 12 * gib, poolWeights: 8 * gib, snapshots: 1 * gib) })
        await c.setServed(srvA, true)
        try? await Task.sleep(for: .milliseconds(200))
        defer { Task { await c.server.stop() } }

        let on = await c.memoryReading()
        #expect(on.servedKeptReady == 5 * gib)
        #expect(on.otherModels == 3 * gib)

        await c.setServerEnabled(false)
        let off = await c.memoryReading()
        #expect(off.servedKeptReady == 0)
        #expect(off.otherModels == 8 * gib)                               // the unpinned weights, not lost

        await c.setServerEnabled(true)
        try? await Task.sleep(for: .milliseconds(200))
        let backOn = await c.memoryReading()
        #expect(backOn.servedKeptReady == 5 * gib)
        #expect(backOn.otherModels == 3 * gib)
    }

    // MARK: Stats

    private func record(_ c: LocalServerController, model: String, ttft: Double?, tps: Double?, prompt: Int?, reused: Int?, status: Int = 200) {
        c.server.requestLog.record(RequestLogEntry(
            time: Date(timeIntervalSince1970: 1_000), userAgent: nil, model: model, promptTokens: prompt, completionTokens: nil,
            durationSeconds: 1, status: status, toolsIgnored: false, ttftSeconds: ttft, decodeTokensPerSecond: tps,
            promptTokensReused: reused))
    }

    @Test func statsShowADashForNoFigureAndZeroForAMiss() {
        let c = controller(served: [srvA, srvB])
        record(c, model: srvA, ttft: 1.25, tps: 30, prompt: 100, reused: 0)          // cache applied, nothing reused: a miss
        record(c, model: srvB, ttft: nil, tps: nil, prompt: nil, reused: nil, status: 503)
        let stats = model(c).stats
        let a = stats.first { $0.id == srvA }, b = stats.first { $0.id == srvB }
        #expect(a?.reused == "0%")
        #expect(a?.ttft == "1.2 s" || a?.ttft == "1.3 s")
        #expect(a?.decode == "30.0 tok/s")
        #expect(b?.reused == "—" && b?.ttft == "—" && b?.decode == "—")
        #expect(b?.unavailable == 1 && b?.requests == 1)
        #expect(a?.lastRequest == Date(timeIntervalSince1970: 1_000))
        record(c, model: srvA, ttft: 1, tps: 30, prompt: 100, reused: 50)
        #expect(model(c).stats.first { $0.id == srvA }?.reused == "25%")
    }

    @Test func aModelWithNoRequestsHasNoStatsRow() {
        #expect(model(controller(served: [srvA])).stats.isEmpty)
    }

    // MARK: Disk (R47)

    @Test func theLowDiskLineAppearsBelowFifteenGB() {
        #expect(model(controller(), free: 14 * gib).lowDiskWarning == "Low disk space — macOS may stop MLXUI while it's serving. Free up space.")
        #expect(model(controller(), free: 15 * gib).lowDiskWarning == nil)
        #expect(model(controller(), free: 200 * gib).lowDiskWarning == nil)
        #expect(model(controller(), free: nil).lowDiskWarning == nil)
        #expect(model(controller(), free: 200 * gib).diskFact.hasPrefix("Free disk space: "))
    }

    // MARK: Status

    @Test func statusAndSwitchComeFromTheController() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let c = controller(served: [srvA])
        #expect(model(c).runState == .stopped && model(c).address == nil)
        await c.autoStart()
        defer { Task { await c.server.stop() } }
        let m = model(c)
        #expect(m.runState == .running && m.address?.hasPrefix("127.0.0.1:") == true && m.switchOn)
        await c.setServerEnabled(false)
        #expect(model(c).switchOn == false)
    }

    // MARK: The listener

    @Test func startRefusesWithTheOptionOff() async {
        LocalServerGate.overrideForTesting = true
        LocalServerGate.optionOverrideForTesting = false
        defer { LocalServerGate.overrideForTesting = nil; LocalServerGate.optionOverrideForTesting = nil }
        let server = LocalServer(settings: ServeSettings(servedModelIDs: [srvA]), defaults: nil, listenPort: 0,
                                 environment: ServeEnvironment(servedModelIDs: { [srvA] }, installed: { InstalledModelIndex(entries: []) },
                                                               appleFoundationReadiness: { nil }, created: 1))
        await server.start()
        #expect(server.status == .stopped)
        LocalServerGate.optionOverrideForTesting = true
        await server.start()
        defer { Task { await server.stop() } }
        if case .running = server.status {} else { Issue.record("expected running, got \(server.status)") }
    }
}
