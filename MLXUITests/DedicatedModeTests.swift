import Testing
import Foundation
@testable import MLXUI

// SC-6 / SC-7 (gates F, G, H — implementer's call, pending owner confirmation): the runtime-adjustable budget,
// Dedicated server mode's transitions and persistence, and the one admission check every model load asks.

private let gb = Int64(1_073_741_824)
private let dmModel = "mlx-community/Ded-A-4bit"
private final class DedBox: Sendable {}

private func suite() -> UserDefaults { UserDefaults(suiteName: "dedicated-tests-\(UUID().uuidString)")! }

private struct DedPreloader: ServePreloading {
    let pool: ResidentPool<DedBox>
    func preload(directory: URL, footprintBytes: Int64?) async -> ModelLoadState {
        await pool.preload(for: ModelContainerPool.key(for: directory), bytes: footprintBytes ?? 1) { DedBox() }
    }
    func addLoadObserver(_ observer: @escaping @Sendable (String, ModelLoadState) -> Void) async { await pool.addObserver(observer) }
    func residentBytes(directory: URL) async -> Int64? { await pool.residentBytes(for: ModelContainerPool.key(for: directory)) }
    func setPinned(directories: [URL]) async { await pool.setPinned(Set(directories.map { ModelContainerPool.key(for: $0) })) }
}

private nonisolated final class DedStage: PipelineStage, @unchecked Sendable {
    let id = "ded-stage"
    var name: String { id }
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media { input }
}

@Suite(.serialized)
@MainActor
struct DedicatedModeTests {
    private func open(_ defaults: UserDefaults) {
        DedicatedMode.store = defaults
        LocalServerGate.overrideForTesting = true
    }
    private func reset() {
        DedicatedMode.store = .standard
        LocalServerGate.overrideForTesting = nil
    }

    private func controller(_ defaults: UserDefaults, budget: MemoryBudget, served: Set<String> = [dmModel],
                            preloader: (any ServePreloading)? = nil) -> LocalServerController {
        let server = LocalServer(settings: ServeSettings(servedModelIDs: served), defaults: nil, listenPort: 0,
                                 environment: ServeEnvironment(servedModelIDs: { [] },
                                                               installed: { InstalledModelIndex(entries: []) },
                                                               appleFoundationReadiness: { nil }, created: 1))
        return LocalServerController(
            server: server, defaults: defaults,
            installed: { InstalledModelIndex(entries: [.init(id: dmModel, hfModelId: dmModel, kind: .llm, ramGB: 4, downloadSizeGB: 2)]) },
            appleFoundation: { .ready }, catalogLoaded: { true }, evict: { _ in }, evictPromptCache: { _ in },
            budget: budget, totalRAMGB: 16, preloadPool: preloader)
    }

    // MARK: The budget

    @Test func raisingTheCapacityMakesRoomAndLoweringEvictsOnlyWhenAsked() async throws {
        let budget = MemoryBudget(capacityBytes: 10 * gb)
        let pool = ResidentPool<DedBox>(budget: budget)
        _ = try await pool.value(for: "a", bytes: 4 * gb) { DedBox() }
        _ = try await pool.value(for: "b", bytes: 4 * gb) { DedBox() }
        #expect(budget.tryReserve(bytes: 3 * gb) == nil)
        #expect(budget.setCapacity(12 * gb))
        #expect(budget.capacityBytes == 12 * gb)
        #expect(budget.tryReserve(bytes: 3 * gb) != nil)
        #expect(budget.baseCapacityBytes == 10 * gb)
        #expect(!budget.setCapacity(12 * gb))                                // unchanged
        #expect(!budget.setCapacity(0))                                      // never zero

        #expect(budget.setCapacity(6 * gb))
        #expect(budget.usedBytes > 6 * gb)                                   // lowering alone evicts nothing
        await budget.enforceCapacity()
        #expect(budget.usedBytes <= 6 * gb)
    }

    @Test func enforcingTheCapacityKeepsAPinnedEntry() async throws {
        let budget = MemoryBudget(capacityBytes: 12 * gb)
        let pool = ResidentPool<DedBox>(budget: budget)
        _ = try await pool.value(for: "kept", bytes: 5 * gb) { DedBox() }
        _ = try await pool.value(for: "other", bytes: 5 * gb) { DedBox() }
        await pool.setPinned(["kept"])
        budget.setCapacity(7 * gb)
        await budget.enforceCapacity()
        #expect(await pool.residentKeys == ["kept"])
        #expect(budget.usedBytes == 5 * gb)
    }

    @Test func theDedicatedCapacityIsThreeQuartersOfRAMAndNeverBelowTheBase() {
        #expect(MemoryBudget.dedicatedCapacity(totalRAMGB: 16, base: 9 * gb) == 12 * gb)
        #expect(MemoryBudget.dedicatedCapacity(totalRAMGB: 16, base: 14 * gb) == 14 * gb)
        #expect(MemoryBudget.dedicatedCapacity(totalRAMGB: 2, base: 4 * gb) == 4 * gb)
    }

    // MARK: The admission check (SC-7)

    @Test func nothingIsRefusedByDefaultOrWithTheGateClosed() {
        let defaults = suite()
        open(defaults)
        defer { reset() }
        for surface in [RunAdmission.Surface.chat, .runWindow, .flowRow] {
            #expect(RunAdmission.canLoadModel(surface: surface) == nil)
        }
        DedicatedMode.set(true, in: defaults)
        LocalServerGate.overrideForTesting = false                           // the build hides serving
        #expect(!DedicatedMode.isActive)
        #expect(RunAdmission.canLoadModel(surface: .chat) == nil)
    }

    @Test func everySurfaceIsRefusedWithTheSameSentenceWhileDedicated() {
        let defaults = suite()
        open(defaults)
        defer { reset() }
        DedicatedMode.set(true, in: defaults)
        for surface in [RunAdmission.Surface.chat, .runWindow, .flowRow] {
            #expect(RunAdmission.canLoadModel(surface: surface)?.sentence == RunAdmission.sentence)
        }
        #expect(RunAdmission.sentence == "This Mac is in Dedicated server mode. Turn it off on the Local Server screen to run models here.")
    }

    @Test func aFlowRowFailsWithTheSentenceBeforeAskingAnythingElse() async {
        let defaults = suite()
        open(defaults)
        defer { reset() }
        DedicatedMode.set(true, in: defaults)
        let gate = FlowServedLoadGate(flowID: "f", bytesNeeded: { _ in 0 }, check: { _ in [] },
                                      pushOut: { _ in }, reloadSeconds: { _ in nil })
        let model = makeEntry(id: "row-model", displayName: "Row Model")
        do {
            try await gate(path: "1", model: model)
            Issue.record("expected a refusal")
        } catch let FlowError.stageFailure(row, message) {
            #expect(row == "1")
            #expect(message == RunAdmission.sentence)
        } catch { Issue.record("unexpected \(error)") }
        DedicatedMode.set(false, in: defaults)
        await #expect(throws: Never.self) { try await gate(path: "1", model: model) }
    }

    @Test func aRunWindowDoesNotRunWhileDedicatedAndRunsAgainAfterwards() async throws {
        let defaults = suite()
        open(defaults)
        defer { reset() }
        let budget = MemoryBudget(capacityBytes: 10 * gb)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)
        let model = makeEntry(id: "ded-asr", modelType: .asr, ramGB: 2)
        let stage = LeasingStage(model: model, config: .default, probe: DedStage(), cache: cache)
        DedicatedMode.set(true, in: defaults)
        do {
            _ = try await stage.run(.text("x")) { _ in }
            Issue.record("expected a refusal")
        } catch let refusal as RunAdmission.Refusal {
            #expect(refusal.sentence == RunAdmission.sentence)
        }
        #expect(budget.usedBytes == 0)                                   // nothing was loaded
        DedicatedMode.set(false, in: defaults)
        _ = try await stage.run(.text("x")) { _ in }
    }

    // MARK: The mode

    @Test func turningItOnRaisesTheBudgetPersistsAndStartsPreloading() async throws {
        let defaults = suite()
        open(defaults)
        defer { reset() }
        let budget = MemoryBudget(capacityBytes: 9 * gb)
        let pool = ResidentPool<DedBox>(budget: budget)
        let c = controller(defaults, budget: budget, preloader: DedPreloader(pool: pool))
        #expect(!c.dedicatedMode)
        await c.setDedicatedMode(true)
        #expect(c.dedicatedMode && c.dedicatedActive)
        #expect(DedicatedMode.isOn(in: defaults))
        #expect(budget.capacityBytes == 12 * gb)
        #expect(c.budgetBytes == 12 * gb)
        #expect(c.enabled)
        // a controller built on the same defaults comes back in the mode
        #expect(controller(defaults, budget: budget).dedicatedMode)
    }

    @Test func turningItOffRestoresTheBaseAndEvictsUnpinnedWeights() async throws {
        let defaults = suite()
        open(defaults)
        defer { reset() }
        let budget = MemoryBudget(capacityBytes: 9 * gb)
        let pool = ResidentPool<DedBox>(budget: budget)
        let c = controller(defaults, budget: budget, preloader: DedPreloader(pool: pool))
        await c.setDedicatedMode(true)
        _ = try await pool.value(for: "chat", bytes: 5 * gb) { DedBox() }
        _ = try await pool.value(for: "other", bytes: 5 * gb) { DedBox() }       // 10 GB: fits 12, not 9
        await c.setDedicatedMode(false)
        #expect(!c.dedicatedMode)
        #expect(!DedicatedMode.isOn(in: defaults))
        #expect(budget.capacityBytes == 9 * gb)
        #expect(c.budgetBytes == 9 * gb)
        #expect(budget.usedBytes <= 9 * gb)
        #expect(c.servedIDs == [dmModel])                                          // the served list is kept
    }

    @Test func itCannotBeTurnedOnWithNothingToServeAndEndsWhenTheLastModelIsUnserved() async {
        let defaults = suite()
        open(defaults)
        defer { reset() }
        let budget = MemoryBudget(capacityBytes: 9 * gb)
        let empty = controller(defaults, budget: budget, served: [])
        await empty.setDedicatedMode(true)
        #expect(!empty.dedicatedMode)
        #expect(budget.capacityBytes == 9 * gb)

        let c = controller(defaults, budget: budget)
        await c.setDedicatedMode(true)
        #expect(c.dedicatedMode)
        await c.setServed(dmModel, false)
        #expect(!c.dedicatedMode)
        #expect(budget.capacityBytes == 9 * gb)
    }

    @Test func switchingTheServerOffEndsTheMode() async {
        let defaults = suite()
        open(defaults)
        defer { reset() }
        let budget = MemoryBudget(capacityBytes: 9 * gb)
        let c = controller(defaults, budget: budget)
        await c.setDedicatedMode(true)
        await c.setServerEnabled(false)
        #expect(!c.dedicatedMode)
        #expect(budget.capacityBytes == 9 * gb)
    }

    @Test func withTheGateClosedNothingChanges() async {
        let defaults = suite()
        open(defaults)
        defer { reset() }
        LocalServerGate.overrideForTesting = false
        let budget = MemoryBudget(capacityBytes: 9 * gb)
        let c = controller(defaults, budget: budget)
        await c.setDedicatedMode(true)
        #expect(!c.dedicatedMode)
        #expect(!DedicatedMode.isOn(in: defaults))
        #expect(budget.capacityBytes == 9 * gb)
    }

    @Test func aLaunchInTheModeComesBackWithTheRaisedBudget() async {
        let defaults = suite()
        open(defaults)
        defer { reset() }
        DedicatedMode.set(true, in: defaults)
        let budget = MemoryBudget(capacityBytes: 9 * gb)
        let c = controller(defaults, budget: budget)
        await c.autoStart()
        #expect(budget.capacityBytes == 12 * gb)
        await c.server.stop()
    }

    @Test func theScreenModelCarriesTheSwitch() {
        #expect(ServerScreenModel.dedicatedCaption(on: true, capacityBytes: 12 * gb).contains("12.0 GB"))
        #expect(!ServerScreenModel.dedicatedCaption(on: false, capacityBytes: 12 * gb).contains("12.0"))
    }
}
