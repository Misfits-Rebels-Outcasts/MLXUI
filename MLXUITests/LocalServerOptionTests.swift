import Testing
import Foundation
@testable import MLXUI

// SC-4a (gate J) — the "Enable Local Server" option: default, migration, `LocalServerGate.isEnabled`, and what the
// controller does when it flips. The stored path is exercised for real: an isolated `UserDefaults` suite stands in for
// `.standard` (`LocalServerOption.store`), and `usesStoredOptionInTests` stops the gate treating `overrideForTesting
// = true` as "option on" (which is what every pre-SC-4a test relies on).

private let optModel = "mlx-community/Opt-A-4bit"
private let optOther = "mlx-community/Opt-B-4bit"
private let gb = Int64(1_073_741_824)
private final class OptBox: Sendable {}

private func suite() -> UserDefaults { UserDefaults(suiteName: "option-tests-\(UUID().uuidString)")! }

private struct OptPreloader: ServePreloading {
    let pool: ResidentPool<OptBox>
    func preload(directory: URL, footprintBytes: Int64?) async -> ModelLoadState {
        await pool.preload(for: ModelContainerPool.key(for: directory), bytes: footprintBytes ?? 1) { OptBox() }
    }
    func addLoadObserver(_ observer: @escaping @Sendable (String, ModelLoadState) -> Void) async { await pool.addObserver(observer) }
    func residentBytes(directory: URL) async -> Int64? { await pool.residentBytes(for: ModelContainerPool.key(for: directory)) }
    func setPinned(directories: [URL]) async { await pool.setPinned(Set(directories.map { ModelContainerPool.key(for: $0) })) }
}

/// A suite that stores like a 0.20/0.21 profile would, then clears the gate seams afterwards.
@Suite(.serialized)
@MainActor
struct LocalServerOptionTests {
    private func useStoredOption(_ defaults: UserDefaults, available: Bool = true) {
        LocalServerOption.store = defaults
        LocalServerGate.overrideForTesting = available
        LocalServerGate.usesStoredOptionInTests = true
        LocalServerGate.optionOverrideForTesting = nil
    }
    private func reset() {
        LocalServerOption.store = .standard
        LocalServerGate.overrideForTesting = nil
        LocalServerGate.usesStoredOptionInTests = false
        LocalServerGate.optionOverrideForTesting = nil
    }

    private func controller(_ defaults: UserDefaults, served: Set<String> = [optModel],
                            preloader: (any ServePreloading)? = nil) -> LocalServerController {
        let server = LocalServer(settings: ServeSettings(servedModelIDs: served), defaults: nil, listenPort: 0,
                                 environment: ServeEnvironment(servedModelIDs: { [] },
                                                               installed: { InstalledModelIndex(entries: []) },
                                                               appleFoundationReadiness: { nil }, created: 1))
        return LocalServerController(
            server: server, defaults: defaults,
            installed: { InstalledModelIndex(entries: [optModel, optOther].map {
                .init(id: $0, hfModelId: $0, kind: .llm, ramGB: 4, downloadSizeGB: 2) }) },
            appleFoundation: { .ready }, catalogLoaded: { true }, evict: { _ in }, evictPromptCache: { _ in },
            budgetBytes: 20 * gb, preloadPool: preloader)
    }

    private func until(_ condition: () async -> Bool) async -> Bool {
        let end = ContinuousClock.now.advanced(by: .seconds(5))
        while ContinuousClock.now < end {
            if await condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return await condition()
    }

    // MARK: Default and migration

    @Test func aFreshInstallStartsWithTheOptionOff() {
        let defaults = suite()
        #expect(!LocalServerOption.isOn(in: defaults))
        useStoredOption(defaults)
        defer { reset() }
        #expect(LocalServerGate.isAvailable)
        #expect(!LocalServerGate.isEnabled)
        #expect(!controller(defaults, served: []).optionEnabled)
    }

    @Test func aNonEmptyServedListTurnsTheOptionOn() throws {
        let defaults = suite()
        ServeSettings().serving(optModel, true).save(to: defaults)
        #expect(LocalServerOption.isOn(in: defaults))
    }

    @Test func aMasterSwitchStoredAsOnTurnsTheOptionOn() {
        let defaults = suite()
        defaults.set(true, forKey: "localServerEnabled")
        #expect(LocalServerOption.isOn(in: defaults))
    }

    @Test func aMasterSwitchStoredAsOffDoesNot() {
        let defaults = suite()
        defaults.set(false, forKey: "localServerEnabled")
        #expect(!LocalServerOption.isOn(in: defaults))
    }

    @Test func theConnectSheetEverShownTurnsTheOptionOn() {
        let defaults = suite()
        defaults.set(true, forKey: "localServerConnectShown")
        #expect(LocalServerOption.isOn(in: defaults))
    }

    @Test func theMigrationRunsOnceAndIsRecorded() {
        let defaults = suite()
        #expect(!LocalServerOption.isOn(in: defaults))
        #expect(defaults.bool(forKey: LocalServerOption.migratedKey))
        // serving history appearing later does not flip it: the migration already ran
        ServeSettings().serving(optModel, true).save(to: defaults)
        defaults.set(true, forKey: "localServerConnectShown")
        #expect(!LocalServerOption.isOn(in: defaults))

        // and a user who turned it off keeps it off across a relaunch even with a served list
        let used = suite()
        ServeSettings().serving(optModel, true).save(to: used)
        #expect(LocalServerOption.isOn(in: used))
        LocalServerOption.set(false, in: used)
        #expect(!LocalServerOption.isOn(in: used))
    }

    @Test func theControllerRunsTheMigrationBeforeAnyoneReadsTheOption() {
        let defaults = suite()
        defaults.set(true, forKey: "localServerConnectShown")
        #expect(controller(defaults, served: []).optionEnabled)
    }

    // MARK: Gate

    @Test func isEnabledNeedsAvailableAndTheOption() {
        let defaults = suite()
        useStoredOption(defaults)
        defer { reset() }
        #expect(!LocalServerGate.isEnabled)
        LocalServerOption.set(true, in: defaults)
        #expect(LocalServerGate.isEnabled)
        LocalServerGate.overrideForTesting = false                     // hideLocalServer: closed beats the option
        #expect(!LocalServerGate.isAvailable)
        #expect(!LocalServerGate.isEnabled)
    }

    @Test func everyServingSurfaceFollowsIsEnabledNotIsAvailable() async {
        let defaults = suite()
        useStoredOption(defaults)                                      // available, option off
        defer { reset() }
        let c = controller(defaults)
        defer { Task { await c.server.stop() } }

        #expect(LocalServerGate.isAvailable)
        #expect(!c.surfacesVisible)
        #expect(!ServeToggleVisibility.shouldShow(state: .installed, isInstalledOnDisk: true, runnerKind: .llm))
        #expect(!ConnectSheetPresentation.isPresented(requested: true))
        #expect(LocalServerPolicy.shouldTerminateAfterLastWindowClosed(status: .running(port: 1212)))
        #expect(LocalServerPolicy.visibleSettingsPanes(gateAvailable: LocalServerGate.isAvailable,
                                                       hideProvidersPrivacy: false).contains(.localServer))   // the option's home

        await c.setServed(optModel, true)                              // the Serve toggle's action
        c.presentConnect(for: optModel)
        await c.setServerEnabled(true)
        await c.setPort(4321)
        await c.autoStart()
        #expect(c.server.status == .stopped)
        #expect(!c.showConnect)
        #expect(c.server.settings.port == ServeSettings.defaultPort)

        LocalServerOption.set(true, in: defaults)
        #expect(ServeToggleVisibility.shouldShow(state: .installed, isInstalledOnDisk: true, runnerKind: .llm))
        #expect(ConnectSheetPresentation.isPresented(requested: true))
        #expect(!LocalServerPolicy.shouldTerminateAfterLastWindowClosed(status: .running(port: 1212)))
    }

    @Test func aClosedGateStillHidesEverythingIncludingTheOption() {
        let defaults = suite()
        LocalServerOption.set(true, in: defaults)
        useStoredOption(defaults, available: false)
        defer { reset() }
        #expect(!LocalServerGate.isEnabled)
        #expect(!controller(defaults).surfacesVisible)
        #expect(!LocalServerPolicy.visibleSettingsPanes(gateAvailable: LocalServerGate.isAvailable,
                                                        hideProvidersPrivacy: false).contains(.localServer))
        #expect(!ServeToggleVisibility.shouldShow(state: .installed, isInstalledOnDisk: true, runnerKind: .llm))
        #expect(LocalServerPolicy.shouldTerminateAfterLastWindowClosed(status: .running(port: 1212)))
    }

    @Test func aClosedGateRefusesToFlipTheOption() async {
        let defaults = suite()
        useStoredOption(defaults, available: false)
        defer { reset() }
        let c = controller(defaults)
        await c.setOptionEnabled(true)
        #expect(!c.optionEnabled)
        #expect(!LocalServerOption.isOn(in: defaults))
    }

    // MARK: Off / on

    @Test func optionOffStopsTheListenerKeepsTheServedListAndOnRestoresIt() async {
        let defaults = suite()
        LocalServerOption.set(true, in: defaults)
        useStoredOption(defaults)
        defer { reset() }
        let pool = OptPreloader(pool: ResidentPool<OptBox>(budget: MemoryBudget(
            capacityBytes: 20 * gb, cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))))
        let c = controller(defaults, preloader: pool)
        defer { Task { await c.server.stop() } }

        await c.autoStart()
        #expect(c.isRunning)
        let key = ModelContainerPool.key(for: ModelStore.shared.directory(forHFModelID: optModel))
        #expect(await until { await pool.pool.pinnedKeys == [key] })

        await c.setOptionEnabled(false)
        #expect(!c.optionEnabled && !c.surfacesVisible)
        #expect(c.server.status == .stopped)                            // listener stopped
        #expect(c.servedIDs == [optModel])                              // list kept
        #expect(await pool.pool.pinnedKeys.isEmpty)                     // unpinned...
        #expect(await pool.pool.residentKeys.contains(key))             // ...not evicted
        #expect(!LocalServerOption.isOn(in: defaults))

        await c.setOptionEnabled(true)
        #expect(c.optionEnabled && c.surfacesVisible)
        #expect(c.isRunning)                                            // auto-started again
        #expect(c.servedIDs == [optModel])
        #expect(await until { await pool.pool.pinnedKeys == [key] })    // pins restored
    }

    @Test func optionOffCancelsAPreloadInFlight() async {
        let defaults = suite()
        LocalServerOption.set(true, in: defaults)
        useStoredOption(defaults)
        defer { reset() }
        let c = controller(defaults, preloader: OptPreloader(pool: ResidentPool<OptBox>(budget: MemoryBudget(
            capacityBytes: 20 * gb, cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1))))))
        defer { Task { await c.server.stop() } }
        await c.autoStart()
        await c.setOptionEnabled(false)
        await c.setServed(optOther, true)                               // refused: surfaces are off
        #expect(!c.isServing(optOther))
        #expect(c.server.status == .stopped)
    }

    @Test func aServeToggleStillTurnsTheRunningSwitchOn() async {
        let defaults = suite()
        LocalServerOption.set(true, in: defaults)
        useStoredOption(defaults)
        defer { reset() }
        let c = controller(defaults, served: [])
        defer { Task { await c.server.stop() } }
        await c.setServerEnabled(false)
        await c.setServed(optModel, true)
        #expect(c.enabled)
        #expect(c.isRunning)
    }
}
