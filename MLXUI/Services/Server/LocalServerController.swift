import Foundation
import Observation

/// The decisions the Local Server UI makes, as pure functions of (gate, server status, served set),
/// so the whole of S1-5's logic is testable without a window. **Every one of them starts from
/// `gateAvailable`** (rule 14): hidden means no surface, no auto-start and no stay-running.
enum LocalServerPolicy {
    /// What a change to the served set / enabled switch means for the listener.
    enum Action: Equatable { case start, stop, none }

    /// Serving any model starts the server if it's stopped; un-serving the last one stops it
    /// (design §5 item 4). Switching it off stops it; nothing starts it while the gate is closed.
    static func action(gateAvailable: Bool, enabled: Bool, servedCount: Int, status: LocalServer.Status) -> Action {
        let running: Bool
        switch status {
        case .starting, .running: running = true
        case .stopped, .failed: running = false
        }
        guard gateAvailable else { return running ? .stop : .none }
        let wanted = enabled && servedCount > 0
        if wanted { return running ? .none : .start }
        return running ? .stop : .none
    }

    /// Launch: "if the reachable set is non-empty at launch, start the server (the user left it
    /// serving)" — and the user hasn't switched it off, and the gate is open. Implementer's call,
    /// pending owner confirmation.
    static func shouldAutoStart(gateAvailable: Bool, enabled: Bool, servedCount: Int) -> Bool {
        gateAvailable && enabled && servedCount > 0
    }

    /// Design F6: closing the last window quits the app — **unless** the server is serving, which
    /// keeps it running (reopen from the Dock). With the gate closed it is always `true`, i.e.
    /// exactly today's behaviour.
    static func shouldTerminateAfterLastWindowClosed(gateAvailable: Bool, status: LocalServer.Status) -> Bool {
        guard gateAvailable else { return true }
        switch status {
        case .starting, .running: return false
        case .stopped, .failed: return true
        }
    }

    /// The tabs Settings shows: Providers and Privacy follow `hideProvidersPrivacy`; **Local Server
    /// follows the gate and nothing else** (R12 — "not behind `hideProvidersPrivacy`").
    static func visibleSettingsPanes(gateAvailable: Bool, hideProvidersPrivacy: Bool) -> [SettingsPane] {
        SettingsPane.allCases.filter { pane in
            switch pane {
            case .providers, .privacy: return !hideProvidersPrivacy
            case .localServer: return gateAvailable
            case .models, .agentTools: return true
            }
        }
    }

    /// A stale `@AppStorage("settingsPane")` pointing at a pane that is hidden now lands on Models.
    static func resolvedPane(_ selected: SettingsPane, visible: [SettingsPane]) -> SettingsPane {
        visible.contains(selected) ? selected : .models
    }

    /// The one-line status ("Serving 2 models at 127.0.0.1:1212").
    static func statusSentence(status: LocalServer.Status, servedCount: Int, enabled: Bool) -> String {
        switch status {
        case .running(let port):
            let noun = servedCount == 1 ? "model" : "models"
            return "Serving \(servedCount) \(noun) at 127.0.0.1:\(port)"
        case .starting:
            return "Starting…"
        case .failed(let reason):
            return reason
        case .stopped:
            if !enabled { return "The server is off." }
            return servedCount == 0 ? "Not serving — turn on Serve for a model to start." : "The server is stopped."
        }
    }
}

/// When the model detail page shows its **Serve** toggle — a pure function so the rule has one
/// home and one test (S1-5b: the toggle had been built into only one of the page's two "installed"
/// branches). Shown iff the Local Server is available (gate), the model is an MLX chat model, and
/// the model is installed — in the `.installed` state, or idle with files on disk.
enum ServeToggleVisibility {
    static func shouldShow(state: InstallState, isInstalledOnDisk: Bool, runnerKind: RunnerKind,
                           gateAvailable: Bool) -> Bool {
        guard gateAvailable, runnerKind == .llm else { return false }
        switch state {
        case .installed: return true
        case .idle: return isInstalledOnDisk
        case .resolving, .downloading, .verifying, .error, .needsAuth: return false
        }
    }
}

/// Which models a Mac can actually serve, and what the user is told (S1-5c, owner rulings 2026-10-02).
nonisolated enum ServeMemory {
    /// "Needs 24 GB RAM" — the same rule that already swaps Run for "Needs N GB RAM" on the detail
    /// page (`ModelEntry.exceedsRAM`). `nil` = the model fits this Mac.
    static func blockedReason(modelRAMGB: Double, systemRAMGB: Double) -> String? {
        guard modelRAMGB > systemRAMGB else { return nil }
        return "Needs \(String(format: "%.0f", modelRAMGB)) GB RAM — this Mac has \(String(format: "%.0f", systemRAMGB)) GB, so it can't be served."
    }

    /// The toggle is disabled for a model that doesn't fit — **unless it is currently reachable**, so
    /// a user can always turn a served model off.
    static func isToggleDisabled(blockedReason: String?, isReachable: Bool) -> Bool {
        blockedReason != nil && !isReachable
    }

    /// The `503` sentence for a request naming a served model that is larger than the whole shared
    /// memory budget (it would load anyway and thrash): "<model> needs N GB; the Local Server can
    /// use M GB on this Mac."
    static func tooLargeSentence(model: String, modelRAMGB: Double, capacityBytes: Int64) -> String {
        "\(model) needs \(String(format: "%.1f", modelRAMGB)) GB; the Local Server can use \(gigabytes(capacityBytes)) GB on this Mac."
    }

    /// True when a model's catalog size is larger than the entire shared budget.
    static func exceedsBudget(modelRAMGB: Double, capacityBytes: Int64) -> Bool {
        Int64(modelRAMGB * 1_073_741_824) > capacityBytes
    }

    /// One honest line when the served MLX models together exceed the budget (pane, S1-5c). Apple
    /// Foundation Models is the OS's and isn't counted.
    static let doesNotAllFitNote = "These models don't all fit in memory at once — switching between them reloads each one."

    static func note(servedIDs: Set<String>, installed: InstalledModelIndex, capacityBytes: Int64) -> String? {
        let total = installed.entries.filter { $0.kind == .llm && servedIDs.contains($0.hfModelId) }
            .reduce(0.0) { $0 + $1.ramGB }
        return Int64(total * 1_073_741_824) > capacityBytes ? doesNotAllFitNote : nil
    }

    private static func gigabytes(_ bytes: Int64) -> String { String(format: "%.1f", Double(bytes) / 1_073_741_824) }
}

/// "Reachable now" (S1-5c, owner ruling 1): a model is reachable by other apps only if it is in the
/// served list **and** the master switch is on. The list is kept when the switch is off; turning the
/// switch on restores every previously served model; serving from a model page turns it on (R24).
nonisolated enum ServeReach {
    static func isReachable(id: String, served: Set<String>, masterEnabled: Bool) -> Bool {
        masterEnabled && served.contains(id)
    }
}

/// S1-5's view-model: owns the `LocalServer`, turns "serve this model" into settings + a start/stop,
/// remembers whether the Connect sheet has ever been shown, and answers the launch / quit questions.
/// Everything is gated by `LocalServerGate.isAvailable`.
@Observable
final class LocalServerController {
    /// The app's one controller (the app delegate needs it for the stay-running decision).
    static let shared = LocalServerController()

    private static let enabledKey = "localServerEnabled"
    private static let connectShownKey = "localServerConnectShown"

    let server: LocalServer
    @ObservationIgnored private let defaults: UserDefaults

    /// Whether the user wants the server on. Defaults to on; serving a model turns it on.
    private(set) var enabled: Bool
    private(set) var hasShownConnect: Bool
    /// Drives the "Connect your app" sheet.
    var showConnect = false
    private(set) var connectModelID: String?

    @ObservationIgnored private let installedIndex: () -> InstalledModelIndex
    @ObservationIgnored private let appleFoundation: () -> Readiness?
    @ObservationIgnored private let catalogLoaded: () -> Bool
    @ObservationIgnored private let evictModel: (String) async -> Void
    @ObservationIgnored private let budgetBytes: Int64

    /// - Parameters: `installed` / `appleFoundation` / `evict` / `budgetBytes` default to the live
    ///   catalog (`catalogLoaded`: did it decode non-empty), the AFM checker, `ModelContainerPool.shared` and `MemoryBudget.shared`; tests
    ///   inject their own. `evict` takes an HF repo id.
    init(server: LocalServer = LocalServer(), defaults: UserDefaults = .standard,
         installed: @escaping () -> InstalledModelIndex = { InstalledModelIndex.loadInstalled() },
         appleFoundation: @escaping () -> Readiness? = { AppleFoundationAvailability.currentReadiness() },
         catalogLoaded: @escaping () -> Bool = { InstalledModelIndex.catalogLoaded },
         evict: @escaping (String) async -> Void = { hfModelID in
             await ModelContainerPool.shared.evict(modelID: ModelStore.repoSlug(for: hfModelID))
         },
         budgetBytes: Int64 = MemoryBudget.shared.capacityBytes) {
        self.server = server
        self.defaults = defaults
        self.installedIndex = installed
        self.appleFoundation = appleFoundation
        self.catalogLoaded = catalogLoaded
        self.evictModel = evict
        self.budgetBytes = budgetBytes
        self.enabled = defaults.object(forKey: Self.enabledKey) as? Bool ?? true
        self.hasShownConnect = defaults.bool(forKey: Self.connectShownKey)
    }

    /// The stored list — what the user chose. Not what other apps can reach: use `reachableIDs`.
    var servedIDs: Set<String> { server.settings.servedModelIDs }
    func isServing(_ id: String) -> Bool { server.settings.isServing(id) }
    /// **The** reachable set (S1-5d): served AND installed (AFM: served AND ready) — exactly what
    /// `/v1/models` lists. The pane, the status sentence, start/stop and auto-start all count this.
    var reachableIDs: Set<String> {
        Set(ServedModels.ids(served: servedIDs, installed: installedIndex(), appleFoundation: appleFoundation()))
    }
    /// What the Serve toggles show: reachable **and** the master switch on (`ServeReach`).
    func isReachable(_ id: String) -> Bool {
        ServeReach.isReachable(id: id, served: reachableIDs, masterEnabled: enabled)
    }
    /// The pane's "these don't all fit" line, or `nil`.
    var memoryNote: String? {
        ServeMemory.note(servedIDs: reachableIDs, installed: installedIndex(), capacityBytes: budgetBytes)
    }
    var statusSentence: String {
        LocalServerPolicy.statusSentence(status: server.status, servedCount: reachableIDs.count, enabled: enabled)
    }
    var isRunning: Bool {
        if case .running = server.status { return true }
        return false
    }

    /// The Serve toggle. **First Serve ever:** start the server, then present Connect your app.
    func setServed(_ id: String, _ on: Bool) async {
        guard LocalServerGate.isAvailable else { return }
        server.update(settings: server.settings.serving(id, on))
        if on { setEnabled(true) }
        await apply()
        if on, !hasShownConnect {
            hasShownConnect = true
            defaults.set(true, forKey: Self.connectShownKey)
            presentConnect(for: id)
        }
    }

    /// The Settings pane's on/off switch.
    func setServerEnabled(_ on: Bool) async {
        guard LocalServerGate.isAvailable else { return }
        setEnabled(on)
        await apply()
    }

    /// A new port applies on restart: the setting is saved now; a running server restarts so the
    /// listener moves, and a bind failure ("port in use") shows in the status sentence.
    func setPort(_ port: Int) async {
        guard LocalServerGate.isAvailable else { return }
        server.update(settings: server.settings.settingPort(port))
        if isRunning || server.status == .starting {
            await server.stop()
            await apply()
        } else if case .failed = server.status {
            await apply()
        }
    }

    /// Launch: drop served models that are no longer installed (deleted outside the app), then
    /// resume serving if any reachable model was left served (gate permitting).
    func autoStart() async {
        await reconcileServed()
        guard LocalServerPolicy.shouldAutoStart(gateAvailable: LocalServerGate.isAvailable, enabled: enabled,
                                                servedCount: reachableIDs.count) else { return }
        await server.start()
    }

    /// Uninstall hook (S1-5d): once `hfModelID`'s files are gone, un-serve it, free its weights, and
    /// stop the server if it was the last reachable model. A model whose repo is still installed
    /// (another card shares it) is left alone.
    func modelUninstalled(_ hfModelID: String) async {
        guard LocalServerGate.isAvailable, catalogLoaded() else { return }
        guard !isInstalledLLM(hfModelID) else { return }
        let wasServed = isServing(hfModelID)
        await reconcileServed()                          // un-serves + evicts it when it was served
        if !wasServed { await evictModel(hfModelID) }    // loaded by chat / a flow, never served
    }

    /// Drops every served MLX id that is no longer an installed chat model — evicting its weights —
    /// then applies start/stop. `apple-foundation` is never dropped (its readiness can change; it
    /// is just not reachable meanwhile). Gated: while hidden the persisted list is left untouched.
    /// Also skipped unless the catalog actually loaded non-empty (S1-5e, R26): an unreadable
    /// `browser.json` makes everything look uninstalled and must not wipe the user's list.
    func reconcileServed() async {
        guard LocalServerGate.isAvailable, catalogLoaded() else { return }
        let gone = servedIDs.filter { $0 != ServedModels.appleFoundationID && !isInstalledLLM($0) }
        for id in gone {
            server.update(settings: server.settings.serving(id, false))
            await evictModel(id)
        }
        await apply()
    }

    private func isInstalledLLM(_ hfModelID: String) -> Bool {
        installedIndex().entries.contains { $0.kind == .llm && $0.hfModelId == hfModelID }
    }

    func presentConnect(for id: String?) {
        guard LocalServerGate.isAvailable else { return }
        connectModelID = id ?? reachableIDs.sorted().first
        showConnect = true
    }

    private func setEnabled(_ on: Bool) {
        enabled = on
        defaults.set(on, forKey: Self.enabledKey)
    }

    private func apply() async {
        switch LocalServerPolicy.action(gateAvailable: LocalServerGate.isAvailable, enabled: enabled,
                                        servedCount: reachableIDs.count, status: server.status) {
        case .start: await server.start()
        case .stop: await server.stop()
        case .none: break
        }
    }
}
