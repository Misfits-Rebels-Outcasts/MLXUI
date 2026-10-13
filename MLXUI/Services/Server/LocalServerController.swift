import Foundation
import Observation
import OSLog

/// The decisions the Local Server UI makes, as pure functions of (gate, server status, served set),
/// so the whole of S1-5's logic is testable without a window. **Every one of them starts from
/// `gateAvailable`** (rule 14): hidden means no surface, no auto-start and no stay-running. SC-4a: callers pass
/// `LocalServerGate.isEnabled` for everything except `visibleSettingsPanes`, which takes `isAvailable` (the Settings
/// pane is where the option lives).
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

    /// SC-4a: the stay-running decision as the app delegate asks it — `gateAvailable` is the gate's `isEnabled` here.
    static func shouldTerminateAfterLastWindowClosed(status: LocalServer.Status) -> Bool {
        shouldTerminateAfterLastWindowClosed(gateAvailable: LocalServerGate.isEnabled, status: status)
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
    /// SC-4a: the toggle exists only when the option is on (enable first, then serve — amends R24).
    static func shouldShow(state: InstallState, isInstalledOnDisk: Bool, runnerKind: RunnerKind) -> Bool {
        shouldShow(state: state, isInstalledOnDisk: isInstalledOnDisk, runnerKind: runnerKind,
                   gateAvailable: LocalServerGate.isEnabled)
    }

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

/// The facts the Connect sheet's snippets are generated from (S2-5): the model's tool capability (S2-1's
/// `ToolCallFormatTable`), the window S2-6 enforces, and whether S2-3's thinking control reaches its template.
/// R51 (2026-10-10) removed the Agent-ready state that used to sit here: a model is just served.
nonisolated enum ServeModelFacts {
    /// Everything the Connect sheet needs about one served model, read from its installed `config.json`.
    static func connectFacts(hfModelID: String) -> ConnectSnippets.ModelFacts {
        let directory = ModelStore.shared.directory(forHFModelID: hfModelID)
        return facts(modelID: hfModelID, configData: ModelContainerPool.configData(directory: directory),
                     weightsBytes: MLXChatBackend.weightsBytes(in: directory))
    }

    /// The pure half: what a `config.json` says about the model, and — with the weights' bytes and the room
    /// (R53) — how much of its window this Mac can actually hold (`usableTokens`).
    static func facts(modelID: String, configData: Data?, weightsBytes: Int64 = 0,
                      roomBytes: Int64 = ServeMemoryAdmission.room()) -> ConnectSnippets.ModelFacts {
        let modelType = ModelContainerPool.modelTypeString(configData: configData) ?? ""
        let window = ModelContextWindow.limit(configData: configData)
        let profile = ServeMemoryAdmission.profile(configData: configData, weightsBytes: weightsBytes, roomBytes: roomBytes)
        return ConnectSnippets.ModelFacts(
            modelID: modelID,
            supportsTools: ToolCallFormatTable.toolSupport(forModelType: modelType).acceptsTools,
            contextLength: window,
            thinkingAvailable: ReasoningTemplateTable.readsEnableThinking(forModelType: modelType),
            usableTokens: ServeMemoryAdmission.usableTokens(profile: profile, contextLength: window))
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
/// Everything is gated by `LocalServerGate.isEnabled`.
@Observable
final class LocalServerController {
    /// The app's one controller (the app delegate needs it for the stay-running decision).
    static let shared = LocalServerController()

    private static let enabledKey = "localServerEnabled"
    private static let connectShownKey = "localServerConnectShown"
    private static let servedOrderKey = "localServerServedOrder"
    /// SC-2: what a resident served model shows when the served set is larger than the budget and it was not kept ready.
    static let notKeptReadyReason = "not enough memory to keep ready"

    let server: LocalServer
    @ObservationIgnored private let defaults: UserDefaults

    /// Whether the user wants the server on. Defaults to on; serving a model turns it on.
    private(set) var enabled: Bool
    private(set) var hasShownConnect: Bool
    /// SC-4a: the "Enable Local Server" option, mirrored here so SwiftUI observes it (`LocalServerOption` is the store).
    private(set) var optionEnabled: Bool
    /// Drives the "Connect your app" sheet.
    var showConnect = false
    private(set) var connectModelID: String?

    @ObservationIgnored private let installedIndex: () -> InstalledModelIndex
    @ObservationIgnored private let appleFoundation: () -> Readiness?
    @ObservationIgnored private let catalogLoaded: () -> Bool
    @ObservationIgnored private let evictModel: (String) async -> Void
    /// PC-1: the prompt cache's snapshots of an uninstalled model go with its weights (F11).
    @ObservationIgnored private let evictPromptCache: (String) async -> Void
    @ObservationIgnored private let budgetBytes: Int64
    /// SC-1: the pool served models are preloaded into (`nil` = no preloading: unit tests that don't inject one).
    @ObservationIgnored private let preloadPool: (any ServePreloading)?
    @ObservationIgnored private var preloads: [String: (run: UUID, task: Task<Void, Never>)] = [:]
    @ObservationIgnored private var observingPool = false
    @ObservationIgnored private static let log = Logger(subsystem: "com.ai-browser", category: "local-server")

    /// SC-1: each served MLX model's load state, mirrored from the pool on every transition (the pool is the source —
    /// this is only what SwiftUI observes). Absent = not loaded.
    private(set) var loadStates: [String: ModelLoadState] = [:]
    /// SC-2: reachable served models that are in memory but did not fit the budget alongside the more recently served
    /// ones, so they are not kept ready (shown as `notLoaded("not enough memory to keep ready")`).
    private(set) var notKeptReady: Set<String> = []
    /// SC-2: pin recomputes coalesce — one runs at a time, and a request that arrives meanwhile makes it run again.
    @ObservationIgnored private var pinTask: Task<Void, Never>?
    @ObservationIgnored private var pinDirty = false

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
         evictPromptCache: @escaping (String) async -> Void = { hfModelID in
             let directory = ModelStore.shared.directory(forHFModelID: hfModelID)
             await PromptCacheShared.store.evict(directory: ModelContainerPool.key(for: directory))
         },
         budgetBytes: Int64 = MemoryBudget.shared.capacityBytes,
         preloadPool: (any ServePreloading)? = TestEnvironment.isRunningTests ? nil : ModelContainerPool.shared) {
        self.server = server
        self.defaults = defaults
        self.installedIndex = installed
        self.appleFoundation = appleFoundation
        self.catalogLoaded = catalogLoaded
        self.evictModel = evict
        self.evictPromptCache = evictPromptCache
        self.budgetBytes = budgetBytes
        self.preloadPool = preloadPool
        self.enabled = defaults.object(forKey: Self.enabledKey) as? Bool ?? true
        self.hasShownConnect = defaults.bool(forKey: Self.connectShownKey)
        self.optionEnabled = LocalServerOption.isOn(in: defaults)      // runs the one-time migration first
    }

    /// Whether the serving surfaces (Serve toggle, Connect sheet, …) are shown: `LocalServerGate.isEnabled`, read
    /// after the observed option so a view that asks re-renders when the option changes.
    var surfacesVisible: Bool {
        _ = optionEnabled
        return LocalServerGate.isEnabled
    }

    /// The stored list — what the user chose. Not what other apps can reach: use `reachableIDs`.
    var servedIDs: Set<String> { server.settings.servedModelIDs }

    /// SC-3: per-model serving numbers over the in-memory request log (last 200). Computed on read — nothing is
    /// stored — so a view re-reads it when it refreshes (the log is not observable).
    var servingStats: [String: ModelServeStats] { ServeStats.summary(entries: server.requestLog.entries) }
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
        guard LocalServerGate.isEnabled else { return }
        server.update(settings: server.settings.serving(id, on))
        if on { setEnabled(true); noteServed(id) } else { server.forgetStoredResponses(for: id) }       // S3-3c
        await apply()
        if on { startPreloads() } else { cancelPreload(id) }
        await recomputePins()
        if on, !hasShownConnect {
            hasShownConnect = true
            defaults.set(true, forKey: Self.connectShownKey)
            presentConnect(for: id)
        }
    }

    /// SC-4a: "Enable Local Server". **Off** stops the listener, cancels preloads and unpins — the served list is
    /// kept. **On** restores it: reconcile, auto-start, preloads, pins. Asks `isAvailable`: it is the one control
    /// that exists while serving is not enabled.
    func setOptionEnabled(_ on: Bool) async {
        guard LocalServerGate.isAvailable else { return }
        LocalServerOption.set(on, in: defaults)
        optionEnabled = on
        if on {
            await autoStart()
        } else {
            showConnect = false
            await apply()
            cancelAllPreloads()
            await recomputePins()
        }
    }

    /// The Settings pane's on/off switch.
    func setServerEnabled(_ on: Bool) async {
        guard LocalServerGate.isEnabled else { return }
        setEnabled(on)
        await apply()
        if on { startPreloads() } else { cancelAllPreloads() }
        await recomputePins()
    }

    /// A new port applies on restart: the setting is saved now; a running server restarts so the
    /// listener moves, and a bind failure ("port in use") shows in the status sentence.
    func setPort(_ port: Int) async {
        guard LocalServerGate.isEnabled else { return }
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
        guard LocalServerPolicy.shouldAutoStart(gateAvailable: LocalServerGate.isEnabled, enabled: enabled,
                                                servedCount: reachableIDs.count) else { return }
        await server.start()
        startPreloads()
        await recomputePins()
    }

    /// Uninstall hook (S1-5d): once `hfModelID`'s files are gone, un-serve it, free its weights, and
    /// stop the server if it was the last reachable model. A model whose repo is still installed
    /// (another card shares it) is left alone.
    func modelUninstalled(_ hfModelID: String) async {
        guard LocalServerGate.isEnabled, catalogLoaded() else { return }
        guard !isInstalledLLM(hfModelID) else { return }
        let wasServed = isServing(hfModelID)
        await reconcileServed()                          // un-serves + evicts it when it was served
        if !wasServed { await evictLoaded(hfModelID) }   // loaded by chat / a flow, never served
    }

    /// A model's weights and its prompt-cache snapshots (PC-1).
    private func evictLoaded(_ hfModelID: String) async {
        await evictModel(hfModelID)
        await evictPromptCache(hfModelID)
        server.forgetStoredResponses(for: hfModelID)
    }

    /// Drops every served MLX id that is no longer an installed chat model — evicting its weights —
    /// then applies start/stop. `apple-foundation` is never dropped (its readiness can change; it
    /// is just not reachable meanwhile). Gated: while hidden the persisted list is left untouched.
    /// Also skipped unless the catalog actually loaded non-empty (S1-5e, R26): an unreadable
    /// `browser.json` makes everything look uninstalled and must not wipe the user's list.
    func reconcileServed() async {
        guard LocalServerGate.isEnabled, catalogLoaded() else { return }
        let gone = servedIDs.filter { $0 != ServedModels.appleFoundationID && !isInstalledLLM($0) }
        for id in gone {
            server.update(settings: server.settings.serving(id, false))
            await evictLoaded(id)
        }
        await apply()
        await recomputePins()
    }

    private func isInstalledLLM(_ hfModelID: String) -> Bool {
        installedIndex().entries.contains { $0.kind == .llm && $0.hfModelId == hfModelID }
    }

    func presentConnect(for id: String?) {
        guard LocalServerGate.isEnabled else { return }
        connectModelID = id ?? reachableIDs.sorted().first
        showConnect = true
    }

    // MARK: - Preload (SC-1, gate B)

    /// The load state the Server screen shows for a served model. Apple Foundation Models loads nothing: ready.
    func loadState(for id: String) -> ModelLoadState {
        if id == ServedModels.appleFoundationID { return .ready }
        let state = loadStates[id] ?? .notLoaded(reason: nil)
        if state == .ready, notKeptReady.contains(id) { return .notLoaded(reason: Self.notKeptReadyReason) }
        return state
    }

    /// Load every reachable served MLX model in the background — unleased, never waiting, never evicting a model in
    /// use (`ModelContainerPool.preload`). Nothing while the gate is closed or the server isn't running. A model
    /// that doesn't fit now stays `notLoaded` with a reason; its first request loads it as before.
    ///
    /// `declinedOnly` is the automatic retry (SC-2: a recompute found room): it touches only models whose last preload
    /// declined for lack of memory — never one that was pushed out (its reason stays until the next request or an
    /// explicit Serve), and never one already loading or ready.
    private func startPreloads(declinedOnly: Bool = false) {
        guard LocalServerGate.isEnabled, enabled, isRunning, let pool = preloadPool else { return }
        observeLoadStates(pool)
        let index = installedIndex()
        for id in reachableIDs where id != ServedModels.appleFoundationID && preloads[id] == nil {
            if declinedOnly, loadStates[id] != .notLoaded(reason: ModelLoadState.declinedReason) { continue }
            guard let entry = index.entries.first(where: { $0.kind == .llm && $0.hfModelId == id }) else { continue }
            let directory = ModelStore.shared.directory(forHFModelID: id)
            let footprint = Int64(entry.ramGB * 1_073_741_824)
            let run = UUID()
            preloads[id] = (run, Task { [weak self] in
                let state = await pool.preload(directory: directory, footprintBytes: footprint)
                await self?.preloadFinished(id, run: run, state)
            })
        }
    }

    private func preloadFinished(_ id: String, run: UUID, _ state: ModelLoadState) {
        guard preloads[id]?.run == run else { return }     // cancelled, and maybe restarted since
        preloads.removeValue(forKey: id)
        loadStates[id] = state
        if state == .ready { Task { await recomputePins() } }      // resident already, so no transition said so
    }

    /// Serve off for one model: stop a preload in flight. It does not evict (owner ruling 2026-10-12).
    private func cancelPreload(_ id: String) {
        preloads.removeValue(forKey: id)?.task.cancel()
        loadStates.removeValue(forKey: id)
        notKeptReady.remove(id)
    }

    private func cancelAllPreloads() {
        preloads.values.forEach { $0.task.cancel() }
        preloads.removeAll()
    }

    /// Mirror the pool's transitions for the served models, and log them (ids only).
    private func observeLoadStates(_ pool: any ServePreloading) {
        guard !observingPool else { return }
        observingPool = true
        Task { [weak self] in
            await pool.addLoadObserver { key, state in
                Task { @MainActor in self?.poolChanged(key: key, state: state) }
            }
        }
    }

    private func poolChanged(key: String, state: ModelLoadState) {
        // Memory freed (any model) or a served model became resident: pins and declined preloads may change. A decline
        // or a load starting is neither, and must not start a recompute (that would loop).
        let memoryMoved: Bool
        switch state {
        case .ready: memoryMoved = true
        case .notLoaded(let reason): memoryMoved = reason != ModelLoadState.declinedReason
        case .loading, .failed: memoryMoved = false
        }
        defer { if memoryMoved { Task { await recomputePins() } } }
        guard let id = servedIDs.first(where: {
            $0 != ServedModels.appleFoundationID
                && ModelContainerPool.key(for: ModelStore.shared.directory(forHFModelID: $0)) == key
        }) else { return }
        loadStates[id] = state
        let name: String
        switch state {
        case .notLoaded: name = "notLoaded"
        case .loading: name = "loading"
        case .ready: name = "ready"
        case .failed: name = "failed"      // the message can carry paths; ids and state only
        }
        if case .notLoaded(let reason?) = state, reason.hasPrefix("pushed out") {
            Self.log.info("model \(id, privacy: .public): \(reason, privacy: .public)")      // a requester kind, never content
        } else {
            Self.log.info("model \(id, privacy: .public): \(name, privacy: .public)")
        }
    }

    // MARK: - Pins (SC-2, gate C)

    /// The reachable served MLX ids, most recently served first (ids never served in this install's history last, by id).
    private func servedByRecency() -> [String] {
        let order = defaults.stringArray(forKey: Self.servedOrderKey) ?? []
        let ids = reachableIDs.filter { $0 != ServedModels.appleFoundationID }
        let ranked = order.filter { ids.contains($0) }
        return ranked + ids.subtracting(ranked).sorted()
    }

    private func noteServed(_ id: String) {
        var order = defaults.stringArray(forKey: Self.servedOrderKey) ?? []
        order.removeAll { $0 == id }
        order.insert(id, at: 0)
        defaults.set(Array(order.prefix(200)), forKey: Self.servedOrderKey)
    }

    /// Keep the reachable served models that are in memory ready: pin them, most recently served first, while they fit
    /// the budget; the rest are `notKeptReady`. Nothing is pinned with the gate closed or the master switch off, and
    /// a model no longer served is unpinned (not evicted — owner, 2026-10-12). Then retry preloads that declined, since
    /// this runs exactly when room may have appeared. Coalesced: concurrent requests share one run, and one that
    /// arrives mid-run makes it go round again.
    func recomputePins() async {
        guard preloadPool != nil else { return }
        pinDirty = true
        if let running = pinTask { await running.value; return }
        let task = Task { [weak self] in
            guard let self else { return }
            while pinDirty {
                pinDirty = false
                await applyPins()
            }
            pinTask = nil
        }
        pinTask = task
        await task.value
    }

    private func applyPins() async {
        guard let pool = preloadPool else { return }
        observeLoadStates(pool)
        var pins: [URL] = []
        var overflow: Set<String> = []
        if LocalServerGate.isEnabled, enabled {
            var used: Int64 = 0
            for id in servedByRecency() {
                let directory = ModelStore.shared.directory(forHFModelID: id)
                guard let bytes = await pool.residentBytes(directory: directory) else { continue }
                if used + bytes <= budgetBytes {
                    used += bytes
                    pins.append(directory)
                } else {
                    overflow.insert(id)
                }
            }
        }
        await pool.setPinned(directories: pins)
        if notKeptReady != overflow { notKeptReady = overflow }
        startPreloads(declinedOnly: true)
    }

    private func setEnabled(_ on: Bool) {
        enabled = on
        defaults.set(on, forKey: Self.enabledKey)
    }

    private func apply() async {
        switch LocalServerPolicy.action(gateAvailable: LocalServerGate.isEnabled, enabled: enabled,
                                        servedCount: reachableIDs.count, status: server.status) {
        case .start: await server.start()
        case .stop: await server.stop()
        case .none: break
        }
    }
}
