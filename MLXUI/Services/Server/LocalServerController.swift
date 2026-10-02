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

    /// Launch: "if `servedModelIDs` is non-empty at launch, start the server (the user left it
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

    init(server: LocalServer = LocalServer(), defaults: UserDefaults = .standard) {
        self.server = server
        self.defaults = defaults
        self.enabled = defaults.object(forKey: Self.enabledKey) as? Bool ?? true
        self.hasShownConnect = defaults.bool(forKey: Self.connectShownKey)
    }

    var servedIDs: Set<String> { server.settings.servedModelIDs }
    func isServing(_ id: String) -> Bool { server.settings.isServing(id) }
    var statusSentence: String {
        LocalServerPolicy.statusSentence(status: server.status, servedCount: servedIDs.count, enabled: enabled)
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

    /// Launch: resume serving if the user left models served (gate permitting).
    func autoStart() async {
        guard LocalServerPolicy.shouldAutoStart(gateAvailable: LocalServerGate.isAvailable, enabled: enabled,
                                                servedCount: servedIDs.count) else { return }
        await server.start()
    }

    func presentConnect(for id: String?) {
        guard LocalServerGate.isAvailable else { return }
        connectModelID = id ?? servedIDs.sorted().first
        showConnect = true
    }

    private func setEnabled(_ on: Bool) {
        enabled = on
        defaults.set(on, forKey: Self.enabledKey)
    }

    private func apply() async {
        switch LocalServerPolicy.action(gateAvailable: LocalServerGate.isAvailable, enabled: enabled,
                                        servedCount: servedIDs.count, status: server.status) {
        case .start: await server.start()
        case .stop: await server.stop()
        case .none: break
        }
    }
}
