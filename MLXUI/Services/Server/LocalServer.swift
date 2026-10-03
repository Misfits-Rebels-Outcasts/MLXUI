import Foundation
import Observation
import os
import FlyingFox
import FlyingSocks

/// The Local Server's lifecycle (S1-2): two loopback listeners on the configured port —
/// `127.0.0.1` (**required**) and `[::1]` (**best-effort**) — owner rulings R10 and R15.
///
/// - IPv4 failing is the server failing, with a human sentence in `status`.
/// - Only IPv6 failing: the server runs IPv4-only and logs it (`ipv6Active == false`). The
///   Connect sheet keeps showing `http://127.0.0.1:<port>/v1`.
/// - Spike finding (journal 2026-362): FlyingFox's `.loopback(port:)` binds `[::1]` *only*, so the
///   IPv4 listener is built explicitly with `.inet(ip4:)`.
/// - **Gate (R14):** `start()` returns without binding when `!LocalServerGate.isAvailable` — a
///   second line of defence behind the UI. Nothing listens in the committed (hidden) default.
/// - **Not started by itself in S1-2** — S1-5's UI asks; tests start it.
///
/// Logs metadata only (port, status, error codes) — never request or response content (rule 13).
@Observable
final class LocalServer {
    enum Status: Equatable, Sendable {
        case stopped
        case starting
        case running(port: UInt16)
        case failed(reason: String)
    }

    private(set) var status: Status = .stopped
    /// Whether the `[::1]` listener is up (`false` while stopped, or when only IPv4 bound).
    private(set) var ipv6Active = false
    private(set) var settings: ServeSettings
    /// The in-memory request log (last 200, metadata only) — Settings → Local Server shows it.
    var requestLog: RequestLog { environment.requestLog }

    @ObservationIgnored private let defaults: UserDefaults?
    @ObservationIgnored private let listenPortOverride: UInt16?
    @ObservationIgnored private let served: ServedSetBox
    @ObservationIgnored private let environment: ServeEnvironment
    @ObservationIgnored private let portBox = PortBox()
    @ObservationIgnored private var servers: [HTTPServer] = []
    @ObservationIgnored private var runTasks: [Task<Void, Never>] = []
    /// Bumped by `stop()` so a `start()` suspended mid-bind notices and tears itself down.
    @ObservationIgnored private var generation = 0

    private static let log = Logger(subsystem: "com.ai-browser", category: "local-server")

    /// - Parameters:
    ///   - settings: defaults to the persisted `ServeSettings`.
    ///   - defaults: where `update(settings:)` persists; `nil` = don't persist (tests).
    ///   - listenPort: overrides the settings' port — **tests only** (0 = pick a free port).
    ///   - environment: what the routes look at; defaults to the live app environment.
    init(settings: ServeSettings = .load(),
         defaults: UserDefaults? = .standard,
         listenPort: UInt16? = nil,
         environment: ServeEnvironment? = nil) {
        let box = ServedSetBox(settings.servedModelIDs)
        self.settings = settings
        self.defaults = defaults
        self.listenPortOverride = listenPort
        self.served = box
        self.environment = environment ?? .live(served: box, created: Int(Date().timeIntervalSince1970))
    }

    /// Replace the settings (persisting them when a `defaults` was given). The served set takes
    /// effect immediately; a port change applies on the next `start()`.
    func update(settings: ServeSettings) {
        self.settings = settings
        served.value = settings.servedModelIDs
        if let defaults { settings.save(to: defaults) }
    }

    // MARK: - Lifecycle

    func start() async {
        guard LocalServerGate.isAvailable else { return }
        switch status {
        case .starting, .running: return
        case .stopped, .failed: break
        }
        status = .starting
        generation += 1
        let myGeneration = generation
        let requestedPort = listenPortOverride ?? UInt16(settings.port)

        // 1. IPv4 — required.
        let handler = LocalServerHandler(port: { [portBox] in portBox.value }, environment: environment)
        let v4Server: HTTPServer
        let v4Task: Task<Void, Never>
        let boundPort: UInt16
        do {
            let address = try sockaddr_in.inet(ip4: "127.0.0.1", port: requestedPort)
            v4Server = Self.makeServer(address: address, handler: handler)
            v4Task = try await Self.run(v4Server)
            boundPort = await Self.port(of: v4Server) ?? requestedPort
        } catch {
            guard myGeneration == generation else { return }
            let sentence = Self.sentence(for: error, port: requestedPort)
            Self.log.error("IPv4 listener failed on port \(requestedPort): \(error.localizedDescription, privacy: .public)")
            status = .failed(reason: sentence)
            return
        }
        portBox.value = boundPort

        // 2. IPv6 — best-effort, on the same port (the actual one, if 0 was requested).
        var v6Server: HTTPServer?
        var v6Task: Task<Void, Never>?
        do {
            let server = Self.makeServer(address: sockaddr_in6.loopback(port: boundPort), handler: handler)
            v6Task = try await Self.run(server)
            v6Server = server
        } catch {
            Self.log.notice("IPv6 listener unavailable on port \(boundPort); running IPv4-only: \(error.localizedDescription, privacy: .public)")
        }

        guard myGeneration == generation else {          // stop() ran while we were binding
            await v4Server.stop(timeout: 1)
            await v6Server?.stop(timeout: 1)
            return
        }
        servers = [v4Server] + (v6Server.map { [$0] } ?? [])
        runTasks = [v4Task] + (v6Task.map { [$0] } ?? [])
        ipv6Active = v6Server != nil
        status = .running(port: boundPort)
        Self.log.notice("serving on port \(boundPort) (IPv6 \(self.ipv6Active ? "on" : "off"))")
    }

    func stop() async {
        generation += 1
        let stopping = servers
        servers = []
        runTasks = []
        ipv6Active = false
        portBox.value = 0
        status = .stopped
        for server in stopping { await server.stop(timeout: 1) }
    }

    // MARK: - Pure helpers (tested directly)

    /// The human sentence for a failed bind (backlog S1-2: "Port-in-use is a `.failed` with a
    /// human sentence, never a crash").
    nonisolated static func sentence(for error: Error, port: UInt16) -> String {
        if case let SocketError.failed(_, errno, _) = error {
            switch errno {
            case EADDRINUSE:
                return "Port \(port) is already in use by another app. Quit that app or choose a different port."
            case EPERM, EACCES:
                return "macOS didn't allow this app to accept local connections on port \(port)."
            default:
                break
            }
        }
        return "The local server couldn't start on port \(port). (\(error.localizedDescription))"
    }

    // MARK: - FlyingFox plumbing

    /// `timeout` is FlyingFox's *per-request handler* timeout (default 15 s), not an idle
    /// timeout. S1-3's non-streaming completions run to completion (R16), so it is set far above
    /// any generation; streaming responses are written after the handler returns.
    nonisolated private static func makeServer(address: some SocketAddress, handler: LocalServerHandler) -> HTTPServer {
        HTTPServer(address: address, timeout: 3600, logger: .disabled, handler: handler)
    }

    /// Start `server` and wait until it is listening, or throw the bind error. `run()` blocks
    /// for the server's lifetime, so it lives in a task; a failed bind surfaces within ms.
    nonisolated private static func run(_ server: HTTPServer) async throws -> Task<Void, Never> {
        let outcome = RunOutcome()
        let task = Task<Void, Never> {
            do { try await server.run(); outcome.finish(nil) } catch { outcome.finish(error) }
        }
        for _ in 0..<500 {                                   // ≤ 5 s
            if await server.isListening { return task }
            if let failure = outcome.failure { throw failure }
            try? await Task.sleep(for: .milliseconds(10))
        }
        task.cancel()
        throw SocketError.timeout(message: "The listener didn't start in time.")
    }

    nonisolated private static func port(of server: HTTPServer) async -> UInt16? {
        switch await server.listeningAddress {
        case .ip4(_, let port)?, .ip6(_, let port)?: return port
        default: return nil
        }
    }
}

/// The port the guard compares `Host` against, set once the IPv4 listener reports it.
nonisolated private final class PortBox: @unchecked Sendable {
    private let lock = NSLock()
    private var port: UInt16 = 0
    var value: UInt16 {
        get { lock.lock(); defer { lock.unlock() }; return port }
        set { lock.lock(); port = newValue; lock.unlock() }
    }
}

/// How a server's `run()` ended, if it ended — read while waiting for the bind.
nonisolated private final class RunOutcome: @unchecked Sendable {
    private let lock = NSLock()
    private var error: Error?
    func finish(_ error: Error?) { lock.lock(); self.error = error; lock.unlock() }
    var failure: Error? { lock.lock(); defer { lock.unlock() }; return error }
}
