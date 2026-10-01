import Testing
import Foundation
import Darwin
import FlyingSocks
@testable import MLXUI

// MARK: - Pure pieces (no sockets)

/// S1-2 — `RequestGuard`: the design §4 table, every rejected case and the accepted ones.
struct RequestGuardTests {
    private let guardRule = RequestGuard(port: 1212)

    private func head(method: String = "GET", host: String? = "127.0.0.1:1212", origin: String? = nil,
                      transferEncoding: String? = nil, contentLength: String? = nil) -> ServeRequestHead {
        ServeRequestHead(method: method, path: "/v1/models", host: host, origin: origin,
                         transferEncoding: transferEncoding, contentLength: contentLength)
    }

    @Test(arguments: ["127.0.0.1:1212", "localhost:1212", "[::1]:1212", "LOCALHOST:1212", "Localhost:1212"])
    func acceptsTheThreeLoopbackHostsOnThePort(host: String) {
        #expect(guardRule.check(head(host: host)) == nil)
    }

    @Test(arguments: [
        "evil.test:1212",            // DNS rebinding: an attacker's name pointing at 127.0.0.1
        "127.0.0.1",                 // no port
        "127.0.0.1:8080",            // wrong port
        "localhost",                 // no port
        "localhost:12120",           // port is a prefix, not equal
        "0.0.0.0:1212",
        "[::1]",
        "127.0.0.1:1212.evil.test",
        "evil.test",
        ""])
    func rejectsAnyOtherHost(host: String) {
        let error = guardRule.check(head(host: host))
        #expect(error?.status == 403)
        #expect(error?.code == "invalid_host")
    }

    @Test func rejectsAMissingHost() {
        #expect(guardRule.check(head(host: nil))?.code == "invalid_host")
    }

    @Test func rejectsAnyOriginHeader() {
        for origin in ["http://evil.test", "https://example.com", "null", ""] {
            let error = guardRule.check(head(origin: origin))
            #expect(error?.status == 403)
            #expect(error?.code == "origin_not_allowed")
        }
    }

    @Test(arguments: ["OPTIONS", "options"])
    func rejectsOptions(method: String) {
        let error = guardRule.check(head(method: method))
        #expect(error?.status == 403)
        #expect(error?.code == "preflight_not_allowed")
    }

    @Test func capsTheBodyAtEightMegabytes() {
        let cap = RequestGuard.maxBodyBytes
        #expect(cap == 8 * 1024 * 1024)
        #expect(guardRule.check(head(method: "POST", contentLength: "\(cap)")) == nil)
        let over = guardRule.check(head(method: "POST", contentLength: "\(cap + 1)"))
        #expect(over?.status == 413)
        #expect(over?.code == "body_too_large")
    }

    @Test func rejectsAnUnparseableOrNegativeContentLength() {
        #expect(guardRule.check(head(method: "POST", contentLength: "abc"))?.status == 400)
        #expect(guardRule.check(head(method: "POST", contentLength: "-1"))?.status == 400)
    }

    @Test func refusesChunkedRequestBodiesBecauseTheyWouldBypassTheCap() {
        let error = guardRule.check(head(method: "POST", transferEncoding: "chunked"))
        #expect(error?.status == 411)
    }

    @Test func theHostCheckComesBeforeEverythingElse() {
        // A rebinding attempt that also carries an Origin and is oversized is reported as a Host failure.
        let error = guardRule.check(head(method: "OPTIONS", host: "evil.test:1212", origin: "http://evil.test",
                                         contentLength: "99999999"))
        #expect(error?.code == "invalid_host")
    }

    @Test func errorBodiesUseTheOpenAIShape() throws {
        let error = try #require(guardRule.check(head(origin: "http://evil.test")))
        let object = try #require(try JSONSerialization.jsonObject(with: error.body()) as? [String: [String: String]])
        #expect(object["error"]?["code"] == "origin_not_allowed")
        #expect(object["error"]?["type"] == "invalid_request_error")
        #expect(object["error"]?["message"]?.isEmpty == false)
    }
}

/// S1-2 — which models `/v1/models` lists, and the responses around them.
struct ServeAPITests {
    private static func entry(_ hf: String, _ kind: RunnerKind) -> InstalledModelIndex.Entry {
        InstalledModelIndex.Entry(id: hf.replacingOccurrences(of: "/", with: "--"), hfModelId: hf, kind: kind, ramGB: 4)
    }

    private let installed = InstalledModelIndex(entries: [
        entry("mlx-community/Qwen3-4B-4bit", .llm),
        entry("mlx-community/gemma-3-1b-it-qat-4bit", .llm),
        entry("mlx-community/bge-m3-mlx-4bit", .embedding),
    ])

    @Test func listsServedAndInstalledChatModelsSorted() {
        let ids = ServedModels.ids(
            served: ["mlx-community/Qwen3-4B-4bit", "mlx-community/gemma-3-1b-it-qat-4bit"],
            installed: installed, appleFoundation: nil)
        #expect(ids == ["mlx-community/Qwen3-4B-4bit", "mlx-community/gemma-3-1b-it-qat-4bit"])
    }

    @Test func aServedModelThatIsNoLongerInstalledDropsOut() {
        let ids = ServedModels.ids(served: ["mlx-community/Qwen3-4B-4bit", "mlx-community/deleted-model"],
                                   installed: installed, appleFoundation: nil)
        #expect(ids == ["mlx-community/Qwen3-4B-4bit"])
    }

    @Test func installedButNotServedIsNotListed() {
        #expect(ServedModels.ids(served: [], installed: installed, appleFoundation: nil).isEmpty)
    }

    @Test func onlyChatModelsAreListedNotEmbeddings() {
        let ids = ServedModels.ids(served: ["mlx-community/bge-m3-mlx-4bit"], installed: installed, appleFoundation: nil)
        #expect(ids.isEmpty)
    }

    @Test func appleFoundationAppearsOnlyWhenServedAndReady() {
        let served: Set<String> = [ServedModels.appleFoundationID]
        #expect(ServedModels.ids(served: served, installed: installed, appleFoundation: .ready) == ["apple-foundation"])
        #expect(ServedModels.ids(served: served, installed: installed, appleFoundation: nil).isEmpty)
        #expect(ServedModels.ids(served: served, installed: installed,
                                 appleFoundation: .needsSetup(reason: "x", action: nil)).isEmpty)
        #expect(ServedModels.ids(served: served, installed: installed,
                                 appleFoundation: .unavailable(reason: "x")).isEmpty)
        #expect(ServedModels.ids(served: [], installed: installed, appleFoundation: .ready).isEmpty)
    }

    @Test func remoteProviderModelsCanNeverAppear() {
        // Design P3 / readme T13: even if one is "served" (stale or hand-edited settings) and
        // even with AFM ready, a remote-provider model must not be exposed — it would proxy the
        // user's own API key through localhost.
        let remote: Set<String> = ["claude-sonnet @ anthropic", "gpt-4o @ openai", "deepseek-chat @ deepseek",
                                   "anthropic", "claude-sonnet-5-5"]
        let ids = ServedModels.ids(served: remote.union(["mlx-community/Qwen3-4B-4bit"]),
                                   installed: installed, appleFoundation: .ready)
        #expect(ids == ["mlx-community/Qwen3-4B-4bit"])
    }

    // MARK: responses

    private func environment(served: Set<String> = ["mlx-community/Qwen3-4B-4bit"],
                             afm: Readiness? = nil) -> ServeEnvironment {
        ServeEnvironment(servedModelIDs: { served }, installed: { installed },
                         appleFoundationReadiness: { afm }, created: 1_700_000_000)
    }

    private func respond(_ method: String, _ path: String, host: String = "127.0.0.1:1212",
                         origin: String? = nil, env: ServeEnvironment? = nil) -> ServeResponse {
        ServeAPI.respond(to: ServeRequestHead(method: method, path: path, host: host, origin: origin),
                         guard: RequestGuard(port: 1212), environment: env ?? environment())
    }

    @Test func modelsReturnsTheOpenAIListShape() throws {
        let response = respond("GET", "/v1/models")
        #expect(response.status == 200)
        #expect(response.headers["Content-Type"] == "application/json")
        let object = try #require(try JSONSerialization.jsonObject(with: response.body) as? [String: Any])
        #expect(object["object"] as? String == "list")
        let data = try #require(object["data"] as? [[String: Any]])
        #expect(data.count == 1)
        #expect(data[0]["id"] as? String == "mlx-community/Qwen3-4B-4bit")
        #expect(data[0]["object"] as? String == "model")
        #expect(data[0]["owned_by"] as? String == "mlxui")
        #expect(data[0]["created"] as? Int == 1_700_000_000)
    }

    @Test func anEmptyServedSetIsAnEmptyList() throws {
        let response = respond("GET", "/v1/models", env: environment(served: []))
        let object = try #require(try JSONSerialization.jsonObject(with: response.body) as? [String: Any])
        #expect((object["data"] as? [Any])?.isEmpty == true)
    }

    @Test func healthIsOk() {
        let response = respond("GET", "/health")
        #expect(response.status == 200)
        #expect(String(data: response.body, encoding: .utf8) == #"{"status":"ok"}"#)
    }

    @Test func unknownPathIs404InTheOpenAIErrorShape() throws {
        let response = respond("GET", "/v1/nope")
        #expect(response.status == 404)
        let object = try #require(try JSONSerialization.jsonObject(with: response.body) as? [String: [String: String]])
        #expect(object["error"]?["code"] == "not_found")
    }

    @Test func wrongMethodIs405WithAllow() {
        let response = respond("POST", "/v1/models")
        #expect(response.status == 405)
        #expect(response.headers["Allow"] == "GET")
    }

    @Test func theGuardRunsBeforeRouting() {
        // A bad Host to a path that doesn't exist is a 403, not a 404 — the guard is first.
        #expect(respond("GET", "/nope", host: "evil.test:1212").status == 403)
        #expect(respond("GET", "/v1/models", origin: "http://evil.test").status == 403)
        #expect(respond("OPTIONS", "/v1/models").status == 403)
        #expect(respond("GET", "/health", host: "evil.test:1212").status == 403)
    }

    @Test func noCORSHeadersAreEverSent() {
        for response in [respond("GET", "/v1/models"), respond("GET", "/health"),
                         respond("GET", "/x", origin: "http://evil.test")] {
            #expect(!response.headers.keys.contains { $0.lowercased().hasPrefix("access-control") })
        }
    }
}

/// S1-2 — the persisted settings.
struct ServeSettingsTests {
    @Test func defaultsAreThePortFromTheRulingAndNothingServed() {
        let settings = ServeSettings()
        #expect(settings.port == 1212)
        #expect(settings.servedModelIDs.isEmpty)
    }

    @Test func portIsClampedToUnprivilegedPorts() {
        #expect(ServeSettings(port: 80).port == 1024)
        #expect(ServeSettings(port: 70_000).port == 65535)
        #expect(ServeSettings().settingPort(8080).port == 8080)
    }

    @Test func servingTogglesTheSet() {
        let on = ServeSettings().serving("a", true).serving("b", true)
        #expect(on.isServing("a") && on.isServing("b"))
        #expect(!on.serving("a", false).isServing("a"))
        #expect(on.serving("a", false).isServing("b"))
    }

    @Test func roundTripsThroughUserDefaults() throws {
        let suite = "ServeSettingsTests-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let original = ServeSettings(port: 4321, servedModelIDs: ["mlx-community/x", "apple-foundation"])
        original.save(to: defaults)
        #expect(ServeSettings.load(from: defaults) == original)
    }

    @Test func missingOrCorruptDataLoadsDefaults() throws {
        let suite = "ServeSettingsTests-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        #expect(ServeSettings.load(from: defaults) == ServeSettings())
        defaults.set(Data("not json".utf8), forKey: "localServerSettings")
        #expect(ServeSettings.load(from: defaults) == ServeSettings())
    }

    @Test func aBlobMissingAFieldStillDecodes() throws {
        let decoded = try JSONDecoder().decode(ServeSettings.self, from: Data(#"{"port":2000}"#.utf8))
        #expect(decoded.port == 2000)
        #expect(decoded.servedModelIDs.isEmpty)
    }
}

/// S1-2 — the human sentences for a failed bind.
struct LocalServerSentenceTests {
    @Test func portInUseNamesThePortAndTheFix() {
        let error = SocketError.failed(type: "Bind", errno: EADDRINUSE, message: "Address already in use")
        let sentence = LocalServer.sentence(for: error, port: 1212)
        #expect(sentence.contains("1212"))
        #expect(sentence.contains("already in use"))
    }

    @Test func aMissingEntitlementIsExplained() {
        let error = SocketError.failed(type: "Bind", errno: EPERM, message: "Operation not permitted")
        #expect(LocalServer.sentence(for: error, port: 1212).contains("didn't allow"))
    }

    @Test func anythingElseStillGetsASentence() {
        struct Odd: Error {}
        #expect(LocalServer.sentence(for: Odd(), port: 1212).contains("couldn't start"))
    }
}

// MARK: - Real sockets (the gate, the listeners, the guard on the wire)

/// Sockets in tests: raw POSIX, so a header like `Host:` or `Content-Length:` goes out exactly as
/// written (URLSession rewrites some of them).
private enum Wire {
    static func freePort() -> UInt16 {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        defer { close(fd) }
        var addr = sockaddr_in(); addr.sin_family = sa_family_t(AF_INET); addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        _ = withUnsafePointer(to: &addr) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }
        var out = sockaddr_in(); var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafeMutablePointer(to: &out) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &len) } }
        return UInt16(bigEndian: out.sin_port)
    }

    /// A listening socket occupying `port` on the given family's loopback. Caller closes it.
    static func occupy(port: UInt16, ipv6: Bool) -> Int32? {
        let fd = socket(ipv6 ? AF_INET6 : AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return nil }
        let ok: Int32
        if ipv6 {
            var addr = sockaddr_in6(); addr.sin6_family = sa_family_t(AF_INET6); addr.sin6_len = UInt8(MemoryLayout<sockaddr_in6>.size)
            addr.sin6_port = port.bigEndian; addr.sin6_addr = in6addr_loopback
            ok = withUnsafePointer(to: &addr) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in6>.size)) } }
        } else {
            var addr = sockaddr_in(); addr.sin_family = sa_family_t(AF_INET); addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
            addr.sin_port = port.bigEndian; addr.sin_addr.s_addr = inet_addr("127.0.0.1")
            ok = withUnsafePointer(to: &addr) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }
        }
        guard ok == 0, listen(fd, 4) == 0 else { close(fd); return nil }
        return fd
    }

    static func canConnect(port: UInt16, ipv6: Bool = false) -> Bool {
        let fd = socket(ipv6 ? AF_INET6 : AF_INET, SOCK_STREAM, 0)
        defer { close(fd) }
        return connectFD(fd, port: port, ipv6: ipv6) == 0
    }

    private static func connectFD(_ fd: Int32, port: UInt16, ipv6: Bool) -> Int32 {
        if ipv6 {
            var addr = sockaddr_in6(); addr.sin6_family = sa_family_t(AF_INET6); addr.sin6_len = UInt8(MemoryLayout<sockaddr_in6>.size)
            addr.sin6_port = port.bigEndian; addr.sin6_addr = in6addr_loopback
            return withUnsafePointer(to: &addr) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in6>.size)) } }
        }
        var addr = sockaddr_in(); addr.sin_family = sa_family_t(AF_INET); addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_port = port.bigEndian; addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        return withUnsafePointer(to: &addr) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }
    }

    struct Reply { let status: Int; let headers: String; let body: String }

    /// Send `raw` and read one response (headers + Content-Length bytes). Nil on connect/read failure.
    static func send(_ raw: String, port: UInt16, ipv6: Bool = false) -> Reply? {
        let fd = socket(ipv6 ? AF_INET6 : AF_INET, SOCK_STREAM, 0)
        defer { close(fd) }
        var timeout = timeval(tv_sec: 3, tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        guard connectFD(fd, port: port, ipv6: ipv6) == 0 else { return nil }
        let bytes = Array(raw.utf8)
        guard write(fd, bytes, bytes.count) == bytes.count else { return nil }

        var received = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while true {
            let n = read(fd, &buffer, buffer.count)
            if n <= 0 { break }
            received.append(buffer, count: n)
            if let text = String(data: received, encoding: .utf8), let end = text.range(of: "\r\n\r\n") {
                let head = String(text[..<end.lowerBound])
                // NB: "\r\n" is ONE Character in Swift — split on the string, never a Character.
                let length = head.components(separatedBy: "\r\n")
                    .first { $0.lowercased().hasPrefix("content-length:") }
                    .flatMap { Int($0.dropFirst("content-length:".count).trimmingCharacters(in: .whitespaces)) } ?? 0
                let body = text[end.upperBound...]
                if body.utf8.count >= length {
                    let status = Int(head.split(separator: " ").dropFirst().first ?? "") ?? 0
                    return Reply(status: status, headers: head, body: String(body))
                }
            }
        }
        return nil
    }

    static func get(_ path: String, port: UInt16, host: String? = nil, extra: String = "", ipv6: Bool = false) -> Reply? {
        send("GET \(path) HTTP/1.1\r\nHost: \(host ?? "127.0.0.1:\(port)")\r\n\(extra)Connection: close\r\n\r\n",
             port: port, ipv6: ipv6)
    }
}

/// Serialized: these flip the shared gate override and bind real ports.
@Suite(.serialized)
@MainActor
struct LocalServerIntegrationTests {
    private static let model = "mlx-community/Qwen3-4B-4bit"

    private static func environment() -> ServeEnvironment {
        ServeEnvironment(
            servedModelIDs: { [model] },
            installed: { InstalledModelIndex(entries: [
                .init(id: "mlx-community--Qwen3-4B-4bit", hfModelId: model, kind: .llm, ramGB: 3)]) },
            appleFoundationReadiness: { nil },
            created: 1_700_000_000)
    }

    private func makeServer(port: UInt16 = 0) -> LocalServer {
        LocalServer(settings: ServeSettings(servedModelIDs: [Self.model]), defaults: nil,
                    listenPort: port, environment: Self.environment())
    }

    @Test func withTheGateOnStartBindsAndServesModels() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let server = makeServer()
        await server.start()
        defer { Task { await server.stop() } }

        guard case .running(let port) = server.status else {
            Issue.record("expected .running, got \(server.status)"); return
        }
        #expect(port != 0)

        // The contract's integration test: GET /v1/models with URLSession.
        let (data, response) = try await URLSession.shared.data(from: URL(string: "http://127.0.0.1:\(port)/v1/models")!)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let ids = (object["data"] as? [[String: Any]])?.compactMap { $0["id"] as? String }
        #expect(ids == [Self.model])

        let (health, _) = try await URLSession.shared.data(from: URL(string: "http://127.0.0.1:\(port)/health")!)
        #expect(String(data: health, encoding: .utf8) == #"{"status":"ok"}"#)
    }

    @Test func withTheGateOffStartDoesNotBind() async {
        LocalServerGate.overrideForTesting = false
        defer { LocalServerGate.overrideForTesting = nil }
        let port = Wire.freePort()
        let server = makeServer(port: port)
        await server.start()
        #expect(server.status == .stopped)
        #expect(!Wire.canConnect(port: port))            // connection refused: nothing listens
        #expect(!Wire.canConnect(port: port, ipv6: true))
        #expect(!server.ipv6Active)
    }

    @Test func listensOnBothLoopbacksOnTheSamePort() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let server = makeServer()
        await server.start()
        defer { Task { await server.stop() } }
        guard case .running(let port) = server.status else { Issue.record("not running: \(server.status)"); return }
        #expect(server.ipv6Active)
        #expect(Wire.get("/health", port: port)?.status == 200)                                  // 127.0.0.1
        #expect(Wire.get("/health", port: port, host: "[::1]:\(port)", ipv6: true)?.status == 200) // [::1]
        #expect(Wire.get("/health", port: port, host: "localhost:\(port)")?.status == 200)
    }

    @Test func theGuardIsEnforcedOnTheWireOnBothListeners() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let server = makeServer()
        await server.start()
        defer { Task { await server.stop() } }
        guard case .running(let port) = server.status else { Issue.record("not running: \(server.status)"); return }

        // The contract's exit checks.
        #expect(Wire.get("/v1/models", port: port, extra: "Origin: http://evil.test\r\n")?.status == 403)
        #expect(Wire.get("/v1/models", port: port, host: "evil.test:\(port)")?.status == 403)
        #expect(Wire.send("OPTIONS /v1/models HTTP/1.1\r\nHost: 127.0.0.1:\(port)\r\nConnection: close\r\n\r\n", port: port)?.status == 403)
        // 413 from the header alone (no body is ever sent).
        #expect(Wire.send("POST /v1/models HTTP/1.1\r\nHost: 127.0.0.1:\(port)\r\nContent-Length: \(RequestGuard.maxBodyBytes + 1)\r\nConnection: close\r\n\r\n", port: port)?.status == 413)
        #expect(Wire.send("POST /v1/models HTTP/1.1\r\nHost: 127.0.0.1:\(port)\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n", port: port)?.status == 411)
        // The same rules on [::1].
        #expect(Wire.get("/v1/models", port: port, host: "[::1]:\(port)", extra: "Origin: http://evil.test\r\n", ipv6: true)?.status == 403)
        #expect(Wire.get("/v1/models", port: port, host: "evil.test:\(port)", ipv6: true)?.status == 403)
        // And the accepted case still works afterwards.
        let ok = Wire.get("/v1/models", port: port)
        #expect(ok?.status == 200)
        #expect(ok?.body.contains(Self.model) == true)
        #expect(Wire.get("/v1/nope", port: port)?.status == 404)
        // No CORS headers on anything.
        #expect(ok?.headers.lowercased().contains("access-control") == false)
    }

    @Test func aPortInUseIsAFailedStatusWithAHumanSentence() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let port = Wire.freePort()
        guard let blocker = Wire.occupy(port: port, ipv6: false) else { Issue.record("couldn't occupy the port"); return }
        defer { close(blocker) }

        let server = makeServer(port: port)
        await server.start()
        guard case .failed(let reason) = server.status else { Issue.record("expected .failed, got \(server.status)"); return }
        #expect(reason.contains("\(port)"))
        #expect(reason.contains("already in use"))
        #expect(!server.ipv6Active)                       // IPv4 is required: nothing else started
    }

    @Test func aFailedStartCanBeRetriedOnceThePortIsFree() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let port = Wire.freePort()
        let blocker = Wire.occupy(port: port, ipv6: false)
        let server = makeServer(port: port)
        await server.start()
        guard case .failed = server.status else { Issue.record("expected .failed"); if let blocker { close(blocker) }; return }
        if let blocker { close(blocker) }
        await server.start()
        #expect(server.status == .running(port: port))
        await server.stop()
    }

    @Test func ifOnlyIPv6FailsTheServerRunsIPv4Only() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let port = Wire.freePort()
        guard let blocker = Wire.occupy(port: port, ipv6: true) else { Issue.record("couldn't occupy [::1]:\(port)"); return }
        defer { close(blocker) }

        let server = makeServer(port: port)
        await server.start()
        defer { Task { await server.stop() } }
        #expect(server.status == .running(port: port))    // R15: IPv6 is best-effort
        #expect(!server.ipv6Active)
        #expect(Wire.get("/health", port: port)?.status == 200)
    }

    @Test func startIsIdempotentAndStopFreesThePort() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let server = makeServer()
        await server.start()
        guard case .running(let port) = server.status else { Issue.record("not running"); return }
        await server.start()                              // no-op, still the same listener
        #expect(server.status == .running(port: port))
        await server.stop()
        #expect(server.status == .stopped)
        #expect(!Wire.canConnect(port: port))
        #expect(!Wire.canConnect(port: port, ipv6: true))
    }

    @Test func theServedSetChangesLiveWithoutARestart() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let box = ServedSetBox([])
        let env = ServeEnvironment(servedModelIDs: { box.value },
                                   installed: Self.environment().installed,
                                   appleFoundationReadiness: { nil }, created: 1)
        let server = LocalServer(settings: ServeSettings(), defaults: nil, listenPort: 0, environment: env)
        await server.start()
        defer { Task { await server.stop() } }
        guard case .running(let port) = server.status else { Issue.record("not running"); return }
        #expect(Wire.get("/v1/models", port: port)?.body.contains(Self.model) == false)
        box.value = [Self.model]
        let after = Wire.get("/v1/models", port: port)
        #expect(after?.body.contains(Self.model) == true)
    }
}

/// S1-2 — the gate itself (rule 14): one helper, a test seam, both states.
@Suite(.serialized)
struct LocalServerGateTests {
    @Test func theOverrideDecidesBothWays() {
        LocalServerGate.overrideForTesting = true
        #expect(LocalServerGate.isAvailable)
        LocalServerGate.overrideForTesting = false
        #expect(!LocalServerGate.isAvailable)
        LocalServerGate.overrideForTesting = nil
    }

    @Test func withoutAnOverrideItFollowsTheFlag() {
        LocalServerGate.overrideForTesting = nil
        #expect(LocalServerGate.isAvailable == !AppState.hideLocalServer)
    }
}
