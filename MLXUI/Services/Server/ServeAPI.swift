import Foundation
import FlyingFox
import FlyingSocks

/// What the server can look at when answering — injected, so tests never touch the disk, the
/// catalog, or Apple Intelligence.
nonisolated struct ServeEnvironment: Sendable {
    var servedModelIDs: @Sendable () -> Set<String>
    var installed: @Sendable () -> InstalledModelIndex
    /// `nil` = Apple Foundation Models is absent on this Mac (macOS < 26, or not opted in).
    /// Reads go through the AFM seam, never `SystemLanguageModel` (readme T8).
    var appleFoundationReadiness: @Sendable () -> Readiness?
    /// Epoch seconds reported as each model's `created`.
    var created: Int
    /// S1-3: the backend that serves a model id (`nil` = no backend for it in this build).
    var backendFor: @Sendable (String) -> ServeBackend? = { _ in nil }
    /// S1-3: one generation per model, global cap 1, depth 4 (design §3).
    var queue: ServeQueue = ServeQueue()
    var requestLog: RequestLog = RequestLog()
    /// S1-A ruling 2: a `: keep-alive` comment whenever nothing was written for this long.
    var keepAliveSeconds: Double = 1.0
    var now: @Sendable () -> Date = { Date() }

    /// The app's real environment: the served set from `served`, the live install markers, the
    /// real AFM checker.
    static func live(served: ServedSetBox, created: Int) -> ServeEnvironment {
        ServeEnvironment(
            servedModelIDs: { served.value },
            installed: { InstalledModelIndex.loadInstalled() },
            appleFoundationReadiness: { AppleFoundationAvailability.currentReadiness() },
            created: created,
            backendFor: { id in
                // An installed MLX chat model, or on-device Apple Foundation Models (S1-4). A
                // remote-provider model never has a backend (design P3).
                if id == ServedModels.appleFoundationID {
                    return AppleFoundationAvailability.makeChatStreamer().map { AFMBackend(streamer: $0) }
                }
                guard let entry = InstalledModelIndex.loadInstalled().entries.first(where: {
                    $0.kind == .llm && $0.hfModelId == id
                }) else { return nil }
                return mlxBackend(for: entry)
            })
    }

    /// The backend for an installed MLX chat model. The directory is keyed by the **HF repo**
    /// (`directory(forHFModelID:)`, `/` → `--`), which is where `InstallManager` put the files — S1-5's
    /// first by-hand run found the unslugged id nesting a `mlx-community/` folder that doesn't exist.
    static func mlxBackend(for entry: InstalledModelIndex.Entry) -> MLXChatBackend {
        MLXChatBackend(directory: ModelStore.shared.directory(forHFModelID: entry.hfModelId),
                       footprintBytes: Int64(entry.ramGB * 1_073_741_824))
    }
}

/// The served set, readable from the server's executors while `LocalServer` (main actor) owns
/// the truth.
nonisolated final class ServedSetBox: @unchecked Sendable {
    private let lock = NSLock()
    private var ids: Set<String>

    init(_ ids: Set<String> = []) { self.ids = ids }

    var value: Set<String> {
        get { lock.lock(); defer { lock.unlock() }; return ids }
        set { lock.lock(); ids = newValue; lock.unlock() }
    }
}

nonisolated struct ServeResponse: Sendable, Equatable {
    var status: Int
    var body: Data
    var headers: [String: String] = [:]
}

/// The decision layer: guard first (every request, before routing), then route. Nothing here
/// logs request content.
nonisolated enum ServeAPI {
    /// The endpoints that need no request body — a pure function over the head (S1-2).
    static func respond(to head: ServeRequestHead, guard requestGuard: RequestGuard,
                        environment: ServeEnvironment) -> ServeResponse {
        if let rejection = requestGuard.check(head) { return error(rejection) }
        return route(head, environment: environment)
    }

    /// The full path (S1-3): the guard still runs first, then `POST /v1/chat/completions` (which
    /// reads the body and may stream), else the body-less routes.
    static func handle(head: ServeRequestHead, guard requestGuard: RequestGuard, environment: ServeEnvironment,
                       body: @Sendable () async throws -> Data) async -> ServeReply {
        if let rejection = requestGuard.check(head) { return .full(error(rejection)) }
        guard head.path == "/v1/chat/completions" else { return .full(route(head, environment: environment)) }
        guard head.method == "POST" else { return .full(methodNotAllowed(["POST"])) }
        let data: Data
        do { data = try await body() } catch {
            return .full(Self.error(ServeError(status: 400, message: "The request body couldn't be read.",
                                               code: "invalid_body")))
        }
        return await ChatCompletionsRoute.handle(head: head, body: data, environment: environment)
    }

    private static func route(_ head: ServeRequestHead, environment: ServeEnvironment) -> ServeResponse {
        switch head.path {
        case "/health":
            return head.method == "GET" ? json(200, Data(#"{"status":"ok"}"#.utf8)) : methodNotAllowed(["GET"])
        case "/v1/models":
            guard head.method == "GET" else { return methodNotAllowed(["GET"]) }
            let ids = ServedModels.ids(served: environment.servedModelIDs(),
                                       installed: environment.installed(),
                                       appleFoundation: environment.appleFoundationReadiness())
            return json(200, ModelsRoute.body(ids: ids, created: environment.created))
        default:
            return error(ServeError(status: 404, message: "Unknown endpoint \(head.method) \(head.path).",
                                    code: "not_found"))
        }
    }

    private static func json(_ status: Int, _ body: Data) -> ServeResponse {
        ServeResponse(status: status, body: body, headers: ["Content-Type": "application/json"])
    }

    private static func error(_ error: ServeError) -> ServeResponse {
        json(error.status, error.body())
    }

    private static func methodNotAllowed(_ allowed: [String]) -> ServeResponse {
        var response = error(ServeError(status: 405, message: "Method not allowed. Use \(allowed.joined(separator: ", ")).",
                                        code: "method_not_allowed"))
        response.headers["Allow"] = allowed.joined(separator: ", ")
        return response
    }
}

/// The thin FlyingFox adapter: one root handler receives **every** request (mounted at `"*"`),
/// so the guard runs before any routing — including for paths that don't exist.
nonisolated struct LocalServerHandler: HTTPHandler {
    let port: @Sendable () -> UInt16
    let environment: ServeEnvironment

    func handleRequest(_ request: HTTPRequest) async throws -> HTTPResponse {
        let head = ServeRequestHead(
            method: request.method.rawValue,
            path: request.path,
            host: request.headers[.host],
            origin: request.headers[HTTPHeader("Origin")],
            transferEncoding: request.headers[.transferEncoding],
            contentLength: request.headers[.contentLength],
            userAgent: request.headers[HTTPHeader("User-Agent")])
        let reply = await ServeAPI.handle(head: head, guard: RequestGuard(port: port()), environment: environment,
                                          body: { try await request.bodyData })

        switch reply {
        case .full(let response):
            return HTTPResponse(statusCode: HTTPStatusCode(response.status, phrase: Self.phrase(for: response.status)),
                                headers: Self.headers(response.headers),
                                body: response.body)
        case .stream(let status, let headers, let bytes):
            // The spike's finding (S1-0): one `nextBuffer` call is one write, so SSE events arrive
            // as they are produced instead of in 4 KB lumps.
            return HTTPResponse(statusCode: HTTPStatusCode(status, phrase: Self.phrase(for: status)),
                                headers: Self.headers(headers),
                                body: HTTPBodySequence(from: SSEByteSequence(stream: bytes)))
        }
    }

    private static func headers(_ values: [String: String]) -> HTTPHeaders {
        var headers = HTTPHeaders()
        for (name, value) in values { headers[HTTPHeader(name)] = value }
        return headers
    }

    static func phrase(for status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 400: return "Bad Request"
        case 403: return "Forbidden"
        case 404: return "Not Found"
        case 405: return "Method Not Allowed"
        case 411: return "Length Required"
        case 413: return "Content Too Large"
        case 499: return "Client Closed Request"
        case 500: return "Internal Server Error"
        case 501: return "Not Implemented"
        case 503: return "Service Unavailable"
        default:  return "Error"
        }
    }
}

/// One element of the stream == one chunk written to the socket (S1-0, journal 2026-362): a custom
/// `AsyncBufferedSequence` whose `nextBuffer` returns exactly what was produced, never waiting to
/// fill a buffer — which is what makes tokens (and heartbeats) reach the client as they happen.
nonisolated struct SSEByteSequence: AsyncBufferedSequence, Sendable {
    typealias Element = UInt8
    let stream: AsyncStream<[UInt8]>

    func makeAsyncIterator() -> Iterator { Iterator(iterator: stream.makeAsyncIterator()) }

    struct Iterator: AsyncBufferedIteratorProtocol {
        var iterator: AsyncStream<[UInt8]>.AsyncIterator
        mutating func next() async throws -> UInt8? { fatalError("FlyingFox reads this through nextBuffer") }
        mutating func nextBuffer(suggested count: Int) async throws -> [UInt8]? { await iterator.next() }
    }
}
