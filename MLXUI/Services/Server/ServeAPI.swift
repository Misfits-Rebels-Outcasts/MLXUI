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
    /// S1-5c: the whole shared memory budget (`MemoryBudget.shared`). A served model larger than this
    /// is refused with a 503 rather than loaded to thrash.
    /// SC-6: read live (Dedicated mode changes the budget at run time) unless a test pinned a figure.
    var memoryCapacityOverride: Int64?
    var memoryCapacityBytes: Int64 {
        get { memoryCapacityOverride ?? MemoryBudget.shared.capacityBytes }
        set { memoryCapacityOverride = newValue }
    }
    /// S2-4: the context window `/v1/models` reports for a model id, or `nil` when the checkpoint
    /// declares none. Injected because it reads `config.json`; the live value reads the same
    /// `ModelContextWindow` the pre-count uses, and 8,192 for Apple Foundation Models.
    var modelContextLength: @Sendable (String) -> Int? = { _ in nil }
    /// WW4-1-5: whether `/v1/models` reports `structured_output: true` for a model id. Injected like `modelContextLength`
    /// (the live value reads `config.json`); the default is `false`, the honest answer for a double.
    var modelStructuredOutput: @Sendable (String) -> Bool = { _ in false }
    var now: @Sendable () -> Date = { Date() }
    /// S3-3c: the `previous_response_id` store, RAM only (rule 13). A class, so every copy of the environment shares it.
    var responseStore: ResponseStore = ResponseStore()

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
            },
            modelContextLength: { id in liveModelContextLength(id) },
            modelStructuredOutput: { id in liveModelStructuredOutput(id) })
    }

    /// WW4-1-5: `structured_output` for `/v1/models` — the allow-list's answer for the checkpoint's own `model_type`
    /// (`ServeStructuredOutputFamilies.allows(configData:)`); `false` for Apple Foundation Models and for an id that is not an
    /// installed MLX chat model.
    static func liveModelStructuredOutput(_ id: String) -> Bool {
        if id == ServedModels.appleFoundationID { return false }
        guard let entry = InstalledModelIndex.loadInstalled().entries.first(where: {
            $0.kind == .llm && $0.hfModelId == id
        }) else { return false }
        return ServeStructuredOutputFamilies.allows(configData: ModelContainerPool.configData(
            directory: ModelStore.shared.directory(forHFModelID: entry.hfModelId)))
    }

    /// S2-4: what `live()`'s `/v1/models` reports as a model's `context_length` — Apple Foundation
    /// Models' measured 8,192, else the checkpoint's own `config.json` through S2-6's
    /// `ModelContextWindow`. Split out so the `apple-foundation` answer is testable without a server
    /// or a disk read (that branch returns before touching either).
    static func liveModelContextLength(_ id: String) -> Int? {
        if id == ServedModels.appleFoundationID { return ConnectSnippets.appleFoundationContextLength }
        guard let entry = InstalledModelIndex.loadInstalled().entries.first(where: {
            $0.kind == .llm && $0.hfModelId == id
        }) else { return nil }
        return ModelContextWindow.limit(configData: ModelContainerPool.configData(
            directory: ModelStore.shared.directory(forHFModelID: entry.hfModelId)))
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
        if let rejection = requestGuard.check(head) { return error(rejection, path: head.path) }
        return route(head, environment: environment)
    }

    /// The full path (S1-3): the guard still runs first, then `POST /v1/chat/completions` (which
    /// reads the body and may stream), S3-2's `POST /v1/messages` and `/v1/messages/count_tokens`, S3-3's `POST /v1/responses`,
    /// else the body-less routes. Errors on `/v1/messages*` — the guard's included — are rendered in
    /// Anthropic's shape; the guard's *decisions* are the same on every path (R35).
    static func handle(head: ServeRequestHead, guard requestGuard: RequestGuard, environment: ServeEnvironment,
                       body: @Sendable () async throws -> Data) async -> ServeReply {
        if let rejection = requestGuard.check(head) { return .full(error(rejection, path: head.path)) }
        let post: ((ServeRequestHead, Data, ServeEnvironment) async -> ServeReply)?
        switch head.path {
        case "/v1/chat/completions": post = { await ChatCompletionsRoute.handle(head: $0, body: $1, environment: $2) }
        case "/v1/responses": post = { await ResponsesRoute.handle(head: $0, body: $1, environment: $2) }
        case "/v1/messages": post = { await MessagesRoute.handle(head: $0, body: $1, environment: $2) }
        case "/v1/messages/count_tokens": post = { await MessagesRoute.countTokens(head: $0, body: $1, environment: $2) }
        default: post = nil
        }
        guard let post else { return .full(route(head, environment: environment)) }
        guard head.method == "POST" else { return .full(methodNotAllowed(["POST"], path: head.path)) }
        let data: Data
        do { data = try await body() } catch {
            return .full(Self.error(ServeError(status: 400, message: "The request body couldn't be read.",
                                               code: "invalid_body"), path: head.path))
        }
        // S-FIX-2: refuse a deeply nested body before any dialect parses it (see `RequestGuard.checkBody`). Logged as metadata only —
        // no model, no tokens, no content.
        if let rejection = requestGuard.checkBody(data) {
            environment.requestLog.record(RequestLogEntry(
                time: environment.now(), userAgent: head.userAgent, model: nil, promptTokens: nil, completionTokens: nil,
                durationSeconds: 0, status: rejection.status, toolsIgnored: false))
            return .full(Self.error(rejection, path: head.path))
        }
        return await post(head, data, environment)
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
            return json(200, ModelsRoute.body(ids: ids, created: environment.created,
                                              contextLength: environment.modelContextLength,
                                              structuredOutput: environment.modelStructuredOutput))
        default:
            return error(ServeError(status: 404, message: "Unknown endpoint \(head.method) \(head.path).",
                                    code: "not_found"), path: head.path)
        }
    }

    private static func json(_ status: Int, _ body: Data) -> ServeResponse {
        ServeResponse(status: status, body: body, headers: ["Content-Type": "application/json"])
    }

    /// The error as a response, in the shape of the dialect that owns `path` (OpenAI's unless the path
    /// is the Anthropic one).
    private static func error(_ error: ServeError, path: String? = nil) -> ServeResponse {
        if let path, AnthropicMessagesDialect.owns(path) {
            return json(error.status, AnthropicMessagesDialect.errorBody(error))
        }
        return json(error.status, error.body())
    }

    private static func methodNotAllowed(_ allowed: [String], path: String? = nil) -> ServeResponse {
        var response = error(ServeError(status: 405, message: "Method not allowed. Use \(allowed.joined(separator: ", ")).",
                                        code: "method_not_allowed"), path: path)
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
