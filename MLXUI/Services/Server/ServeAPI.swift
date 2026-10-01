import Foundation
import FlyingFox

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

    /// The app's real environment: the served set from `served`, the live install markers, the
    /// real AFM checker.
    static func live(served: ServedSetBox, created: Int) -> ServeEnvironment {
        ServeEnvironment(
            servedModelIDs: { served.value },
            installed: { InstalledModelIndex.loadInstalled() },
            appleFoundationReadiness: { AppleFoundationAvailability.currentReadiness() },
            created: created)
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

/// The whole S1-2 decision layer as a pure function: guard first (every request, before
/// routing), then route. Nothing here logs request content; there is none to log.
nonisolated enum ServeAPI {
    static func respond(to head: ServeRequestHead, guard requestGuard: RequestGuard,
                        environment: ServeEnvironment) -> ServeResponse {
        if let rejection = requestGuard.check(head) { return error(rejection) }

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
            contentLength: request.headers[.contentLength])
        let response = ServeAPI.respond(to: head, guard: RequestGuard(port: port()), environment: environment)

        var headers = HTTPHeaders()
        for (name, value) in response.headers { headers[HTTPHeader(name)] = value }
        return HTTPResponse(statusCode: HTTPStatusCode(response.status, phrase: Self.phrase(for: response.status)),
                            headers: headers,
                            body: response.body)
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
        case 503: return "Service Unavailable"
        default:  return "Error"
        }
    }
}
