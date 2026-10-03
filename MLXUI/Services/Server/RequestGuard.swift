import Foundation

/// An OpenAI-shaped error: `{"error":{"message","type","code"}}` (design §4, backlog S1-2).
nonisolated struct ServeError: Error, Sendable, Equatable {
    let status: Int
    let message: String
    let type: String
    let code: String

    init(status: Int, message: String, type: String = "invalid_request_error", code: String) {
        self.status = status
        self.message = message
        self.type = type
        self.code = code
    }

    /// The JSON body. Fixed key order so the output is deterministic.
    func body() -> Data {
        struct Wire: Encodable {
            struct Inner: Encodable { let message: String; let type: String; let code: String }
            let error: Inner
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return (try? encoder.encode(Wire(error: .init(message: message, type: type, code: code)))) ?? Data()
    }
}

/// The few request fields the guard and router need, lifted off the HTTP library's request so
/// the whole decision layer is a pure function over values (unit-testable without a socket).
nonisolated struct ServeRequestHead: Sendable, Equatable {
    var method: String
    var path: String
    var host: String?
    var origin: String?
    var transferEncoding: String?
    var contentLength: String?
    var userAgent: String?
}

/// The browser guard and size cap — design §4, run on **every** request, before routing.
///
/// - `Host` must be exactly `127.0.0.1:<port>`, `localhost:<port>` or `[::1]:<port>` (DNS
///   rebinding — Ollama's CVE-2024-28224). Host names compare case-insensitively (RFC 9110).
/// - Any `Origin` header → 403 (a page in the user's browser must not drive this server).
/// - `OPTIONS` → 403 (no CORS preflight is ever answered; no CORS headers are ever sent).
/// - Body > 8 MB → 413, judged from `Content-Length`.
///
/// Two additions the contract didn't list, **implementer's call, pending owner confirmation**:
/// FlyingFox reads a request body only by `Content-Length` and does not decode chunked bodies,
/// so a request carrying `Transfer-Encoding` would slip past the size cap and desynchronise the
/// connection — it is refused with 411; and an unparseable `Content-Length` is a 400.
///
/// Per design §4 / P2 this guard is always on and not user-configurable.
nonisolated struct RequestGuard: Sendable {
    static let maxBodyBytes = 8 * 1024 * 1024

    let port: UInt16

    /// `nil` when the request may proceed; otherwise the rejection to send.
    func check(_ head: ServeRequestHead) -> ServeError? {
        let allowedHosts: Set<String> = ["127.0.0.1:\(port)", "localhost:\(port)", "[::1]:\(port)"]
        guard let host = head.host?.trimmingCharacters(in: .whitespaces).lowercased(),
              allowedHosts.contains(host) else {
            return ServeError(status: 403,
                              message: "This server only answers requests addressed to 127.0.0.1, localhost or [::1] on port \(port).",
                              code: "invalid_host")
        }
        if head.origin != nil {
            return ServeError(status: 403,
                              message: "Requests from web pages are not allowed.",
                              code: "origin_not_allowed")
        }
        if head.method.uppercased() == "OPTIONS" {
            return ServeError(status: 403,
                              message: "Cross-origin preflight requests are not allowed.",
                              code: "preflight_not_allowed")
        }
        if head.transferEncoding != nil {
            return ServeError(status: 411,
                              message: "Chunked request bodies aren't supported; send a Content-Length.",
                              code: "length_required")
        }
        if let raw = head.contentLength {
            guard let length = Int(raw.trimmingCharacters(in: .whitespaces)), length >= 0 else {
                return ServeError(status: 400, message: "Invalid Content-Length.", code: "invalid_content_length")
            }
            if length > Self.maxBodyBytes {
                return ServeError(status: 413,
                                  message: "Request body is larger than the \(Self.maxBodyBytes / 1_048_576) MB limit.",
                                  code: "body_too_large")
            }
        }
        return nil
    }
}
