import Testing
import Foundation
@testable import MLXUI

/// SC-2a-b (1, 2) — "the model it needs is in use by someone else and the wait ran out" is a retryable `503`
/// (`waiting_for_memory`) in every dialect: non-streaming carries `Retry-After: 2`; a streaming reply has already
/// sent its `200` head (the wait happens inside the generation, under the keep-alives), so it is an in-stream error
/// with the same code and message. "Too large for memory" stays a `400` without `Retry-After`.
struct ServeWaitingForMemoryTests {

    private static let model = "mlx-community/Qwen3-4B-4bit"
    private static let sentence = "Waiting for memory: Qwen3-8B-4bit is in use by a flow. Try again when it finishes."

    private struct Dialect: Sendable {
        let name: String, path: String, body: String
        /// What this dialect's error shape carries for a `waiting_for_memory` 503 (Anthropic has no `code`, only a type).
        let busyMarker: String
    }
    private static let dialects = [
        Dialect(name: "chat", path: "/v1/chat/completions",
                body: #"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}]__STREAM__}"#,
                busyMarker: "waiting_for_memory"),
        Dialect(name: "messages", path: "/v1/messages",
                body: #"{"model":"\#(model)","max_tokens":16,"messages":[{"role":"user","content":"hi"}]__STREAM__}"#,
                busyMarker: "overloaded_error"),
        Dialect(name: "responses", path: "/v1/responses",
                body: #"{"model":"\#(model)","input":"hi"__STREAM__}"#,
                busyMarker: "waiting_for_memory"),
    ]

    private func environment(_ backend: ServeBackend) -> ServeEnvironment {
        ServeEnvironment(
            servedModelIDs: { [Self.model] },
            installed: { InstalledModelIndex(entries: [
                .init(id: "mlx-community--Qwen3-4B-4bit", hfModelId: Self.model, kind: .llm, ramGB: 3, downloadSizeGB: 2)]) },
            appleFoundationReadiness: { nil },
            created: 1_700_000_000,
            backendFor: { $0 == Self.model ? backend : nil },
            queue: ServeQueue(), requestLog: RequestLog(), keepAliveSeconds: 1.0)
    }

    private func call(_ dialect: Dialect, stream: Bool, backend: ServeBackend) async -> ServeReply {
        let body = Data(dialect.body.replacingOccurrences(of: "__STREAM__", with: stream ? #","stream":true"# : "").utf8)
        let head = ServeRequestHead(method: "POST", path: dialect.path, host: "127.0.0.1:1212", origin: nil,
                                    transferEncoding: nil, contentLength: String(body.count), userAgent: "test/1.0")
        return await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212), environment: environment(backend),
                                     body: { body })
    }

    // MARK: the error itself

    @Test func theBusyErrorIsA503ThatCarriesItsRetryHint() {
        let error = ServeError.waitingForMemory(Self.sentence)
        #expect(error.status == 503)
        #expect(error.code == "waiting_for_memory")
        #expect(error.type == "server_error")
        #expect(error.message == Self.sentence)
        #expect(error.retryAfterSeconds == 2)
        #expect(ServeMemoryAdmission.refusal(needBytes: 2, roomBytes: 1).retryAfterSeconds == nil)   // "too large" is final
        #expect(ServeError(status: 500, message: "x", code: "internal_error").retryAfterSeconds == nil)
    }

    // MARK: non-streaming — every dialect

    @Test(arguments: dialects)
    private func nonStreamingIs503WithRetryAfterInTheDialectsShape(_ dialect: Dialect) async throws {
        let backend = FakeBackend([.error(.waitingForMemory(Self.sentence))])
        guard case .full(let response) = await call(dialect, stream: false, backend: backend) else {
            Issue.record("\(dialect.name): expected a full response"); return
        }
        #expect(response.status == 503, "\(dialect.name)")
        #expect(response.headers["Retry-After"] == "2", "\(dialect.name)")
        let text = String(decoding: response.body, as: UTF8.self)
        #expect(text.contains(dialect.busyMarker), "\(dialect.name): \(text)")
        #expect(text.contains(Self.sentence), "\(dialect.name): \(text)")
    }

    @Test(arguments: dialects)
    private func tooLargeStaysA400WithoutRetryAfter(_ dialect: Dialect) async throws {
        let backend = FakeBackend([.error(ServeMemoryAdmission.refusal(needBytes: 20 << 30, roomBytes: 10 << 30))])
        guard case .full(let response) = await call(dialect, stream: false, backend: backend) else {
            Issue.record("\(dialect.name): expected a full response"); return
        }
        #expect(response.status == 400, "\(dialect.name)")
        #expect(response.headers["Retry-After"] == nil, "\(dialect.name)")
        #expect(String(decoding: response.body, as: UTF8.self).contains("Use a shorter prompt"), "\(dialect.name)")
    }

    // MARK: streaming — the head is already out; the error rides the stream

    @Test(arguments: dialects)
    private func streamingKeepsItsHeadAndCarriesTheErrorInStream(_ dialect: Dialect) async throws {
        let backend = FakeBackend([.error(.waitingForMemory(Self.sentence))])
        guard case .stream(let status, _, let body) = await call(dialect, stream: true, backend: backend) else {
            Issue.record("\(dialect.name): expected a stream"); return
        }
        #expect(status == 200, "\(dialect.name): the head was sent before the wait, so it can't become a 503")
        var bytes: [UInt8] = []
        for await chunk in body { bytes += chunk }
        let text = String(decoding: bytes, as: UTF8.self)
        #expect(text.contains(dialect.busyMarker), "\(dialect.name): \(text)")
        #expect(text.contains(Self.sentence), "\(dialect.name): \(text)")
    }
}
