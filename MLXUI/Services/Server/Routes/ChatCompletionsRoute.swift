import Foundation
import os

/// What a route hands back to the transport: a whole response, or an open SSE byte stream.
nonisolated enum ServeReply: Sendable {
    case full(ServeResponse)
    case stream(status: Int, headers: [String: String], body: AsyncStream<[UInt8]>)
}

/// `POST /v1/chat/completions` (S1-3): decode → is the model served → queue → backend → encode.
///
/// - Unknown / unserved model → `404`; line full → `503` + `Retry-After: 2` (decided *before* any
///   byte is written, so it is still a proper status).
/// - Non-streaming runs to completion (it cannot notice a disconnect — S1-A ruling 3).
/// - Streaming writes `: keep-alive` whenever nothing was written for `keepAliveSeconds` (S1-A
///   ruling 2), including while queued and during prefill; a client that went away makes the next
///   write fail, which cancels the generation (S1-0).
/// - **Nothing here logs request or response content.** The request log has no field for it.
nonisolated enum ChatCompletionsRoute {
    private static let log = Logger(subsystem: "com.ai-browser", category: "local-server")

    static func handle(head: ServeRequestHead, body: Data, environment: ServeEnvironment) async -> ServeReply {
        let started = environment.now()
        let tracker = RequestTracker(head: head, started: started, environment: environment)

        let request: ServeRequest
        do { request = try OpenAIChatDialect.decode(body) } catch {
            return tracker.fail(error as? ServeError ?? Self.internalError)
        }
        tracker.model = request.model
        tracker.toolsIgnored = request.toolsIgnored
        if request.toolsIgnored { log.notice("tools ignored") }

        // Apple Foundation Models that is served but not ready is a 503 with the readiness sentence
        // (S1-4), not a 404: `/v1/models` omits it until ready, but the client named a model the
        // user *is* serving. Readiness comes only through the injected checker (readme T8).
        if request.model == ServedModels.appleFoundationID,
           environment.servedModelIDs().contains(request.model),
           let state = environment.appleFoundationReadiness(), state != .ready {
            return tracker.fail(ReadinessSentence.error(for: state), headers: ["Retry-After": "2"])
        }

        let servable = ServedModels.ids(served: environment.servedModelIDs(),
                                        installed: environment.installed(),
                                        appleFoundation: environment.appleFoundationReadiness())
        guard servable.contains(request.model) else {
            return tracker.fail(ServeError(status: 404, message: "The model '\(request.model)' is not served.",
                                           code: "model_not_found"))
        }
        guard let backend = environment.backendFor(request.model) else {
            return tracker.fail(ServeError(status: 501, message: "The model '\(request.model)' can't be served by this build yet.",
                                           type: "server_error", code: "not_implemented"))
        }
        guard let ticket = await environment.queue.reserve(model: request.model) else {
            return tracker.fail(ServeError(status: 503, message: "The server is busy with other requests for this model. Try again shortly.",
                                           type: "server_error", code: "server_busy"),
                                headers: ["Retry-After": "2"])
        }

        return request.stream
            ? stream(request, backend: backend, ticket: ticket, tracker: tracker, environment: environment)
            : await complete(request, backend: backend, ticket: ticket, tracker: tracker, environment: environment)
    }

    // MARK: - Non-streaming

    private static func complete(_ request: ServeRequest, backend: ServeBackend, ticket: ServeQueue.Ticket,
                                 tracker: RequestTracker, environment: ServeEnvironment) async -> ServeReply {
        let queue = environment.queue
        do { try await queue.waitTurn(ticket) } catch {
            await queue.release(ticket)
            return tracker.fail(Self.internalError)
        }
        var text = ""
        var finish = ServeFinishReason.stop
        var usage = ServeUsage(promptTokens: 0, completionTokens: 0)
        var failure: ServeError?
        do {
            for try await event in backend.generate(request) {
                switch event {
                case .textDelta(let piece): text += piece
                case .usage(let value): usage = value
                case .finish(let reason): finish = reason
                case .error(let error): failure = error
                case .reasoningDelta, .toolCallDelta: break         // S2
                }
            }
        } catch {
            log.error("generation failed: \(String(describing: type(of: error)), privacy: .public)")
            failure = Self.generationFailed
        }
        await queue.release(ticket)
        if let failure { return tracker.fail(failure) }

        tracker.usage = usage
        let body = OpenAIChatDialect.completionBody(
            id: OpenAIChatDialect.makeCompletionID(), created: Int(environment.now().timeIntervalSince1970),
            model: request.model, text: text, finish: finish, usage: usage)
        return tracker.succeed(.full(ServeResponse(status: 200, body: body,
                                                   headers: ["Content-Type": "application/json"])))
    }

    // MARK: - Streaming

    private static func stream(_ request: ServeRequest, backend: ServeBackend, ticket: ServeQueue.Ticket,
                               tracker: RequestTracker, environment: ServeEnvironment) -> ServeReply {
        let (bytes, continuation) = AsyncStream<[UInt8]>.makeStream()
        let clock = WriteClock(now: environment.now)
        let id = OpenAIChatDialect.makeCompletionID()
        let created = Int(environment.now().timeIntervalSince1970)
        let model = request.model
        let includeUsage = request.includeUsage
        let queue = environment.queue
        let write: @Sendable ([UInt8]) -> Void = { chunk in continuation.yield(chunk); clock.touch() }

        let producer = Task {
            defer { continuation.finish() }
            do { try await queue.waitTurn(ticket) } catch {
                await queue.release(ticket)
                _ = tracker.fail(Self.internalError)
                return
            }
            var wroteRole = false
            var usage: ServeUsage?
            var finish: ServeFinishReason?
            var failure: ServeError?
            func ensureRole() {
                guard !wroteRole else { return }
                wroteRole = true
                write(OpenAIChatDialect.sse(OpenAIChatDialect.roleChunk(id: id, created: created, model: model,
                                                                         includeUsage: includeUsage)))
            }
            do {
                for try await event in backend.generate(request) {
                    switch event {
                    case .textDelta(let piece):
                        ensureRole()
                        write(OpenAIChatDialect.sse(OpenAIChatDialect.contentChunk(
                            id: id, created: created, model: model, text: piece, includeUsage: includeUsage)))
                    case .usage(let value): usage = value
                    case .finish(let reason): finish = reason
                    case .error(let error): failure = error
                    case .reasoningDelta, .toolCallDelta: break          // S2
                    }
                }
            } catch {
                log.error("generation failed: \(String(describing: type(of: error)), privacy: .public)")
                failure = Self.generationFailed
            }
            await queue.release(ticket)

            if Task.isCancelled {                                          // the client went away
                _ = tracker.fail(ServeError(status: 499, message: "Client closed the request.", code: "client_closed"))
                return
            }
            if let failure {
                write(OpenAIChatDialect.sse(OpenAIChatDialect.streamErrorChunk(failure)))
                write(OpenAIChatDialect.done)
                _ = tracker.fail(failure, response: false)
                return
            }
            ensureRole()
            write(OpenAIChatDialect.sse(OpenAIChatDialect.finishChunk(
                id: id, created: created, model: model, finish: finish ?? .stop, includeUsage: includeUsage)))
            if includeUsage, let usage {
                write(OpenAIChatDialect.sse(OpenAIChatDialect.usageChunk(id: id, created: created, model: model, usage: usage)))
            }
            write(OpenAIChatDialect.done)
            tracker.usage = usage
            _ = tracker.succeed(nil)
        }

        let heartbeat = Task {
            let interval = environment.keepAliveSeconds
            while !Task.isCancelled {
                let idle = clock.secondsSinceLastWrite()
                if idle >= interval {
                    write(OpenAIChatDialect.keepAlive)
                    try? await Task.sleep(for: .seconds(interval))
                } else {
                    try? await Task.sleep(for: .seconds(interval - idle))
                }
            }
        }
        continuation.onTermination = { _ in
            producer.cancel()
            heartbeat.cancel()
        }
        return .stream(status: 200,
                       headers: ["Content-Type": "text/event-stream", "Cache-Control": "no-cache"],
                       body: bytes)
    }

    // MARK: - Shared

    fileprivate static let internalError = ServeError(status: 500, message: "The server couldn't handle this request.",
                                                      type: "server_error", code: "internal_error")
    fileprivate static let generationFailed = ServeError(status: 500, message: "The model failed to generate a reply.",
                                                         type: "server_error", code: "generation_failed")
}

/// When the last byte (or heartbeat) was written — the heartbeat's only input.
nonisolated final class WriteClock: @unchecked Sendable {
    private let lock = NSLock()
    private var last: Date
    private let now: @Sendable () -> Date

    init(now: @escaping @Sendable () -> Date) { self.now = now; self.last = now() }

    func touch() { lock.lock(); last = now(); lock.unlock() }
    func secondsSinceLastWrite() -> Double { lock.lock(); defer { lock.unlock() }; return now().timeIntervalSince(last) }
}

/// Accumulates one request's log line and writes it exactly once (metadata only).
nonisolated final class RequestTracker: @unchecked Sendable {
    private let head: ServeRequestHead
    private let started: Date
    private let environment: ServeEnvironment
    private let lock = NSLock()
    private var recorded = false

    var model: String?
    var toolsIgnored = false
    var usage: ServeUsage?

    init(head: ServeRequestHead, started: Date, environment: ServeEnvironment) {
        self.head = head
        self.started = started
        self.environment = environment
    }

    /// Log `status` and return the error as a response (`response: false` = only log it — the
    /// stream already carried the error).
    @discardableResult
    func fail(_ error: ServeError, headers: [String: String] = [:], response: Bool = true) -> ServeReply {
        record(status: error.status)
        var all = ["Content-Type": "application/json"]
        for (name, value) in headers { all[name] = value }
        return .full(ServeResponse(status: error.status, body: error.body(), headers: all))
    }

    @discardableResult
    func succeed(_ reply: ServeReply?) -> ServeReply {
        record(status: 200)
        return reply ?? .full(ServeResponse(status: 200, body: Data()))
    }

    private func record(status: Int) {
        lock.lock()
        defer { lock.unlock() }
        guard !recorded else { return }
        recorded = true
        let finished = environment.now()
        environment.requestLog.record(RequestLogEntry(
            time: started, userAgent: head.userAgent, model: model,
            promptTokens: usage?.promptTokens, completionTokens: usage?.completionTokens,
            durationSeconds: finished.timeIntervalSince(started), status: status, toolsIgnored: toolsIgnored))
    }
}
