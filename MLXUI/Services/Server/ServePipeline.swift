import Foundation
import os

// S3-2a — the dialect-neutral request pipeline, extracted from `ChatCompletionsRoute` (R34). A dialect
// route is now decode → `ServePipeline.prepare` → `collect` / `stream` → encode; everything between is
// here, once, so the Anthropic and Responses dialects (S3-2, S3-3) cannot fork it. A pure move: the
// OpenAI route's bodies, headers and SSE frames are byte-identical to before.

/// What a route hands back to the transport: a whole response, or an open SSE byte stream.
nonisolated enum ServeReply: Sendable {
    case full(ServeResponse)
    case stream(status: Int, headers: [String: String], body: AsyncStream<[UInt8]>)
}

/// A dialect-neutral refusal: the status, code and message (`ServeError`) plus the `Retry-After` the
/// wire carries where one exists. A dialect renders it in its own error shape; the OpenAI dialect's
/// rendering is `ServeError.body()` and `Retry-After` as a header, exactly as before.
nonisolated struct ServeFailure: Error, Sendable, Equatable {
    var error: ServeError
    var retryAfterSeconds: Int?

    init(_ error: ServeError, retryAfterSeconds: Int? = nil) {
        self.error = error
        self.retryAfterSeconds = retryAfterSeconds
    }

    var headers: [String: String] {
        retryAfterSeconds.map { ["Retry-After": String($0)] } ?? [:]
    }
}

/// A request that cleared every check and holds a place in the queue.
nonisolated struct ServePrepared: Sendable {
    let request: ServeRequest
    let backend: ServeBackend
    let ticket: ServeQueue.Ticket
    /// The S2-6 render's count of the prompt, when the backend could make one (S3-2c). Additive: the
    /// OpenAI route ignores it; the Anthropic stream seeds `message_start.usage.input_tokens` with it.
    var promptTokens: Int? = nil
}

/// Everything a non-streaming generation produced.
nonisolated struct ServeCollected: Sendable {
    var text = ""
    var reasoning = ""
    var toolCalls: [ServeToolCall] = []
    var finish = ServeFinishReason.stop
    var usage = ServeUsage(promptTokens: 0, completionTokens: 0)
}

/// A dialect's streaming encoder: turns neutral events into the bytes of its own wire format. The
/// pipeline owns ordering, the queue, the heartbeat and cancellation; the encoder owns framing (and
/// whatever per-stream state its format needs, e.g. OpenAI's one-time `role` chunk).
nonisolated protocol ServeStreamEncoder: Sendable {
    /// The frames for a `textDelta`, `reasoningDelta` or `toolCallDelta`. Other events never arrive.
    mutating func frames(for event: ServeEvent) -> [[UInt8]]
    /// The closing frames of a stream that ended normally.
    mutating func completion(finish: ServeFinishReason, usage: ServeUsage?) -> [[UInt8]]
    /// The closing frames of a stream that failed after the head was already out.
    func failure(_ error: ServeError) -> [[UInt8]]
    /// A frame that is ignored by every client but proves the peer is still there.
    var keepAlive: [UInt8] { get }
}

nonisolated enum ServePipeline {
    private static let log = Logger(subsystem: "com.ai-browser", category: "local-server")

    // MARK: - Prepare

    /// Model resolution → the `tools` gate → the thinking decision → assembly → the S2-6 pre-count (and
    /// S2-8 template refusal) → a place in the queue. Nothing here writes a byte, so every refusal is
    /// still a proper status.
    ///
    /// `tracker` is told what the request turned out to be (model, tools ignored, thinking on) so the
    /// log line is right whichever step refuses; the caller logs the failure itself.
    static func prepare(_ decoded: ServeRequest, tracker: RequestTracker,
                        environment: ServeEnvironment) async -> Result<ServePrepared, ServeFailure> {
        let resolved: Resolved
        switch await resolve(decoded, tracker: tracker, environment: environment, enforcingContextLimit: true) {
        case .failure(let failure): return .failure(failure)
        case .success(let value): resolved = value
        }
        guard let ticket = await environment.queue.reserve(model: resolved.request.model) else {
            return .failure(ServeFailure(ServeError(status: 503, message: "The server is busy with other requests for this model. Try again shortly.",
                                                    type: "server_error", code: "server_busy"),
                                         retryAfterSeconds: 2))
        }
        return .success(ServePrepared(request: resolved.request, backend: resolved.backend, ticket: ticket,
                                      promptTokens: resolved.promptTokens))
    }

    /// S3-2: the prompt's token count for `/v1/messages/count_tokens` — the same resolution, `tools`
    /// gate, thinking decision, assembly and S2-6 render as a real request, with **no generation and no
    /// queue slot**. Unlike a real request it does not refuse a prompt past the window: a counting call
    /// is how a client finds that out. A template refusal is still a `400`. A backend that can't count
    /// (no tokenizer, Apple Foundation Models below macOS 26.4) is a `501`, never an estimate.
    static func countPrompt(_ decoded: ServeRequest, tracker: RequestTracker,
                            environment: ServeEnvironment) async -> Result<Int, ServeFailure> {
        switch await resolve(decoded, tracker: tracker, environment: environment, enforcingContextLimit: false) {
        case .failure(let failure): return .failure(failure)
        case .success(let resolved):
            guard let count = resolved.promptTokens else {
                return .failure(ServeFailure(ServeError(
                    status: 501, message: "This model's tokens can't be counted by this build.",
                    type: "server_error", code: "count_unavailable")))
            }
            return .success(count)
        }
    }

    private struct Resolved {
        let request: ServeRequest
        let backend: ServeBackend
        /// The S2-6 render's count; `nil` when the backend couldn't make one (fail-open).
        let promptTokens: Int?
    }

    private static func resolve(_ decoded: ServeRequest, tracker: RequestTracker, environment: ServeEnvironment,
                                enforcingContextLimit: Bool) async -> Result<Resolved, ServeFailure> {
        // The pipeline reshapes the request below (tool traffic, `system` placement), so it works on
        // a mutable copy of what the dialect decoded.
        var request = decoded
        tracker.model = request.model
        // S1's shape, provisional: a request that carried `tools` is "ignored" until the model is
        // resolved below and says otherwise (an early 404/503 still logs what the client sent).
        tracker.toolsIgnored = request.toolsRequested

        // Apple Foundation Models that is served but not ready is a 503 with the readiness sentence
        // (S1-4), not a 404: `/v1/models` omits it until ready, but the client named a model the
        // user *is* serving. Readiness comes only through the injected checker (readme T8).
        if request.model == ServedModels.appleFoundationID,
           environment.servedModelIDs().contains(request.model),
           let state = environment.appleFoundationReadiness(), state != .ready {
            return .failure(ServeFailure(ReadinessSentence.error(for: state), retryAfterSeconds: 2))
        }

        let servable = ServedModels.ids(served: environment.servedModelIDs(),
                                        installed: environment.installed(),
                                        appleFoundation: environment.appleFoundationReadiness())
        guard servable.contains(request.model) else {
            return .failure(ServeFailure(ServeError(status: 404, message: "The model '\(request.model)' is not served.",
                                                    code: "model_not_found")))
        }
        // S1-5c: a model bigger than the whole shared budget would load anyway and thrash — say so.
        // (The catalog size of an installed MLX model; Apple Foundation Models isn't in the index.)
        if let entry = environment.installed().entries.first(where: { $0.kind == .llm && $0.hfModelId == request.model }),
           ServeMemory.exceedsBudget(modelRAMGB: entry.ramGB, capacityBytes: environment.memoryCapacityBytes) {
            return .failure(ServeFailure(ServeError(
                status: 503,
                message: ServeMemory.tooLargeSentence(model: request.model, modelRAMGB: entry.ramGB,
                                                      capacityBytes: environment.memoryCapacityBytes),
                type: "server_error", code: "model_too_large")))
        }
        guard let backend = environment.backendFor(request.model) else {
            return .failure(ServeFailure(ServeError(status: 501, message: "The model '\(request.model)' can't be served by this build yet.",
                                                    type: "server_error", code: "not_implemented")))
        }

        // S2-2: the client's `tools` are honoured only for a family whose template renders them and
        // whose output a parser reads (`ToolCallFormatTable`). Otherwise S1 exactly: the history's
        // tool traffic is flattened back to plain text (owner ruling R20, 2026-10-02) and the log
        // says "tools ignored" — the same answer the backend reaches on its own `modelType`, since
        // both ask the same table.
        let modelType = backend.modelType ?? ""
        let acceptsTools = ToolCallFormatTable.acceptsTools(forModelType: modelType)
        // S2-3: thinking is on only when the client asked *and* the family's template reads
        // `enable_thinking`. Nothing is refused either way — a family without thinking simply
        // ignores the request (decision 2) — so the log records what actually happened.
        tracker.reasoningOn = ReasoningTemplateTable.thinkingOn(requested: request.reasoning,
                                                                 forModelType: modelType)
        // S2-3b: an unrecognised `reasoning_effort` is honoured as "on" and named in the log — a
        // parameter value, not request content.
        if let unknown = request.unknownReasoningEffort {
            log.notice("unknown reasoning_effort '\(unknown, privacy: .public)' treated as on")
        }
        tracker.toolsIgnored = request.toolsRequested && !acceptsTools
        if tracker.toolsIgnored { log.notice("tools ignored") }
        if !acceptsTools { request.tools = [] }
        // The request's final shape, in the order the assembly takes: tool traffic settled, then
        // the family's `system` placement (S2-1, owner ruling R22).
        request.messages = ServeMessageAssembly.assembling(
            request.messages, acceptsTools: acceptsTools,
            // S2-8: a family whose template takes one call per assistant turn gets the packed
            // history split, so the client can replay a multi-call turn at all.
            oneCallPerTurn: ToolCallHistoryTable.oneCallPerTurn(forModelType: modelType),
            placement: ToolCallFormatTable.systemPlacement(forModelType: modelType))

        // R23 (S2-6): count the **real** prompt before any response byte is chosen, so an
        // over-limit request is a plain `400` — no 200 head, no first SSE byte, and no queue slot
        // taken (nothing is generated). The message carries both numbers, which is what the
        // documented error does.
        // https://developers.openai.com/api/docs/guides/error-codes (fetched 2026-10-07) for the
        // `invalid_request_error` envelope, and `context_length_exceeded` as the code OpenAI and its
        // clients have long used for this case (Portkey's error library reproduces the original
        // message: "The total number of tokens exceeds the model's maximum context length.").
        var promptTokens: Int?
        do {
            // S2-8: this render is also where a *template refusal* surfaces, which is why it runs even
            // for a model with no declared window — a refusal must be a 400 before the head, in a
            // streaming request as much as outside one.
            let count = try await backend.countPromptTokens(request)
            promptTokens = count
            if enforcingContextLimit, let limit = backend.contextLimit, let count, count > limit {
                return .failure(ServeFailure(ServeError(status: 400,
                                                        message: contextLengthMessage(count: count, limit: limit),
                                                        code: "context_length_exceeded")))
            }
        } catch let refusal as ServeError {
            // The checkpoint's template refused the prompt: the caller's problem to fix, so a plain
            // 400 rather than a 200 that dies mid-SSE.
            return .failure(ServeFailure(refusal))
        } catch {
            // S2-6 fail-open: a count we can't make is not a reason to refuse the request.
        }

        return .success(Resolved(request: request, backend: backend, promptTokens: promptTokens))
    }

    // MARK: - Non-streaming

    /// Non-streaming runs to completion (it cannot notice a disconnect — S1-A ruling 3). On success the
    /// tracker has the usage and the tool-call count; the caller encodes and calls `tracker.succeed`.
    static func collect(_ prepared: ServePrepared, tracker: RequestTracker,
                        environment: ServeEnvironment) async -> Result<ServeCollected, ServeFailure> {
        let queue = environment.queue
        let ticket = prepared.ticket
        do { try await queue.waitTurn(ticket) } catch {
            await queue.release(ticket)
            return .failure(ServeFailure(internalError))
        }
        var collected = ServeCollected()
        var failure: ServeError?
        do {
            for try await event in prepared.backend.generate(prepared.request) {
                switch event {
                case .textDelta(let piece): collected.text += piece
                case .reasoningDelta(let piece): collected.reasoning += piece      // S2-3
                case .usage(let value): collected.usage = value
                case .finish(let reason): collected.finish = reason
                case .error(let error): failure = error
                case .toolCallDelta(let delta): append(delta, to: &collected.toolCalls)
                }
            }
        } catch {
            log.error("generation failed: \(String(describing: type(of: error)), privacy: .public)")
            failure = generationFailed
        }
        await queue.release(ticket)
        if let failure { return .failure(ServeFailure(failure)) }

        tracker.usage = collected.usage
        tracker.toolCalls = collected.toolCalls.count
        return .success(collected)
    }

    /// Fold one `delta.tool_calls` fragment into the running call list. The parsers hand back a
    /// finished call, so today each event is a whole call; keying on `index` keeps this right if a
    /// format ever fragments.
    static func append(_ delta: ServeToolCallDelta, to calls: inout [ServeToolCall]) {
        while calls.count <= delta.index { calls.append(ServeToolCall(id: nil, name: "", argumentsJSON: "")) }
        if let id = delta.id { calls[delta.index].id = id }
        if let name = delta.name { calls[delta.index].name = name }
        calls[delta.index].argumentsJSON += delta.argumentsFragment ?? ""
    }

    // MARK: - Streaming

    /// Streaming writes the encoder's `keepAlive` whenever nothing was written for `keepAliveSeconds`
    /// (S1-A ruling 2), including while queued and during prefill; a client that went away makes the
    /// next write fail, which cancels the generation (S1-0).
    static func stream<Encoder: ServeStreamEncoder>(_ prepared: ServePrepared, encoder initial: Encoder,
                                                    tracker: RequestTracker,
                                                    environment: ServeEnvironment) -> ServeReply {
        let (bytes, continuation) = AsyncStream<[UInt8]>.makeStream()
        let clock = WriteClock(now: environment.now)
        let queue = environment.queue
        let ticket = prepared.ticket
        let request = prepared.request
        let backend = prepared.backend
        let keepAlive = initial.keepAlive
        let write: @Sendable ([UInt8]) -> Void = { chunk in continuation.yield(chunk); clock.touch() }

        let producer = Task {
            defer { continuation.finish() }
            var encoder = initial
            do { try await queue.waitTurn(ticket) } catch {
                await queue.release(ticket)
                _ = tracker.fail(internalError)
                return
            }
            var usage: ServeUsage?
            var finish: ServeFinishReason?
            var failure: ServeError?
            var toolCallCount = 0
            do {
                for try await event in backend.generate(request) {
                    switch event {
                    case .textDelta, .reasoningDelta:
                        for frame in encoder.frames(for: event) { write(frame) }
                    case .toolCallDelta(let delta):
                        if delta.id != nil { toolCallCount += 1 }
                        for frame in encoder.frames(for: event) { write(frame) }
                    case .usage(let value): usage = value
                    case .finish(let reason): finish = reason
                    case .error(let error): failure = error
                    }
                }
            } catch {
                log.error("generation failed: \(String(describing: type(of: error)), privacy: .public)")
                failure = generationFailed
            }
            await queue.release(ticket)

            if Task.isCancelled {                                          // the client went away
                _ = tracker.fail(ServeError(status: 499, message: "Client closed the request.", code: "client_closed"))
                return
            }
            if let failure {
                for frame in encoder.failure(failure) { write(frame) }
                _ = tracker.fail(failure, response: false)
                return
            }
            for frame in encoder.completion(finish: finish ?? .stop, usage: usage) { write(frame) }
            tracker.usage = usage
            tracker.toolCalls = toolCallCount
            _ = tracker.succeed(nil)
        }

        let heartbeat = Task {
            let interval = environment.keepAliveSeconds
            while !Task.isCancelled {
                let idle = clock.secondsSinceLastWrite()
                if idle >= interval {
                    write(keepAlive)
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

    static let internalError = ServeError(status: 500, message: "The server couldn't handle this request.",
                                          type: "server_error", code: "internal_error")
    static let generationFailed = ServeError(status: 500, message: "The model failed to generate a reply.",
                                             type: "server_error", code: "generation_failed")

    /// The `400` for a prompt that doesn't fit (R23, S2-6), carrying the limit **and** the count —
    /// the shape the documented OpenAI error uses ("This model's maximum context length is N tokens.
    /// However, your messages resulted in M tokens."). Implementer's call, pending owner
    /// confirmation: plain numbers, and no `param` (the whole request is at fault, not one field).
    static func contextLengthMessage(count: Int, limit: Int) -> String {
        "This model's maximum context length is \(limit) tokens, but this request's prompt is \(count) tokens. "
            + "Shorten the messages and try again."
    }
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
    /// How this dialect renders an error body. The OpenAI shape is the default; S3's dialects pass
    /// their own.
    private let renderError: @Sendable (ServeError) -> Data
    private let lock = NSLock()
    private var recorded = false

    var model: String?
    var toolsIgnored = false
    /// Whether thinking was actually on for this request (S2-3) — metadata only, like everything
    /// else here (rule 13).
    var reasoningOn = false
    var usage: ServeUsage?
    /// How many tool calls the model produced — **a count only** (rule 13: no tool names, no
    /// arguments, no results).
    var toolCalls = 0

    init(head: ServeRequestHead, started: Date, environment: ServeEnvironment,
         renderError: @escaping @Sendable (ServeError) -> Data = { $0.body() }) {
        self.head = head
        self.started = started
        self.environment = environment
        self.renderError = renderError
    }

    /// Log `status` and return the error as a response (`response: false` = only log it — the
    /// stream already carried the error).
    @discardableResult
    func fail(_ error: ServeError, headers: [String: String] = [:], response: Bool = true) -> ServeReply {
        record(status: error.status)
        var all = ["Content-Type": "application/json"]
        for (name, value) in headers { all[name] = value }
        return .full(ServeResponse(status: error.status, body: renderError(error), headers: all))
    }

    @discardableResult
    func fail(_ failure: ServeFailure) -> ServeReply {
        fail(failure.error, headers: failure.headers)
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
            durationSeconds: finished.timeIntervalSince(started), status: status, toolsIgnored: toolsIgnored,
            reasoningOn: reasoningOn, toolCalls: toolCalls,
            cachedTokens: usage?.cachedTokens ?? 0, cacheFellBack: usage?.cacheFellBack ?? false))
    }
}
