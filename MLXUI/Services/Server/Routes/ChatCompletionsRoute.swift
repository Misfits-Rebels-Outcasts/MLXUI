import Foundation

/// `POST /v1/chat/completions` (S1-3): decode → `ServePipeline` → encode.
///
/// Since S3-2a this route is only the OpenAI dialect's two ends. Model resolution, the `tools` gate,
/// thinking, message assembly, the S2-6 pre-count, queueing and generation are `ServePipeline`'s, so
/// the S3 dialects reuse them instead of forking them.
///
/// - Unknown / unserved model → `404`; line full → `503` + `Retry-After: 2` (decided *before* any
///   byte is written, so it is still a proper status).
/// - Non-streaming runs to completion (it cannot notice a disconnect — S1-A ruling 3).
/// - Streaming heartbeats and cancellation: see `ServePipeline.stream`.
/// - **Nothing here logs request or response content.** The request log has no field for it.
nonisolated enum ChatCompletionsRoute {

    static func handle(head: ServeRequestHead, body: Data, environment: ServeEnvironment) async -> ServeReply {
        let started = environment.now()
        let tracker = RequestTracker(head: head, started: started, environment: environment)

        let decoded: ServeRequest
        do { decoded = try OpenAIChatDialect.decode(body) } catch {
            return tracker.fail(error as? ServeError ?? ServePipeline.internalError)
        }

        let prepared: ServePrepared
        switch await ServePipeline.prepare(decoded, tracker: tracker, environment: environment) {
        case .failure(let failure): return tracker.fail(failure)
        case .success(let value): prepared = value
        }
        let request = prepared.request

        guard request.stream else {
            let collected: ServeCollected
            switch await ServePipeline.collect(prepared, tracker: tracker, environment: environment) {
            case .failure(let failure): return tracker.fail(failure)
            case .success(let value): collected = value
            }
            let body = OpenAIChatDialect.completionBody(
                id: OpenAIChatDialect.makeCompletionID(), created: Int(environment.now().timeIntervalSince1970),
                model: request.model, text: collected.text, reasoning: collected.reasoning,
                toolCalls: collected.toolCalls, finish: collected.finish, usage: collected.usage)
            return tracker.succeed(.full(ServeResponse(status: 200, body: body,
                                                       headers: ["Content-Type": "application/json"])))
        }
        let encoder = OpenAIChatStreamEncoder(
            id: OpenAIChatDialect.makeCompletionID(), created: Int(environment.now().timeIntervalSince1970),
            model: request.model, includeUsage: request.includeUsage)
        return ServePipeline.stream(prepared, encoder: encoder, tracker: tracker, environment: environment)
    }

    // The pipeline's helpers keep their historical names here: they are the OpenAI route's public
    // face as far as the existing tests are concerned.
    static func append(_ delta: ServeToolCallDelta, to calls: inout [ServeToolCall]) {
        ServePipeline.append(delta, to: &calls)
    }

    static func contextLengthMessage(count: Int, limit: Int) -> String {
        ServePipeline.contextLengthMessage(count: count, limit: limit)
    }
}
