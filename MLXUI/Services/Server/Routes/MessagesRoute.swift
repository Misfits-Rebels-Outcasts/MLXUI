import Foundation

/// `POST /v1/messages` (streaming or not) and `POST /v1/messages/count_tokens` (S3-2): the Anthropic dialect's two ends
/// around `ServePipeline`. Decode, hand to the pipeline, encode — there is no pipeline logic here.
///
/// Errors come back in Anthropic's shape (`AnthropicMessagesDialect.errorBody`) with the pipeline's
/// own status codes and `Retry-After`. **Nothing here logs request or response content.**
nonisolated enum MessagesRoute {

    static func handle(head: ServeRequestHead, body: Data, environment: ServeEnvironment) async -> ServeReply {
        let tracker = RequestTracker(head: head, started: environment.now(), environment: environment,
                                     renderError: AnthropicMessagesDialect.errorBody)
        let decoded: ServeRequest
        do { decoded = try AnthropicMessagesDialect.decode(body) } catch {
            return tracker.fail(error as? ServeError ?? ServePipeline.internalError)
        }
        let prepared: ServePrepared
        switch await ServePipeline.prepare(decoded, tracker: tracker, environment: environment) {
        case .failure(let failure): return tracker.fail(failure)
        case .success(let value): prepared = value
        }
        guard !prepared.request.stream else {
            return ServePipeline.stream(prepared,
                                        encoder: AnthropicStreamEncoder(id: AnthropicMessagesDialect.makeMessageID(),
                                                                        model: prepared.request.model),
                                        tracker: tracker, environment: environment)
        }
        let collected: ServeCollected
        switch await ServePipeline.collect(prepared, tracker: tracker, environment: environment) {
        case .failure(let failure): return tracker.fail(failure)
        case .success(let value): collected = value
        }
        let body = AnthropicMessagesDialect.messageBody(id: AnthropicMessagesDialect.makeMessageID(),
                                                        model: prepared.request.model, collected: collected)
        return tracker.succeed(.full(ServeResponse(status: 200, body: body,
                                                   headers: ["Content-Type": "application/json"])))
    }

    static func countTokens(head: ServeRequestHead, body: Data, environment: ServeEnvironment) async -> ServeReply {
        let tracker = RequestTracker(head: head, started: environment.now(), environment: environment,
                                     renderError: AnthropicMessagesDialect.errorBody)
        let decoded: ServeRequest
        do { decoded = try AnthropicMessagesDialect.decode(body, requireMaxTokens: false) } catch {
            return tracker.fail(error as? ServeError ?? ServePipeline.internalError)
        }
        switch await ServePipeline.countPrompt(decoded, tracker: tracker, environment: environment) {
        case .failure(let failure): return tracker.fail(failure)
        case .success(let tokens):
            tracker.usage = ServeUsage(promptTokens: tokens, completionTokens: 0)
            return tracker.succeed(.full(ServeResponse(status: 200, body: AnthropicMessagesDialect.countBody(tokens),
                                                       headers: ["Content-Type": "application/json"])))
        }
    }
}
