import Foundation

/// `POST /v1/responses` (S3-3): the OpenAI Responses dialect around `ServePipeline`. Decode, hand to the
/// pipeline, encode; errors use the existing OpenAI envelope. **Nothing here logs request or response
/// content.**
nonisolated enum ResponsesRoute {

    static func handle(head: ServeRequestHead, body: Data, environment: ServeEnvironment) async -> ServeReply {
        let tracker = RequestTracker(head: head, started: environment.now(), environment: environment)
        let decoded: OpenAIResponsesDialect.Decoded
        do { decoded = try OpenAIResponsesDialect.decode(body) } catch {
            return tracker.fail(error as? ServeError ?? ServePipeline.internalError)
        }
        tracker.ignoredInputItems = decoded.ignoredInputItems
        let prepared: ServePrepared
        switch await ServePipeline.prepare(decoded.request, tracker: tracker, environment: environment) {
        case .failure(let failure):
            if decoded.droppedHostedTools { tracker.toolsIgnored = true }
            return tracker.fail(failure)
        case .success(let value): prepared = value
        }
        // The pipeline sets `toolsIgnored` from `tools` the *model* can't use; hosted tools dropped at
        // decode are the same note in the same field (R39 #1).
        if decoded.droppedHostedTools { tracker.toolsIgnored = true }

        let collected: ServeCollected
        switch await ServePipeline.collect(prepared, tracker: tracker, environment: environment) {
        case .failure(let failure): return tracker.fail(failure)
        case .success(let value): collected = value
        }
        let body = OpenAIResponsesDialect.responseBody(
            id: OpenAIResponsesDialect.makeID("resp"), createdAt: Int(environment.now().timeIntervalSince1970),
            model: prepared.request.model, collected: collected)
        return tracker.succeed(.full(ServeResponse(status: 200, body: body,
                                                   headers: ["Content-Type": "application/json"])))
    }
}
