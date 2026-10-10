import Foundation

/// `POST /v1/responses` (S3-3): the OpenAI Responses dialect around `ServePipeline`. Decode, hand to the
/// pipeline, encode; errors use the existing OpenAI envelope. **Nothing here logs request or response
/// content.**
nonisolated enum ResponsesRoute {

    static func handle(head: ServeRequestHead, body: Data, environment: ServeEnvironment) async -> ServeReply {
        let tracker = RequestTracker(head: head, started: environment.now(), environment: environment)
        var decoded: OpenAIResponsesDialect.Decoded
        do { decoded = try OpenAIResponsesDialect.decode(body) } catch {
            return tracker.fail(error as? ServeError ?? ServePipeline.internalError)
        }
        tracker.ignoredInputItems = decoded.ignoredInputItems

        // S3-3c. The conversation this response continues = the stored one + this request's input (merged as at decode).
        // A new `instructions` replaces the old ones (they are not stored). An unknown or expired id is the error
        // OpenAI returns, never a silent fresh start. The prepended conversation is part of the prompt that S3-5a
        // admission, the window clamp and the prompt cache all see: `request.messages` is the one list.
        var history: [ServeMessage] = decoded.inputMessages
        if let previous = decoded.previousResponseID {
            guard let stored = environment.responseStore.conversation(for: previous) else {
                return tracker.fail(ResponseStore.notFound(previous))
            }
            history = ServeMessageAssembly.mergingAdjacentRoles(stored + decoded.inputMessages)
            decoded.request.messages = ServeMessageAssembly.mergingAdjacentRoles(decoded.instructionMessages + history)
        } else {
            history = ServeMessageAssembly.mergingAdjacentRoles(history)
        }
        let responseID = OpenAIResponsesDialect.makeID("resp")
        let store = environment.responseStore
        let model = decoded.request.model
        var remember: (@Sendable (ServeMessage) -> Void)?
        if decoded.store {
            let saved = history
            remember = { turn in
                store.save(id: responseID, model: model,
                           conversation: ServeMessageAssembly.mergingAdjacentRoles(saved + [turn]))
            }
        }

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

        guard !prepared.request.stream else {
            return ServePipeline.stream(prepared,
                                        encoder: OpenAIResponsesStreamEncoder(
                                            id: responseID, model: prepared.request.model,
                                            createdAt: Int(environment.now().timeIntervalSince1970),
                                            onCompleted: remember),
                                        tracker: tracker, environment: environment)
        }
        let collected: ServeCollected
        switch await ServePipeline.collect(prepared, tracker: tracker, environment: environment) {
        case .failure(let failure): return tracker.fail(failure)
        case .success(let value): collected = value
        }
        let output = OpenAIResponsesDialect.outputItems(collected: collected)
        let body = OpenAIResponsesDialect.responseBody(
            id: responseID, createdAt: Int(environment.now().timeIntervalSince1970),
            model: prepared.request.model, collected: collected, output: output)
        remember?(OpenAIResponsesDialect.assistantTurn(fromOutput: output))
        return tracker.succeed(.full(ServeResponse(status: 200, body: body,
                                                   headers: ["Content-Type": "application/json"])))
    }
}
