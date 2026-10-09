import Foundation

/// The Responses event stream for `ServePipeline.stream` (S3-3b).
///
/// Frames are `event: <type>\ndata: <json>\n\n`, and every event carries `type` and a running
/// `sequence_number`. The order is `response.created` → `response.in_progress` → per output item
/// `output_item.added`, its deltas, its `.done` events, `output_item.done` (with the **whole** item) →
/// `response.completed` (or `response.incomplete` on length) carrying the whole response.
///
/// What the two clients that matter require (journal `2026-433`, from the sources read 2026-10-09):
///
/// * **Codex** (`codex-api/src/sse/responses.rs`) builds its items from `response.output_item.done` alone, takes text
///   from `response.output_text.delta`, reads `response.reasoning_text.delta` only with a `content_index`, needs
///   `response.created` with `response.id`, **fails the turn if the stream closes without `response.completed`**
///   (`response.incomplete` is an error unless its reason is `interrupted`), treats `response.failed` as the error
///   (a bare `error` event is ignored), and never reads `sequence_number`.
/// * **the OpenAI Python SDK's accumulator** needs `response.created` to be the **first** event, and `output_index`,
///   `content_index` and `item_id` consistent between `added`, the deltas and `done`.
///
/// Keep-alive is an SSE comment: any other event written before `response.created` would break the SDK. Codex's
/// idle timer (default 300 s, `stream_idle_timeout_ms`) is **not** reset by a comment, only by a parsed event.
nonisolated struct OpenAIResponsesStreamEncoder: ServeStreamEncoder {
    let id: String
    let model: String
    let createdAt: Int

    private enum Open {
        case reasoning(id: String, text: String)
        case message(id: String, text: String)
        case functionCall(id: String, callID: String, name: String, arguments: String)
    }
    private var started = false
    private var sequence = 0
    private var nextOutputIndex = 0
    private var open: Open?
    private var items: [[String: Any]] = []
    private var heldText = ""

    init(id: String, model: String, createdAt: Int) {
        self.id = id
        self.model = model
        self.createdAt = createdAt
    }

    var keepAlive: [UInt8] { OpenAIChatDialect.keepAlive }

    // MARK: - Events

    mutating func frames(for event: ServeEvent) -> [[UInt8]] {
        var out = startIfNeeded()
        switch event {
        case .reasoningDelta(let piece):
            heldText = ""
            if case .reasoning = open {} else {
                out += closeOpen()
                let itemID = OpenAIResponsesDialect.makeID("rs")
                open = .reasoning(id: itemID, text: "")
                out.append(emit("response.output_item.added", ["output_index": nextOutputIndex,
                    "item": ["id": itemID, "type": "reasoning", "summary": [Any](), "status": "in_progress"] as [String: Any]]))
            }
            if case .reasoning(let itemID, let text) = open {
                open = .reasoning(id: itemID, text: text + piece)
                out.append(emit("response.reasoning_text.delta", ["item_id": itemID, "output_index": nextOutputIndex,
                                                                   "content_index": 0, "delta": piece]))
            }
        case .textDelta(let piece):
            if case .message(let itemID, let text) = open {
                open = .message(id: itemID, text: text + piece)
                out.append(textDelta(piece, itemID: itemID))
            } else {
                // R41: leading whitespace-only text is held until real text arrives (then flushed in order). A
                // call or reasoning that starts first discards it, so no message item is opened for nothing —
                // the same items the non-streaming reply produces.
                heldText += piece
                if heldText.contains(where: { !$0.isWhitespace }) {
                    out += closeOpen()
                    out += openMessage()
                    out += appendText(heldText)
                    heldText = ""
                }
            }
        case .toolCallDelta(let call):
            heldText = ""
            // A delta carrying an `id` is a new call; one without continues the open call's arguments.
            var continuing = false
            if call.id == nil, case .functionCall = open { continuing = true }
            if !continuing {
                out += closeOpen()
                let itemID = OpenAIResponsesDialect.makeID("fc")
                let callID = call.id ?? OpenAIChatDialect.makeToolCallID()
                open = .functionCall(id: itemID, callID: callID, name: call.name ?? "", arguments: "")
                out.append(emit("response.output_item.added", ["output_index": nextOutputIndex,
                    "item": ["id": itemID, "type": "function_call", "status": "in_progress", "call_id": callID,
                             "name": call.name ?? "", "arguments": ""] as [String: Any]]))
            }
            if case .functionCall(let itemID, let callID, let name, let arguments) = open,
               let fragment = call.argumentsFragment, !fragment.isEmpty {
                open = .functionCall(id: itemID, callID: callID, name: name, arguments: arguments + fragment)
                out.append(emit("response.function_call_arguments.delta", ["item_id": itemID, "output_index": nextOutputIndex,
                                                                             "delta": fragment]))
            }
        case .usage, .finish, .error:
            break
        }
        return out
    }

    mutating func completion(finish: ServeFinishReason, usage: ServeUsage?) -> [[UInt8]] {
        var out = startIfNeeded()
        out += closeOpen()
        // Held whitespace is a reply of its own only when nothing else was produced (the non-streaming rule); an
        // otherwise empty reply is the same empty message the non-streaming reply has.
        if items.isEmpty {
            out += openMessage()
            if !heldText.isEmpty { out += appendText(heldText) }
            out += closeOpen()
        }
        heldText = ""
        let incomplete = finish == .length
        let response = OpenAIResponsesDialect.responseObject(
            id: id, createdAt: createdAt, model: model, incomplete: incomplete, output: items,
            usage: OpenAIResponsesDialect.usageObject(usage ?? ServeUsage(promptTokens: 0, completionTokens: 0)))
        out.append(emit(incomplete ? "response.incomplete" : "response.completed", ["response": response]))
        return out
    }

    /// `response.failed` with the error, and nothing after it. If nothing was written yet the stream still opens
    /// with `response.created` (the SDK refuses any other first event). This is a non-mutating requirement, so
    /// the sequence numbers continue from the current counter without storing the advance.
    func failure(_ error: ServeError) -> [[UInt8]] {
        var copy = self
        var out = copy.startIfNeeded()
        var response = OpenAIResponsesDialect.responseObject(id: id, createdAt: createdAt, model: model,
                                                              output: copy.items, usage: nil, status: "failed")
        response["error"] = ["code": error.code, "message": error.message]
        out.append(copy.emit("response.failed", ["response": response]))
        return out
    }

    // MARK: - Pieces

    /// Open a message item (`output_item.added` with empty content, then `content_part.added`).
    private mutating func openMessage() -> [[UInt8]] {
        let itemID = OpenAIResponsesDialect.makeID("msg")
        open = .message(id: itemID, text: "")
        return [emit("response.output_item.added", ["output_index": nextOutputIndex,
                    "item": ["id": itemID, "type": "message", "role": "assistant", "status": "in_progress",
                             "content": [Any]()] as [String: Any]]),
                emit("response.content_part.added", ["item_id": itemID, "output_index": nextOutputIndex,
                    "content_index": 0, "part": ["type": "output_text", "text": "", "annotations": [Any]()] as [String: Any]])]
    }

    private mutating func appendText(_ piece: String) -> [[UInt8]] {
        guard case .message(let itemID, let text) = open else { return [] }
        open = .message(id: itemID, text: text + piece)
        return [textDelta(piece, itemID: itemID)]
    }

    private mutating func textDelta(_ piece: String, itemID: String) -> [UInt8] {
        emit("response.output_text.delta", ["item_id": itemID, "output_index": nextOutputIndex,
                                            "content_index": 0, "delta": piece, "logprobs": [Any]()])
    }

    private mutating func startIfNeeded() -> [[UInt8]] {
        guard !started else { return [] }
        started = true
        let response = OpenAIResponsesDialect.responseObject(id: id, createdAt: createdAt, model: model, output: [], usage: nil)
        return [emit("response.created", ["response": response]), emit("response.in_progress", ["response": response])]
    }

    /// Finish the open item: its `.done` events, then `output_item.done` with the whole item, which is also
    /// what lands in the final response.
    private mutating func closeOpen() -> [[UInt8]] {
        guard let current = open else { return [] }
        var out: [[UInt8]] = []
        let index = nextOutputIndex
        let item: [String: Any]
        switch current {
        case .reasoning(let itemID, let text):
            out.append(emit("response.reasoning_text.done", ["item_id": itemID, "output_index": index,
                                                             "content_index": 0, "text": text]))
            item = ["id": itemID, "type": "reasoning", "summary": [Any](), "status": "completed",
                    "content": [["type": "reasoning_text", "text": text]]]
        case .message(let itemID, let text):
            let part: [String: Any] = ["type": "output_text", "text": text, "annotations": [Any]()]
            out.append(emit("response.output_text.done", ["item_id": itemID, "output_index": index, "content_index": 0,
                                                           "text": text, "logprobs": [Any]()]))
            out.append(emit("response.content_part.done", ["item_id": itemID, "output_index": index,
                                                            "content_index": 0, "part": part]))
            item = ["id": itemID, "type": "message", "role": "assistant", "status": "completed", "content": [part]]
        case .functionCall(let itemID, let callID, let name, let arguments):
            let arguments = arguments.isEmpty ? "{}" : arguments
            out.append(emit("response.function_call_arguments.done", ["item_id": itemID, "output_index": index,
                                                                       "name": name, "arguments": arguments]))
            item = ["id": itemID, "type": "function_call", "status": "completed", "call_id": callID,
                    "name": name, "arguments": arguments]
        }
        out.append(emit("response.output_item.done", ["output_index": index, "item": item]))
        items.append(item)
        open = nil
        nextOutputIndex += 1
        return out
    }

    /// One frame: the event name is the payload's `type`, and every event gets the next `sequence_number`.
    private mutating func emit(_ type: String, _ fields: [String: Any]) -> [UInt8] {
        var payload = fields
        payload["type"] = type
        payload["sequence_number"] = sequence
        sequence += 1
        let json = (try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys, .withoutEscapingSlashes])) ?? Data()
        return Array("event: \(type)\ndata: ".utf8) + Array(json) + Array("\n\n".utf8)
    }
}
