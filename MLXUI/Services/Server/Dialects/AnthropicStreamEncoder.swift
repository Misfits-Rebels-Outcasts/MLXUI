import Foundation

/// The Anthropic Messages event stream for `ServePipeline.stream` (S3-2b).
///
/// Frames are `event: <type>\ndata: <json>\n\n`. A stream is `message_start`, then for each content block a
/// `content_block_start`, one or more `content_block_delta`, and a `content_block_stop` (with a running
/// `index` that is the block's place in the final `content` array), then `message_delta` and
/// `message_stop` (https://platform.claude.com/docs/en/build-with-claude/streaming, fetched 2026-10-09).
///
/// * **Blocks.** thinking → `thinking_delta`; text → `text_delta`; tool_use → `input_json_delta` with
///   `partial_json`. The open block is closed whenever the kind changes, and a tool delta carrying an `id`
///   always starts a new `tool_use` block.
/// * **`signature_delta`.** Documented as sent "just before `content_block_stop`" on a thinking block. We
///   can't sign, so it carries `""` — implementer's call, pending owner confirmation. The official Python
///   accumulator copes either way (the block starts with `signature: ""` and a `signature_delta` only
///   overwrites it), so the frame is there to match the documented sequence for stricter clients.
/// * **Usage.** `message_start` carries `input_tokens` (the prepare-time count when the route has one;
///   otherwise 0) and `output_tokens: 0`. `message_delta.usage` is cumulative and **overwrites**: the SDK's
///   `MessageDeltaUsage` may carry `input_tokens` and `cache_read_input_tokens`, so the final
///   cache-adjusted numbers (R34 #10) go there.
/// * **Errors.** `failure` is the `event: error` frame and nothing after it — no `message_stop`.
nonisolated struct AnthropicStreamEncoder: ServeStreamEncoder {
    let id: String
    let model: String
    /// The prompt's token count from the S2-6 render, when the route can supply it; it only seeds
    /// `message_start` (the final figure is `message_delta`'s).
    let promptTokens: Int?

    private enum Block: Equatable { case thinking, text, toolUse }
    private var started = false
    private var nextIndex = 0
    private var open: Block?

    init(id: String, model: String, promptTokens: Int? = nil) {
        self.id = id
        self.model = model
        self.promptTokens = promptTokens
    }

    /// Anthropic's own heartbeat. The official SDK drops `ping` at the SSE layer, so it is safe before
    /// `message_start` too (the pipeline sends one whenever nothing was written for a second).
    var keepAlive: [UInt8] { Self.frame("ping", ["type": "ping"]) }

    mutating func frames(for event: ServeEvent) -> [[UInt8]] {
        var out = startIfNeeded()
        switch event {
        case .reasoningDelta(let piece):
            if open != .thinking {
                out += closeOpen()
                out.append(startBlock(.thinking, ["type": "thinking", "thinking": "", "signature": ""]))
            }
            out.append(delta(["type": "thinking_delta", "thinking": piece]))
        case .textDelta(let piece):
            if open != .text {
                out += closeOpen()
                out.append(startBlock(.text, ["type": "text", "text": ""]))
            }
            out.append(delta(["type": "text_delta", "text": piece]))
        case .toolCallDelta(let call):
            if call.id != nil || open != .toolUse {
                out += closeOpen()
                out.append(startBlock(.toolUse, ["type": "tool_use", "id": call.id ?? OpenAIChatDialect.makeToolCallID(),
                                                 "name": call.name ?? "", "input": [String: Any]()]))
            }
            if let fragment = call.argumentsFragment, !fragment.isEmpty {
                out.append(delta(["type": "input_json_delta", "partial_json": fragment]))
            }
        case .usage, .finish, .error:
            break
        }
        return out
    }

    mutating func completion(finish: ServeFinishReason, usage: ServeUsage?) -> [[UInt8]] {
        var out = startIfNeeded()
        out += closeOpen()
        var cumulative: [String: Any] = ["output_tokens": usage?.completionTokens ?? 0]
        if let usage {
            for (key, value) in AnthropicMessagesDialect.usageObject(usage) where key != "output_tokens" {
                cumulative[key] = value                                  // input_tokens, cache_read_input_tokens
            }
        }
        out.append(Self.frame("message_delta", [
            "type": "message_delta",
            "delta": ["stop_reason": AnthropicMessagesDialect.stopReason(finish), "stop_sequence": NSNull()],
            "usage": cumulative]))
        out.append(Self.frame("message_stop", ["type": "message_stop"]))
        return out
    }

    func failure(_ error: ServeError) -> [[UInt8]] {
        let envelope = (try? JSONSerialization.jsonObject(with: AnthropicMessagesDialect.errorBody(error))) as? [String: Any] ?? [:]
        return [Self.frame("error", envelope)]
    }

    // MARK: - Pieces

    private mutating func startIfNeeded() -> [[UInt8]] {
        guard !started else { return [] }
        started = true
        return [Self.frame("message_start", [
            "type": "message_start",
            "message": ["id": id, "type": "message", "role": "assistant", "model": model,
                        "content": [Any](), "stop_reason": NSNull(), "stop_sequence": NSNull(),
                        "usage": ["input_tokens": promptTokens ?? 0, "output_tokens": 0]]])]
    }

    private mutating func startBlock(_ kind: Block, _ block: [String: Any]) -> [UInt8] {
        open = kind
        return Self.frame("content_block_start", ["type": "content_block_start", "index": nextIndex, "content_block": block])
    }

    private func delta(_ delta: [String: Any]) -> [UInt8] {
        Self.frame("content_block_delta", ["type": "content_block_delta", "index": nextIndex, "delta": delta])
    }

    /// Close the open block (a thinking block gets its `signature_delta` first) and advance the index.
    private mutating func closeOpen() -> [[UInt8]] {
        guard let kind = open else { return [] }
        var out: [[UInt8]] = []
        if kind == .thinking { out.append(delta(["type": "signature_delta", "signature": ""])) }
        out.append(Self.frame("content_block_stop", ["type": "content_block_stop", "index": nextIndex]))
        open = nil
        nextIndex += 1
        return out
    }

    static func frame(_ event: String, _ object: [String: Any]) -> [UInt8] {
        let json = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])) ?? Data()
        return Array("event: \(event)\ndata: ".utf8) + Array(json) + Array("\n\n".utf8)
    }
}
