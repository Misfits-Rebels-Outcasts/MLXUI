import Foundation

/// The Anthropic Messages wire format ⇄ the neutral core (backlog S3-2). The event stream is
/// `AnthropicStreamEncoder` (S3-2b).
///
/// Wire facts are from https://platform.claude.com/docs/en/api/messages, `…/api/messages-count-tokens`,
/// `…/build-with-claude/prompt-caching` and `…/api/errors` (fetched 2026-10-09):
///
/// * `max_tokens` is a required body parameter; `system` is a string or `[text blocks]`; `tools[]` are
///   `{name, description, input_schema}`; `tool_choice` is `auto` / `any` / `tool` (+ `name`) / `none`;
/// * a `tool_result` block is `{tool_use_id, content, is_error}` and `tool_use` is `{id, name, input}`;
/// * `usage.input_tokens` **excludes** tokens read from the cache; `cache_read_input_tokens` is the
///   reused part (the core's `cachedTokens` is *part of* `promptTokens`, so the dialect subtracts);
/// * errors are `{"type":"error","error":{"type","message"}}`.
///
/// Everything else the request can carry (`top_k`, `metadata`, `cache_control`, `service_tier`,
/// `disable_parallel_tool_use`, the `anthropic-*` headers, an API key) is accepted and ignored — never
/// a `400` (owner ruling R35, P1).
nonisolated enum AnthropicMessagesDialect {

    /// Whether `path` is one of this dialect's endpoints (or under it), which decides the error shape
    /// even for a request the guard turns away before routing.
    static func owns(_ path: String) -> Bool {
        path == "/v1/messages" || path.hasPrefix("/v1/messages/")
    }

    // MARK: - Request

    /// - Parameter requireMaxTokens: `/v1/messages` needs it; `/v1/messages/count_tokens` has none.
    static func decode(_ body: Data, requireMaxTokens: Bool = true) throws -> ServeRequest {
        let root: Any
        do { root = try JSONSerialization.jsonObject(with: body) } catch {
            throw invalid("The request body isn't valid JSON.", code: "invalid_json")
        }
        guard let object = root as? [String: Any] else {
            throw invalid("The request body must be a JSON object.", code: "invalid_json")
        }
        guard let model = object["model"] as? String, !model.isEmpty else {
            throw invalid("model: Field required", code: "missing_model")
        }
        guard let rawMessages = object["messages"] as? [Any], !rawMessages.isEmpty else {
            throw invalid("messages: Field required and must be a non-empty array", code: "invalid_messages")
        }

        var messages: [ServeMessage] = []
        if let system = try decodeSystem(object["system"]), !system.isEmpty {
            messages.append(ServeMessage(role: .system, text: system))
        }
        // S3-8A: a `system` entry that comes *after* the first user/assistant entry stays where the client put it
        // (see `inPlaceReminder`); only the leading ones are system messages for the assembly to place.
        var sawConversation = false
        for (index, raw) in rawMessages.enumerated() {
            let decoded = try decodeMessage(raw, at: index)
            if sawConversation {
                messages.append(contentsOf: decoded.map { $0.role == .system ? inPlaceReminder($0) : $0 })
            } else {
                messages.append(contentsOf: decoded)
            }
            if (raw as? [String: Any])?["role"] as? String != "system" { sawConversation = true }
        }

        var request = ServeRequest(model: model, messages: ServeMessageAssembly.mergingAdjacentRoles(messages))
        request.tools = try decodeTools(object["tools"])
        request.toolsRequested = !request.tools.isEmpty
        request.toolChoice = try decodeToolChoice(object["tool_choice"])
        request.reasoning = decodeThinking(object["thinking"])
        request.temperature = try number(object["temperature"], "temperature")
        request.topP = try number(object["top_p"], "top_p")
        request.maxTokens = try integer(object["max_tokens"], "max_tokens")
        if requireMaxTokens, request.maxTokens == nil {
            throw invalid("max_tokens: Field required", code: "missing_max_tokens")
        }
        if let limit = request.maxTokens, limit < 1 {
            // The reference allows 0 ("populate the prompt cache without generating"); this server
            // generates or refuses. Implementer's call, pending owner confirmation.
            throw invalid("max_tokens: must be at least 1", code: "invalid_max_tokens")
        }
        request.stop = try decodeStopSequences(object["stop_sequences"])
        if let stream = object["stream"] {
            guard let flag = stream as? Bool else { throw invalid("stream: must be true or false", code: "invalid_stream") }
            request.stream = flag
        }
        return request
    }

    /// S3-8A (amends R43). Claude Code appends a `system` entry to `messages` on every turn and replays the earlier
    /// ones unchanged, in place (journal `2026-452`). Hoisting them all into one leading system message made that
    /// message differ on every request, which moved every later token and left an exact-match (hybrid) prompt
    /// cache nothing of the previous turn to reuse. A mid-conversation entry therefore becomes a **user** turn at
    /// its own position, wrapped in the `<system-reminder>` form Claude Code itself uses for such notes, so the
    /// model can tell it from the user's words; the assembly's same-role merge folds it into an adjacent user turn.
    private static func inPlaceReminder(_ message: ServeMessage) -> ServeMessage {
        var reminder = message
        reminder.role = .user
        reminder.text = "<system-reminder>\n\(message.text)\n</system-reminder>"
        return reminder
    }

    /// `system`: a string, or `[{type:"text", text}]` blocks (joined with a blank line).
    private static func decodeSystem(_ value: Any?) throws -> String? {
        switch value {
        case nil, is NSNull: return nil
        case let text as String: return text
        case let blocks as [Any]:
            return try blocks.map { block -> String in
                guard let block = block as? [String: Any], block["type"] as? String == "text",
                      let text = block["text"] as? String else {
                    throw invalid("system: only text blocks are supported", code: "unsupported_content_block")
                }
                return text
            }.filter { !$0.isEmpty }.joined(separator: "\n\n")
        default:
            throw invalid("system: must be a string or an array of text blocks", code: "invalid_system")
        }
    }

    /// One Anthropic message becomes one or more neutral ones, **in order**: a user turn
    /// `[tool_result, tool_result, text]` is two `.tool` messages and then the user's text.
    private static func decodeMessage(_ raw: Any, at index: Int) throws -> [ServeMessage] {
        let param = "messages.\(index)"
        guard let message = raw as? [String: Any], let role = message["role"] as? String else {
            throw invalid("\(param): each message needs a role", code: "invalid_message")
        }
        // R43: Claude Code puts `system` entries *inside* `messages` (the `mid-conversation-system` beta; seen at
        // messages[1] and again after a tool_result, journal 2026-436). They become `.system` messages, and the
        // assembly places them per family as usual. Only an unknown role is refused.
        guard role == "user" || role == "assistant" || role == "system" else {
            throw invalid("\(param).role: must be 'user', 'assistant' or 'system'", code: "unsupported_role")
        }
        let blocks: [[String: Any]]
        switch message["content"] {
        case nil, is NSNull: blocks = []
        case let text as String: blocks = [["type": "text", "text": text]]
        case let list as [Any]:
            blocks = try list.map {
                guard let block = $0 as? [String: Any], block["type"] is String else {
                    throw invalid("\(param).content: each block needs a type", code: "invalid_content_block")
                }
                return block
            }
        default:
            throw invalid("\(param).content: must be a string or an array of content blocks", code: "invalid_content")
        }

        if role == "system" {
            let text = try blocks.map { block -> String in
                guard block["type"] as? String == "text", let piece = block["text"] as? String else {
                    throw invalid("\(param).content: a system message may only contain text", code: "unsupported_content_block")
                }
                return piece                                    // `cache_control` and the like are ignored
            }.joined(separator: "\n\n")
            return text.isEmpty ? [] : [ServeMessage(role: .system, text: text)]
        }

        var out: [ServeMessage] = []
        var text = ""
        var calls: [ServeToolCall] = []
        func flushText() {
            if !text.isEmpty { out.append(ServeMessage(role: .user, text: text)); text = "" }
        }
        for block in blocks {
            let type = block["type"] as? String ?? ""
            switch (role, type) {
            case (_, "text"):
                guard let piece = block["text"] as? String else {
                    throw invalid("\(param).content: a text block needs a 'text' string", code: "invalid_content_block")
                }
                text += piece
            case ("assistant", "tool_use"):
                guard let id = block["id"] as? String, let name = block["name"] as? String, !name.isEmpty else {
                    throw invalid("\(param).content: a tool_use block needs 'id' and 'name'", code: "invalid_content_block")
                }
                calls.append(ServeToolCall(id: id, name: name, argumentsJSON: try inputJSON(block["input"], param: param)))
            case ("assistant", "thinking"), ("assistant", "redacted_thinking"):
                continue                       // dropped; the signature is never checked (R34 #3)
            case ("user", "image"), ("user", "document"):
                // R36: a placeholder, never a 400 — an agent resends the whole history every turn, so one
                // refused block would end the session for good.
                if !text.isEmpty, !text.hasSuffix("\n") { text += "\n" }
                text += placeholder(forBlockType: type)
            case ("user", "tool_result"):
                guard let id = block["tool_use_id"] as? String else {
                    throw invalid("\(param).content: a tool_result block needs 'tool_use_id'", code: "invalid_content_block")
                }
                flushText()                    // text that came first stays before this result
                var result = try resultText(block["content"], param: param)
                if block["is_error"] as? Bool == true { result = "Error: " + result }       // R34 #6
                out.append(ServeMessage(role: .tool, text: result, toolCallID: id))
            default:
                throw invalid("\(param).content: block type '\(type)' isn't supported", code: "unsupported_content_block")
            }
        }
        if role == "assistant" {
            if !text.isEmpty || !calls.isEmpty || out.isEmpty {
                out.append(ServeMessage(role: .assistant, text: text, toolCalls: calls))
            }
        } else {
            flushText()
            if out.isEmpty { out.append(ServeMessage(role: .user, text: "")) }
        }
        return out
    }

    /// What stands in for an image or document block this model can't read (owner ruling R36).
    static func placeholder(forBlockType type: String) -> String {
        type == "image" ? "[image omitted: this model can't see images]"
                        : "[document omitted: this model can't read documents]"
    }

    /// A `tool_result`'s `content`: a string, or text blocks (anything else — images — is refused).
    private static func resultText(_ value: Any?, param: String) throws -> String {
        switch value {
        case nil, is NSNull: return ""
        case let text as String: return text
        case let blocks as [Any]:
            return try blocks.map { block -> String in
                guard let block = block as? [String: Any], let type = block["type"] as? String else {
                    throw invalid("\(param).content: a tool_result block needs a type", code: "unsupported_content_block")
                }
                if type == "image" || type == "document" { return placeholder(forBlockType: type) }     // R36
                guard type == "text", let text = block["text"] as? String else {
                    throw invalid("\(param).content: a tool_result may only contain text, images or documents",
                                  code: "unsupported_content_block")
                }
                return text
            }.joined()
        default:
            throw invalid("\(param).content: a tool_result's content must be a string or text blocks",
                          code: "invalid_content_block")
        }
    }

    /// A `tool_use`'s `input` object as the JSON string the core carries.
    private static func inputJSON(_ value: Any?, param: String) throws -> String {
        switch value {
        case nil, is NSNull: return "{}"
        case let object as [String: Any]:
            guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes]),
                  let json = String(data: data, encoding: .utf8) else {
                throw invalid("\(param).content: a tool_use input couldn't be read", code: "invalid_content_block")
            }
            return json
        default:
            throw invalid("\(param).content: a tool_use 'input' must be an object", code: "invalid_content_block")
        }
    }

    /// `tools[]`: `{name, description?, input_schema}`, re-shaped into the OpenAI chat form the chat
    /// templates and the tool-call parsers read (`ServeToolSpec`). Anthropic's own server tools
    /// (`web_search_…`, `bash_…`) have no `input_schema` and a `type` other than `custom` — a `400`
    /// naming the type rather than a silent drop.
    private static func decodeTools(_ value: Any?) throws -> [ServeToolSpec] {
        switch value {
        case nil, is NSNull: return []
        case let entries as [Any]:
            return try entries.enumerated().map { index, entry in
                let param = "tools.\(index)"
                guard let tool = entry as? [String: Any] else {
                    throw invalid("\(param): must be an object", code: "invalid_tools")
                }
                if let type = tool["type"] as? String, type != "custom" {
                    throw invalid("\(param).type: tool type '\(type)' isn't supported; only custom tools are",
                                  code: "unsupported_tool_type")
                }
                guard let name = tool["name"] as? String, !name.isEmpty else {
                    throw invalid("\(param).name: Field required", code: "invalid_tools")
                }
                guard let schema = tool["input_schema"] as? [String: Any] else {
                    throw invalid("\(param).input_schema: Field required", code: "invalid_tools")
                }
                var function: [String: Any] = ["name": name, "parameters": schema]
                if let description = tool["description"] as? String { function["description"] = description }
                guard let data = try? JSONSerialization.data(withJSONObject: ["type": "function", "function": function],
                                                             options: [.sortedKeys]),
                      let json = String(data: data, encoding: .utf8) else {
                    throw invalid("\(param): couldn't be read", code: "invalid_tools")
                }
                return ServeToolSpec(name: name, rawJSON: json)
            }
        default:
            throw invalid("tools: must be an array", code: "invalid_tools")
        }
    }

    private static func decodeToolChoice(_ value: Any?) throws -> ServeToolChoice {
        switch value {
        case nil, is NSNull: return .auto
        case let choice as [String: Any]:
            switch choice["type"] as? String {
            case "auto": return .auto
            case "any": return .required
            case "none": return .none
            case "tool":
                guard let name = choice["name"] as? String, !name.isEmpty else {
                    throw invalid("tool_choice.name: Field required", code: "invalid_tool_choice")
                }
                return .function(name)
            default:
                throw invalid("tool_choice.type: must be auto, any, tool or none", code: "invalid_tool_choice")
            }
        default:
            throw invalid("tool_choice: must be an object", code: "invalid_tool_choice")
        }
    }

    /// `thinking.type`: only `enabled` turns thinking on; `disabled`, `adaptive`, `between_tools`, anything else,
    /// or no field → off (R30; owner ruling R44, which amends R34 #4: Claude Code sends `adaptive` with
    /// `display: "omitted"` on every call, and it asked for no thinking text). `budget_tokens` and `display` are
    /// ignored. `nil` when the client didn't send a `thinking` object.
    private static func decodeThinking(_ value: Any?) -> Bool? {
        guard let thinking = value as? [String: Any] else { return nil }
        switch thinking["type"] as? String {
        case "enabled": return true
        default: return false
        }
    }

    private static func decodeStopSequences(_ value: Any?) throws -> [String] {
        switch value {
        case nil, is NSNull: return []
        case let many as [Any]:
            guard many.allSatisfy({ $0 is String }) else {
                throw invalid("stop_sequences: must be an array of strings", code: "invalid_stop")
            }
            return many.compactMap { $0 as? String }.filter { !$0.isEmpty }
        default:
            throw invalid("stop_sequences: must be an array of strings", code: "invalid_stop")
        }
    }

    private static func number(_ value: Any?, _ name: String) throws -> Double? {
        switch value {
        case nil, is NSNull: return nil
        case let n as NSNumber where CFGetTypeID(n) != CFBooleanGetTypeID(): return n.doubleValue
        default: throw invalid("\(name): must be a number", code: "invalid_\(name)")
        }
    }

    private static func integer(_ value: Any?, _ name: String) throws -> Int? {
        guard let double = try number(value, name) else { return nil }
        guard double == double.rounded(), abs(double) < 1e15 else {
            throw invalid("\(name): must be an integer", code: "invalid_\(name)")
        }
        return Int(double)
    }

    private static func invalid(_ message: String, code: String) -> ServeError {
        ServeError(status: 400, message: message, code: code)
    }

    // MARK: - Response

    /// A fresh `msg_…` id.
    static func makeMessageID() -> String {
        "msg_" + String(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(24))
    }

    /// `stop` → `end_turn`, `length` → `max_tokens`, `toolCalls` → `tool_use`. A matched stop
    /// sequence is **not** distinguished (`stop_sequence` stays null): the core doesn't carry which
    /// string matched — a known gap, owner ruling R34 #2.
    static func stopReason(_ finish: ServeFinishReason) -> String {
        switch finish {
        case .stop: return "end_turn"
        case .length: return "max_tokens"
        case .toolCalls: return "tool_use"
        }
    }

    /// The usage object. `input_tokens` is the prompt **minus** what the prompt cache supplied (the
    /// Anthropic meaning); the reused part is `cache_read_input_tokens`, omitted when 0 (R34 #10).
    static func usageObject(_ usage: ServeUsage) -> [String: Any] {
        let cached = min(max(usage.cachedTokens, 0), usage.promptTokens)
        var object: [String: Any] = ["input_tokens": usage.promptTokens - cached,
                                     "output_tokens": usage.completionTokens]
        if cached > 0 { object["cache_read_input_tokens"] = cached }
        return object
    }

    /// The non-streaming `message` object.
    static func messageBody(id: String, model: String, collected: ServeCollected) -> Data {
        var content: [[String: Any]] = []
        if !collected.reasoning.isEmpty {
            // We can't sign a thinking block, and our decode never checks one (R34 #3): `""`.
            content.append(["type": "thinking", "thinking": collected.reasoning, "signature": ""])
        }
        if !collected.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            content.append(["type": "text", "text": collected.text])
        }
        for call in collected.toolCalls {
            content.append(["type": "tool_use",
                            "id": call.id ?? OpenAIChatDialect.makeToolCallID(),
                            "name": call.name,
                            "input": inputObject(call.argumentsJSON)])
        }
        if content.isEmpty { content.append(["type": "text", "text": ""]) }       // never an empty array
        return encode(["id": id, "type": "message", "role": "assistant", "model": model,
                       "content": content,
                       "stop_reason": stopReason(collected.finish),
                       "stop_sequence": NSNull(),
                       "usage": usageObject(collected.usage)])
    }

    /// `{"input_tokens": n}` for `/v1/messages/count_tokens`.
    static func countBody(_ tokens: Int) -> Data { encode(["input_tokens": tokens]) }

    /// A tool call's arguments string as the JSON **object** `input` must be (`{}` when it isn't one).
    private static func inputObject(_ json: String) -> [String: Any] {
        guard let data = json.data(using: .utf8),
              let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return [:] }
        return object
    }

    // MARK: - Errors

    /// `{"type":"error","error":{"type","message"}}`. Our status picks the type; the status code
    /// itself is kept as it was (`503` stays `503`). Names from https://platform.claude.com/docs/en/api/errors.
    static func errorBody(_ error: ServeError) -> Data {
        encode(["type": "error", "error": ["type": errorType(forStatus: error.status), "message": error.message]])
    }

    static func errorType(forStatus status: Int) -> String {
        switch status {
        case 401: return "authentication_error"
        case 403: return "permission_error"
        case 404: return "not_found_error"
        case 413: return "request_too_large"
        case 429: return "rate_limit_error"
        case 503, 529: return "overloaded_error"
        case 504: return "timeout_error"
        case 500...: return "api_error"
        default: return "invalid_request_error"           // 400, and "other 4XX" per the reference
        }
    }

    private static func encode(_ object: [String: Any]) -> Data {
        (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])) ?? Data()
    }
}
