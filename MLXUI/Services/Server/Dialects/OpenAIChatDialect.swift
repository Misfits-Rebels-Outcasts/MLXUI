import Foundation

/// The OpenAI Chat Completions wire format ⇄ the neutral core (backlog S1-3).
///
/// Decoding follows https://platform.openai.com/docs/api-reference/chat/create : `content` is a
/// string **or** an array of typed parts (only `text` is accepted — anything else is a `400`
/// naming the part type); unknown request fields are ignored; `tools` / `tool_choice` are
/// ignored and flagged (owner ruling R9, 2026-09-27: "OQ2 go with ignore.").
/// Encoding follows the `chat.completion` / `chat.completion.chunk` objects and the SSE framing
/// (`data: <json>\n\n`, terminated by `data: [DONE]\n\n`).
nonisolated enum OpenAIChatDialect {

    // MARK: - Request

    static func decode(_ body: Data) throws -> ServeRequest {
        let root: Any
        do { root = try JSONSerialization.jsonObject(with: body) } catch {
            throw invalid("The request body isn't valid JSON.", code: "invalid_json")
        }
        guard let object = root as? [String: Any] else {
            throw invalid("The request body must be a JSON object.", code: "invalid_json")
        }

        guard let model = object["model"] as? String, !model.isEmpty else {
            throw invalid("'model' is required.", code: "missing_model", param: "model")
        }
        guard let rawMessages = object["messages"] as? [Any], !rawMessages.isEmpty else {
            throw invalid("'messages' must be a non-empty array.", code: "invalid_messages", param: "messages")
        }

        var messages: [ServeMessage] = []
        for (index, raw) in rawMessages.enumerated() {
            messages.append(try decodeMessage(raw, at: index))
        }

        var request = ServeRequest(model: model, messages: mergingAdjacentRoles(messages))
        request.tools = try decodeTools(object["tools"])
        request.toolChoice = try decodeToolChoice(object["tool_choice"])
        request.toolsRequested = !request.tools.isEmpty
        let reasoning = try decodeReasoning(object)
        request.reasoning = reasoning.reasoning
        request.unknownReasoningEffort = reasoning.unknown
        request.temperature = try number(object["temperature"], "temperature")
        request.topP = try number(object["top_p"], "top_p")
        // `max_completion_tokens` is the newer name; either one limits the reply.
        request.maxTokens = try integer(object["max_completion_tokens"], "max_completion_tokens")
            ?? integer(object["max_tokens"], "max_tokens")
        if let limit = request.maxTokens, limit < 1 {
            throw invalid("'max_tokens' must be at least 1.", code: "invalid_max_tokens", param: "max_tokens")
        }
        request.seed = try integer(object["seed"], "seed")
        request.stop = try decodeStop(object["stop"])
        if let stream = object["stream"] {
            guard let flag = stream as? Bool else {
                throw invalid("'stream' must be true or false.", code: "invalid_stream", param: "stream")
            }
            request.stream = flag
        }
        if let options = object["stream_options"] as? [String: Any] {
            request.includeUsage = (options["include_usage"] as? Bool) ?? false
        }
        return request
    }

    // MARK: - Tools (S2-2)

    /// `tools[]` — each entry kept **verbatim** so the template and the parsers see exactly what
    /// the client sent (https://platform.openai.com/docs/api-reference/chat/create, the `tools`
    /// parameter: an array of `{"type": "function", "function": {name, description, parameters}}`).
    /// Only `function` tools exist in the Chat Completions API; anything else is a `400` naming it
    /// rather than a silent drop.
    private static func decodeTools(_ value: Any?) throws -> [ServeToolSpec] {
        switch value {
        case nil, is NSNull: return []
        case let entries as [Any]:
            return try entries.enumerated().map { index, entry in
                let param = "tools[\(index)]"
                guard let object = entry as? [String: Any] else {
                    throw invalid("Each entry of 'tools' must be an object.", code: "invalid_tools", param: param)
                }
                if let type = object["type"] as? String, type != "function" {
                    throw invalid("Tool type '\(type)' isn't supported; only 'function' is.",
                                  code: "unsupported_tool_type", param: "\(param).type")
                }
                guard let function = object["function"] as? [String: Any],
                      let name = function["name"] as? String, !name.isEmpty else {
                    throw invalid("Each tool needs a 'function.name'.", code: "invalid_tools", param: "\(param).function.name")
                }
                guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
                      let json = String(data: data, encoding: .utf8) else {
                    throw invalid("That tool couldn't be read.", code: "invalid_tools", param: param)
                }
                return ServeToolSpec(name: name, rawJSON: json)
            }
        default:
            throw invalid("'tools' must be an array.", code: "invalid_tools", param: "tools")
        }
    }

    // MARK: - Reasoning (S2-3)

    /// Whether the client asked for thinking — `nil` when it didn't say.
    ///
    /// Two controls, each cited in the journal:
    ///
    /// - `reasoning_effort`
    ///   (https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create):
    ///   the documented set is `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` — the first
    ///   two mean off, the rest mean on. There is **no budget to pass on**, and the string is not
    ///   forwarded to the template (see `MLXChatBackend`: the Qwen3.8/Bonsai templates raise on
    ///   anything outside their own three values, which would kill the render and silently turn
    ///   thinking off);
    /// - `chat_template_kwargs: {"enable_thinking": true|false}` — the **vLLM** convention for
    ///   handing a value to `apply_chat_template`
    ///   (https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server; Qwen's own
    ///   docs show the same call at https://qwen.readthedocs.io/en/latest/deployment/vllm.html). The
    ///   S2-3 brief called this "the vLLM / LM Studio convention"; LM Studio's OpenAI-compatible
    ///   endpoint documents no such key (its reasoning switch is native-REST `/api/v1/chat` only),
    ///   so the citation here is vLLM and Qwen, not LM Studio.
    ///
    /// An explicit `enable_thinking` **wins** when both are sent (S2-3 decision 1). Other keys
    /// inside `chat_template_kwargs` are ignored: they belong to templates this server doesn't know
    /// (vLLM filters undeclared kwargs for the same reason).
    ///
    /// An **unknown** effort string is treated as *on* rather than refused (S2-3b, review call 3) —
    /// a newer OpenAI value should not break a request that clearly asks for thinking — and the
    /// value is handed back for the request log's notice. A non-string is still a `400`: that is a
    /// malformed field, not a value from the future. Returns the on/off decision and the unknown
    /// string (metadata only, never logged with any content).
    private static func decodeReasoning(_ object: [String: Any]) throws -> (reasoning: Bool?, unknown: String?) {
        let (effort, unknown) = try decodeReasoningEffort(object["reasoning_effort"], name: "reasoning_effort")

        guard let kwargs = object["chat_template_kwargs"] else { return (effort, unknown) }
        guard let dictionary = kwargs as? [String: Any] else {
            throw invalid("'chat_template_kwargs' must be an object.",
                          code: "invalid_chat_template_kwargs", param: "chat_template_kwargs")
        }
        switch dictionary["enable_thinking"] {
        case nil, is NSNull: return (effort, unknown)
        case let explicit as Bool: return (explicit, unknown)      // explicit wins (decision 1)
        default:
            throw invalid("'chat_template_kwargs.enable_thinking' must be true or false.",
                          code: "invalid_chat_template_kwargs", param: "chat_template_kwargs.enable_thinking")
        }
    }

    /// S2-3's `reasoning_effort` mapping as one function (S3-3 reuses it for the Responses API's
    /// `reasoning.effort`, with no new rule): `none` / `minimal` → off; `low` … `max` → on; an unknown
    /// string → on, handed back for the log's notice; absent → `nil` (the client didn't ask); a
    /// non-string → `400` naming `name`.
    static func decodeReasoningEffort(_ value: Any?, name: String) throws -> (reasoning: Bool?, unknown: String?) {
        switch value {
        case nil, is NSNull: return (nil, nil)
        case let value as String:
            switch value {
            case "none", "minimal": return (false, nil)
            case "low", "medium", "high", "xhigh", "max": return (true, nil)
            default:
                // Unknown but well-formed: honour it as "on" and let the log say what was seen.
                return (true, value)
            }
        default:
            throw invalid("'\(name)' must be a string.", code: "invalid_reasoning_effort", param: name)
        }
    }

    /// `tool_choice` — `"auto"` / `"none"` / `"required"`, or `{"type":"function","function":{"name":…}}`.
    private static func decodeToolChoice(_ value: Any?) throws -> ServeToolChoice {
        switch value {
        case nil, is NSNull: return .auto
        case let name as String:
            switch name {
            case "auto": return .auto
            case "none": return .none
            case "required": return .required
            default:
                throw invalid("'tool_choice' must be 'auto', 'none', 'required', or a named function.",
                              code: "invalid_tool_choice", param: "tool_choice")
            }
        case let object as [String: Any]:
            guard let function = object["function"] as? [String: Any],
                  let name = function["name"] as? String, !name.isEmpty else {
                throw invalid("A named 'tool_choice' needs a 'function.name'.",
                              code: "invalid_tool_choice", param: "tool_choice.function.name")
            }
            return .function(name)
        default:
            throw invalid("'tool_choice' must be a string or a function object.",
                          code: "invalid_tool_choice", param: "tool_choice")
        }
    }

    /// Adjacent messages with the same role become one, joined by a blank line (owner ruling R21,
    /// 2026-10-02). Flattening turns each tool result into a user turn, and strict-alternation
    /// templates — the Mistral family, which includes the catalog's Ministral 3B — reject two user
    /// turns in a row; a client may also send consecutive same-role messages itself. Applies to every
    /// role; an empty side adds no stray blank line.
    ///
    /// **S2-2 adds two exceptions**, because text is no longer all a message carries: a `tool` turn
    /// is its own result — gluing two of them into one string would feed the template one result
    /// where the client sent two — and an assistant turn carrying `tool_calls` must keep them as a
    /// separate structured turn rather than being folded into plain text.
    ///
    /// The rule itself lives in `ServeMessageAssembly`, beside the `system` hoisting that has to
    /// re-apply it; this entry point stays as the dialect's own name for it.
    static func mergingAdjacentRoles(_ messages: [ServeMessage]) -> [ServeMessage] {
        ServeMessageAssembly.mergingAdjacentRoles(messages)
    }

    /// Tool traffic in the history stays **structured** (S2-2). For a model that can't use tools —
    /// and for the deferral in S1 (owner ruling R20, 2026-10-02) — the route runs the same messages
    /// through `ServeMessageAssembly.flatteningToolTraffic`, which is where S1's `[called …]` /
    /// `[tool result for …]` wording now lives.
    private static func decodeMessage(_ raw: Any, at index: Int) throws -> ServeMessage {
        let param = "messages[\(index)]"
        guard let message = raw as? [String: Any], let roleName = message["role"] as? String else {
            throw invalid("Each message needs a 'role'.", code: "invalid_message", param: param)
        }
        let content = try decodeContent(message["content"], param: "\(param).content")
        switch roleName {
        case "system", "developer":                      // OpenAI renamed system → developer for newer models
            return ServeMessage(role: .system, text: content)
        case "user":
            return ServeMessage(role: .user, text: content)
        case "assistant":
            // An incoming `reasoning_content` is accepted and **dropped** (S2-3 decision 6): it is
            // not part of `content`, and the Qwen templates strip prior thinking out of history
            // themselves, so forwarding it would double it.
            return ServeMessage(role: .assistant, text: content,
                                toolCalls: try decodeToolCalls(message["tool_calls"], param: "\(param).tool_calls"))
        case "tool":
            return ServeMessage(role: .tool, text: content,
                                toolCallID: message["tool_call_id"] as? String,
                                toolName: message["name"] as? String)
        default:
            throw invalid("Message role '\(roleName)' isn't supported; use system, user, assistant or tool.",
                          code: "unsupported_role", param: "\(param).role")
        }
    }

    /// An assistant turn's `tool_calls[]`: `{id, type: "function", function: {name, arguments}}`,
    /// where `arguments` is a JSON **string** (an object is accepted and re-serialized, since some
    /// clients send one).
    private static func decodeToolCalls(_ value: Any?, param: String) throws -> [ServeToolCall] {
        switch value {
        case nil, is NSNull: return []
        case let entries as [Any]:
            return try entries.enumerated().map { index, entry in
                let callParam = "\(param)[\(index)]"
                guard let object = entry as? [String: Any],
                      let function = object["function"] as? [String: Any],
                      let name = function["name"] as? String, !name.isEmpty else {
                    throw invalid("Each tool call needs a 'function.name'.", code: "invalid_tool_call", param: callParam)
                }
                return ServeToolCall(id: object["id"] as? String, name: name,
                                     argumentsJSON: try argumentsString(function["arguments"], param: "\(callParam).function.arguments"))
            }
        default:
            throw invalid("'tool_calls' must be an array.", code: "invalid_tool_call", param: param)
        }
    }

    /// `function.arguments` as a JSON string: passed through when it already is one, encoded when
    /// the client sent an object, `"{}"` when absent (which is what S1's flattening printed).
    private static func argumentsString(_ value: Any?, param: String) throws -> String {
        switch value {
        case nil, is NSNull: return "{}"
        case let text as String: return text.isEmpty ? "{}" : text
        case let object as [String: Any]:
            guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
                  let json = String(data: data, encoding: .utf8) else {
                throw invalid("That tool call's arguments couldn't be read.", code: "invalid_tool_call", param: param)
            }
            return json
        default:
            throw invalid("'function.arguments' must be a JSON string or object.",
                          code: "invalid_tool_call", param: param)
        }
    }

    /// A string, `null` (an assistant turn with no text), or an array of `{type:"text"}` parts.
    private static func decodeContent(_ content: Any?, param: String) throws -> String {
        switch content {
        case nil, is NSNull:
            return ""
        case let text as String:
            return text
        case let parts as [Any]:
            var pieces: [String] = []
            for part in parts {
                guard let part = part as? [String: Any], let type = part["type"] as? String else {
                    throw invalid("Each content part needs a 'type'.", code: "invalid_content_part", param: param)
                }
                guard type == "text" else {
                    throw invalid("Content part type '\(type)' isn't supported; only 'text' is.",
                                  code: "unsupported_content_part", param: param)
                }
                guard let text = part["text"] as? String else {
                    throw invalid("A 'text' content part needs a 'text' string.", code: "invalid_content_part", param: param)
                }
                pieces.append(text)
            }
            return pieces.joined()
        default:
            throw invalid("'content' must be a string or an array of content parts.",
                          code: "invalid_content", param: param)
        }
    }

    private static func decodeStop(_ value: Any?) throws -> [String] {
        switch value {
        case nil, is NSNull: return []
        case let one as String: return one.isEmpty ? [] : [one]
        case let many as [Any]:
            guard many.count <= 4, many.allSatisfy({ $0 is String }) else {
                throw invalid("'stop' must be a string or an array of at most 4 strings.", code: "invalid_stop", param: "stop")
            }
            return many.compactMap { $0 as? String }.filter { !$0.isEmpty }
        default:
            throw invalid("'stop' must be a string or an array of at most 4 strings.", code: "invalid_stop", param: "stop")
        }
    }

    private static func number(_ value: Any?, _ name: String) throws -> Double? {
        switch value {
        case nil, is NSNull: return nil
        case let n as NSNumber where CFGetTypeID(n) != CFBooleanGetTypeID(): return n.doubleValue
        default: throw invalid("'\(name)' must be a number.", code: "invalid_\(name)", param: name)
        }
    }

    private static func integer(_ value: Any?, _ name: String) throws -> Int? {
        guard let double = try number(value, name) else { return nil }
        guard double == double.rounded(), abs(double) < 1e15 else {
            throw invalid("'\(name)' must be an integer.", code: "invalid_\(name)", param: name)
        }
        return Int(double)
    }

    private static func invalid(_ message: String, code: String, param: String? = nil) -> ServeError {
        ServeError(status: 400, message: message, code: code)
    }

    // MARK: - Response encoding

    /// A fresh `chatcmpl-…` id.
    static func makeCompletionID() -> String {
        "chatcmpl-" + String(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(24))
    }

    /// The non-streaming `chat.completion` object (S2-2).
    ///
    /// `content` follows the Chat Completions reference: a turn that is **only** tool calls reports
    /// `"content": null`; a turn with text keeps that text even when it also called a tool. A turn
    /// with no calls keeps S1's shape exactly — a string, empty or not.
    static func completionBody(id: String, created: Int, model: String, text: String,
                               reasoning: String = "",
                               toolCalls: [ServeToolCall] = [],
                               finish: ServeFinishReason, usage: ServeUsage) -> Data {
        let onlyCalls = !toolCalls.isEmpty
            && text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        return encode(Completion(id: id, created: created, model: model,
                                 choices: [.init(index: 0,
                                                 message: .init(role: "assistant",
                                                                content: onlyCalls ? nil : text,
                                                                reasoning_content: reasoning.isEmpty ? nil : reasoning,
                                                                tool_calls: toolCalls.isEmpty ? nil
                                                                    : toolCalls.map(ToolCallWire.init)),
                                                 finish_reason: finish.rawValue)],
                                 usage: .init(usage)))
    }

    /// One `delta.reasoning_content` event (S2-3): the thinking half, on its own, with no
    /// `content` key. Byte-exact — the split happened on the raw text.
    static func reasoningChunk(id: String, created: Int, model: String,
                               text: String, includeUsage: Bool) -> Data {
        chunk(id: id, created: created, model: model,
              delta: .init(role: nil, content: nil, reasoning: text, tool_calls: nil),
              finish: nil, usage: includeUsage ? .some(nil) : nil)
    }

    /// One `delta.tool_calls[]` event: `index` on every fragment, `id` and `name` on the first
    /// (S2-2). The parsers hand back a finished call, so today the arguments arrive in one
    /// fragment — the field stays a fragment because that is the shape the wire format uses.
    static func toolCallChunk(id: String, created: Int, model: String,
                              call: ServeToolCallDelta, includeUsage: Bool) -> Data {
        chunk(id: id, created: created, model: model,
              delta: .init(role: nil, content: nil, reasoning: nil,
                           tool_calls: [.init(index: call.index, id: call.id, name: call.name,
                                              arguments: call.argumentsFragment)]),
              finish: nil, usage: includeUsage ? .some(nil) : nil)
    }

    /// A fresh `call_…` id for a tool call the model produced.
    static func makeToolCallID() -> String {
        "call_" + String(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(24))
    }

    /// The SSE chunk events. `includeUsage` makes every chunk carry `"usage": null` until the last,
    /// as OpenAI's `stream_options.include_usage` does.
    static func roleChunk(id: String, created: Int, model: String, includeUsage: Bool) -> Data {
        chunk(id: id, created: created, model: model, delta: .init(role: "assistant", content: "", reasoning: nil, tool_calls: nil),
              finish: nil, usage: includeUsage ? .some(nil) : nil)
    }

    static func contentChunk(id: String, created: Int, model: String, text: String, includeUsage: Bool) -> Data {
        chunk(id: id, created: created, model: model, delta: .init(role: nil, content: text, reasoning: nil, tool_calls: nil),
              finish: nil, usage: includeUsage ? .some(nil) : nil)
    }

    static func finishChunk(id: String, created: Int, model: String, finish: ServeFinishReason,
                            includeUsage: Bool) -> Data {
        chunk(id: id, created: created, model: model, delta: .init(role: nil, content: nil, reasoning: nil, tool_calls: nil),
              finish: finish.rawValue, usage: includeUsage ? .some(nil) : nil)
    }

    /// The trailing usage chunk: `choices: []` plus `usage` (only when `include_usage` was asked for).
    static func usageChunk(id: String, created: Int, model: String, usage: ServeUsage) -> Data {
        encode(Chunk(id: id, created: created, model: model, choices: [], usage: .some(.init(usage))))
    }

    /// An in-stream error, after the 200 head is already out: `{"error": {...}}`.
    static func streamErrorChunk(_ error: ServeError) -> Data { error.body() }

    // MARK: - SSE framing

    static func sse(_ json: Data) -> [UInt8] {
        Array("data: ".utf8) + Array(json) + Array("\n\n".utf8)
    }
    static let done: [UInt8] = Array("data: [DONE]\n\n".utf8)
    /// An SSE comment line — ignored by every SSE client, but it's a write, so a dead peer is noticed.
    static let keepAlive: [UInt8] = Array(": keep-alive\n\n".utf8)

    // MARK: - Wire shapes

    private struct Usage: Encodable {
        let prompt_tokens: Int
        let completion_tokens: Int
        let total_tokens: Int
        /// PC-3: `usage.prompt_tokens_details.cached_tokens`, as in OpenAI's `CompletionUsage`
        /// (integer). Present **only when something was served from the cache**, so a response from
        /// a server with the cache off or a miss is byte-identical to before.
        struct PromptTokensDetails: Encodable { let cached_tokens: Int }
        let prompt_tokens_details: PromptTokensDetails?
        init(_ usage: ServeUsage) {
            prompt_tokens = usage.promptTokens
            completion_tokens = usage.completionTokens
            total_tokens = usage.totalTokens
            prompt_tokens_details = usage.cachedTokens > 0 ? PromptTokensDetails(cached_tokens: usage.cachedTokens) : nil
        }
    }

    /// One tool call as the wire carries it (S2-2): `{id, type: "function", function: {name, arguments}}`
    /// with `arguments` a JSON **string**, per the Chat Completions reference.
    private struct ToolCallWire: Encodable {
        struct Function: Encodable { let name: String; let arguments: String }
        let id: String
        let type = "function"
        let function: Function

        init(_ call: ServeToolCall) {
            id = call.id ?? OpenAIChatDialect.makeToolCallID()
            function = Function(name: call.name, arguments: call.argumentsJSON)
        }
    }

    private struct Completion: Encodable {
        struct Message: Encodable {
            let role: String
            /// `nil` → `"content": null` (a turn that is only tool calls). The reference sends the
            /// key with a null rather than omitting it.
            let content: String?
            /// The thinking half (S2-3), omitted entirely when the model did none.
            let reasoning_content: String?
            let tool_calls: [ToolCallWire]?
            func encode(to encoder: Encoder) throws {
                var c = encoder.container(keyedBy: Keys.self)
                try c.encode(role, forKey: .role)
                if let content { try c.encode(content, forKey: .content) }
                else { try c.encodeNil(forKey: .content) }
                try c.encodeIfPresent(reasoning_content, forKey: .reasoning_content)
                try c.encodeIfPresent(tool_calls, forKey: .tool_calls)
            }
            private enum Keys: String, CodingKey { case role, content, reasoning_content, tool_calls }
        }
        struct Choice: Encodable { let index: Int; let message: Message; let finish_reason: String }
        let id: String
        let object = "chat.completion"
        let created: Int
        let model: String
        let choices: [Choice]
        let usage: Usage
    }

    private struct Delta: Encodable {
        struct ToolCall: Encodable {
            let index: Int
            let id: String?
            let name: String?
            let arguments: String?
            func encode(to encoder: Encoder) throws {
                var c = encoder.container(keyedBy: Keys.self)
                try c.encode(index, forKey: .index)
                try c.encodeIfPresent(id, forKey: .id)
                // `type` and `function.name` ride along on the fragment that carries the id, the
                // way OpenAI's first fragment does; later fragments are arguments-only.
                if id != nil { try c.encode("function", forKey: .type) }
                if arguments != nil {
                    var f = c.nestedContainer(keyedBy: FunctionKeys.self, forKey: .function)
                    try f.encodeIfPresent(name, forKey: .name)
                    try f.encodeIfPresent(arguments, forKey: .arguments)
                }
            }
            private enum Keys: String, CodingKey { case index, id, type, function }
            private enum FunctionKeys: String, CodingKey { case name, arguments }
        }
        let role: String?
        let content: String?
        let reasoning: String?
        let tool_calls: [ToolCall]?
        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: Keys.self)
            try c.encodeIfPresent(role, forKey: .role)
            try c.encodeIfPresent(content, forKey: .content)
            try c.encodeIfPresent(reasoning, forKey: .reasoning_content)
            try c.encodeIfPresent(tool_calls, forKey: .tool_calls)
        }
        private enum Keys: String, CodingKey { case role, content, reasoning_content, tool_calls }
    }

    private struct Chunk: Encodable {
        struct Choice: Encodable {
            let index: Int
            let delta: Delta
            let finish_reason: String?
            func encode(to encoder: Encoder) throws {
                var c = encoder.container(keyedBy: Keys.self)
                try c.encode(index, forKey: .index)
                try c.encode(delta, forKey: .delta)
                if let finish_reason { try c.encode(finish_reason, forKey: .finish_reason) } else { try c.encodeNil(forKey: .finish_reason) }
            }
            private enum Keys: String, CodingKey { case index, delta, finish_reason }
        }
        let id: String
        let object = "chat.completion.chunk"
        let created: Int
        let model: String
        let choices: [Choice]
        /// `nil` = omit the key; `.some(nil)` = `"usage": null`; `.some(x)` = the usage object.
        let usage: Usage??

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: Keys.self)
            try c.encode(id, forKey: .id)
            try c.encode(object, forKey: .object)
            try c.encode(created, forKey: .created)
            try c.encode(model, forKey: .model)
            try c.encode(choices, forKey: .choices)
            switch usage {
            case .none: break
            case .some(.none): try c.encodeNil(forKey: .usage)
            case .some(.some(let value)): try c.encode(value, forKey: .usage)
            }
        }
        private enum Keys: String, CodingKey { case id, object, created, model, choices, usage }
    }

    private static func chunk(id: String, created: Int, model: String, delta: Delta,
                              finish: String?, usage: Usage??) -> Data {
        encode(Chunk(id: id, created: created, model: model,
                     choices: [.init(index: 0, delta: delta, finish_reason: finish)], usage: usage))
    }

    private static func encode<T: Encodable>(_ value: T) -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return (try? encoder.encode(value)) ?? Data()
    }
}
