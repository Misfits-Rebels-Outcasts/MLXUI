import Foundation

/// The OpenAI Responses wire format ⇄ the neutral core (backlog S3-3). Non-streaming only; the event
/// stream is S3-3b and `previous_response_id` is S3-3c.
///
/// Wire facts are from the OpenAI Python SDK's types (`openai` 3.26.1, `openai/types/responses/…`,
/// fetched 2026-10-09: `Response`, `ResponseOutputMessage`, `ResponseFunctionToolCall`,
/// `ResponseReasoningItem`, `ResponseUsage`, `ResponseStatus`), the API reference
/// (https://developers.openai.com/api/reference/resources/responses/methods/create) and the Codex client
/// source (`codex-rs/core/src/client.rs`, `codex-api/src/sse/responses.rs` at `99aa053`), which is the
/// client this dialect has to satisfy. Errors use the existing OpenAI envelope (`ServeError.body()`).
///
/// What Codex sends and this decode copes with (journal `2026-431`): no `instructions` (its base
/// instructions are leading `developer`/`system` input messages), `store:false`, `include`,
/// `prompt_cache_key`, `parallel_tool_calls`, `text`, `reasoning {summary}`, reasoning input items, and
/// the hosted `web_search` tool.
nonisolated enum OpenAIResponsesDialect {

    /// A decoded request, plus the one fact the core has no field for: hosted tools that were
    /// accepted and dropped (the route notes it in the log's tools-ignored metadata, R39 #1).
    struct Decoded {
        var request: ServeRequest
        var droppedHostedTools: Bool
        /// Input items of a type this server doesn't know, dropped (R40): Codex resends its whole history
        /// every turn, so one 400 would end the session. A count only — the route notes it in the log.
        var ignoredInputItems: Int = 0
    }

    /// Hosted tools the model could only call on OpenAI's side, and `namespace` groups. Accepted and dropped
    /// (owner rulings R39 #1, and R55 which amends R39 #2: Codex 0.162.1 sends `multi_agent_v1` as a
    /// `namespace` on every request, so a `400` ended every session). Matched by prefix so dated variants
    /// (`web_search_2025_08_26`, `web_search_preview_2025_03_11`) are covered. The grouped tools are not
    /// flattened into function tools: MCP/connector support is S4.
    static let droppedToolTypePrefixes = ["web_search", "image_generation", "tool_search", "namespace"]

    // MARK: - Request

    static func decode(_ body: Data) throws -> Decoded {
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
        var stream = false
        if let value = object["stream"], !(value is NSNull) {
            guard let flag = value as? Bool else { throw invalid("'stream' must be true or false.", code: "invalid_stream", param: "stream") }
            stream = flag
        }
        if let previous = object["previous_response_id"], !(previous is NSNull) {
            throw invalid("'previous_response_id' is not supported yet.", code: "previous_response_id_not_supported",
                          param: "previous_response_id")
        }

        var messages: [ServeMessage] = []
        switch object["instructions"] {
        case nil, is NSNull: break
        case let text as String: if !text.isEmpty { messages.append(ServeMessage(role: .system, text: text)) }
        default: throw invalid("'instructions' must be a string.", code: "invalid_instructions", param: "instructions")
        }
        let input = try decodeInput(object["input"])
        messages += input.messages
        guard !messages.isEmpty else {
            throw invalid("'input' is required.", code: "missing_input", param: "input")
        }

        var request = ServeRequest(model: model, messages: ServeMessageAssembly.mergingAdjacentRoles(messages))
        request.stream = stream
        let tools = try decodeTools(object["tools"])
        request.tools = tools.specs
        request.toolsRequested = !tools.specs.isEmpty
        request.toolChoice = try decodeToolChoice(object["tool_choice"])
        let effort = try decodeReasoning(object["reasoning"])
        request.reasoning = effort.reasoning
        request.unknownReasoningEffort = effort.unknown
        request.temperature = try number(object["temperature"], "temperature")
        request.topP = try number(object["top_p"], "top_p")
        request.maxTokens = try integer(object["max_output_tokens"], "max_output_tokens")
        request.fillsContextWindow = request.maxTokens == nil          // R41: Codex sends none; see ServeRequest
        if let limit = request.maxTokens, limit < 1 {
            throw invalid("'max_output_tokens' must be at least 1.", code: "invalid_max_output_tokens", param: "max_output_tokens")
        }
        // Accepted and ignored, never a 400 (R38): metadata, user, store, include, text, prompt_cache_key,
        // parallel_tool_calls, service_tier, truncation, client_metadata, safety_identifier, background.
        return Decoded(request: request, droppedHostedTools: tools.dropped, ignoredInputItems: input.ignored)
    }

    // MARK: - Input

    private static func decodeInput(_ value: Any?) throws -> (messages: [ServeMessage], ignored: Int) {
        switch value {
        case nil, is NSNull: return ([], 0)
        case let text as String: return (text.isEmpty ? [] : [ServeMessage(role: .user, text: text)], 0)
        case let items as [Any]:
            var out: [ServeMessage] = []
            var ignored = 0
            for (index, raw) in items.enumerated() {
                if try !decodeItem(raw, at: index, into: &out) { ignored += 1 }
            }
            return (out, ignored)
        default:
            throw invalid("'input' must be a string or an array of items.", code: "invalid_input", param: "input")
        }
    }

    /// - Returns: `false` when the item's type is unknown and it was dropped (R40); a malformed item of a
    ///   known type, an unknown message role, or an item with no type at all is still a `400`.
    @discardableResult
    private static func decodeItem(_ raw: Any, at index: Int, into out: inout [ServeMessage]) throws -> Bool {
        let param = "input[\(index)]"
        guard let item = raw as? [String: Any] else {
            throw invalid("Each input item must be an object.", code: "invalid_input_item", param: param)
        }
        // An item with a `role` and no `type` is a message (the "easy" form).
        let type = item["type"] as? String ?? (item["role"] != nil ? "message" : "")
        switch type {
        case "message":
            guard let role = item["role"] as? String else {
                throw invalid("A message item needs a 'role'.", code: "invalid_input_item", param: param)
            }
            let text = try decodeContent(item["content"], param: "\(param).content")
            switch role {
            case "user": out.append(ServeMessage(role: .user, text: text))
            case "assistant": out.append(ServeMessage(role: .assistant, text: text))
            // Codex's base instructions arrive as `developer` (or `system`) input messages, not as
            // `instructions` (journal 2026-431); the assembly hoists them per family as usual.
            case "system", "developer": out.append(ServeMessage(role: .system, text: text))
            default:
                throw invalid("Message role '\(role)' isn't supported.", code: "unsupported_role", param: "\(param).role")
            }
        case "function_call":
            guard let callID = item["call_id"] as? String, let name = item["name"] as? String, !name.isEmpty else {
                throw invalid("A function_call item needs 'call_id' and 'name'.", code: "invalid_input_item", param: param)
            }
            let call = ServeToolCall(id: callID, name: name, argumentsJSON: try argumentsString(item["arguments"], param: param))
            // Text and calls from one assistant turn arrive as separate items; they are one turn again here
            // (and parallel calls share it), which is how the templates expect to see them.
            if let last = out.last, last.role == .assistant {
                out[out.count - 1].toolCalls.append(call)
            } else {
                out.append(ServeMessage(role: .assistant, text: "", toolCalls: [call]))
            }
        case "function_call_output":
            guard let callID = item["call_id"] as? String else {
                throw invalid("A function_call_output item needs 'call_id'.", code: "invalid_input_item", param: param)
            }
            out.append(ServeMessage(role: .tool, text: try decodeContent(item["output"], param: "\(param).output"),
                                    toolCallID: callID))
        case "reasoning":
            return true     // dropped, with or without `encrypted_content`: never a 400 (R39)
        case "":
            throw invalid("Each input item needs a 'type' (or a 'role').", code: "invalid_input_item", param: param)
        default:
            return false    // an item type we don't know: dropped and counted, never a 400 (R40)
        }
        return true
    }

    /// A string, or content parts: text parts are joined; an image or file becomes the R36 placeholder.
    private static func decodeContent(_ value: Any?, param: String) throws -> String {
        switch value {
        case nil, is NSNull: return ""
        case let text as String: return text
        case let parts as [Any]:
            var text = ""
            for part in parts {
                guard let part = part as? [String: Any], let type = part["type"] as? String else {
                    throw invalid("Each content part needs a 'type'.", code: "invalid_content_part", param: param)
                }
                switch type {
                case "input_text", "output_text", "text":
                    guard let piece = part["text"] as? String else {
                        throw invalid("A '\(type)' part needs a 'text' string.", code: "invalid_content_part", param: param)
                    }
                    text += piece
                case "refusal":
                    text += part["refusal"] as? String ?? ""
                case "input_image", "input_file":
                    if !text.isEmpty, !text.hasSuffix("\n") { text += "\n" }
                    text += AnthropicMessagesDialect.placeholder(forBlockType: type == "input_image" ? "image" : "document")
                default:
                    throw invalid("Content part type '\(type)' isn't supported.", code: "unsupported_content_part", param: param)
                }
            }
            return text
        default:
            throw invalid("Content must be a string or an array of content parts.", code: "invalid_content", param: param)
        }
    }

    private static func argumentsString(_ value: Any?, param: String) throws -> String {
        switch value {
        case nil, is NSNull: return "{}"
        case let text as String: return text.isEmpty ? "{}" : text
        case let object as [String: Any]:
            guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes]),
                  let json = String(data: data, encoding: .utf8) else {
                throw invalid("That function_call's arguments couldn't be read.", code: "invalid_input_item", param: param)
            }
            return json
        default:
            throw invalid("'arguments' must be a JSON string.", code: "invalid_input_item", param: param)
        }
    }

    // MARK: - Tools

    /// `tools[]`: flat `{type:"function", name, description, parameters}` re-shaped into the chat form the
    /// templates read. Hosted `web_search*`, `image_generation` and `tool_search` are accepted and dropped
    /// (R39 #1), and so is `namespace` (R55); everything else — `custom`, `file_search`, `code_interpreter`,
    /// `computer_use*`, `mcp`, … — is a `400` naming the type.
    private static func decodeTools(_ value: Any?) throws -> (specs: [ServeToolSpec], dropped: Bool) {
        switch value {
        case nil, is NSNull: return ([], false)
        case let entries as [Any]:
            var specs: [ServeToolSpec] = []
            var dropped = false
            for (index, entry) in entries.enumerated() {
                let param = "tools[\(index)]"
                guard let tool = entry as? [String: Any], let type = tool["type"] as? String else {
                    throw invalid("Each tool needs a 'type'.", code: "invalid_tools", param: param)
                }
                if droppedToolTypePrefixes.contains(where: { type.hasPrefix($0) }) { dropped = true; continue }
                guard type == "function" else {
                    throw invalid("Tool type '\(type)' isn't supported; only 'function' is (hosted web_search, "
                                  + "image_generation, tool_search and namespace are accepted and ignored).",
                                  code: "unsupported_tool_type", param: "\(param).type")
                }
                guard let name = tool["name"] as? String, !name.isEmpty else {
                    throw invalid("Each function tool needs a 'name'.", code: "invalid_tools", param: "\(param).name")
                }
                var function: [String: Any] = ["name": name]
                if let description = tool["description"] as? String { function["description"] = description }
                function["parameters"] = (tool["parameters"] as? [String: Any]) ?? ["type": "object", "properties": [String: Any]()]
                guard let data = try? JSONSerialization.data(withJSONObject: ["type": "function", "function": function],
                                                             options: [.sortedKeys]),
                      let json = String(data: data, encoding: .utf8) else {
                    throw invalid("That tool couldn't be read.", code: "invalid_tools", param: param)
                }
                specs.append(ServeToolSpec(name: name, rawJSON: json))
            }
            return (specs, dropped)
        default:
            throw invalid("'tools' must be an array.", code: "invalid_tools", param: "tools")
        }
    }

    /// `tool_choice`: `"auto"` / `"none"` / `"required"`, or `{"type":"function","name":…}`.
    private static func decodeToolChoice(_ value: Any?) throws -> ServeToolChoice {
        switch value {
        case nil, is NSNull: return .auto
        case let name as String:
            switch name {
            case "auto": return .auto
            case "none": return .none
            case "required": return .required
            default: throw invalid("'tool_choice' must be 'auto', 'none', 'required', or a named function.",
                                   code: "invalid_tool_choice", param: "tool_choice")
            }
        case let object as [String: Any]:
            guard object["type"] as? String == "function", let name = object["name"] as? String, !name.isEmpty else {
                throw invalid("A named 'tool_choice' must be {\"type\":\"function\",\"name\":…}.",
                              code: "invalid_tool_choice", param: "tool_choice")
            }
            return .function(name)
        default:
            throw invalid("'tool_choice' must be a string or an object.", code: "invalid_tool_choice", param: "tool_choice")
        }
    }

    /// `reasoning.effort` through S2-3's `reasoning_effort` mapping — one function, no new rule. `summary`
    /// does not turn thinking on.
    private static func decodeReasoning(_ value: Any?) throws -> (reasoning: Bool?, unknown: String?) {
        switch value {
        case nil, is NSNull: return (nil, nil)
        case let object as [String: Any]:
            return try OpenAIChatDialect.decodeReasoningEffort(object["effort"], name: "reasoning.effort")
        default:
            throw invalid("'reasoning' must be an object.", code: "invalid_reasoning", param: "reasoning")
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

    // MARK: - Response

    static func makeID(_ prefix: String) -> String {
        prefix + "_" + String(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(24))
    }

    /// The non-streaming `response` object. `output` is, in order: an optional `reasoning` item (the raw
    /// thinking as `reasoning_text` content, with an empty `summary` — it is not a summary), a `message`
    /// with one `output_text`, and one `function_call` per call.
    static func responseBody(id: String, createdAt: Int, model: String, collected: ServeCollected) -> Data {
        var output: [[String: Any]] = []
        let incomplete = collected.finish == .length
        if !collected.reasoning.isEmpty {
            output.append(["id": makeID("rs"), "type": "reasoning", "summary": [Any](),
                           "content": [["type": "reasoning_text", "text": collected.reasoning]]])
        }
        if !collected.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || (collected.toolCalls.isEmpty && collected.reasoning.isEmpty) {
            output.append(["id": makeID("msg"), "type": "message", "role": "assistant",
                           "status": incomplete ? "incomplete" : "completed",
                           "content": [["type": "output_text", "text": collected.text, "annotations": [Any]()]]])
        }
        for call in collected.toolCalls {
            output.append(["id": makeID("fc"), "type": "function_call", "status": "completed",
                           "call_id": call.id ?? OpenAIChatDialect.makeToolCallID(),
                           "name": call.name, "arguments": call.argumentsJSON])
        }
        let body = responseObject(id: id, createdAt: createdAt, model: model, incomplete: incomplete,
                                  output: output, usage: usageObject(collected.usage))
        return (try? JSONSerialization.data(withJSONObject: body, options: [.sortedKeys, .withoutEscapingSlashes])) ?? Data()
    }

    /// The `response` object, whole: the body of a non-streaming reply and the payload of
    /// `response.completed` / `response.incomplete` (S3-3b), so the two cannot drift. `usage: nil` is the
    /// in-progress response of `response.created`.
    static func responseObject(id: String, createdAt: Int, model: String, incomplete: Bool = false,
                               output: [[String: Any]], usage: [String: Any]?, status: String? = nil) -> [String: Any] {
        [
            "id": id, "object": "response", "created_at": createdAt, "model": model,
            "status": status ?? (usage == nil ? "in_progress" : (incomplete ? "incomplete" : "completed")),
            "incomplete_details": incomplete ? ["reason": "max_output_tokens"] : NSNull(),
            "error": NSNull(), "instructions": NSNull(), "metadata": [String: Any](),
            "parallel_tool_calls": true, "tool_choice": "auto", "tools": [Any](),
            "output": output, "usage": usage ?? NSNull()]
    }

    /// `input_tokens` is the whole prompt; `input_tokens_details.cached_tokens` is the part the prompt cache
    /// supplied (inclusive — the core's meaning); the details object is always present, 0 when nothing was cached (R40).
    static func usageObject(_ usage: ServeUsage) -> [String: Any] {
        var object: [String: Any] = ["input_tokens": usage.promptTokens, "output_tokens": usage.completionTokens,
                                     "total_tokens": usage.totalTokens]
        object["input_tokens_details"] = ["cached_tokens": usage.cachedTokens]       // always present, 0 when none (R40)
        return object
    }
}
