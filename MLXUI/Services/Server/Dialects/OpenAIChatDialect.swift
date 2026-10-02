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
        var toolNames: [String: String] = [:]          // tool_call id → function name, for later `tool` results
        for (index, raw) in rawMessages.enumerated() {
            messages.append(try decodeMessage(raw, at: index, toolNames: &toolNames))
        }

        var request = ServeRequest(model: model, messages: messages)
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
        request.toolsIgnored = (object["tools"] != nil && !(object["tools"] is NSNull))
            || (object["tool_choice"] != nil && !(object["tool_choice"] is NSNull))
        return request
    }

    /// Tool traffic in the history is **flattened to plain text** (owner ruling R20, 2026-10-02: don't
    /// 400 — S1 ignores `tools`, but a harness replays its whole conversation, tool turns included):
    ///
    /// - an assistant turn's `tool_calls` become lines `[called <name> with <arguments>]` after any
    ///   text it had;
    /// - a `tool` message becomes a **user** turn `[tool result for <name>: <content>]`, where
    ///   `<name>` is the message's `name`, else the name of the call with that `tool_call_id`,
    ///   else `tool`.
    ///
    /// Wording is the implementer's call, pending owner confirmation; S2 replaces it with real tool
    /// messages for models that have a tool template.
    private static func decodeMessage(_ raw: Any, at index: Int, toolNames: inout [String: String]) throws -> ServeMessage {
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
            var lines: [String] = content.isEmpty ? [] : [content]
            for call in (message["tool_calls"] as? [[String: Any]]) ?? [] {
                let function = call["function"] as? [String: Any]
                let name = (function?["name"] as? String) ?? "tool"
                if let id = call["id"] as? String { toolNames[id] = name }
                lines.append("[called \(name) with \((function?["arguments"] as? String) ?? "{}")]")
            }
            return ServeMessage(role: .assistant, text: lines.joined(separator: "\n"))
        case "tool":
            let name = (message["name"] as? String)
                ?? (message["tool_call_id"] as? String).flatMap { toolNames[$0] } ?? "tool"
            return ServeMessage(role: .user, text: "[tool result for \(name): \(content)]")
        default:
            throw invalid("Message role '\(roleName)' isn't supported; use system, user, assistant or tool.",
                          code: "unsupported_role", param: "\(param).role")
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

    /// The non-streaming `chat.completion` object.
    static func completionBody(id: String, created: Int, model: String, text: String,
                               finish: ServeFinishReason, usage: ServeUsage) -> Data {
        encode(Completion(id: id, created: created, model: model,
                          choices: [.init(index: 0, message: .init(role: "assistant", content: text),
                                          finish_reason: finish.rawValue)],
                          usage: .init(usage)))
    }

    /// The SSE chunk events. `includeUsage` makes every chunk carry `"usage": null` until the last,
    /// as OpenAI's `stream_options.include_usage` does.
    static func roleChunk(id: String, created: Int, model: String, includeUsage: Bool) -> Data {
        chunk(id: id, created: created, model: model, delta: .init(role: "assistant", content: ""),
              finish: nil, usage: includeUsage ? .some(nil) : nil)
    }

    static func contentChunk(id: String, created: Int, model: String, text: String, includeUsage: Bool) -> Data {
        chunk(id: id, created: created, model: model, delta: .init(role: nil, content: text),
              finish: nil, usage: includeUsage ? .some(nil) : nil)
    }

    static func finishChunk(id: String, created: Int, model: String, finish: ServeFinishReason,
                            includeUsage: Bool) -> Data {
        chunk(id: id, created: created, model: model, delta: .init(role: nil, content: nil),
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
        init(_ usage: ServeUsage) {
            prompt_tokens = usage.promptTokens
            completion_tokens = usage.completionTokens
            total_tokens = usage.totalTokens
        }
    }

    private struct Completion: Encodable {
        struct Message: Encodable { let role: String; let content: String }
        struct Choice: Encodable { let index: Int; let message: Message; let finish_reason: String }
        let id: String
        let object = "chat.completion"
        let created: Int
        let model: String
        let choices: [Choice]
        let usage: Usage
    }

    private struct Delta: Encodable {
        let role: String?
        let content: String?
        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: Keys.self)
            try c.encodeIfPresent(role, forKey: .role)
            try c.encodeIfPresent(content, forKey: .content)
        }
        private enum Keys: String, CodingKey { case role, content }
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
