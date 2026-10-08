import Testing
import Foundation
@testable import MLXUI

// S3-3 — `POST /v1/responses` (non-streaming; no store, no streaming yet).

private let model = "mlx-community/Qwen3-4B-4bit"

private func environment(_ backend: ServeBackend?, queue: ServeQueue = ServeQueue(),
                         log: RequestLog = RequestLog()) -> ServeEnvironment {
    ServeEnvironment(
        servedModelIDs: { [model] },
        installed: { InstalledModelIndex(entries: [
            .init(id: "mlx-community--Qwen3-4B-4bit", hfModelId: model, kind: .llm, ramGB: 3, downloadSizeGB: 2)]) },
        appleFoundationReadiness: { nil },
        created: 1_700_000_000,
        backendFor: { $0 == model ? backend : nil },
        queue: queue, requestLog: log)
}

private func send(_ json: String, env: ServeEnvironment, path: String = "/v1/responses") async -> ServeResponse? {
    let body = Data(json.utf8)
    let head = ServeRequestHead(method: "POST", path: path, host: "127.0.0.1:1212", origin: nil, transferEncoding: nil,
                                contentLength: String(body.count), userAgent: "codex/test")
    if case .full(let response) = await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212),
                                                         environment: env, body: { body }) { return response }
    return nil
}

private func object(_ data: Data) -> [String: Any]? { (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] }

private func req(_ fields: String) -> String { #"{"model":"\#(model)",\#(fields)}"# }

private func decode(_ fields: String) throws -> OpenAIResponsesDialect.Decoded {
    try OpenAIResponsesDialect.decode(Data(req(fields).utf8))
}

private func refused(_ fields: String) -> ServeError? {
    do { _ = try decode(fields); return nil } catch { return error as? ServeError }
}

// MARK: - Decode

struct ResponsesDecodeTests {
    @Test func aStringInputIsOneUserMessageAndInstructionsALeadingSystemOne() throws {
        let plain = try decode(#""input":"hello""#)
        #expect(plain.request.messages == [ServeMessage(role: .user, text: "hello")])
        let withInstructions = try decode(#""instructions":"Be brief.","input":"hello""#)
        #expect(withInstructions.request.messages == [ServeMessage(role: .system, text: "Be brief."),
                                                      ServeMessage(role: .user, text: "hello")])
    }

    /// Codex's base instructions arrive as `developer` input messages, not as `instructions` (journal 2026-431).
    @Test func developerAndSystemMessagesBecomeSystemMessages() throws {
        let decoded = try decode(#"""
        "input":[{"type":"message","role":"developer","content":[{"type":"input_text","text":"BASE"}]},
                 {"role":"system","content":"SYS"},
                 {"type":"message","role":"user","content":[{"type":"input_text","text":"hi "},{"type":"input_text","text":"there"}]}]
        """#)
        #expect(decoded.request.messages == [ServeMessage(role: .system, text: "BASE\n\nSYS"),
                                             ServeMessage(role: .user, text: "hi there")])
    }

    @Test func reasoningItemsAreDroppedWithOrWithoutEncryptedContent() throws {
        let decoded = try decode(#"""
        "input":[{"type":"message","role":"user","content":"q"},
                 {"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"gAAAA…"},
                 {"type":"reasoning","summary":[{"type":"summary_text","text":"s"}]},
                 {"type":"message","role":"assistant","content":[{"type":"output_text","text":"a"}]}]
        """#)
        #expect(decoded.request.messages == [ServeMessage(role: .user, text: "q"), ServeMessage(role: .assistant, text: "a")])
    }

    @Test func functionCallAndOutputItemsRoundTripIntoToolTraffic() throws {
        let decoded = try decode(#"""
        "input":[{"type":"message","role":"user","content":"weather?"},
                 {"type":"message","role":"assistant","content":[{"type":"output_text","text":"Checking."}]},
                 {"type":"reasoning","summary":[]},
                 {"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Oslo\"}"},
                 {"type":"function_call","call_id":"call_2","name":"get_time","arguments":"{}"},
                 {"type":"function_call_output","call_id":"call_1","output":"3C"},
                 {"type":"function_call_output","call_id":"call_2","output":[{"type":"input_text","text":"noon"}]}]
        """#)
        #expect(decoded.request.messages == [
            ServeMessage(role: .user, text: "weather?"),
            ServeMessage(role: .assistant, text: "Checking.", toolCalls: [
                ServeToolCall(id: "call_1", name: "get_weather", argumentsJSON: #"{"city":"Oslo"}"#),
                ServeToolCall(id: "call_2", name: "get_time", argumentsJSON: "{}")]),
            ServeMessage(role: .tool, text: "3C", toolCallID: "call_1"),
            ServeMessage(role: .tool, text: "noon", toolCallID: "call_2"),
        ])
    }

    @Test func aCallWithNoPrecedingAssistantTextIsItsOwnTurn() throws {
        let decoded = try decode(#"""
        "input":[{"type":"message","role":"user","content":"x"},
                 {"type":"function_call","call_id":"c","name":"f","arguments":"{}"}]
        """#)
        #expect(decoded.request.messages.last == ServeMessage(role: .assistant, text: "",
                                                              toolCalls: [ServeToolCall(id: "c", name: "f", argumentsJSON: "{}")]))
    }

    @Test func imagesAndFilesArePlaceholdersInMessagesAndToolOutputs() throws {
        let decoded = try decode(#"""
        "input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"look"},
                   {"type":"input_image","image_url":"data:image/png;base64,AA"},{"type":"input_file","file_id":"f"}]},
                 {"type":"function_call_output","call_id":"c","output":[{"type":"input_image","image_url":"x"}]}]
        """#)
        #expect(decoded.request.messages[0].text ==
                "look\n[image omitted: this model can't see images]\n[document omitted: this model can't read documents]")
        #expect(decoded.request.messages[1].text == "[image omitted: this model can't see images]")
    }

    @Test func functionToolsAreReshapedIntoTheChatForm() throws {
        let decoded = try decode(#"""
        "input":"x","tools":[{"type":"function","name":"get_weather","description":"W","strict":false,
          "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]
        """#)
        let tool = try #require(decoded.request.tools.first)
        #expect(tool.name == "get_weather" && decoded.request.toolsRequested && !decoded.droppedHostedTools)
        let function = try #require(tool.object?["function"] as? [String: Any])
        #expect(tool.object?["type"] as? String == "function")
        #expect(function["description"] as? String == "W" && (function["parameters"] as? [String: Any])?["type"] as? String == "object")
    }

    /// R39 #1: hosted web_search / web_search_preview (and dated variants) / image_generation / tool_search
    /// are accepted and dropped.
    @Test func hostedToolsAreDroppedAndFlagged() throws {
        let hosted = ["web_search", "web_search_preview", "web_search_2025_08_26", "web_search_preview_2025_03_11",
                      "image_generation", "tool_search"]
        for type in hosted {
            let decoded = try decode(#""input":"x","tools":[{"type":"\#(type)","external_web_access":false},{"type":"function","name":"f","parameters":{}}]"#)
            #expect(decoded.droppedHostedTools, "\(type)")
            #expect(decoded.request.tools.map(\.name) == ["f"], "\(type)")
        }
        let only = try decode(#""input":"x","tools":[{"type":"web_search","external_web_access":false}]"#)
        #expect(only.droppedHostedTools && only.request.tools.isEmpty && !only.request.toolsRequested)
    }

    @Test func everyOtherBuiltInToolIsA400NamingTheType() {
        let refusedTypes = ["namespace", "custom", "file_search", "code_interpreter", "computer_use_preview",
                            "computer_use", "mcp", "local_shell", "apply_patch", "shell"]
        for type in refusedTypes {
            let error = refused(#""input":"x","tools":[{"type":"\#(type)","name":"n"}]"#)
            #expect(error?.status == 400 && error?.code == "unsupported_tool_type", "\(type)")
            #expect(error?.message.contains("'\(type)'") == true, "\(type)")
        }
    }

    @Test func toolChoiceForms() throws {
        func choice(_ json: String) throws -> ServeToolChoice { try decode(#""input":"x","tool_choice":\#(json)"#).request.toolChoice }
        #expect(try choice(#""auto""#) == .auto)
        #expect(try choice(#""none""#) == .none)
        #expect(try choice(#""required""#) == .required)
        #expect(try choice(#"{"type":"function","name":"f"}"#) == .function("f"))
        #expect(refused(#""input":"x","tool_choice":{"type":"allowed_tools"}"#)?.status == 400)
    }

    /// `reasoning.effort` goes through S2-3's mapping unchanged; `summary` alone never turns thinking on.
    @Test func reasoningEffortUsesTheSharedMapping() throws {
        func thinking(_ reasoning: String?) throws -> Bool? {
            try decode(#""input":"x"\#(reasoning.map { #","reasoning":\#($0)"# } ?? "")"#).request.reasoning
        }
        #expect(try thinking(#"{"summary":"auto"}"#) == nil)                 // Codex's default shape
        #expect(try thinking(#"{"effort":"medium","summary":"auto"}"#) == true)
        #expect(try thinking(#"{"effort":"none"}"#) == false)
        #expect(try thinking(#"{"effort":"minimal"}"#) == false)
        #expect(try thinking(#"{"effort":"xhigh"}"#) == true)
        #expect(try thinking(#"{"effort":"from-the-future"}"#) == true)
        #expect(try thinking(nil) == nil)
        #expect(refused(#""input":"x","reasoning":{"effort":3}"#)?.status == 400)
        #expect(try decode(#""input":"x","reasoning":{"effort":"from-the-future"}"#).request.unknownReasoningEffort == "from-the-future")
    }

    @Test func limitsAndSamplingAreCarried() throws {
        let decoded = try decode(#""input":"x","max_output_tokens":128,"temperature":0.3,"top_p":0.8"#)
        #expect(decoded.request.maxTokens == 128 && decoded.request.temperature == 0.3 && decoded.request.topP == 0.8)
        #expect(refused(#""input":"x","max_output_tokens":0"#)?.status == 400)
    }

    /// R38: accepted and ignored, never a 400 — and exactly what Codex sends alongside the input.
    @Test func codexsExtraFieldsAreAcceptedAndIgnored() throws {
        let decoded = try decode(#"""
        "input":"x","store":false,"include":["reasoning.encrypted_content"],"prompt_cache_key":"abc","parallel_tool_calls":true,
        "text":{"verbosity":"low"},"service_tier":"auto","truncation":"auto","metadata":{"k":"v"},"user":"u",
        "client_metadata":{"a":"b"},"tool_choice":"auto","reasoning":{"summary":"auto"},"safety_identifier":"s","background":false
        """#)
        #expect(decoded.request.messages == [ServeMessage(role: .user, text: "x")])
    }

    @Test func streamAndPreviousResponseIdAreRefusedForNow() {
        let stream = refused(#""input":"x","stream":true"#)
        #expect(stream?.status == 400 && stream?.message.contains("S3-3b") == true)
        let previous = refused(#""input":"x","previous_response_id":"resp_1""#)
        #expect(previous?.status == 400 && previous?.message.contains("not supported yet") == true)
        #expect(refused(#""input":"x","previous_response_id":null,"stream":false"#) == nil)
    }

    @Test func unknownItemsAndMissingInputAreClear400s() {
        #expect(refused(#""input":[{"type":"item_reference","id":"msg_1"}]"#)?.code == "unsupported_input_item")
        #expect(refused(#""input":[{"type":"local_shell_call","call_id":"c"}]"#)?.status == 400)
        #expect(refused(#""model2":1"#)?.code == "missing_input")
        #expect(refused(#""input":[{"type":"message","role":"user","content":[{"type":"input_audio"}]}]"#)?.code == "unsupported_content_part")
    }
}

// MARK: - Encode

struct ResponsesEncodeTests {
    private func body(_ collected: ServeCollected) throws -> [String: Any] {
        try #require(object(OpenAIResponsesDialect.responseBody(id: "resp_x", createdAt: 1_700_000_000, model: "m", collected: collected)))
    }

    @Test func aTextReplyHasTheResponseEnvelope() throws {
        var collected = ServeCollected()
        collected.text = "Hello"
        collected.usage = ServeUsage(promptTokens: 7, completionTokens: 2)
        let response = try body(collected)
        #expect(response["id"] as? String == "resp_x" && response["object"] as? String == "response")
        #expect(response["created_at"] as? Int == 1_700_000_000 && response["model"] as? String == "m")
        #expect(response["status"] as? String == "completed" && response["incomplete_details"] is NSNull)
        #expect(response["error"] is NSNull)
        let output = try #require(response["output"] as? [[String: Any]])
        #expect(output.count == 1)
        let message = output[0]
        #expect(message["type"] as? String == "message" && message["role"] as? String == "assistant")
        #expect(message["status"] as? String == "completed" && (message["id"] as? String)?.hasPrefix("msg_") == true)
        let part = try #require((message["content"] as? [[String: Any]])?.first)
        #expect(part["type"] as? String == "output_text" && part["text"] as? String == "Hello")
        #expect((part["annotations"] as? [Any])?.isEmpty == true)
    }

    @Test func reasoningThenMessageThenFunctionCallsInOrder() throws {
        var collected = ServeCollected()
        collected.reasoning = "plan"
        collected.text = "Looking."
        collected.toolCalls = [ServeToolCall(id: "call_1", name: "f", argumentsJSON: #"{"a":1}"#),
                               ServeToolCall(id: nil, name: "g", argumentsJSON: "{}")]
        collected.finish = .toolCalls
        let output = try #require(try body(collected)["output"] as? [[String: Any]])
        #expect(output.map { $0["type"] as? String } == ["reasoning", "message", "function_call", "function_call"])
        #expect(output[0]["summary"] as? [Any] != nil && (output[0]["summary"] as? [Any])?.isEmpty == true)
        let reasoningText = try #require((output[0]["content"] as? [[String: Any]])?.first)
        #expect(reasoningText["type"] as? String == "reasoning_text" && reasoningText["text"] as? String == "plan")
        #expect(output[0]["encrypted_content"] == nil)
        let call = output[2]
        #expect(call["call_id"] as? String == "call_1" && call["name"] as? String == "f")
        #expect(call["arguments"] as? String == #"{"a":1}"#)                         // a string, per the reference
        #expect((call["id"] as? String)?.hasPrefix("fc_") == true && call["status"] as? String == "completed")
        #expect((output[3]["call_id"] as? String)?.hasPrefix("call_") == true)        // minted when the model had none
    }

    @Test func aToolOnlyTurnHasNoMessageItemAndAnEmptyReplyIsOneEmptyMessage() throws {
        var calls = ServeCollected()
        calls.toolCalls = [ServeToolCall(id: "c", name: "f", argumentsJSON: "{}")]
        #expect((try body(calls)["output"] as? [[String: Any]])?.map { $0["type"] as? String } == ["function_call"])
        let empty = try #require(try body(ServeCollected())["output"] as? [[String: Any]])
        #expect(empty.count == 1 && empty[0]["type"] as? String == "message")
    }

    @Test func lengthIsIncompleteWithMaxOutputTokens() throws {
        var collected = ServeCollected()
        collected.text = "cut"
        collected.finish = .length
        let response = try body(collected)
        #expect(response["status"] as? String == "incomplete")
        #expect((response["incomplete_details"] as? [String: String])?["reason"] == "max_output_tokens")
        #expect(((response["output"] as? [[String: Any]])?.first)?["status"] as? String == "incomplete")
    }

    /// Inclusive, as OpenAI's: `input_tokens` is the whole prompt and `cached_tokens` the part reused;
    /// the details object is omitted when nothing was reused.
    @Test func usageIsInclusiveAndCachedTokensOmittedWhenZero() throws {
        let miss = OpenAIResponsesDialect.usageObject(ServeUsage(promptTokens: 100, completionTokens: 5))
        #expect(miss["input_tokens"] as? Int == 100 && miss["output_tokens"] as? Int == 5 && miss["total_tokens"] as? Int == 105)
        #expect(miss["input_tokens_details"] == nil)
        let hit = OpenAIResponsesDialect.usageObject(ServeUsage(promptTokens: 100, completionTokens: 5, cachedTokens: 80))
        #expect(hit["input_tokens"] as? Int == 100)                                    // NOT 20
        #expect((hit["input_tokens_details"] as? [String: Int]) == ["cached_tokens": 80])
        #expect(hit["total_tokens"] as? Int == 105)
    }
}

// MARK: - Route

struct ResponsesRouteTests {
    @Test func answersAResponse() async throws {
        let backend = MessagesBackend()
        let response = try #require(await send(req(#""input":"hi""#), env: environment(backend)))
        #expect(response.status == 200 && response.headers["Content-Type"] == "application/json")
        let body = try #require(object(response.body))
        #expect(body["object"] as? String == "response" && body["model"] as? String == model)
        #expect((body["id"] as? String)?.hasPrefix("resp_") == true)
        let usage = try #require(body["usage"] as? [String: Int])
        #expect(usage == ["input_tokens": 10, "output_tokens": 2, "total_tokens": 12])
        #expect(backend.generated.first?.messages == [ServeMessage(role: .user, text: "hi")])
    }

    @Test func aCacheHitIsInclusiveOnTheWire() async throws {
        let backend = MessagesBackend(events: [.textDelta("ok"),
            .usage(ServeUsage(promptTokens: 1000, completionTokens: 3, cachedTokens: 900)), .finish(.stop)])
        let response = try #require(await send(req(#""input":"hi""#), env: environment(backend)))
        let usage = try #require(object(response.body)?["usage"] as? [String: Any])
        #expect(usage["input_tokens"] as? Int == 1000)
        #expect((usage["input_tokens_details"] as? [String: Int]) == ["cached_tokens": 900])
    }

    @Test func aFunctionCallRoundTripsAcrossTwoTurns() async throws {
        let tools = #","tools":[{"type":"function","name":"get_weather","parameters":{"type":"object"}}]"#
        // Turn 1: the model calls the tool.
        let first = MessagesBackend(events: [
            .toolCallDelta(ServeToolCallDelta(index: 0, id: "call_9", name: "get_weather", argumentsFragment: #"{"city":"Oslo"}"#)),
            .usage(ServeUsage(promptTokens: 4, completionTokens: 6)), .finish(.toolCalls)])
        let turn1 = try #require(await send(req(#""input":"weather in Oslo?""# + tools), env: environment(first)))
        let output = try #require(object(turn1.body)?["output"] as? [[String: Any]])
        let call = try #require(output.first { $0["type"] as? String == "function_call" })
        #expect(call["call_id"] as? String == "call_9" && call["arguments"] as? String == #"{"city":"Oslo"}"#)

        // Turn 2: the client replays the call and its output; the backend sees ordinary tool traffic.
        let second = MessagesBackend()
        let replay = #"""
        "input":[{"type":"message","role":"user","content":"weather in Oslo?"},
                 {"type":"function_call","call_id":"call_9","name":"get_weather","arguments":"{\"city\":\"Oslo\"}"},
                 {"type":"function_call_output","call_id":"call_9","output":"3C"}]
        """#
        let turn2 = try #require(await send(req(replay + tools), env: environment(second)))
        #expect(turn2.status == 200)
        #expect(second.generated.first?.messages == [
            ServeMessage(role: .user, text: "weather in Oslo?"),
            ServeMessage(role: .assistant, text: "", toolCalls: [ServeToolCall(id: "call_9", name: "get_weather", argumentsJSON: #"{"city":"Oslo"}"#)]),
            ServeMessage(role: .tool, text: "3C", toolCallID: "call_9"),
        ])
    }

    /// R39 #1: dropped hosted tools never reach the backend and the log's tools-ignored flag says so.
    @Test func droppedHostedToolsAreNotedInTheLogMetadata() async throws {
        let backend = MessagesBackend()
        let log = RequestLog()
        let body = req(#""input":"hi","tools":[{"type":"web_search","external_web_access":false},{"type":"function","name":"f","parameters":{"type":"object"}}]"#)
        let response = try #require(await send(body, env: environment(backend, log: log)))
        #expect(response.status == 200)
        #expect(backend.generated.first?.tools.map(\.name) == ["f"])
        #expect(log.entries.last?.toolsIgnored == true)

        // A hosted tool alone leaves no tools at all, and is still flagged.
        let aloneLog = RequestLog()
        _ = await send(req(#""input":"hi","tools":[{"type":"web_search"}]"#), env: environment(MessagesBackend(), log: aloneLog))
        #expect(aloneLog.entries.last?.toolsIgnored == true)

        // …and a request with only a function tool the model accepts is not flagged.
        let plainLog = RequestLog()
        _ = await send(req(#""input":"hi","tools":[{"type":"function","name":"f","parameters":{"type":"object"}}]"#),
                       env: environment(MessagesBackend(), log: plainLog))
        #expect(plainLog.entries.last?.toolsIgnored == false)
    }

    @Test func aRefusedToolTypeIs400WithTheOpenAIEnvelope() async throws {
        let backend = MessagesBackend()
        let response = try #require(await send(req(#""input":"hi","tools":[{"type":"namespace","name":"mcp__x","tools":[]}]"#),
                                               env: environment(backend)))
        #expect(response.status == 400)
        let error = try #require(object(response.body)?["error"] as? [String: Any])
        #expect(error["code"] as? String == "unsupported_tool_type" && error["type"] as? String == "invalid_request_error")
        #expect(object(response.body)?["type"] == nil)                          // OpenAI's envelope, not Anthropic's
        #expect(backend.generated.isEmpty)
    }

    @Test func errorsKeepTheOpenAIEnvelopeStatusesAndRetryAfter() async throws {
        let missing = try #require(await send(req(#""input":"x""#).replacingOccurrences(of: model, with: "nope"), env: environment(MessagesBackend())))
        #expect(missing.status == 404 && (object(missing.body)?["error"] as? [String: Any])?["code"] as? String == "model_not_found")
        let queue = ServeQueue(globalCap: 1, perModelDepth: 0)
        _ = await queue.reserve(model: model)
        let busy = try #require(await send(req(#""input":"x""#), env: environment(MessagesBackend(), queue: queue)))
        #expect(busy.status == 503 && busy.headers["Retry-After"] == "2")
        let streaming = try #require(await send(req(#""input":"x","stream":true"#), env: environment(MessagesBackend())))
        #expect(streaming.status == 400)
    }

    @Test func theLogLineIsMetadataOnly() async throws {
        let log = RequestLog()
        _ = await send(req(#""input":"hi""#), env: environment(MessagesBackend(), log: log))
        let entry = try #require(log.entries.last)
        #expect(entry.status == 200 && entry.model == model && entry.promptTokens == 10 && entry.userAgent == "codex/test")
    }

    /// Codex's default request: developer instructions in `input`, `store:false`, `include`, a
    /// `reasoning` with only a `summary`, the hosted web_search tool — and thinking stays off.
    @Test func aCodexShapedRequestRuns() async throws {
        let backend = MessagesBackend()
        let body = req(#"""
        "input":[{"type":"message","role":"developer","content":[{"type":"input_text","text":"You are Codex."}]},
                 {"type":"message","role":"user","content":[{"type":"input_text","text":"fix it"}]}],
        "tools":[{"type":"function","name":"exec_command","description":"run","strict":false,"parameters":{"type":"object","properties":{}}},
                 {"type":"web_search","external_web_access":false}],
        "tool_choice":"auto","parallel_tool_calls":true,"reasoning":{"summary":"auto"},"store":false,"stream":false,
        "include":["reasoning.encrypted_content"],"prompt_cache_key":"k","text":{"verbosity":"low"}
        """#)
        let response = try #require(await send(body, env: environment(backend)))
        #expect(response.status == 200)
        let seen = try #require(backend.generated.first)
        #expect(seen.messages == [ServeMessage(role: .system, text: "You are Codex."), ServeMessage(role: .user, text: "fix it")])
        #expect(seen.tools.map(\.name) == ["exec_command"] && seen.reasoning == nil)
    }
}
