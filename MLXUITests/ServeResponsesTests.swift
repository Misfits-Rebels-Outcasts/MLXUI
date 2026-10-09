import Testing
import Foundation
@testable import MLXUI

// S3-3 / S3-3b — `POST /v1/responses`, non-streaming and streaming (no store yet).

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

    @Test func previousResponseIdIsRefusedForNowAndStreamIsAccepted() throws {
        #expect(try decode(#""input":"x","stream":true"#).request.stream)
        #expect(refused(#""input":"x","stream":"yes""#)?.status == 400)
        let previous = refused(#""input":"x","previous_response_id":"resp_1""#)
        #expect(previous?.status == 400 && previous?.message.contains("not supported yet") == true)
        #expect(refused(#""input":"x","previous_response_id":null,"stream":false"#) == nil)
    }

    /// R40: an unknown item *type* is dropped and counted, never a 400 (Codex resends its whole history).
    @Test func unknownItemTypesAreDroppedAndCounted() throws {
        let decoded = try decode(#"""
        "input":[{"type":"message","role":"user","content":"q"},
                 {"type":"item_reference","id":"msg_1"},
                 {"type":"local_shell_call","call_id":"c","action":{}},
                 {"type":"web_search_call","id":"ws_1","status":"completed"},
                 {"type":"message","role":"assistant","content":"a"}]
        """#)
        #expect(decoded.request.messages == [ServeMessage(role: .user, text: "q"), ServeMessage(role: .assistant, text: "a")])
        #expect(decoded.ignoredInputItems == 3)
        #expect(try decode(#""input":"plain""#).ignoredInputItems == 0)
        // reasoning items are known (dropped by R39) and are not counted as unknown.
        #expect(try decode(#""input":[{"type":"reasoning","summary":[]},{"role":"user","content":"x"}]"#).ignoredInputItems == 0)
    }

    /// …but malformed items of a known type, an unknown role and an item with no type at all stay 400s.
    @Test func malformedKnownItemsUnknownRolesAndMissingInputAreStill400s() {
        #expect(refused(#""input":[{"type":"function_call","name":"f"}]"#)?.code == "invalid_input_item")
        #expect(refused(#""input":[{"type":"function_call_output"}]"#)?.code == "invalid_input_item")
        #expect(refused(#""input":[{"type":"message","content":"x"}]"#)?.code == "invalid_input_item")
        #expect(refused(#""input":[{"type":"message","role":"narrator","content":"x"}]"#)?.code == "unsupported_role")
        #expect(refused(#""input":[{"content":"x"}]"#)?.code == "invalid_input_item")
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
    /// the details object is always present, 0 when nothing was reused (R40).
    @Test func usageIsInclusiveAndCachedTokensAlwaysPresent() throws {
        let miss = OpenAIResponsesDialect.usageObject(ServeUsage(promptTokens: 100, completionTokens: 5))
        #expect(miss["input_tokens"] as? Int == 100 && miss["output_tokens"] as? Int == 5 && miss["total_tokens"] as? Int == 105)
        #expect((miss["input_tokens_details"] as? [String: Int]) == ["cached_tokens": 0])
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
        let usage = try #require(body["usage"] as? [String: Any])
        #expect(usage["input_tokens"] as? Int == 10 && usage["output_tokens"] as? Int == 2 && usage["total_tokens"] as? Int == 12)
        #expect((usage["input_tokens_details"] as? [String: Int]) == ["cached_tokens": 0])
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
    }

    @Test func droppedUnknownItemsAreCountedInTheLogMetadata() async throws {
        let log = RequestLog()
        let body = req(#""input":[{"type":"message","role":"user","content":"hi"},{"type":"item_reference","id":"x"},{"type":"mystery"}]"#)
        let response = try #require(await send(body, env: environment(MessagesBackend(), log: log)))
        #expect(response.status == 200)
        #expect(log.entries.last?.ignoredInputItems == 2)
        let plain = RequestLog()
        _ = await send(req(#""input":"hi""#), env: environment(MessagesBackend(), log: plain))
        #expect(plain.entries.last?.ignoredInputItems == 0)
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


// MARK: - Streaming (S3-3b)

private struct RSEvent {
    let name: String
    let json: [String: Any]
    var index: Int? { json["output_index"] as? Int }
    /// `name[@output_index]`: a whole sequence reads at a glance.
    var summary: String { index.map { "\(name)@\($0)" } ?? name }
}

private func parseResponsesSSE(_ text: String) -> [RSEvent] {
    text.components(separatedBy: "\n\n").filter { !$0.isEmpty && !$0.hasPrefix(":") }.compactMap { chunk in
        let lines = chunk.components(separatedBy: "\n")
        guard let name = lines.first(where: { $0.hasPrefix("event: ") })?.dropFirst(7),
              let data = lines.first(where: { $0.hasPrefix("data: ") })?.dropFirst(6),
              let json = object(Data(data.utf8)) else { return nil }
        return RSEvent(name: String(name), json: json)
    }
}

private func streamRun(_ events: [ServeEvent], backend: ServeBackend? = nil, keepAlive: Double = 1.0)
    async -> (status: Int?, headers: [String: String], events: [RSEvent], text: String) {
    let body = Data(req(#""input":"hi","stream":true"#).utf8)
    let head = ServeRequestHead(method: "POST", path: "/v1/responses", host: "127.0.0.1:1212", origin: nil,
                                transferEncoding: nil, contentLength: String(body.count), userAgent: nil)
    var env = environment(backend ?? MessagesBackend(events: events))
    env.keepAliveSeconds = keepAlive
    guard case .stream(let status, let headers, let bytes) = await ServeAPI.handle(
        head: head, guard: RequestGuard(port: 1212), environment: env, body: { body }) else { return (nil, [:], [], "") }
    var all: [UInt8] = []
    for await chunk in bytes { all += chunk }
    let text = String(decoding: all, as: UTF8.self)
    return (status, headers, parseResponsesSSE(text), text)
}

private func tokens(_ prompt: Int, _ completion: Int, cached: Int = 0) -> ServeEvent {
    .usage(ServeUsage(promptTokens: prompt, completionTokens: completion, cachedTokens: cached))
}

private func fcall(_ index: Int, id: String?, name: String? = nil, args: String? = nil) -> ServeEvent {
    .toolCallDelta(ServeToolCallDelta(index: index, id: id, name: name, argumentsFragment: args))
}

/// What every stream must satisfy, whatever it contains.
private func expectWellFormed(_ events: [RSEvent], sourceLocation: SourceLocation = #_sourceLocation) {
    #expect(events.first?.name == "response.created", sourceLocation: sourceLocation)            // the SDK needs it first
    #expect(events.allSatisfy { $0.json["type"] as? String == $0.name }, sourceLocation: sourceLocation)
    #expect(events.map { $0.json["sequence_number"] as? Int } == Array(0..<events.count).map { Optional($0) },
            sourceLocation: sourceLocation)                                                      // running, from 0, no gaps
}

struct ResponsesStreamTests {
    @Test func textOnly() async throws {
        let run = await streamRun([.textDelta("Hel"), .textDelta("lo"), tokens(10, 2), .finish(.stop)])
        #expect(run.status == 200 && run.headers["Content-Type"] == "text/event-stream")
        #expect(run.events.map(\.summary) == [
            "response.created", "response.in_progress",
            "response.output_item.added@0", "response.content_part.added@0",
            "response.output_text.delta@0", "response.output_text.delta@0",
            "response.output_text.done@0", "response.content_part.done@0", "response.output_item.done@0",
            "response.completed"])
        expectWellFormed(run.events)
        let created = try #require(run.events[0].json["response"] as? [String: Any])
        #expect(created["status"] as? String == "in_progress" && (created["output"] as? [Any])?.isEmpty == true)
        #expect((created["id"] as? String)?.hasPrefix("resp_") == true)
        let added = try #require(run.events[2].json["item"] as? [String: Any])
        #expect(added["type"] as? String == "message" && added["status"] as? String == "in_progress")
        let deltas = run.events.filter { $0.name == "response.output_text.delta" }
        #expect(deltas.compactMap { $0.json["delta"] as? String }.joined() == "Hello")
        #expect(deltas.allSatisfy { $0.json["content_index"] as? Int == 0 && $0.json["item_id"] as? String == added["id"] as? String })
        let done = try #require(run.events[8].json["item"] as? [String: Any])
        #expect(done["id"] as? String == added["id"] as? String && done["status"] as? String == "completed")
        let part = try #require((done["content"] as? [[String: Any]])?.first)
        #expect(part["text"] as? String == "Hello" && part["type"] as? String == "output_text")
    }

    @Test func theFinalResponseCarriesTheWholeOutputAndUsage() async throws {
        let run = await streamRun([.reasoningDelta("plan"), .textDelta("Hi"), fcall(0, id: "call_1", name: "f", args: #"{"a":1}"#),
                                   tokens(1000, 7, cached: 900), .finish(.toolCalls)])
        let response = try #require(run.events.last?.json["response"] as? [String: Any])
        #expect(run.events.last?.name == "response.completed" && response["status"] as? String == "completed")
        #expect(response["id"] as? String == (run.events[0].json["response"] as? [String: Any])?["id"] as? String)
        let finalOutput = try #require(response["output"] as? [[String: Any]])
        let doneItems = run.events.filter { $0.name == "response.output_item.done" }.compactMap { $0.json["item"] as? [String: Any] }
        #expect(finalOutput.map { $0["id"] as? String } == doneItems.map { $0["id"] as? String })
        #expect(finalOutput.map { $0["type"] as? String } == ["reasoning", "message", "function_call"])
        let usage = try #require(response["usage"] as? [String: Any])
        #expect(usage["input_tokens"] as? Int == 1000 && usage["output_tokens"] as? Int == 7 && usage["total_tokens"] as? Int == 1007)
        #expect((usage["input_tokens_details"] as? [String: Int]) == ["cached_tokens": 900])
        // A miss still carries the details object (R40).
        let miss = await streamRun([.textDelta("x"), tokens(5, 1), .finish(.stop)])
        let missUsage = try #require((miss.events.last?.json["response"] as? [String: Any])?["usage"] as? [String: Any])
        #expect((missUsage["input_tokens_details"] as? [String: Int]) == ["cached_tokens": 0])
    }

    @Test func reasoningThenText() async throws {
        let run = await streamRun([.reasoningDelta("pl"), .reasoningDelta("an"), .textDelta("Hi"), tokens(5, 4), .finish(.stop)])
        #expect(run.events.map(\.summary) == [
            "response.created", "response.in_progress",
            "response.output_item.added@0", "response.reasoning_text.delta@0", "response.reasoning_text.delta@0",
            "response.reasoning_text.done@0", "response.output_item.done@0",
            "response.output_item.added@1", "response.content_part.added@1", "response.output_text.delta@1",
            "response.output_text.done@1", "response.content_part.done@1", "response.output_item.done@1",
            "response.completed"])
        expectWellFormed(run.events)
        let delta = try #require(run.events.first { $0.name == "response.reasoning_text.delta" }?.json)
        #expect(delta["content_index"] as? Int == 0 && delta["delta"] as? String == "pl")      // Codex needs content_index
        let done = try #require(run.events[6].json["item"] as? [String: Any])
        #expect(done["type"] as? String == "reasoning" && (done["summary"] as? [Any])?.isEmpty == true)
        #expect(((done["content"] as? [[String: Any]])?.first)?["text"] as? String == "plan")
    }

    @Test func textThenFunctionCall() async throws {
        let run = await streamRun([.textDelta("Looking."), fcall(0, id: "call_1", name: "get_weather", args: #"{"city":"Oslo"}"#),
                                   tokens(8, 9), .finish(.toolCalls)])
        #expect(run.events.map(\.summary) == [
            "response.created", "response.in_progress",
            "response.output_item.added@0", "response.content_part.added@0", "response.output_text.delta@0",
            "response.output_text.done@0", "response.content_part.done@0", "response.output_item.done@0",
            "response.output_item.added@1", "response.function_call_arguments.delta@1",
            "response.function_call_arguments.done@1", "response.output_item.done@1",
            "response.completed"])
        expectWellFormed(run.events)
        let added = try #require(run.events[8].json["item"] as? [String: Any])
        #expect(added["type"] as? String == "function_call" && added["call_id"] as? String == "call_1")
        #expect(added["name"] as? String == "get_weather" && added["arguments"] as? String == "")
        let delta = try #require(run.events[9].json)
        #expect(delta["item_id"] as? String == added["id"] as? String && delta["delta"] as? String == #"{"city":"Oslo"}"#)
        let done = try #require(run.events[11].json["item"] as? [String: Any])
        #expect(done["id"] as? String == added["id"] as? String && done["arguments"] as? String == #"{"city":"Oslo"}"#)
        #expect(done["call_id"] as? String == "call_1" && done["status"] as? String == "completed")
    }

    @Test func functionCallOnlyStartsAtIndexZero() async throws {
        let run = await streamRun([fcall(0, id: "call_1", name: "f", args: "{}"), tokens(3, 4), .finish(.toolCalls)])
        #expect(run.events.map(\.summary) == [
            "response.created", "response.in_progress", "response.output_item.added@0",
            "response.function_call_arguments.delta@0", "response.function_call_arguments.done@0",
            "response.output_item.done@0", "response.completed"])
        expectWellFormed(run.events)
    }

    @Test func twoFunctionCallsAreTwoItemsWithRunningIndexes() async throws {
        let run = await streamRun([fcall(0, id: "call_a", name: "f", args: #"{"a":1}"#),
                                   fcall(1, id: "call_b", name: "g", args: #"{"b":2}"#), tokens(3, 8), .finish(.toolCalls)])
        #expect(run.events.map(\.summary) == [
            "response.created", "response.in_progress",
            "response.output_item.added@0", "response.function_call_arguments.delta@0",
            "response.function_call_arguments.done@0", "response.output_item.done@0",
            "response.output_item.added@1", "response.function_call_arguments.delta@1",
            "response.function_call_arguments.done@1", "response.output_item.done@1",
            "response.completed"])
        expectWellFormed(run.events)
        let ids = run.events.filter { $0.name == "response.output_item.done" }.compactMap { ($0.json["item"] as? [String: Any])?["call_id"] as? String }
        #expect(ids == ["call_a", "call_b"])
        let itemIDs = run.events.filter { $0.name == "response.output_item.added" }.compactMap { ($0.json["item"] as? [String: Any])?["id"] as? String }
        #expect(Set(itemIDs).count == 2)
    }

    @Test func aContinuationFragmentStaysInTheOpenCall() async throws {
        let run = await streamRun([fcall(0, id: "call_a", name: "f", args: #"{"a":"#), fcall(0, id: nil, args: #"1}"#),
                                   tokens(3, 8), .finish(.toolCalls)])
        #expect(run.events.filter { $0.name == "response.output_item.added" }.count == 1)
        let done = try #require(run.events.first { $0.name == "response.output_item.done" }?.json["item"] as? [String: Any])
        #expect(done["arguments"] as? String == #"{"a":1}"#)
    }

    /// Codex treats `response.incomplete` as an error unless the reason is `interrupted` (journal 2026-433);
    /// the documented event is still the right one for a length stop.
    @Test func lengthEndsWithResponseIncomplete() async throws {
        let run = await streamRun([.textDelta("cut"), tokens(5, 4096), .finish(.length)])
        #expect(run.events.last?.name == "response.incomplete")
        expectWellFormed(run.events)
        let response = try #require(run.events.last?.json["response"] as? [String: Any])
        #expect(response["status"] as? String == "incomplete")
        #expect((response["incomplete_details"] as? [String: String])?["reason"] == "max_output_tokens")
        #expect(!run.text.contains("response.completed"))
    }

    @Test func anEmptyReplyIsOneEmptyMessage() async throws {
        let run = await streamRun([tokens(4, 0), .finish(.stop)])
        let output = try #require((run.events.last?.json["response"] as? [String: Any])?["output"] as? [[String: Any]])
        #expect(output.count == 1 && output[0]["type"] as? String == "message")
        expectWellFormed(run.events)
    }

    /// Codex ignores a bare `error` event and fails the turn on `response.failed`; after it, nothing.
    @Test func aMidStreamErrorIsResponseFailedAndNothingAfter() async throws {
        let run = await streamRun([.textDelta("par"), .error(ServeError(status: 500, message: "The model failed to generate a reply.",
                                                                          type: "server_error", code: "generation_failed"))])
        #expect(run.events.map(\.name) == ["response.created", "response.in_progress", "response.output_item.added",
                                           "response.content_part.added", "response.output_text.delta", "response.failed"])
        expectWellFormed(run.events)
        let response = try #require(run.events.last?.json["response"] as? [String: Any])
        #expect(response["status"] as? String == "failed")
        let error = try #require(response["error"] as? [String: String])
        #expect(error["code"] == "generation_failed" && error["message"] == "The model failed to generate a reply.")
        #expect(!run.text.contains("response.completed") && !run.text.contains("response.incomplete"))
        #expect(!run.text.contains("event: error"))
    }

    @Test func aFailureBeforeAnyOutputStillOpensWithResponseCreated() async throws {
        let run = await streamRun([.error(ServeError(status: 500, message: "boom", type: "server_error", code: "generation_failed"))])
        #expect(run.events.map(\.name) == ["response.created", "response.in_progress", "response.failed"])
        expectWellFormed(run.events)
    }

    @Test func aFailureBeforeTheHeadIsTheJSONError() async throws {
        let body = Data(req(#""input":"x","stream":true"#).replacingOccurrences(of: model, with: "nope").utf8)
        let head = ServeRequestHead(method: "POST", path: "/v1/responses", host: "127.0.0.1:1212", origin: nil,
                                    transferEncoding: nil, contentLength: String(body.count), userAgent: nil)
        guard case .full(let response) = await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212),
                                                                environment: environment(MessagesBackend()), body: { body }) else {
            Issue.record("expected a whole response, not a stream"); return
        }
        #expect(response.status == 404 && (object(response.body)?["error"] as? [String: Any])?["code"] as? String == "model_not_found")
    }

    /// An SSE comment, never an event: the OpenAI SDK needs `response.created` first. (Codex's idle timer is
    /// reset only by a parsed event; its default is 300 s — journal 2026-433.)
    @Test func keepAliveIsAnSSECommentWrittenWhileTheModelWorks() async throws {
        let encoder = OpenAIResponsesStreamEncoder(id: "resp_x", model: "m", createdAt: 1)
        #expect(String(decoding: encoder.keepAlive, as: UTF8.self) == ": keep-alive\n\n")

        let slow = FakeBackend([.wait(0.45), .text("late"), .usage(prompt: 2, completion: 1), .finish(.stop)], modelType: "qwen3")
        let run = await streamRun([], backend: slow, keepAlive: 0.1)
        #expect(run.text.hasPrefix(": keep-alive\n\n"))
        #expect(run.events.first?.name == "response.created" && run.events.last?.name == "response.completed")
        expectWellFormed(run.events)
    }
}

// MARK: - S3-3d: the output limit and the whitespace hold (R41)

struct ResponsesOutputLimitTests {
    private func maxTokens(_ path: String, _ body: String, window: Int?, count: Int?) async throws -> Int? {
        let backend = MessagesBackend(count: count, contextLimit: window)
        let response = try #require(await send(body, env: environment(backend), path: path))
        #expect(response.status == 200)
        return backend.generated.first?.effectiveMaxTokens
    }

    /// No `max_output_tokens` → the room the prompt leaves: min(window, 32768) − the S2-6 count (R42).
    @Test func noLimitMeansTheCappedWindowMinusThePrompt() async throws {
        #expect(try await maxTokens("/v1/responses", req(#""input":"hi""#), window: 8_192, count: 1_000) == 7_192)       // under the cap: the window
        #expect(try await maxTokens("/v1/responses", req(#""input":"hi""#), window: 40_960, count: 1_000) == 32_768)      // the reply cap; the window leaves 39,960
        #expect(try await maxTokens("/v1/responses", req(#""input":"hi""#), window: 262_144, count: 1_000) == 32_768)     // the cap, not 261,144
        #expect(try await maxTokens("/v1/responses", req(#""input":"hi""#), window: 1000, count: 1000) == 1)             // never below 1
        #expect(try await maxTokens("/v1/responses", req(#""input":"hi""#), window: 40_960, count: 40_000) == 960)        // the window bounds prompt + reply
    }

    @Test func anExplicitLimitWins() async throws {
        #expect(try await maxTokens("/v1/responses", req(#""input":"hi","max_output_tokens":50"#), window: 40_960, count: 1_000) == 50)
    }

    /// R16's 4096 stands wherever there is no window or no count to compute from.
    @Test func noWindowOrNoCountKeepsTheDefault() async throws {
        #expect(try await maxTokens("/v1/responses", req(#""input":"hi""#), window: nil, count: 1_000) == ServeRequest.defaultMaxTokens)
        #expect(try await maxTokens("/v1/responses", req(#""input":"hi""#), window: 40_960, count: nil) == ServeRequest.defaultMaxTokens)
    }

    /// The other dialects keep R16 unchanged: chat with no limit gets 4096, and a messages request keeps the
    /// `max_tokens` it must send.
    @Test func chatAndMessagesAreUnchanged() async throws {
        let chat = #"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}]}"#
        #expect(try await maxTokens("/v1/chat/completions", chat, window: 40_960, count: 1_000) == ServeRequest.defaultMaxTokens)
        let messages = #"{"model":"\#(model)","max_tokens":77,"messages":[{"role":"user","content":"hi"}]}"#
        #expect(try await maxTokens("/v1/messages", messages, window: 40_960, count: 1_000) == 77)
    }

    @Test func aWindowFillingRequestIsOnlyMarkedByTheResponsesDecode() throws {
        #expect(try decode(#""input":"hi""#).request.fillsContextWindow)
        #expect(!(try decode(#""input":"hi","max_output_tokens":9"#).request.fillsContextWindow))
        let chat = try OpenAIChatDialect.decode(Data(#"{"model":"m","messages":[{"role":"user","content":"x"}]}"#.utf8))
        #expect(!chat.fillsContextWindow)
    }

    /// A length stop that still happens stays `response.incomplete`.
    @Test func aLengthStopStaysIncomplete() async throws {
        let backend = MessagesBackend(events: [.textDelta("cut"), .usage(ServeUsage(promptTokens: 5, completionTokens: 99)), .finish(.length)],
                                      count: 5, contextLimit: 104)
        let response = try #require(await send(req(#""input":"hi""#), env: environment(backend)))
        #expect(object(response.body)?["status"] as? String == "incomplete")
        #expect(backend.generated.first?.effectiveMaxTokens == 99)
    }
}

struct ResponsesWhitespaceTests {
    /// The output items of a reply, as `(type, payload)` so ids don't matter.
    private func shape(_ items: [[String: Any]]) -> [String] {
        items.map { item in
            switch item["type"] as? String {
            case "message": return "message:" + ((item["content"] as? [[String: Any]])?.first?["text"] as? String ?? "")
            case "function_call": return "function_call:\(item["name"] as? String ?? "")(\(item["arguments"] as? String ?? ""))"
            case let other: return other ?? "?"
            }
        }
    }

    private func nonStreamed(_ events: [ServeEvent]) async throws -> [String] {
        let response = try #require(await send(req(#""input":"hi""#), env: environment(MessagesBackend(events: events))))
        return shape(try #require(object(response.body)?["output"] as? [[String: Any]]))
    }

    private func streamed(_ events: [ServeEvent]) async throws -> (items: [String], run: [String]) {
        let run = await streamRun(events)
        let final = try #require((run.events.last?.json["response"] as? [String: Any])?["output"] as? [[String: Any]])
        let done = run.events.filter { $0.name == "response.output_item.done" }.compactMap { $0.json["item"] as? [String: Any] }
        #expect(shape(final) == shape(done))                                 // the final response is the .done items
        return (shape(final), run.events.map(\.name))
    }

    private func same(_ events: [ServeEvent], _ expected: [String], sourceLocation: SourceLocation = #_sourceLocation) async throws {
        let plain = try await nonStreamed(events)
        let live = try await streamed(events)
        #expect(plain == expected, sourceLocation: sourceLocation)
        #expect(live.items == expected, sourceLocation: sourceLocation)
    }

    @Test func spacesThenTextAreFlushedInOrder() async throws {
        try await same([.textDelta("  "), .textDelta("\n"), .textDelta("hi"), .usage(ServeUsage(promptTokens: 1, completionTokens: 1)), .finish(.stop)],
                       ["message:  \nhi"])
        // …as one ordered stream of deltas: nothing was emitted for the held part until "hi" arrived.
        let run = await streamRun([.textDelta("  "), .textDelta("hi"), .finish(.stop)])
        let deltas = run.events.filter { $0.name == "response.output_text.delta" }.compactMap { $0.json["delta"] as? String }
        #expect(deltas == ["  hi"])
    }

    @Test func spacesThenACallOpenNoMessage() async throws {
        let events: [ServeEvent] = [.textDelta("  "), .textDelta("\n"),
                                    .toolCallDelta(ServeToolCallDelta(index: 0, id: "c1", name: "f", argumentsFragment: "{}")),
                                    .usage(ServeUsage(promptTokens: 1, completionTokens: 1)), .finish(.toolCalls)]
        try await same(events, ["function_call:f({})"])
        let live = try await streamed(events)
        #expect(!live.run.contains("response.content_part.added"))              // no message item was ever opened
    }

    @Test func textThenSpacesThenACallKeepsTheSpacesInTheMessage() async throws {
        try await same([.textDelta("hi"), .textDelta("   "),
                        .toolCallDelta(ServeToolCallDelta(index: 0, id: "c1", name: "f", argumentsFragment: "{}")),
                        .finish(.toolCalls)],
                       ["message:hi   ", "function_call:f({})"])
    }

    @Test func whitespaceAloneIsAMessageOnlyWhenNothingElseCame() async throws {
        try await same([.textDelta("  "), .finish(.stop)], ["message:  "])
        try await same([.reasoningDelta("plan"), .textDelta("  "), .finish(.stop)], ["reasoning"])
        try await same([.finish(.stop)], ["message:"])
    }
}

// MARK: - S3-3f: the output clamp, every dialect (R44)

struct OutputClampTests {
    private func limit(_ path: String, _ body: String, window: Int?, count: Int?) async throws -> Int? {
        let backend = MessagesBackend(count: count, contextLimit: window)
        let response = try #require(await send(body, env: environment(backend), path: path))
        #expect(response.status == 200)
        return backend.generated.first?.effectiveMaxTokens
    }
    private let chat = #"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}]}"#
    private func chat(max: Int) -> String { #"{"model":"\#(model)","max_tokens":\#(max),"messages":[{"role":"user","content":"hi"}]}"# }
    private func messages(max: Int) -> String { #"{"model":"\#(model)","max_tokens":\#(max),"messages":[{"role":"user","content":"hi"}]}"# }

    /// Chat's default 4096 with a prompt that nearly fills the window is lowered to what is left.
    @Test func chatDefaultIsClampedWhenThePromptNearsTheWindow() async throws {
        #expect(try await limit("/v1/chat/completions", chat, window: 8_192, count: 8_000) == 192)
        #expect(try await limit("/v1/chat/completions", chat, window: 40_960, count: 30_000) == ServeRequest.defaultMaxTokens)   // 10,960 left: the default stands
    }

    /// Room to spare: the default is left alone (a clamp never raises).
    @Test func roomToSpareLeavesTheDefaultAndSmallLimitsUntouched() async throws {
        #expect(try await limit("/v1/chat/completions", chat, window: 40_960, count: 1_000) == ServeRequest.defaultMaxTokens)
        #expect(try await limit("/v1/chat/completions", chat, window: 262_144, count: 1_000) == ServeRequest.defaultMaxTokens)
        #expect(try await limit("/v1/chat/completions", chat(max: 50), window: 40_960, count: 1_000) == 50)
        #expect(try await limit("/v1/messages", messages(max: 64), window: 262_144, count: 1_000) == 64)
        #expect(try await limit("/v1/responses", req(#""input":"hi","max_output_tokens":10"#), window: 40_960, count: 1_000) == 10)
    }

    /// A Messages request for 32000 on a 40,960 window is lowered to what the window leaves after the prompt.
    @Test func messagesBigLimitIsClampedToWhatTheWindowLeaves() async throws {
        #expect(try await limit("/v1/messages", messages(max: 32_000), window: 40_960, count: 20_000) == 20_960)
        #expect(try await limit("/v1/messages", messages(max: 32_000), window: 40_960, count: 1_000) == 32_000)   // fits: unchanged
        #expect(try await limit("/v1/messages", messages(max: 32_000), window: 262_144, count: 1_000) == 32_000)
    }

    /// An explicit limit above the room is lowered too, on every dialect.
    @Test func anExplicitLimitAboveTheRoomIsLowered() async throws {
        #expect(try await limit("/v1/chat/completions", chat(max: 9_000), window: 8_192, count: 1_000) == 7_192)
        #expect(try await limit("/v1/responses", req(#""input":"hi","max_output_tokens":9000"#), window: 8_192, count: 1_000) == 7_192)
    }

    /// No window, or no count: unchanged.
    @Test func noWindowOrNoCountLeavesEveryLimitAlone() async throws {
        #expect(try await limit("/v1/chat/completions", chat, window: nil, count: 1_000) == ServeRequest.defaultMaxTokens)
        #expect(try await limit("/v1/messages", messages(max: 32_000), window: nil, count: 1_000) == 32_000)
        #expect(try await limit("/v1/messages", messages(max: 32_000), window: 40_960, count: nil) == 32_000)
        #expect(try await limit("/v1/chat/completions", chat, window: 40_960, count: nil) == ServeRequest.defaultMaxTokens)
    }

    /// R46: the window bounds prompt + reply, and the 32k caps the reply only, so a long prompt no longer starves the reply
    /// (R44's `min(window, 32768) − prompt` gave 1 token past a 32k prompt).
    @Test func aLongPromptKeepsTheRoomTheWindowHasLeft() async throws {
        // window 40,960, prompt 35,000 → 5,960 of room (a request for more is lowered to it)
        #expect(try await limit("/v1/messages", messages(max: 32_000), window: 40_960, count: 35_000) == 5_960)
        // window 262,144, prompt 100,000 → 162,144 of room, but the reply is capped at 32,768
        #expect(try await limit("/v1/responses", req(#""input":"hi""#), window: 262_144, count: 100_000) == 32_768)
        // window 40,960, prompt 1,000 → min(requested, 32,768)
        #expect(try await limit("/v1/responses", req(#""input":"hi""#), window: 40_960, count: 1_000) == 32_768)
        #expect(try await limit("/v1/messages", messages(max: 32_000), window: 40_960, count: 1_000) == 32_000)
        #expect(try await limit("/v1/messages", messages(max: 50_000), window: 262_144, count: 1_000) == 32_768)  // the reply cap
        // the default 4096 is left alone while the window has room, even for a 100k prompt
        #expect(try await limit("/v1/chat/completions", chat, window: 262_144, count: 100_000) == ServeRequest.defaultMaxTokens)
        // and the floor is still 1
        #expect(try await limit("/v1/chat/completions", chat, window: 40_960, count: 40_960) == 1)
    }

    /// The Responses fill is the same rule with "requested" = the cap.
    @Test func theResponsesFillIsTheSameRule() async throws {
        #expect(try await limit("/v1/responses", req(#""input":"hi""#), window: 40_960, count: 1_000) == 32_768)
        #expect(try await limit("/v1/responses", req(#""input":"hi""#), window: 262_144, count: 1_000) == 32_768)
        #expect(try await limit("/v1/responses", req(#""input":"hi""#), window: 8_192, count: 1_000) == 7_192)
    }

    /// A length stop is still reported as one: chat says `length`, messages `max_tokens`.
    @Test func aClampedLimitStillEndsAsALengthStop() async throws {
        let events: [ServeEvent] = [.textDelta("cut"), .usage(ServeUsage(promptTokens: 8_000, completionTokens: 192)), .finish(.length)]
        let chatReply = try #require(await send(chat, env: environment(MessagesBackend(events: events, count: 8_000, contextLimit: 8_192)),
                                                path: "/v1/chat/completions"))
        let choice = try #require((object(chatReply.body)?["choices"] as? [[String: Any]])?.first)
        #expect(choice["finish_reason"] as? String == "length")
        let messagesReply = try #require(await send(messages(max: 9_000), env: environment(MessagesBackend(events: events, count: 8_000, contextLimit: 8_192)),
                                                    path: "/v1/messages"))
        #expect(object(messagesReply.body)?["stop_reason"] as? String == "max_tokens")
    }
}
