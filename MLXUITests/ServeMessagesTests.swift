import Testing
import Foundation
@testable import MLXUI

// S3-2 — `POST /v1/messages` (non-streaming) and `POST /v1/messages/count_tokens`.

// MARK: - Helpers

private let model = "mlx-community/Qwen3-4B-4bit"

/// Records what the pipeline hands the pre-count and the generation, and answers a scripted count and
/// usage (including a cache hit, which `FakeBackend` can't express).
nonisolated final class MessagesBackend: ServeBackend, @unchecked Sendable {
    private let lock = NSLock()
    private var _counted: [ServeRequest] = []
    private var _generated: [ServeRequest] = []
    let modelType: String? = "qwen3"
    var contextLimit: Int?
    var count: Int?
    let events: [ServeEvent]

    init(events: [ServeEvent] = [.textDelta("ok"), .usage(ServeUsage(promptTokens: 10, completionTokens: 2)), .finish(.stop)],
         count: Int? = 10, contextLimit: Int? = nil) {
        self.events = events
        self.count = count
        self.contextLimit = contextLimit
    }

    var counted: [ServeRequest] { lock.lock(); defer { lock.unlock() }; return _counted }
    var generated: [ServeRequest] { lock.lock(); defer { lock.unlock() }; return _generated }

    func countPromptTokens(_ request: ServeRequest) async throws -> Int? {
        lock.lock(); _counted.append(request); lock.unlock()
        return count
    }

    func generate(_ request: ServeRequest) -> AsyncThrowingStream<ServeEvent, Error> {
        lock.lock(); _generated.append(request); lock.unlock()
        return AsyncThrowingStream { continuation in
            for event in events { continuation.yield(event) }
            continuation.finish()
        }
    }
}

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

private func send(_ json: String, path: String = "/v1/messages", method: String = "POST",
                  host: String = "127.0.0.1:1212", env: ServeEnvironment) async -> ServeResponse? {
    let body = Data(json.utf8)
    let head = ServeRequestHead(method: method, path: path, host: host, origin: nil, transferEncoding: nil,
                                contentLength: String(body.count), userAgent: "claude-cli/test")
    if case .full(let response) = await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212),
                                                         environment: env, body: { body }) { return response }
    return nil
}

private func object(_ data: Data) -> [String: Any]? { (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] }

private func request(_ extra: String = "") -> String {
    #"{"model":"\#(model)","max_tokens":64,"messages":[{"role":"user","content":"hi"}]\#(extra)}"#
}

// MARK: - Decode

struct AnthropicMessagesDecodeTests {
    @Test func aSystemStringAndSystemBlocksBothBecomeOneLeadingSystemMessage() throws {
        let one = try AnthropicMessagesDialect.decode(Data(request(#","system":"Be brief.""#).utf8))
        #expect(one.messages.first == ServeMessage(role: .system, text: "Be brief."))
        let blocks = try AnthropicMessagesDialect.decode(Data(
            request(#","system":[{"type":"text","text":"A","cache_control":{"type":"ephemeral"}},{"type":"text","text":"B"}]"#).utf8))
        #expect(blocks.messages.first == ServeMessage(role: .system, text: "A\n\nB"))
        #expect(blocks.messages.count == 2)
    }

    @Test func toolTrafficKeepsItsOrderAndIds() throws {
        let body = request("").replacingOccurrences(of: #""messages":[{"role":"user","content":"hi"}]"#, with: #"""
        "messages":[
          {"role":"user","content":"weather?"},
          {"role":"assistant","content":[
            {"type":"thinking","thinking":"hm","signature":"SIG"},
            {"type":"redacted_thinking","data":"xx"},
            {"type":"text","text":"Checking."},
            {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Oslo"}}]},
          {"role":"user","content":[
            {"type":"tool_result","tool_use_id":"toolu_1","content":"3C"},
            {"type":"tool_result","tool_use_id":"toolu_2","content":[{"type":"text","text":"boom"}],"is_error":true},
            {"type":"text","text":"and then?"}]}]
        """#)
        let decoded = try AnthropicMessagesDialect.decode(Data(body.utf8))
        #expect(decoded.messages == [
            ServeMessage(role: .user, text: "weather?"),
            ServeMessage(role: .assistant, text: "Checking.",
                         toolCalls: [ServeToolCall(id: "toolu_1", name: "get_weather", argumentsJSON: #"{"city":"Oslo"}"#)]),
            ServeMessage(role: .tool, text: "3C", toolCallID: "toolu_1"),
            ServeMessage(role: .tool, text: "Error: boom", toolCallID: "toolu_2"),
            ServeMessage(role: .user, text: "and then?"),
        ])
    }

    @Test func aTextBlockBeforeAToolResultStaysBeforeIt() throws {
        let body = request("").replacingOccurrences(of: #""messages":[{"role":"user","content":"hi"}]"#, with: #"""
        "messages":[{"role":"user","content":[{"type":"text","text":"first"},{"type":"tool_result","tool_use_id":"t","content":"r"}]}]
        """#)
        let decoded = try AnthropicMessagesDialect.decode(Data(body.utf8))
        #expect(decoded.messages.map(\.role) == [.user, .tool])
    }

    @Test func toolsAreReshapedIntoTheChatFormTheTemplatesRead() throws {
        let decoded = try AnthropicMessagesDialect.decode(Data(request(#"""
        ,"tools":[{"name":"get_weather","description":"Weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}},"cache_control":{"type":"ephemeral"}}]
        """#).utf8))
        let tool = try #require(decoded.tools.first)
        #expect(tool.name == "get_weather" && decoded.toolsRequested)
        let function = try #require(tool.object?["function"] as? [String: Any])
        #expect(tool.object?["type"] as? String == "function")
        #expect(function["name"] as? String == "get_weather")
        #expect(function["description"] as? String == "Weather")
        #expect((function["parameters"] as? [String: Any])?["type"] as? String == "object")
    }

    @Test func toolChoiceMapsAnyToRequiredAndToolToFunction() throws {
        func choice(_ json: String) throws -> ServeToolChoice {
            try AnthropicMessagesDialect.decode(Data(request(#","tool_choice":\#(json)"#).utf8)).toolChoice
        }
        #expect(try choice(#"{"type":"auto","disable_parallel_tool_use":true}"#) == .auto)
        #expect(try choice(#"{"type":"any"}"#) == .required)
        #expect(try choice(#"{"type":"tool","name":"x"}"#) == .function("x"))
        #expect(try choice(#"{"type":"none"}"#) == .none)
        #expect(throws: ServeError.self) { try choice(#"{"type":"bogus"}"#) }
    }

    /// R34 #4: enabled/adaptive → on; disabled / between_tools / anything else → off; absent → not asked.
    @Test func thinkingTypeMapsPerTheRuling() throws {
        func thinking(_ json: String?) throws -> Bool? {
            try AnthropicMessagesDialect.decode(Data(request(json.map { #","thinking":\#($0)"# } ?? "").utf8)).reasoning
        }
        #expect(try thinking(#"{"type":"enabled","budget_tokens":2048}"#) == true)
        #expect(try thinking(#"{"type":"adaptive","display":"omitted"}"#) == true)
        #expect(try thinking(#"{"type":"disabled"}"#) == false)
        #expect(try thinking(#"{"type":"between_tools"}"#) == false)
        #expect(try thinking(#"{"type":"from-the-future"}"#) == false)
        #expect(try thinking(nil) == nil)
    }

    @Test func maxTokensIsRequiredForMessagesButNotForCountTokens() throws {
        let body = Data(#"{"model":"m","messages":[{"role":"user","content":"x"}]}"#.utf8)
        let error = #expect(throws: ServeError.self) { try AnthropicMessagesDialect.decode(body) }
        #expect(error?.status == 400 && error?.message.contains("max_tokens") == true)
        #expect(try AnthropicMessagesDialect.decode(body, requireMaxTokens: false).maxTokens == nil)
    }

    @Test func limitsStopSequencesAndSamplingAreCarried() throws {
        let decoded = try AnthropicMessagesDialect.decode(Data(
            request(#","stop_sequences":["END","STOP"],"temperature":0.2,"top_p":0.9"#).utf8))
        #expect(decoded.maxTokens == 64 && decoded.stop == ["END", "STOP"])
        #expect(decoded.temperature == 0.2 && decoded.topP == 0.9)
    }

    /// R35 #9: accepted and ignored, never a 400.
    @Test func listedFieldsAreAcceptedAndIgnored() throws {
        let decoded = try AnthropicMessagesDialect.decode(Data(request(#"""
        ,"top_k":5,"metadata":{"user_id":"u"},"service_tier":"auto","tool_choice":{"type":"auto","disable_parallel_tool_use":true},"container":"c","output_config":{"effort":"high"}
        """#).utf8))
        #expect(decoded.messages == [ServeMessage(role: .user, text: "hi")])
    }

    @Test func unsupportedBlocksAndServerToolsAre400sNamingThem() {
        let image = request("").replacingOccurrences(of: #""content":"hi""#,
            with: #""content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA"}}]"#)
        let error = #expect(throws: ServeError.self) { try AnthropicMessagesDialect.decode(Data(image.utf8)) }
        #expect(error?.message.contains("image") == true)
        let server = #expect(throws: ServeError.self) {
            try AnthropicMessagesDialect.decode(Data(request(#","tools":[{"type":"web_search_20250305","name":"web_search"}]"#).utf8))
        }
        #expect(server?.message.contains("web_search_20250305") == true)
    }
}

// MARK: - Encode

struct AnthropicMessagesEncodeTests {
    private func body(_ collected: ServeCollected) throws -> [String: Any] {
        try #require(object(AnthropicMessagesDialect.messageBody(id: "msg_x", model: "m", collected: collected)))
    }

    @Test func aTextReplyHasTheMessageEnvelope() throws {
        var collected = ServeCollected()
        collected.text = "Hello"
        collected.usage = ServeUsage(promptTokens: 7, completionTokens: 2)
        let message = try body(collected)
        #expect(message["type"] as? String == "message" && message["role"] as? String == "assistant")
        #expect(message["id"] as? String == "msg_x" && message["model"] as? String == "m")
        #expect(message["stop_reason"] as? String == "end_turn")
        #expect(message["stop_sequence"] is NSNull)
        let content = try #require(message["content"] as? [[String: Any]])
        #expect(content.count == 1 && content[0]["type"] as? String == "text" && content[0]["text"] as? String == "Hello")
    }

    @Test func thinkingThenTextThenToolUsesInThatOrderWithUnsignedThinking() throws {
        var collected = ServeCollected()
        collected.reasoning = "plan"
        collected.text = "Looking."
        collected.toolCalls = [ServeToolCall(id: "call_1", name: "f", argumentsJSON: #"{"a":1,"b":[true]}"#)]
        collected.finish = .toolCalls
        let message = try body(collected)
        #expect(message["stop_reason"] as? String == "tool_use")
        let content = try #require(message["content"] as? [[String: Any]])
        #expect(content.map { $0["type"] as? String } == ["thinking", "text", "tool_use"])
        #expect(content[0]["thinking"] as? String == "plan" && content[0]["signature"] as? String == "")
        #expect(content[2]["id"] as? String == "call_1" && content[2]["name"] as? String == "f")
        let input = try #require(content[2]["input"] as? [String: Any])        // an object, not a string
        #expect(input["a"] as? Int == 1 && (input["b"] as? [Bool]) == [true])
    }

    @Test func aToolOnlyTurnHasNoEmptyTextBlockAndAnEmptyReplyIsNeverAnEmptyArray() throws {
        var calls = ServeCollected()
        calls.text = "  \n"
        calls.toolCalls = [ServeToolCall(id: "c", name: "f", argumentsJSON: "{}")]
        #expect((try body(calls)["content"] as? [[String: Any]])?.map { $0["type"] as? String } == ["tool_use"])
        let content = try #require(try body(ServeCollected())["content"] as? [[String: Any]])
        #expect(content.count == 1 && content[0]["text"] as? String == "")
    }

    @Test func theStopReasonMap() {
        #expect(AnthropicMessagesDialect.stopReason(.stop) == "end_turn")
        #expect(AnthropicMessagesDialect.stopReason(.length) == "max_tokens")
        #expect(AnthropicMessagesDialect.stopReason(.toolCalls) == "tool_use")
    }

    /// R34 #10: `input_tokens` excludes what the cache supplied; `cache_read_input_tokens` is that part.
    @Test func usageSubtractsTheCachedTokensAndOmitsTheFieldWhenZero() {
        let miss = AnthropicMessagesDialect.usageObject(ServeUsage(promptTokens: 100, completionTokens: 5))
        #expect(miss["input_tokens"] as? Int == 100 && miss["output_tokens"] as? Int == 5)
        #expect(miss["cache_read_input_tokens"] == nil)

        let hit = AnthropicMessagesDialect.usageObject(ServeUsage(promptTokens: 100, completionTokens: 5, cachedTokens: 80))
        #expect(hit["input_tokens"] as? Int == 20)
        #expect(hit["cache_read_input_tokens"] as? Int == 80)
        #expect(hit["output_tokens"] as? Int == 5)
        // Anthropic's own identity: total input = cache_read + input_tokens.
        #expect((hit["input_tokens"] as? Int ?? 0) + (hit["cache_read_input_tokens"] as? Int ?? 0) == 100)
    }

    @Test func errorsUseAnthropicsEnvelopeAndTypes() throws {
        let body = AnthropicMessagesDialect.errorBody(ServeError(status: 404, message: "no", code: "model_not_found"))
        let decoded = try #require(object(body))
        #expect(decoded["type"] as? String == "error")
        let inner = try #require(decoded["error"] as? [String: Any])
        #expect(inner["type"] as? String == "not_found_error" && inner["message"] as? String == "no")
        #expect(inner["code"] == nil)
        let types = [400: "invalid_request_error", 403: "permission_error", 404: "not_found_error",
                     405: "invalid_request_error", 411: "invalid_request_error", 413: "request_too_large",
                     500: "api_error", 501: "api_error", 503: "overloaded_error"]
        for (status, type) in types { #expect(AnthropicMessagesDialect.errorType(forStatus: status) == type) }
    }
}

// MARK: - Route

struct AnthropicMessagesRouteTests {
    @Test func answersANonStreamingMessage() async throws {
        let backend = MessagesBackend()
        let response = try #require(await send(request(), env: environment(backend)))
        #expect(response.status == 200 && response.headers["Content-Type"] == "application/json")
        let message = try #require(object(response.body))
        #expect(message["type"] as? String == "message" && message["model"] as? String == model)
        #expect((message["content"] as? [[String: Any]])?.first?["text"] as? String == "ok")
        let usage = try #require(message["usage"] as? [String: Int])
        #expect(usage == ["input_tokens": 10, "output_tokens": 2])
        #expect((message["id"] as? String)?.hasPrefix("msg_") == true)
    }

    @Test func aCacheHitIsReportedTheAnthropicWay() async throws {
        let backend = MessagesBackend(events: [.textDelta("ok"),
            .usage(ServeUsage(promptTokens: 1000, completionTokens: 3, cachedTokens: 900)), .finish(.stop)])
        let response = try #require(await send(request(), env: environment(backend)))
        let usage = try #require(object(response.body)?["usage"] as? [String: Int])
        #expect(usage == ["input_tokens": 100, "cache_read_input_tokens": 900, "output_tokens": 3])
    }

    /// R34 #2: a matched stop sequence is still `end_turn` with `stop_sequence: null` — the core
    /// doesn't carry which string matched. Pinned so the gap is on the record.
    @Test func aMatchedStopSequenceStillAnswersEndTurn() async throws {
        let backend = MessagesBackend()
        let response = try #require(await send(request(#","stop_sequences":["ok"]"#), env: environment(backend)))
        let message = try #require(object(response.body))
        #expect(message["stop_reason"] as? String == "end_turn")
        #expect(message["stop_sequence"] is NSNull)
        #expect(backend.generated.first?.stop == ["ok"])
    }

    @Test func aToolCallBecomesAToolUseBlockAndEndsToolUse() async throws {
        let backend = MessagesBackend(events: [
            .toolCallDelta(ServeToolCallDelta(index: 0, id: "call_9", name: "get_weather", argumentsFragment: #"{"city":"Oslo"}"#)),
            .usage(ServeUsage(promptTokens: 4, completionTokens: 6)), .finish(.toolCalls)])
        let tools = #","tools":[{"name":"get_weather","input_schema":{"type":"object"}}]"#
        let response = try #require(await send(request(tools), env: environment(backend)))
        let message = try #require(object(response.body))
        #expect(message["stop_reason"] as? String == "tool_use")
        let block = try #require((message["content"] as? [[String: Any]])?.first)
        #expect(block["type"] as? String == "tool_use" && block["id"] as? String == "call_9")
        #expect((block["input"] as? [String: Any])?["city"] as? String == "Oslo")
        #expect(backend.generated.first?.tools.map(\.name) == ["get_weather"])        // the gate let them through
    }

    @Test func streamTrueIsAnAnthropicShaped400ForNow() async throws {
        let backend = MessagesBackend()
        let response = try #require(await send(request(#","stream":true"#), env: environment(backend)))
        #expect(response.status == 400)
        let error = try #require(object(response.body)?["error"] as? [String: Any])
        #expect(object(response.body)?["type"] as? String == "error")
        #expect(error["type"] as? String == "invalid_request_error")
        #expect((error["message"] as? String)?.contains("S3-2b") == true)
        #expect(backend.generated.isEmpty)
    }

    @Test func errorsKeepOurStatusesInAnthropicsShape() async throws {
        let missing = try #require(await send(request().replacingOccurrences(of: model, with: "nope"), env: environment(MessagesBackend())))
        #expect(missing.status == 404)
        #expect((object(missing.body)?["error"] as? [String: Any])?["type"] as? String == "not_found_error")

        let noMax = try #require(await send(#"{"model":"\#(model)","messages":[{"role":"user","content":"x"}]}"#,
                                            env: environment(MessagesBackend())))
        #expect(noMax.status == 400)
        #expect((object(noMax.body)?["error"] as? [String: Any])?["type"] as? String == "invalid_request_error")

        // A full line is a 503 with Retry-After, typed overloaded_error.
        let queue = ServeQueue(globalCap: 1, perModelDepth: 0)
        _ = await queue.reserve(model: model)
        let busy = try #require(await send(request(), env: environment(MessagesBackend(), queue: queue)))
        #expect(busy.status == 503 && busy.headers["Retry-After"] == "2")
        #expect((object(busy.body)?["error"] as? [String: Any])?["type"] as? String == "overloaded_error")
    }

    @Test func theLogLineIsMetadataOnly() async throws {
        let log = RequestLog()
        _ = await send(request(), env: environment(MessagesBackend(), log: log))
        let entry = try #require(log.entries.last)
        #expect(entry.status == 200 && entry.model == model && entry.promptTokens == 10 && entry.userAgent == "claude-cli/test")
    }
}

// MARK: - count_tokens

struct AnthropicCountTokensTests {
    private let countPath = "/v1/messages/count_tokens"

    @Test func returnsTheCountersNumberWithoutGeneratingOrTakingASlot() async throws {
        let backend = MessagesBackend(count: 1234)
        let queue = ServeQueue(globalCap: 1, perModelDepth: 0)
        _ = await queue.reserve(model: model)                          // the only slot is taken…
        let body = #"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}]}"#      // …and no max_tokens
        let response = try #require(await send(body, path: countPath, env: environment(backend, queue: queue)))
        #expect(response.status == 200)
        #expect(object(response.body) as? [String: Int] == ["input_tokens": 1234])
        #expect(backend.generated.isEmpty)
    }

    /// The count is made from the **same** request the real call would generate from: system, tools and
    /// thinking all reach the counter, and it is byte-for-byte what `/v1/messages` pre-counts.
    @Test func countsTheSameRenderAsARealRequest() async throws {
        let extra = #","system":"S","thinking":{"type":"enabled","budget_tokens":1024},"tools":[{"name":"f","input_schema":{"type":"object"}}]"#
        let backend = MessagesBackend()
        _ = await send(request(extra), env: environment(backend))
        _ = await send(request(extra), path: countPath, env: environment(backend))
        let counted = backend.counted
        #expect(counted.count == 2)
        #expect(counted[0] == counted[1])
        #expect(counted[1].messages.first == ServeMessage(role: .system, text: "S"))
        #expect(counted[1].tools.map(\.name) == ["f"] && counted[1].reasoning == true)
    }

    @Test func aPromptPastTheWindowIsStillCounted() async throws {
        let backend = MessagesBackend(count: 5000, contextLimit: 100)
        let response = try #require(await send(request(), path: countPath, env: environment(backend)))
        #expect(response.status == 200 && object(response.body) as? [String: Int] == ["input_tokens": 5000])
        let real = try #require(await send(request(), env: environment(backend)))     // …while a real request is refused
        #expect(real.status == 400)
    }

    @Test func aBackendThatCantCountIsAnAnthropicErrorNotAnEstimate() async throws {
        let response = try #require(await send(request(), path: countPath, env: environment(MessagesBackend(count: nil))))
        #expect(response.status == 501)
        let decoded = try #require(object(response.body))
        #expect(decoded["type"] as? String == "error" && decoded["input_tokens"] == nil)
        #expect((decoded["error"] as? [String: Any])?["type"] as? String == "api_error")
    }

    @Test func aTemplateRefusalIsA400() async throws {
        final class Refusing: ServeBackend, @unchecked Sendable {
            let modelType: String? = "qwen3"
            func generate(_ request: ServeRequest) -> AsyncThrowingStream<ServeEvent, Error> { AsyncThrowingStream { $0.finish() } }
            func countPromptTokens(_ request: ServeRequest) async throws -> Int? {
                throw ServeError(status: 400, message: "template said no", code: "template_error")
            }
        }
        let response = try #require(await send(request(), path: countPath, env: environment(Refusing())))
        #expect(response.status == 400)
        #expect((object(response.body)?["error"] as? [String: Any])?["message"] as? String == "template said no")
    }
}

// MARK: - The request guard's renderer (R35)

struct AnthropicGuardRenderingTests {
    @Test func aRejectionOnMessagesIsAnthropicShapedAndOnChatCompletionsUnchanged() async throws {
        let env = environment(MessagesBackend())
        for path in ["/v1/messages", "/v1/messages/count_tokens"] {
            let response = try #require(await send(request(), path: path, host: "evil.example:1212", env: env))
            #expect(response.status == 403)
            let decoded = try #require(object(response.body))
            #expect(decoded["type"] as? String == "error")
            #expect((decoded["error"] as? [String: Any])?["type"] as? String == "permission_error")
        }
        let chat = try #require(await send(request(), path: "/v1/chat/completions", host: "evil.example:1212", env: env))
        #expect(chat.status == 403)
        let inner = try #require(object(chat.body)?["error"] as? [String: Any])
        #expect(inner["code"] as? String == "invalid_host" && inner["type"] as? String == "invalid_request_error")
        #expect(object(chat.body)?["type"] == nil)
    }

    @Test func theGuardDecidesTheSameOnEveryPath() async throws {
        let env = environment(MessagesBackend())
        for path in ["/v1/messages", "/v1/chat/completions"] {
            let origin = ServeRequestHead(method: "POST", path: path, host: "127.0.0.1:1212", origin: "http://x")
            if case .full(let response) = await ServeAPI.handle(head: origin, guard: RequestGuard(port: 1212), environment: env, body: { Data() }) {
                #expect(response.status == 403)
            }
        }
    }

    @Test func wrongMethodAndUnknownSubpathsAreAnthropicShapedUnderMessages() async throws {
        let env = environment(MessagesBackend())
        let get = try #require(await send("", method: "GET", env: env))
        #expect(get.status == 405 && get.headers["Allow"] == "POST")
        #expect(object(get.body)?["type"] as? String == "error")
        let unknown = try #require(await send(request(), path: "/v1/messages/batches", env: env))
        #expect(unknown.status == 404 && object(unknown.body)?["type"] as? String == "error")
        #expect(AnthropicMessagesDialect.owns("/v1/messages") && !AnthropicMessagesDialect.owns("/v1/messagesX"))
    }

    @Test func authHeadersAreNeverRequired() async throws {
        // No `x-api-key`, no `Authorization`, no `anthropic-version`: the request head has no field for
        // them, and the route answers regardless (P1).
        let response = try #require(await send(request(), env: environment(MessagesBackend())))
        #expect(response.status == 200)
    }
}
