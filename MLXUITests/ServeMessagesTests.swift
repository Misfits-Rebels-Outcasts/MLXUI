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
                         log: RequestLog = RequestLog(), keepAlive: Double = 1.0) -> ServeEnvironment {
    ServeEnvironment(
        servedModelIDs: { [model] },
        installed: { InstalledModelIndex(entries: [
            .init(id: "mlx-community--Qwen3-4B-4bit", hfModelId: model, kind: .llm, ramGB: 3, downloadSizeGB: 2)]) },
        appleFoundationReadiness: { nil },
        created: 1_700_000_000,
        backendFor: { $0 == model ? backend : nil },
        queue: queue, requestLog: log, keepAliveSeconds: keepAlive)
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

    /// R43: Claude Code sends `system` entries inside `messages` (string or text blocks).
    @Test func systemEntriesInsideMessagesAreAcceptedAsSystemMessages() throws {
        func messages(_ json: String, top: String = "") throws -> [ServeMessage] {
            try AnthropicMessagesDialect.decode(Data(
                #"{"model":"m","max_tokens":9\#(top),"messages":\#(json)}"#.utf8)).messages
        }
        // at messages[0], as a string
        #expect(try messages(#"[{"role":"system","content":"S0"},{"role":"user","content":"hi"}]"#)
                == [ServeMessage(role: .system, text: "S0"), ServeMessage(role: .user, text: "hi")])
        // as text blocks (with cache_control), after a user turn
        #expect(try messages(#"[{"role":"user","content":"hi"},{"role":"system","content":[{"type":"text","text":"A","cache_control":{"type":"ephemeral"}},{"type":"text","text":"B"}]}]"#)
                == [ServeMessage(role: .user, text: "hi"), ServeMessage(role: .system, text: "A\n\nB")])
        // mid-conversation, between tool traffic, with a top-level system as well (it stays first)
        let mixed = try messages(#"""
        [{"role":"user","content":"go"},{"role":"system","content":"S1"},
         {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{}}]},
         {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]},
         {"role":"system","content":[{"type":"text","text":"S2"}]}]
        """#, top: #","system":"TOP""#)
        #expect(mixed.map(\.role) == [.system, .user, .system, .assistant, .tool, .system])
        #expect(mixed.map(\.text) == ["TOP", "go", "S1", "", "ok", "S2"])
        // an empty system entry adds nothing
        #expect(try messages(#"[{"role":"system","content":""},{"role":"user","content":"hi"}]"#) == [ServeMessage(role: .user, text: "hi")])
    }

    /// …and the assembly places them per family, as for every dialect.
    @Test func systemEntriesAreHoistedPerFamilyByTheAssembly() throws {
        let decoded = try AnthropicMessagesDialect.decode(Data(
            #"{"model":"m","max_tokens":9,"system":"TOP","messages":[{"role":"user","content":"go"},{"role":"system","content":"MID"},{"role":"user","content":"again"}]}"#.utf8))
        let firstOnly = ServeMessageAssembly.assembling(decoded.messages, acceptsTools: true, placement: .firstOnly)
        #expect(firstOnly.map(\.role) == [.system, .user])
        #expect(firstOnly.first?.text == "TOP\n\nMID")
        let anywhere = ServeMessageAssembly.assembling(decoded.messages, acceptsTools: true, placement: .anywhere)
        #expect(anywhere.map(\.role) == [.system, .user, .system, .user])
    }

    @Test func anUnknownRoleOrANonTextSystemBlockIsStillA400() {
        let role = #expect(throws: ServeError.self) {
            try AnthropicMessagesDialect.decode(Data(#"{"model":"m","max_tokens":9,"messages":[{"role":"narrator","content":"x"}]}"#.utf8))
        }
        #expect(role?.code == "unsupported_role")
        let block = #expect(throws: ServeError.self) {
            try AnthropicMessagesDialect.decode(Data(#"{"model":"m","max_tokens":9,"messages":[{"role":"system","content":[{"type":"image","source":{}}]}]}"#.utf8))
        }
        #expect(block?.code == "unsupported_content_block")
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

    /// R44 (amends R34 #4): only `enabled` → on; adaptive / disabled / between_tools / anything else → off; absent → not asked.
    @Test func thinkingTypeMapsPerTheRuling() throws {
        func thinking(_ json: String?) throws -> Bool? {
            try AnthropicMessagesDialect.decode(Data(request(json.map { #","thinking":\#($0)"# } ?? "").utf8)).reasoning
        }
        #expect(try thinking(#"{"type":"enabled","budget_tokens":2048}"#) == true)
        #expect(try thinking(#"{"type":"adaptive","display":"omitted"}"#) == false)      // what Claude Code sends on every call
        #expect(try thinking(#"{"type":"adaptive"}"#) == false)
        #expect(try thinking(#"{"type":"disabled"}"#) == false)
        #expect(try thinking(#"{"type":"between_tools"}"#) == false)
        #expect(try thinking(#"{"type":"from-the-future"}"#) == false)
        #expect(try thinking(nil) == nil)
    }

    /// The decision reaches the template switch: adaptive (as Claude Code sends it) leaves thinking off, `enabled` turns it on.
    @Test func adaptiveThinkingIsOffAndEnabledIsOnAtTheTemplate() throws {
        func on(_ thinking: String) throws -> Bool {
            let r = try AnthropicMessagesDialect.decode(Data(request(#","thinking":\#(thinking)"#).utf8))
            return ReasoningTemplateTable.thinkingOn(requested: r.reasoning, forModelType: "qwen3")
        }
        #expect(try on(#"{"type":"adaptive","display":"omitted"}"#) == false)
        #expect(try on(#"{"type":"enabled","budget_tokens":2048}"#) == true)
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

    /// R36: an image or document block is a text placeholder, never a 400 — in user content and inside a
    /// tool_result. Server tools stay a 400.
    @Test func imagesAndDocumentsBecomePlaceholdersNotErrors() throws {
        let image = #"{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA"}}"#
        let document = #"{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"AA"}}"#
        let body = request("").replacingOccurrences(of: #""messages":[{"role":"user","content":"hi"}]"#, with: """
        "messages":[{"role":"user","content":[{"type":"text","text":"look"},\(image),\(document),
          {"type":"tool_result","tool_use_id":"t","content":[\(image),\(document)]}]}]
        """)
        let decoded = try AnthropicMessagesDialect.decode(Data(body.utf8))
        #expect(decoded.messages == [
            ServeMessage(role: .user, text: "look\n[image omitted: this model can't see images]\n[document omitted: this model can't read documents]"),
            ServeMessage(role: .tool, text: "[image omitted: this model can't see images][document omitted: this model can't read documents]",
                         toolCallID: "t"),
        ])
    }

    @Test func serverToolsAndUnknownBlocksAreStill400s() {
        let server = #expect(throws: ServeError.self) {
            try AnthropicMessagesDialect.decode(Data(request(#","tools":[{"type":"web_search_20250305","name":"web_search"}]"#).utf8))
        }
        #expect(server?.message.contains("web_search_20250305") == true)
        let unknown = request("").replacingOccurrences(of: #""content":"hi""#, with: #""content":[{"type":"server_tool_use","id":"x"}]"#)
        #expect(throws: ServeError.self) { try AnthropicMessagesDialect.decode(Data(unknown.utf8)) }
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


// MARK: - Streaming (S3-2b)

/// One SSE event of a stream: its `event:` name and decoded `data:` JSON.
private struct SSEEvent {
    let name: String
    let json: [String: Any]
    /// `type[:index][:kind]` — enough to read a whole sequence at a glance.
    var summary: String {
        var parts = [name]
        if let index = json["index"] as? Int { parts.append(String(index)) }
        if let kind = (json["delta"] as? [String: Any])?["type"] as? String ?? (json["content_block"] as? [String: Any])?["type"] as? String {
            parts.append(kind)
        }
        return parts.joined(separator: ":")
    }
}

private func parseSSE(_ text: String) -> [SSEEvent] {
    text.components(separatedBy: "\n\n").filter { !$0.isEmpty }.compactMap { chunk in
        let lines = chunk.components(separatedBy: "\n")
        guard let name = lines.first(where: { $0.hasPrefix("event: ") })?.dropFirst(7),
              let data = lines.first(where: { $0.hasPrefix("data: ") })?.dropFirst(6),
              let json = object(Data(data.utf8)) else { return nil }
        return SSEEvent(name: String(name), json: json)
    }
}

private func stream(_ events: [ServeEvent], _ extra: String = #","stream":true"#, backend: ServeBackend? = nil,
                    keepAlive: Double = 1.0) async -> (head: (status: Int, headers: [String: String])?, events: [SSEEvent], text: String) {
    let body = Data(request(extra).utf8)
    let head = ServeRequestHead(method: "POST", path: "/v1/messages", host: "127.0.0.1:1212", origin: nil,
                                transferEncoding: nil, contentLength: String(body.count), userAgent: nil)
    let env = environment(backend ?? MessagesBackend(events: events), keepAlive: keepAlive)
    guard case .stream(let status, let headers, let bytes) = await ServeAPI.handle(
        head: head, guard: RequestGuard(port: 1212), environment: env, body: { body }) else { return (nil, [], "") }
    var all: [UInt8] = []
    for await chunk in bytes { all += chunk }
    let text = String(decoding: all, as: UTF8.self)
    return ((status, headers), parseSSE(text), text)
}

private func usage(_ prompt: Int, _ completion: Int, cached: Int = 0) -> ServeEvent {
    .usage(ServeUsage(promptTokens: prompt, completionTokens: completion, cachedTokens: cached))
}

private func call(_ index: Int, id: String?, name: String? = nil, args: String? = nil) -> ServeEvent {
    .toolCallDelta(ServeToolCallDelta(index: index, id: id, name: name, argumentsFragment: args))
}

struct AnthropicStreamTests {
    @Test func textOnly() async throws {
        let run = await stream([.textDelta("Hel"), .textDelta("lo"), usage(10, 2), .finish(.stop)])
        #expect(run.head?.status == 200 && run.head?.headers["Content-Type"] == "text/event-stream")
        #expect(run.events.map(\.summary) == [
            "message_start", "content_block_start:0:text", "content_block_delta:0:text_delta",
            "content_block_delta:0:text_delta", "content_block_stop:0", "message_delta", "message_stop"])
        let start = try #require(run.events[0].json["message"] as? [String: Any])
        #expect(start["role"] as? String == "assistant" && (start["content"] as? [Any])?.isEmpty == true)
        #expect(start["stop_reason"] is NSNull && (start["id"] as? String)?.hasPrefix("msg_") == true)
        #expect(run.events[1].json["content_block"] as? [String: String] == ["type": "text", "text": ""])
        let text = run.events.compactMap { ($0.json["delta"] as? [String: Any])?["text"] as? String }.joined()
        #expect(text == "Hello")
        let delta = run.events[5].json
        #expect((delta["delta"] as? [String: Any])?["stop_reason"] as? String == "end_turn")
        #expect((delta["delta"] as? [String: Any])?["stop_sequence"] is NSNull)
        #expect(delta["usage"] as? [String: Int] == ["input_tokens": 10, "output_tokens": 2])
        // Every frame is `event:` + `data:` whose `type` matches the event name.
        #expect(run.events.allSatisfy { $0.json["type"] as? String == $0.name })
    }

    @Test func thinkingThenText() async throws {
        let run = await stream([.reasoningDelta("pl"), .reasoningDelta("an"), .textDelta("Hi"), usage(5, 4), .finish(.stop)])
        #expect(run.events.map(\.summary) == [
            "message_start",
            "content_block_start:0:thinking", "content_block_delta:0:thinking_delta", "content_block_delta:0:thinking_delta",
            "content_block_delta:0:signature_delta", "content_block_stop:0",
            "content_block_start:1:text", "content_block_delta:1:text_delta", "content_block_stop:1",
            "message_delta", "message_stop"])
        #expect(run.events[1].json["content_block"] as? [String: String] == ["type": "thinking", "thinking": "", "signature": ""])
        #expect((run.events[4].json["delta"] as? [String: Any])?["signature"] as? String == "")
    }

    @Test func textThenToolUse() async throws {
        let run = await stream([.textDelta("Looking."), call(0, id: "call_1", name: "get_weather", args: #"{"city":"Oslo"}"#),
                                usage(8, 9), .finish(.toolCalls)])
        #expect(run.events.map(\.summary) == [
            "message_start",
            "content_block_start:0:text", "content_block_delta:0:text_delta", "content_block_stop:0",
            "content_block_start:1:tool_use", "content_block_delta:1:input_json_delta", "content_block_stop:1",
            "message_delta", "message_stop"])
        let block = try #require(run.events[4].json["content_block"] as? [String: Any])
        #expect(block["id"] as? String == "call_1" && block["name"] as? String == "get_weather")
        #expect((block["input"] as? [String: Any])?.isEmpty == true)
        #expect((run.events[5].json["delta"] as? [String: String])?["partial_json"] == #"{"city":"Oslo"}"#)
        #expect((run.events[7].json["delta"] as? [String: Any])?["stop_reason"] as? String == "tool_use")
    }

    @Test func aToolOnlyTurnStartsAtIndexZero() async throws {
        let run = await stream([call(0, id: "call_1", name: "f", args: "{}"), usage(3, 4), .finish(.toolCalls)])
        #expect(run.events.map(\.summary) == [
            "message_start", "content_block_start:0:tool_use", "content_block_delta:0:input_json_delta",
            "content_block_stop:0", "message_delta", "message_stop"])
    }

    @Test func twoToolCallsAreTwoBlocksWithRunningIndexes() async throws {
        let run = await stream([call(0, id: "call_a", name: "f", args: #"{"a":1}"#),
                                call(1, id: "call_b", name: "g", args: #"{"b":2}"#), usage(3, 8), .finish(.toolCalls)])
        #expect(run.events.map(\.summary) == [
            "message_start",
            "content_block_start:0:tool_use", "content_block_delta:0:input_json_delta", "content_block_stop:0",
            "content_block_start:1:tool_use", "content_block_delta:1:input_json_delta", "content_block_stop:1",
            "message_delta", "message_stop"])
        let ids = run.events.compactMap { ($0.json["content_block"] as? [String: Any])?["id"] as? String }
        #expect(ids == ["call_a", "call_b"])
    }

    @Test func aContinuationFragmentStaysInTheOpenToolBlock() async throws {
        let run = await stream([call(0, id: "call_a", name: "f", args: #"{"a":"#), call(0, id: nil, args: #"1}"#),
                                usage(3, 8), .finish(.toolCalls)])
        #expect(run.events.map(\.summary).filter { $0 == "content_block_start:0:tool_use" }.count == 1)
        let pieces = run.events.compactMap { ($0.json["delta"] as? [String: Any])?["partial_json"] as? String }
        #expect(pieces.joined() == #"{"a":1}"#)
    }

    @Test func blockSwitchingFollowsTheKindWithAnIncreasingIndex() async throws {
        let run = await stream([.textDelta("a"), .reasoningDelta("b"), .textDelta("c"), usage(1, 3), .finish(.length)])
        let starts = run.events.filter { $0.name == "content_block_start" }
        #expect(starts.map { $0.json["index"] as? Int } == [0, 1, 2])
        #expect(starts.compactMap { ($0.json["content_block"] as? [String: Any])?["type"] as? String } == ["text", "thinking", "text"])
        let stops = run.events.filter { $0.name == "content_block_stop" }
        #expect(stops.map { $0.json["index"] as? Int } == [0, 1, 2])
        #expect((run.events.first { $0.name == "message_delta" }?.json["delta"] as? [String: Any])?["stop_reason"] as? String == "max_tokens")
    }

    @Test func anEmptyReplyIsStartDeltaStop() async throws {
        let run = await stream([usage(4, 0), .finish(.stop)])
        #expect(run.events.map(\.summary) == ["message_start", "message_delta", "message_stop"])
    }

    /// `message_delta.usage` is cumulative: output tokens, the cache-adjusted input tokens, and
    /// `cache_read_input_tokens` only when something was reused (R34 #10).
    @Test func messageDeltaCarriesTheCacheAdjustedUsage() async throws {
        let hit = await stream([.textDelta("x"), usage(1000, 3, cached: 900), .finish(.stop)])
        let hitUsage = try #require(hit.events.first { $0.name == "message_delta" }?.json["usage"] as? [String: Int])
        #expect(hitUsage == ["input_tokens": 100, "cache_read_input_tokens": 900, "output_tokens": 3])
        let miss = await stream([.textDelta("x"), usage(1000, 3), .finish(.stop)])
        let missUsage = try #require(miss.events.first { $0.name == "message_delta" }?.json["usage"] as? [String: Int])
        #expect(missUsage == ["input_tokens": 1000, "output_tokens": 3])
    }

    @Test func aMidStreamErrorIsAnErrorEventAndNoMessageStop() async throws {
        let run = await stream([.textDelta("par"), .error(ServeError(status: 500, message: "The model failed to generate a reply.",
                                                                     type: "server_error", code: "generation_failed"))])
        #expect(run.events.map(\.name) == ["message_start", "content_block_start", "content_block_delta", "error"])
        let error = try #require(run.events.last?.json["error"] as? [String: String])
        #expect(run.events.last?.json["type"] as? String == "error")
        #expect(error["type"] == "api_error" && error["message"] == "The model failed to generate a reply.")
        #expect(!run.text.contains("message_stop") && !run.text.contains("message_delta"))
    }

    @Test func aFailureBeforeTheHeadIsTheJSONError() async throws {
        let body = Data(request(#","stream":true"#).replacingOccurrences(of: model, with: "nope").utf8)
        let head = ServeRequestHead(method: "POST", path: "/v1/messages", host: "127.0.0.1:1212", origin: nil,
                                    transferEncoding: nil, contentLength: String(body.count), userAgent: nil)
        guard case .full(let response) = await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212),
                                                                environment: environment(MessagesBackend()), body: { body }) else {
            Issue.record("expected a whole response, not a stream"); return
        }
        #expect(response.status == 404 && object(response.body)?["type"] as? String == "error")
    }

    /// S3-2c: `message_start` carries the prepare-time count (what the counter said), 0 when it couldn't count.
    @Test func messageStartCarriesThePrepareTimeCount() async throws {
        func startUsage(count: Int?) async throws -> [String: Int] {
            let backend = MessagesBackend(events: [.textDelta("x"), usage(999, 1), .finish(.stop)], count: count)
            let run = await stream([], backend: backend)
            let message = try #require(run.events.first?.json["message"] as? [String: Any])
            return try #require(message["usage"] as? [String: Int])
        }
        #expect(try await startUsage(count: 321) == ["input_tokens": 321, "output_tokens": 0])
        #expect(try await startUsage(count: nil) == ["input_tokens": 0, "output_tokens": 0])
    }

    @Test func keepAliveIsAnthropicsPing() async throws {
        let encoder = AnthropicStreamEncoder(id: "msg_x", model: "m")
        #expect(String(decoding: encoder.keepAlive, as: UTF8.self) == "event: ping\ndata: {\"type\":\"ping\"}\n\n")

        // …and the pipeline writes it while the model is still thinking about the first token.
        let slow = FakeBackend([.wait(0.45), .text("late"), .usage(prompt: 2, completion: 1), .finish(.stop)], modelType: "qwen3")
        let run = await stream([], backend: slow, keepAlive: 0.1)
        #expect(run.text.hasPrefix("event: ping\n"))
        #expect(run.events.first?.name == "ping" && run.events.contains { $0.name == "message_stop" })
        #expect(run.events.last?.name == "message_stop")
    }
}
