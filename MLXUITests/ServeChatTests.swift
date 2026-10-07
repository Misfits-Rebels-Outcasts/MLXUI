import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import MLXUI

// S1-3 — `POST /v1/chat/completions`: dialect, token pipeline, queue, route, wire.

// MARK: - Test doubles

/// A scripted backend: no model, no GPU. Records what it was asked and whether it was cancelled.
nonisolated final class FakeBackend: ServeBackend, @unchecked Sendable {
    enum Step: Sendable {
        case wait(Double)               // seconds
        case text(String)
        case usage(prompt: Int, completion: Int)
        case finish(ServeFinishReason)
        case toolCall(index: Int, id: String?, name: String?, arguments: String?)
        case reasoning(String)
        case fail
        /// S2-8: throw a specific `ServeError` (a backend failure — the route keeps it opaque).
        case failWith(ServeError)
        /// S2-8: report it on the stream's **error channel**, which is how a template refusal
        /// travels — the client sees this error, not a generic 500.
        case error(ServeError)
    }

    private let lock = NSLock()
    private var _requests: [ServeRequest] = []
    private var _cancelled = 0
    private var _active = 0
    private var _maxActive = 0
    let script: [Step]
    /// What the checkpoint reports as its `model_type` (S2-2): the route asks this to decide
    /// whether the client's `tools` are honoured. `nil` stands in for Apple Foundation Models.
    var modelType: String?
    /// The context window the checkpoint declares, and the prompt count the route pre-checks
    /// (S2-6). Both default to `nil` — "this backend can't count", which is S1's behaviour.
    var contextLimit: Int?
    var promptTokenCount: Int?

    /// S2-8: what the pre-flight render reports — a refusal, or a count.
    var refusal: ServeError?

    init(_ script: [Step], modelType: String? = nil,
         contextLimit: Int? = nil, promptTokenCount: Int? = nil, refusal: ServeError? = nil) {
        self.script = script
        self.modelType = modelType
        self.contextLimit = contextLimit
        self.promptTokenCount = promptTokenCount
        self.refusal = refusal
    }

    func countPromptTokens(_ request: ServeRequest) async throws -> Int? {
        if let refusal { throw refusal }
        return promptTokenCount
    }

    var requests: [ServeRequest] { lock.lock(); defer { lock.unlock() }; return _requests }
    var cancelled: Int { lock.lock(); defer { lock.unlock() }; return _cancelled }
    var maxActive: Int { lock.lock(); defer { lock.unlock() }; return _maxActive }

    func generate(_ request: ServeRequest) -> AsyncThrowingStream<ServeEvent, Error> {
        lock.lock(); _requests.append(request); _active += 1; _maxActive = max(_maxActive, _active); lock.unlock()
        return AsyncThrowingStream { continuation in
            let task = Task {
                for step in script {
                    if Task.isCancelled { break }
                    switch step {
                    case .wait(let seconds): try? await Task.sleep(for: .seconds(seconds))
                    case .text(let text): continuation.yield(.textDelta(text))
                    case .usage(let p, let c): continuation.yield(.usage(ServeUsage(promptTokens: p, completionTokens: c)))
                    case .finish(let reason): continuation.yield(.finish(reason))
                    case .toolCall(let index, let id, let name, let arguments):
                        continuation.yield(.toolCallDelta(ServeToolCallDelta(
                            index: index, id: id, name: name, argumentsFragment: arguments)))
                    case .reasoning(let text): continuation.yield(.reasoningDelta(text))
                    case .fail: continuation.finish(throwing: ServeError(status: 500, message: "boom", code: "boom")); return
                    case .failWith(let error): continuation.finish(throwing: error); return
                    // The real backend yields the error **and then finishes** — without the finish
                    // the consumer waits forever (which is how this fake was first written).
                    case .error(let error): continuation.yield(.error(error)); continuation.finish(); return
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { [self] reason in
                lock.lock(); _active -= 1; if case .cancelled = reason { _cancelled += 1 }; lock.unlock()
                task.cancel()
            }
        }
    }
}

private let testModel = "mlx-community/Qwen3-4B-4bit"

private func chatEnvironment(backend: ServeBackend?, queue: ServeQueue = ServeQueue(),
                             served: Set<String> = [testModel], keepAlive: Double = 1.0,
                             log: RequestLog = RequestLog()) -> ServeEnvironment {
    ServeEnvironment(
        servedModelIDs: { served },
        installed: { InstalledModelIndex(entries: [
            .init(id: "mlx-community--Qwen3-4B-4bit", hfModelId: testModel, kind: .llm, ramGB: 3, downloadSizeGB: 2)]) },
        appleFoundationReadiness: { nil },
        created: 1_700_000_000,
        backendFor: { id in id == testModel ? backend : nil },
        queue: queue, requestLog: log, keepAliveSeconds: keepAlive)
}

private func post(_ json: String, host: String = "127.0.0.1:1212", path: String = "/v1/chat/completions",
                  method: String = "POST", userAgent: String? = "test-agent/1.0") -> (ServeRequestHead, Data) {
    let body = Data(json.utf8)
    return (ServeRequestHead(method: method, path: path, host: host, origin: nil, transferEncoding: nil,
                             contentLength: String(body.count), userAgent: userAgent), body)
}

private func call(_ json: String, _ env: ServeEnvironment, host: String = "127.0.0.1:1212",
                  method: String = "POST") async -> ServeReply {
    let (head, body) = post(json, host: host, method: method)
    return await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212), environment: env, body: { body })
}

private func full(_ reply: ServeReply) -> ServeResponse? {
    if case .full(let response) = reply { return response }
    return nil
}

private func collect(_ reply: ServeReply) async -> (status: Int, headers: [String: String], text: String)? {
    guard case .stream(let status, let headers, let body) = reply else { return nil }
    var bytes: [UInt8] = []
    for await chunk in body { bytes += chunk }
    return (status, headers, String(decoding: bytes, as: UTF8.self))
}

private func jsonObject(_ data: Data) -> [String: Any]? {
    (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
}

private let hello = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"SECRET-PROMPT-TEXT"}]}"#

// MARK: - Dialect

struct OpenAIChatDialectTests {
    private func decode(_ json: String) throws -> ServeRequest { try OpenAIChatDialect.decode(Data(json.utf8)) }
    private func rejection(_ json: String) -> ServeError? {
        do { _ = try decode(json); return nil } catch { return error as? ServeError }
    }

    @Test func decodesAStringContentRequest() throws {
        let r = try decode(#"""
        {"model":"m","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}],
         "temperature":0.2,"top_p":0.9,"max_tokens":64,"stop":"END","stream":true,"seed":7,
         "stream_options":{"include_usage":true},"unknown_field":{"x":1}}
        """#)
        #expect(r.model == "m")
        #expect(r.messages == [ServeMessage(role: .system, text: "be brief"), ServeMessage(role: .user, text: "hi")])
        #expect(r.temperature == 0.2 && r.topP == 0.9 && r.maxTokens == 64 && r.seed == 7)
        #expect(r.stop == ["END"] && r.stream && r.includeUsage && !r.toolsRequested)
    }

    @Test func aContentArrayOfTextPartsIsJoined() throws {
        let r = try decode(#"{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"Hello, "},{"type":"text","text":"world"}]}]}"#)
        #expect(r.messages.first?.text == "Hello, world")
    }

    @Test func aNonTextPartIs400NamingTheType() {
        let error = rejection(#"{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}"#)
        #expect(error?.status == 400)
        #expect(error?.message.contains("image_url") == true)
    }

    @Test func toolsAreDecodedAndTheRouteDecidesWhetherTheyAreHonoured() throws {
        let r = try decode(#"{"model":"m","messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"auto"}"#)
        // S2-2: the dialect reports what the client sent; whether it is honoured depends on the
        // model, which only the route (through the backend's `modelType`) knows. `toolsIgnored` is
        // a request-log outcome now, not a decode result.
        #expect(r.toolsRequested)
        #expect(r.tools.map(\.name) == ["f"])
        #expect(r.toolChoice == .auto)
    }

    @Test func maxCompletionTokensIsAnAliasAndTheDefaultIs4096() throws {
        #expect(try decode(#"{"model":"m","messages":[{"role":"user","content":"x"}],"max_completion_tokens":10}"#).effectiveMaxTokens == 10)
        #expect(try decode(#"{"model":"m","messages":[{"role":"user","content":"x"}]}"#).effectiveMaxTokens == 4096)
    }

    @Test func stopAcceptsAnArray() throws {
        #expect(try decode(#"{"model":"m","messages":[{"role":"user","content":"x"}],"stop":["a","b"]}"#).stop == ["a", "b"])
        #expect(rejection(#"{"model":"m","messages":[{"role":"user","content":"x"}],"stop":["a","b","c","d","e"]}"#)?.status == 400)
    }

    @Test func developerIsSystemAndNullAssistantContentIsEmpty() throws {
        let r = try decode(#"{"model":"m","messages":[{"role":"developer","content":"d"},{"role":"assistant","content":null}]}"#)
        #expect(r.messages.map(\.role) == [.system, .assistant])
        #expect(r.messages.last?.text == "")
    }

    @Test(arguments: [
        "not json",
        "[1,2]",
        #"{"messages":[{"role":"user","content":"x"}]}"#,
        #"{"model":"m"}"#,
        #"{"model":"m","messages":[]}"#,
        #"{"model":"m","messages":[{"role":"critic","content":"x"}]}"#,
        #"{"model":"m","messages":[{"role":"user","content":5}]}"#,
        #"{"model":"m","messages":[{"role":"user","content":"x"}],"stream":"yes"}"#,
        #"{"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":0}"#,
        #"{"model":"m","messages":[{"role":"user","content":"x"}],"temperature":"hot"}"#,
    ])
    func malformedRequestsAre400(json: String) {
        #expect(rejection(json)?.status == 400)
    }

    private static let toolConversation = #"""
    {"model":"m","messages":[
      {"role":"user","content":"Read a.txt"},
      {"role":"assistant","content":null,"tool_calls":[
        {"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.txt\"}"}},
        {"id":"call_2","type":"function","function":{"name":"list_dir","arguments":"{}"}}]},
      {"role":"tool","tool_call_id":"call_1","content":"hello"},
      {"role":"tool","tool_call_id":"call_2","name":"explicit","content":[{"type":"text","text":"x"},{"type":"text","text":"y"}]},
      {"role":"tool","tool_call_id":"unknown","content":"z"},
      {"role":"assistant","content":"Done.","tool_calls":[{"id":"c3","type":"function","function":{"name":"f"}}]}
    ]}
    """#

    @Test func toolTrafficInTheHistoryIsFlattenedToPlainText() throws {
        // S2-2: `decode` keeps tool traffic structured (the tool-capable path); the flattening S1
        // shipped now lives in `ServeMessageAssembly.flatteningToolTraffic`, which the route runs
        // for a model that can't take tools. The wording below is S1's, unchanged.
        let decoded = try decode(Self.toolConversation)
        #expect(decoded.messages.map(\.role) == [.user, .assistant, .tool, .tool, .tool, .assistant])
        let r = ServeMessageAssembly.flatteningToolTraffic(decoded.messages)
        // Flattened first, then adjacent same-role turns merged (R21): the three tool results are one user turn.
        #expect(r.map(\.role) == [.user, .assistant, .user, .assistant])
        #expect(r[1].text == #"[called read_file with {"path":"a.txt"}]"# + "\n[called list_dir with {}]")
        // name found via tool_call_id; `name` wins and parts are joined; unknown id → "tool"
        #expect(r[2].text == "[tool result for read_file: hello]\n\n[tool result for explicit: xy]\n\n[tool result for tool: z]")
        #expect(r[3].text == "Done.\n[called f with {}]")                         // text kept, call appended
        #expect(!decoded.toolsRequested)                                           // only a `tools` field sets that
    }

    @Test func twoConsecutiveToolResultsBecomeOneUserTurn() throws {
        let decoded = try decode(#"{"model":"m","messages":[{"role":"user","content":"go"},{"role":"assistant","tool_calls":[{"id":"a","function":{"name":"f","arguments":"{}"}},{"id":"b","function":{"name":"g","arguments":"{}"}}]},{"role":"tool","tool_call_id":"a","content":"one"},{"role":"tool","tool_call_id":"b","content":"two"}]}"#)
        // Two results stay two turns for a tool-capable model (S2-2)…
        #expect(decoded.messages.map(\.role) == [.user, .assistant, .tool, .tool])
        // …and flatten into the single user turn S1 shipped for one that can't take tools.
        let r = ServeMessageAssembly.flatteningToolTraffic(decoded.messages)
        #expect(r.map(\.role) == [.user, .assistant, .user])
        #expect(r[2].text == "[tool result for f: one]\n\n[tool result for g: two]")
    }

    @Test func aToolResultFollowedByAUserMessageIsOneUserTurn() throws {
        let decoded = try decode(#"{"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"a","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"a","content":"result"},{"role":"user","content":"now summarise"}]}"#)
        #expect(decoded.messages.map(\.role) == [.assistant, .tool, .user])
        let r = ServeMessageAssembly.flatteningToolTraffic(decoded.messages)
        #expect(r.map(\.role) == [.assistant, .user])
        #expect(r[1].text == "[tool result for f: result]\n\nnow summarise")
    }

    @Test func twoPlainConsecutiveUserMessagesFromTheClientAreMerged() throws {
        let r = try decode(#"{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":"first"},{"role":"user","content":"second"},{"role":"assistant","content":"a"},{"role":"user","content":"third"}]}"#)
        #expect(r.messages.map(\.role) == [.system, .user, .assistant, .user])
        #expect(r.messages[1].text == "first\n\nsecond")
        #expect(r.messages[3].text == "third")                                     // alternation left alone
    }

    @Test func mergingNeverAddsABlankLineForAnEmptySide() {
        let merged = OpenAIChatDialect.mergingAdjacentRoles([ServeMessage(role: .assistant, text: ""),
                                                             ServeMessage(role: .assistant, text: "x")])
        #expect(merged == [ServeMessage(role: .assistant, text: "x")])
    }

    @Test func completionBodyMatchesTheOpenAIShape() throws {
        let body = OpenAIChatDialect.completionBody(id: "chatcmpl-1", created: 5, model: "m", text: "Hi!",
                                                    finish: .length, usage: ServeUsage(promptTokens: 3, completionTokens: 2))
        #expect(String(decoding: body, as: UTF8.self) ==
            #"{"choices":[{"finish_reason":"length","index":0,"message":{"content":"Hi!","role":"assistant"}}],"created":5,"id":"chatcmpl-1","model":"m","object":"chat.completion","usage":{"completion_tokens":2,"prompt_tokens":3,"total_tokens":5}}"#)
    }

    @Test func chunksMatchTheOpenAIShape() {
        func s(_ d: Data) -> String { String(decoding: d, as: UTF8.self) }
        #expect(s(OpenAIChatDialect.roleChunk(id: "i", created: 1, model: "m", includeUsage: false)) ==
            #"{"choices":[{"delta":{"content":"","role":"assistant"},"finish_reason":null,"index":0}],"created":1,"id":"i","model":"m","object":"chat.completion.chunk"}"#)
        #expect(s(OpenAIChatDialect.contentChunk(id: "i", created: 1, model: "m", text: "a/b", includeUsage: false)) ==
            #"{"choices":[{"delta":{"content":"a/b"},"finish_reason":null,"index":0}],"created":1,"id":"i","model":"m","object":"chat.completion.chunk"}"#)
        #expect(s(OpenAIChatDialect.finishChunk(id: "i", created: 1, model: "m", finish: .stop, includeUsage: true)) ==
            #"{"choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":1,"id":"i","model":"m","object":"chat.completion.chunk","usage":null}"#)
        #expect(s(OpenAIChatDialect.usageChunk(id: "i", created: 1, model: "m", usage: ServeUsage(promptTokens: 1, completionTokens: 2))) ==
            #"{"choices":[],"created":1,"id":"i","model":"m","object":"chat.completion.chunk","usage":{"completion_tokens":2,"prompt_tokens":1,"total_tokens":3}}"#)
    }

    @Test func sseFramingIsDataLinesAndABlankLine() {
        #expect(String(decoding: OpenAIChatDialect.sse(Data("{}".utf8)), as: UTF8.self) == "data: {}\n\n")
        #expect(String(decoding: OpenAIChatDialect.done, as: UTF8.self) == "data: [DONE]\n\n")
        #expect(String(decoding: OpenAIChatDialect.keepAlive, as: UTF8.self) == ": keep-alive\n\n")
    }
}

// MARK: - Token pipeline (TCP-1)

struct ServeTextPipelineTests {
    /// A tokenizer stand-in: id → piece, decode = concatenation. Pieces are chosen to split `<div><span>`
    /// exactly where `ToolCallProcessor` used to eat a character.
    private static let vocabulary: [Int: String] = [
        1: "<div", 2: "><", 3: "span", 4: ">", 5: "hello", 6: " world", 7: "<think>", 8: "reasoning", 9: "</think>",
        10: "\n\n", 11: "answer", 12: "ST", 13: "OP", 14: "x",
    ]
    private static func decode(_ ids: [Int]) -> String { ids.compactMap { vocabulary[$0] }.joined() }

    private func run(_ ids: [Int], stop: [String] = []) -> (deltas: [String], stopped: Bool) {
        var pipeline = ServeTextPipeline(decode: Self.decode, stop: stop)
        var deltas: [String] = []
        for id in ids {
            let out = pipeline.push(id)
            if !out.text.isEmpty { deltas.append(out.text) }
            if out.stopped { break }
        }
        let tail = pipeline.finish()
        if !tail.isEmpty { deltas.append(tail) }
        return (deltas, pipeline.stopped)
    }

    @Test func divSpanArrivesIntact() {
        // TCP-1 regression (journal 2026-230): `<div><span>` must not become `<div<span>`.
        let result = run([1, 2, 3, 4])
        #expect(result.deltas.joined() == "<div><span>")
        #expect(!result.deltas.joined().contains("<div<span"))
    }

    @Test func divSpanStaysIntactWhenToolsWerePresentInTheRequest() throws {
        // The request's `tools` are ignored in S1 (R9) and the pipeline never sees them; S2 keeps this test.
        let request = try OpenAIChatDialect.decode(Data(#"{"model":"m","messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f"}}]}"#.utf8))
        #expect(request.toolsRequested)
        #expect(run([1, 2, 3, 4]).deltas.joined() == "<div><span>")
    }

    @Test func deltasAreProgressiveNotOneLump() {
        #expect(run([5, 6]).deltas == ["hello", " world"])
    }

    @Test func incompleteUTF8IsHeldUntilItCompletes() {
        // 🙂 = F0 9F 99 82. A byte-level tokenizer decodes a partial sequence to U+FFFD.
        let bytes: [Int: [UInt8]] = [1: [0xF0, 0x9F], 2: [0x99, 0x82], 3: Array("ok".utf8)]
        func decode(_ ids: [Int]) -> String { String(decoding: ids.flatMap { bytes[$0] ?? [] }, as: UTF8.self) }
        var pipeline = ServeTextPipeline(decode: decode, stop: [])
        #expect(pipeline.push(1).text == "")             // held, not "\u{FFFD}"
        #expect(pipeline.push(2).text == "🙂")
        #expect(pipeline.push(3).text == "ok")
        #expect(pipeline.finish() == "")
    }

    @Test func aLeadingClosedThinkBlockIsStripped() {
        #expect(run([7, 8, 9, 10, 11]).deltas.joined() == "answer")
    }

    @Test func anUnclosedThinkBlockIsLeftAsIs() {
        #expect(run([7, 8]).deltas.joined() == "<think>reasoning")
    }

    @Test(arguments: [
        "<think>r</think>\n\nanswer", "  <think>r</think> x", "plain", "<thinking>x", "<think>never closed",
        "<th", "x<think>r</think>y", "", "   ",
    ])
    func theStreamingStripperMatchesLLMEngineStripThinkBlock(text: String) {
        var stripper = LeadingThinkStripper()
        var out = ""
        for character in text { out += stripper.push(String(character)) }
        out += stripper.finish()
        #expect(out == LLMEngine.stripThinkBlock(text))
    }

    @Test func aStopSequenceCutsTheText() {
        let result = run([5, 12, 13, 6], stop: ["STOP"])
        #expect(result.deltas.joined() == "hello")
        #expect(result.stopped)
    }

    @Test func aStopSequenceSplitAcrossTokensIsStillCaught() {
        let result = run([14, 12, 13, 14], stop: ["STOP"])
        #expect(result.deltas.joined() == "x")
        #expect(result.stopped)
    }

    @Test func textThatOnlyLooksLikeAStopIsNotLost() {
        let result = run([12, 14], stop: ["STOP"])                  // "STx"
        #expect(result.deltas.joined() == "STx")
        #expect(!result.stopped)
    }
}

// MARK: - Sampling defaults (R20)

struct SamplingDefaultsTests {
    private func parse(_ json: String?) -> SamplingDefaults { SamplingDefaults.parse(json.map { Data($0.utf8) }) }

    @Test func theCheckpointsGenerationConfigSuppliesBoth() {
        #expect(parse(#"{"temperature":0.6,"top_p":0.95,"top_k":20}"#) == SamplingDefaults(temperature: 0.6, topP: 0.95))
    }

    @Test func noFileMeansPointSevenAndOne() {
        #expect(parse(nil) == SamplingDefaults(temperature: 0.7, topP: 1.0))
        #expect(SamplingDefaults.fallback == SamplingDefaults(temperature: 0.7, topP: 1.0))
    }

    @Test func eachKeyFallsBackOnItsOwn() {
        #expect(parse(#"{"top_p":0.8}"#) == SamplingDefaults(temperature: 0.7, topP: 0.8))
        #expect(parse(#"{"temperature":0.3}"#) == SamplingDefaults(temperature: 0.3, topP: 1.0))
    }

    @Test(arguments: ["not json", "[]", #"{"temperature":"hot","top_p":true}"#, #"{"temperature":-1,"top_p":0}"#, #"{"top_p":5}"#])
    func anUnusableFileFallsBack(json: String) {
        #expect(parse(json) == .fallback)
    }

    @Test func loadReadsTheFileFromTheModelDirectory() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("sampling-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(SamplingDefaults.load(directory: dir) == .fallback)                // file absent
        try Data(#"{"temperature":0.6,"top_p":0.95}"#.utf8).write(to: dir.appendingPathComponent("generation_config.json"))
        #expect(SamplingDefaults.load(directory: dir) == SamplingDefaults(temperature: 0.6, topP: 0.95))   // file present
    }

    @Test func anExplicitRequestValueAlwaysWins() {
        let config = SamplingDefaults(temperature: 0.6, topP: 0.95)
        #expect(config.applying(temperature: 0.0, topP: nil) == SamplingDefaults(temperature: 0.0, topP: 0.95))   // 0 is explicit, not "missing"
        #expect(config.applying(temperature: nil, topP: 0.5) == SamplingDefaults(temperature: 0.6, topP: 0.5))
        #expect(config.applying(temperature: 1.2, topP: 0.4) == SamplingDefaults(temperature: 1.2, topP: 0.4))
        #expect(config.applying(temperature: nil, topP: nil) == config)
    }
}

// MARK: - Queue

@Suite(.timeLimit(.minutes(1)))
struct ServeQueueTests {
    @Test func theSecondRequestForAModelWaitsForTheFirst() async throws {
        let queue = ServeQueue()
        let a = try #require(await queue.reserve(model: "m"))
        let b = try #require(await queue.reserve(model: "m"))
        try await queue.waitTurn(a)
        let (r1, w1) = (await queue.runningCount, await queue.waitingCount)
        #expect(r1 == 1 && w1 == 1)

        let admitted = Flag()
        let waiter = Task { try await queue.waitTurn(b); admitted.set() }
        try await Task.sleep(for: .milliseconds(80))
        #expect(!admitted.isSet)                                       // b is still waiting
        await queue.release(a)
        try await waiter.value
        #expect(admitted.isSet)
        await queue.release(b)
        let (r2, w2) = (await queue.runningCount, await queue.waitingCount)
        #expect(r2 == 0 && w2 == 0)
    }

    @Test func theGlobalCapIsOneAcrossModels() async throws {
        let queue = ServeQueue()
        let a = try #require(await queue.reserve(model: "one"))
        let b = try #require(await queue.reserve(model: "two"))
        try await queue.waitTurn(a)
        #expect(await queue.runningCount == 1)                         // "two" waits although it is another model
        await queue.release(a)
        try await queue.waitTurn(b)
        #expect(await queue.runningCount == 1)
        await queue.release(b)
    }

    @Test func aFullLineIsRefused() async throws {
        let queue = ServeQueue()
        let running = try #require(await queue.reserve(model: "m"))
        for _ in 0..<4 { #expect(await queue.reserve(model: "m") != nil) }   // depth 4 waiting
        #expect(await queue.reserve(model: "m") == nil)                      // the fifth is the 503
        #expect(await queue.reserve(model: "other") != nil)                  // depth is per model
        await queue.release(running)
    }

    @Test func waitersAreAdmittedInArrivalOrder() async throws {
        let queue = ServeQueue()
        let first = try #require(await queue.reserve(model: "m"))
        let order = Order()
        var tasks: [Task<Void, Error>] = []
        for n in 1...3 {
            let t = try #require(await queue.reserve(model: "m"))
            tasks.append(Task { try await queue.waitTurn(t); order.add(n); await queue.release(t) })
            try await Task.sleep(for: .milliseconds(20))
        }
        await queue.release(first)
        for task in tasks { try await task.value }
        #expect(order.values == [1, 2, 3])
    }

    @Test func aCancelledWaiterGivesItsPlaceBack() async throws {
        let queue = ServeQueue()
        let running = try #require(await queue.reserve(model: "m"))
        let waiting = try #require(await queue.reserve(model: "m"))
        let task = Task { try await queue.waitTurn(waiting) }
        try await Task.sleep(for: .milliseconds(50))
        task.cancel()
        _ = await task.result
        try await Task.sleep(for: .milliseconds(50))
        #expect(await queue.waitingCount == 0)
        await queue.release(running)
        #expect(await queue.runningCount == 0)
    }
}

nonisolated final class Flag: @unchecked Sendable {
    private let lock = NSLock(); private var value = false
    func set() { lock.lock(); value = true; lock.unlock() }
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }
}
nonisolated final class Order: @unchecked Sendable {
    private let lock = NSLock(); private var items: [Int] = []
    func add(_ n: Int) { lock.lock(); items.append(n); lock.unlock() }
    var values: [Int] { lock.lock(); defer { lock.unlock() }; return items }
}

// MARK: - Route (through ServeAPI.handle, no socket)

@Suite(.timeLimit(.minutes(1)))
struct ChatCompletionsRouteTests {
    @Test func answersANonStreamingRequestWithUsage() async throws {
        let backend = FakeBackend([.text("Hel"), .text("lo"), .usage(prompt: 7, completion: 2), .finish(.stop)])
        let reply = await call(hello, chatEnvironment(backend: backend))
        let response = try #require(full(reply))
        #expect(response.status == 200)
        let object = try #require(jsonObject(response.body))
        #expect(object["object"] as? String == "chat.completion")
        let choice = try #require((object["choices"] as? [[String: Any]])?.first)
        #expect((choice["message"] as? [String: Any])?["content"] as? String == "Hello")
        #expect(choice["finish_reason"] as? String == "stop")
        let usage = try #require(object["usage"] as? [String: Int])
        #expect(usage == ["prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9])
    }

    @Test func reportsLengthWhenTheBackendHitsTheLimit() async throws {
        let backend = FakeBackend([.text("a"), .usage(prompt: 1, completion: 1), .finish(.length)])
        let response = try #require(full(await call(hello, chatEnvironment(backend: backend))))
        let choice = try #require((jsonObject(response.body)?["choices"] as? [[String: Any]])?.first)
        #expect(choice["finish_reason"] as? String == "length")
    }

    @Test func streamsRoleThenDeltasThenFinishThenDone() async throws {
        let backend = FakeBackend([.text("Hel"), .text("lo"), .usage(prompt: 7, completion: 2), .finish(.stop)])
        let request = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"x"}],"stream":true,"stream_options":{"include_usage":true}}"#
        let streamed = try #require(await collect(await call(request, chatEnvironment(backend: backend))))
        #expect(streamed.status == 200)
        #expect(streamed.headers["Content-Type"] == "text/event-stream")

        let events = streamed.text.components(separatedBy: "\n\n").filter { !$0.isEmpty }
        #expect(events.last == "data: [DONE]")
        let chunks = try events.dropLast().map { event -> [String: Any] in
            #expect(event.hasPrefix("data: "))
            return try #require(jsonObject(Data(event.dropFirst(6).utf8)))
        }
        func delta(_ c: [String: Any]) -> [String: Any]? { ((c["choices"] as? [[String: Any]])?.first)?["delta"] as? [String: Any] }
        #expect(delta(chunks[0])?["role"] as? String == "assistant")                   // first chunk carries the role
        #expect(chunks.compactMap { delta($0)?["content"] as? String }.joined() == "Hello")
        let finishes = chunks.compactMap { ($0["choices"] as? [[String: Any]])?.first?["finish_reason"] as? String }
        #expect(finishes == ["stop"])                                                    // only on the last content chunk
        let usageChunk = try #require(chunks.last)
        #expect((usageChunk["choices"] as? [Any])?.isEmpty == true)
        #expect((usageChunk["usage"] as? [String: Int])?["total_tokens"] == 9)
    }

    @Test func aStreamWithoutIncludeUsageHasNoUsageChunk() async throws {
        let backend = FakeBackend([.text("a"), .usage(prompt: 1, completion: 1), .finish(.stop)])
        let request = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"x"}],"stream":true}"#
        let streamed = try #require(await collect(await call(request, chatEnvironment(backend: backend))))
        #expect(!streamed.text.contains("\"usage\""))
    }

    @Test func aMissingMaxTokensGetsTheServerDefault() async throws {
        let backend = FakeBackend([.finish(.stop)])
        _ = await call(hello, chatEnvironment(backend: backend))
        #expect(backend.requests.first?.effectiveMaxTokens == 4096)
    }

    @Test func toolsAreIgnoredAnsweredAndNoted() async throws {
        let backend = FakeBackend([.text("plain text"), .finish(.stop)])
        let log = RequestLog()
        let request = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f"}}]}"#
        let response = try #require(full(await call(request, chatEnvironment(backend: backend, log: log))))
        #expect(response.status == 200)
        #expect(backend.requests.first?.tools.isEmpty == true)
        #expect(log.entries.last?.toolsIgnored == true)
    }

    @Test func aConversationWithToolTrafficAnswers200AndTheFlattenedTextReachesTheBackend() async throws {
        let backend = FakeBackend([.text("ok"), .usage(prompt: 1, completion: 1), .finish(.stop)])
        let log = RequestLog()
        let request = #"""
        {"model":"mlx-community/Qwen3-4B-4bit","tools":[{"type":"function","function":{"name":"read_file"}}],
         "messages":[{"role":"user","content":"Read a.txt"},
          {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.txt\"}"}}]},
          {"role":"tool","tool_call_id":"c1","content":"FILE-BODY"}]}
        """#
        let response = try #require(full(await call(request, chatEnvironment(backend: backend, log: log))))
        #expect(response.status == 200)
        let texts = try #require(backend.requests.first).messages.map(\.text)
        #expect(texts == ["Read a.txt", #"[called read_file with {"path":"a.txt"}]"#, "[tool result for read_file: FILE-BODY]"])
        #expect(log.entries.last?.toolsIgnored == true)                            // `tools` was present
    }

    @Test func anUnservedModelIs404() async throws {
        let env = chatEnvironment(backend: FakeBackend([]), served: [])
        let response = try #require(full(await call(hello, env)))
        #expect(response.status == 404)
        #expect((jsonObject(response.body)?["error"] as? [String: Any])?["code"] as? String == "model_not_found")
    }

    @Test func aServedModelWithNoBackendIs501() async throws {
        let response = try #require(full(await call(hello, chatEnvironment(backend: nil))))
        #expect(response.status == 501)
    }

    @Test func aMalformedBodyIs400() async throws {
        let response = try #require(full(await call("{", chatEnvironment(backend: FakeBackend([])))))
        #expect(response.status == 400)
    }

    @Test func onlyPostIsAllowed() async throws {
        let response = try #require(full(await call(hello, chatEnvironment(backend: FakeBackend([])), method: "GET")))
        #expect(response.status == 405)
        #expect(response.headers["Allow"] == "POST")
    }

    @Test func theGuardStillRunsFirst() async throws {
        let backend = FakeBackend([.text("x"), .finish(.stop)])
        let response = try #require(full(await call(hello, chatEnvironment(backend: backend), host: "evil.test:1212")))
        #expect(response.status == 403)
        #expect(backend.requests.isEmpty)                              // the body was never even looked at
    }

    @Test func aFullLineIs503WithRetryAfter() async throws {
        let queue = ServeQueue()
        _ = await queue.reserve(model: testModel)                      // occupies the one slot
        for _ in 0..<4 { _ = await queue.reserve(model: testModel) }   // and the 4 waiting places
        let response = try #require(full(await call(hello, chatEnvironment(backend: FakeBackend([]), queue: queue))))
        #expect(response.status == 503)
        #expect(response.headers["Retry-After"] == "2")
    }

    @Test func twoRequestsForOneModelNeverRunTogether() async throws {
        let backend = FakeBackend([.wait(0.15), .text("x"), .finish(.stop)])
        let env = chatEnvironment(backend: backend)
        async let a = call(hello, env)
        async let b = call(hello, env)
        let (ra, rb) = await (a, b)
        #expect(full(ra)?.status == 200 && full(rb)?.status == 200)
        #expect(backend.requests.count == 2)
        #expect(backend.maxActive == 1)
    }

    @Test func aBackendFailureIs500WithoutLeakingDetail() async throws {
        let response = try #require(full(await call(hello, chatEnvironment(backend: FakeBackend([.fail])))))
        #expect(response.status == 500)
        #expect(!String(decoding: response.body, as: UTF8.self).contains("boom"))
    }

    @Test func aKeepAliveCommentIsWrittenDuringPrefill() async throws {
        // The backend is silent for 0.45 s before its first token; with a 0.1 s interval several
        // `: keep-alive` lines must precede the first data chunk.
        let backend = FakeBackend([.wait(0.45), .text("first"), .finish(.stop)])
        let request = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"x"}],"stream":true}"#
        let streamed = try #require(await collect(await call(request, chatEnvironment(backend: backend, keepAlive: 0.1))))
        let firstData = try #require(streamed.text.range(of: "data: "))
        let before = streamed.text[..<firstData.lowerBound]
        #expect(before.components(separatedBy: ": keep-alive\n\n").count - 1 >= 2)
    }

    @Test func droppingTheStreamCancelsTheGenerationAndFreesTheQueue() async throws {
        let backend = FakeBackend([.text("a"), .wait(30), .text("never"), .finish(.stop)])
        let queue = ServeQueue()
        let request = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"x"}],"stream":true}"#
        let reply = await call(request, chatEnvironment(backend: backend, queue: queue))
        guard case .stream(_, _, let body) = reply else { Issue.record("expected a stream"); return }
        let connected = Flag()
        let consumer = Task { for await _ in body { connected.set() } }
        try await Task.sleep(for: .milliseconds(200))
        #expect(connected.isSet)                                       // the role chunk arrived — the client is connected
        consumer.cancel()                                              // …and then goes away
        _ = await consumer.result
        try await Task.sleep(for: .milliseconds(300))
        #expect(backend.cancelled == 1)
        #expect(await queue.runningCount == 0)
    }

    // MARK: - S2-2 tool calls

    /// A model whose family can't take tools keeps S1 exactly: the client's `tools` never reach the
    /// backend, the history's tool traffic is flattened (R20) and the log says "tools ignored".
    @Test func anUnsupportedFamilyKeepsS1sIgnoreAndFlatten() async throws {
        let backend = FakeBackend([.text("ok"), .finish(.stop)], modelType: "gemma2")
        let log = RequestLog()
        let request = #"""
        {"model":"mlx-community/Qwen3-4B-4bit","tools":[{"type":"function","function":{"name":"read_file"}}],
         "messages":[{"role":"user","content":"Read a.txt"},
          {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.txt\"}"}}]},
          {"role":"tool","tool_call_id":"c1","content":"FILE-BODY"}]}
        """#
        let response = try #require(full(await call(request, chatEnvironment(backend: backend, log: log))))
        #expect(response.status == 200)
        let seen = try #require(backend.requests.first)
        #expect(seen.tools.isEmpty)                                                   // not rendered at all
        #expect(seen.messages.map(\.role) == [.user, .assistant, .user])              // flattened (R20)
        #expect(seen.messages[2].text == "[tool result for read_file: FILE-BODY]")
        #expect(log.entries.last?.toolsIgnored == true)
        #expect(log.entries.last?.toolCalls == 0)
    }

    /// A tool-capable family gets the client's `tools` and the history's structured turns.
    @Test func aSupportedFamilyGetsToolsAndStructuredHistory() async throws {
        let backend = FakeBackend([.text("ok"), .finish(.stop)], modelType: "qwen3")
        let log = RequestLog()
        let request = #"""
        {"model":"mlx-community/Qwen3-4B-4bit","tools":[{"type":"function","function":{"name":"read_file"}}],
         "messages":[{"role":"user","content":"Read a.txt"},
          {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.txt\"}"}}]},
          {"role":"tool","tool_call_id":"c1","content":"FILE-BODY"}]}
        """#
        _ = await call(request, chatEnvironment(backend: backend, log: log))
        let seen = try #require(backend.requests.first)
        #expect(seen.tools.map(\.name) == ["read_file"])
        #expect(seen.messages.map(\.role) == [.user, .assistant, .tool])
        #expect(seen.messages[1].toolCalls.map(\.name) == ["read_file"])
        #expect(seen.messages[2].toolCallID == "c1")
        #expect(log.entries.last?.toolsIgnored == false)
    }

    /// `tool_choice: "none"` (S2-2 requirement 7): the array is still the client's, but nothing is
    /// rendered into the prompt — which is also what stops the backend scanning for calls.
    @Test func toolChoiceNoneRendersNothing() async throws {
        let backend = FakeBackend([.text("ok"), .finish(.stop)], modelType: "qwen3")
        let request = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"none"}"#
        _ = await call(request, chatEnvironment(backend: backend))
        let seen = try #require(backend.requests.first)
        #expect(seen.renderedTools.isEmpty)
        #expect(seen.tools.map(\.name) == ["f"])                                      // still logged as requested
    }

    /// Non-streaming: the calls a backend reports become `message.tool_calls` with
    /// `finish_reason: "tool_calls"`.
    @Test func toolCallsAreEncodedNonStreaming() async throws {
        let backend = FakeBackend([
            .text("Checking. "),
            .toolCall(index: 0, id: "call_a", name: "get_weather", arguments: #"{"location":"Paris"}"#),
            .usage(prompt: 5, completion: 6), .finish(.toolCalls),
        ], modelType: "qwen3")
        let request = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"weather?"}],"tools":[{"type":"function","function":{"name":"get_weather"}}]}"#
        let response = try #require(full(await call(request, chatEnvironment(backend: backend))))
        let object = try #require(jsonObject(response.body))
        let choice = try #require((object["choices"] as? [[String: Any]])?.first)
        #expect(choice["finish_reason"] as? String == "tool_calls")
        let message = try #require(choice["message"] as? [String: Any])
        #expect(message["content"] as? String == "Checking. ")
        let calls = try #require(message["tool_calls"] as? [[String: Any]])
        #expect(message["content"] as? String == "Checking. ")      // text came with the call, so it stays
        #expect(calls[0]["id"] as? String == "call_a")
        #expect(calls[0]["type"] as? String == "function")
        #expect((calls[0]["function"] as? [String: Any])?["name"] as? String == "get_weather")
        #expect((calls[0]["function"] as? [String: Any])?["arguments"] as? String == #"{"location":"Paris"}"#)
    }

    /// Streaming: `delta.tool_calls[]` fragments, then the finish chunk.
    @Test func toolCallsAreEncodedStreaming() async throws {
        let backend = FakeBackend([
            .toolCall(index: 0, id: "call_a", name: "get_weather", arguments: #"{"location":"Paris"}"#),
            .usage(prompt: 5, completion: 6), .finish(.toolCalls),
        ], modelType: "qwen3")
        let request = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"weather?"}],"stream":true,"tools":[{"type":"function","function":{"name":"get_weather"}}]}"#
        let streamed = try #require(await collect(await call(request, chatEnvironment(backend: backend))))
        let events = streamed.text.components(separatedBy: "\n\n").filter { !$0.isEmpty }
        #expect(events.last == "data: [DONE]")
        let chunks = try events.dropLast().map { try #require(jsonObject(Data($0.dropFirst(6).utf8))) }
        func delta(_ c: [String: Any]) -> [String: Any]? { ((c["choices"] as? [[String: Any]])?.first)?["delta"] as? [String: Any] }
        let calls = try #require(delta(chunks[1])?["tool_calls"] as? [[String: Any]])
        #expect(calls[0]["index"] as? Int == 0)
        #expect(calls[0]["id"] as? String == "call_a")
        #expect((calls[0]["function"] as? [String: Any])?["name"] as? String == "get_weather")
        let finishes = chunks.compactMap { ($0["choices"] as? [[String: Any]])?.first?["finish_reason"] as? String }
        #expect(finishes == ["tool_calls"])
    }

    /// A turn that is **only** tool calls answers `"content": null` (S2-2b, review call 7), which
    /// is what the reference shows; a turn with text keeps the text.
    @Test func aCallsOnlyTurnHasNullContent() async throws {
        let backend = FakeBackend([
            .toolCall(index: 0, id: "call_a", name: "get_weather", arguments: "{}"),
            .finish(.toolCalls),
        ], modelType: "qwen3")
        let request = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"weather?"}],"tools":[{"type":"function","function":{"name":"get_weather"}}]}"#
        let response = try #require(full(await call(request, chatEnvironment(backend: backend))))
        let message = try #require(((jsonObject(response.body)?["choices"] as? [[String: Any]])?.first)?["message"] as? [String: Any])
        #expect(message["content"] is NSNull)
        #expect(message.keys.contains("content"))
        #expect((message["tool_calls"] as? [[String: Any]])?.count == 1)
    }

    /// Two calls in one turn keep their indices, in both wire shapes (S2-2 requirement 5).
    // MARK: - S2-8: a template refusal is a 400

    /// A stand-in for `swift-jinja`'s `TemplateException`: the production code recognises a refusal
    /// **by type name** (the module isn't a dependency of this target, and adding one needs owner
    /// approval), so the name is what this mirrors. The *real* type is covered by the gated test
    /// below, against Llama's own template.
    private struct TemplateException: Error { var message: String? }

    /// The refusal → wire error mapping, with the template's own words.
    @Test func aTemplateRefusalMapsToA400WithTheTemplatesWords() throws {
        let raised = "This model only supports single tool-calls at once!"
        let error = try #require(ChatTemplateFailure.serveError(for: TemplateException(message: raised)))
        #expect(error.status == 400)
        #expect(error.code == "template_error")
        #expect(error.type == "invalid_request_error")
        #expect(error.message == "This model's chat template refused the request: \(raised)")

        // A template that raises without words still says what happened — nothing is invented.
        let silent = try #require(ChatTemplateFailure.serveError(for: TemplateException(message: nil)))
        #expect(silent.status == 400)
        #expect(silent.message.hasPrefix("This model's chat template refused the request: "))
        #expect(silent.message.contains("raised an exception"))

        // Anything that isn't a template refusal is not this error (it keeps S2's mapping).
        #expect(ChatTemplateFailure.serveError(for: ServeError(status: 500, message: "x", code: "boom")) == nil)
    }

    /// A refusal reaches the client as a **400 before the head**, for a streaming request too —
    /// which is why the backend's pre-flight reports it rather than the generation.
    @Test func aRefusalIsA400BeforeTheHeadStreamingAndNot() async throws {
        for stream in [true, false] {
            let refusal = try #require(ChatTemplateFailure.serveError(for: TemplateException(
                message: "This model only supports single tool-calls at once!")))
            let backend = FakeBackend([.text("never")], modelType: "llama",
                                      contextLimit: 131_072, promptTokenCount: 10, refusal: refusal)
            let request = stream
                ? #"{"model":"\#(testModel)","messages":[{"role":"user","content":"x"}],"stream":true}"#
                : hello
            let reply = await call(request, chatEnvironment(backend: backend))
            let response = try #require(full(reply))              // a whole response: no head, no SSE byte
            #expect(response.status == 400, "stream=\(stream)")
            let inner = try #require(jsonObject(response.body)?["error"] as? [String: Any])
            #expect(inner["code"] as? String == "template_error", "stream=\(stream)")
            #expect((inner["message"] as? String)?.contains("single tool-calls") == true, "stream=\(stream)")
            #expect(backend.requests.isEmpty, "stream=\(stream)")   // nothing was generated
        }
    }

    /// The route's own wiring of S2-8's flag: a family whose template takes one call per turn gets a
    /// **split** history, a family that loops over calls keeps the client's packed turn — and both
    /// see the same wire response.
    @Test func theRouteSplitsTheHistoryForTheLlamaFamilyOnly() async throws {
        let packed = #"""
        {"model":"\#(testModel)","messages":[{"role":"user","content":"go"},
          {"role":"assistant","content":null,"tool_calls":[
            {"id":"c1","type":"function","function":{"name":"read","arguments":"{}"}},
            {"id":"c2","type":"function","function":{"name":"write","arguments":"{}"}}]},
          {"role":"tool","tool_call_id":"c1","content":"body"},
          {"role":"tool","tool_call_id":"c2","content":"ok"}]}
        """#

        for (family, expected) in [("llama", 5), ("qwen3", 4)] {
            let backend = FakeBackend([.text("done"), .finish(.stop)], modelType: family,
                                      contextLimit: 131_072, promptTokenCount: 10)
            _ = await call(packed, chatEnvironment(backend: backend))
            let sent = try #require(backend.requests.first)
            #expect(sent.messages.count == expected, "\(family): \(sent.messages.map(\.role))")
            if family == "llama" {
                #expect(sent.messages.map(\.role) == [.user, .assistant, .tool, .assistant, .tool])
                #expect(sent.messages[1].toolCalls.map(\.name) == ["read"])
                #expect(sent.messages[3].toolCalls.map(\.name) == ["write"])
            } else {
                #expect(sent.messages[1].toolCalls.count == 2)       // the packed turn, untouched
            }
        }
    }

    /// Belt and braces: a refusal the backend reports **during** generation (the container path, after
    /// the pre-flight found nothing) arrives on the error channel and keeps its 400 — while a genuine
    /// backend *failure* stays the opaque 500 (`aBackendFailureIs500WithoutLeakingDetail`).
    @Test func aRefusalReportedByTheBackendIsA400NotA500() async throws {
        let refusal = try #require(ChatTemplateFailure.serveError(for: TemplateException(message: "no")))
        let backend = FakeBackend([.error(refusal)], modelType: "llama")
        let response = try #require(full(await call(hello, chatEnvironment(backend: backend))))
        #expect(response.status == 400)
        #expect((jsonObject(response.body)?["error"] as? [String: Any])?["code"] as? String == "template_error")

        // …and a backend failure keeps S2's opaque 500 (no `template_error`, no internals).
        let failure = FakeBackend([.failWith(ServeError(status: 500, message: "boom", code: "boom"))])
        let opaque = try #require(full(await call(hello, chatEnvironment(backend: failure))))
        #expect(opaque.status == 500)
        #expect((jsonObject(opaque.body)?["error"] as? [String: Any])?["code"] as? String == "generation_failed")
    }

    // MARK: - S2-6 pre-count (R23)

    /// An over-limit prompt is a `400` **before the head**: the reply is a whole response, never a
    /// stream, so not one SSE byte can have been written.
    @Test func anOverLimitStreamIs400BeforeTheHead() async throws {
        let backend = FakeBackend([.text("never")], modelType: "qwen3", contextLimit: 40960,
                                  promptTokenCount: 41208)
        let queue = ServeQueue()
        let log = RequestLog()
        let request = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"x"}],"stream":true}"#
        let reply = await call(request, chatEnvironment(backend: backend, queue: queue, log: log))

        let response = try #require(full(reply))                       // not `.stream`: no head, no bytes
        #expect(response.status == 400)
        let error = try #require((jsonObject(response.body)?["error"] as? [String: Any]))
        #expect(error["code"] as? String == "context_length_exceeded")
        #expect(error["type"] as? String == "invalid_request_error")
        let message = try #require(error["message"] as? String)
        #expect(message.contains("40960") && message.contains("41208"))
        #expect(backend.requests.isEmpty)                              // nothing was generated
        let running = await queue.runningCount
        let waiting = await queue.waitingCount
        #expect(running == 0 && waiting == 0)                          // and no slot was taken
        #expect(log.entries.last?.status == 400)
    }

    /// The same for a non-streaming request.
    @Test func anOverLimitRequestIs400NonStreamingToo() async throws {
        let backend = FakeBackend([.text("never")], modelType: "qwen3", contextLimit: 100, promptTokenCount: 101)
        let response = try #require(full(await call(hello, chatEnvironment(backend: backend))))
        #expect(response.status == 400)
        #expect((jsonObject(response.body)?["error"] as? [String: Any])?["code"] as? String == "context_length_exceeded")
        #expect(backend.requests.isEmpty)
    }

    /// A prompt that exactly fills the window is fine — the boundary is `>`, not `>=`.
    @Test func aPromptAtTheLimitRuns() async throws {
        let backend = FakeBackend([.text("ok"), .finish(.stop)], modelType: "qwen3", contextLimit: 40960,
                                  promptTokenCount: 40960)
        let response = try #require(full(await call(hello, chatEnvironment(backend: backend))))
        #expect(response.status == 200)
        #expect(backend.requests.count == 1)
    }

    /// When a backend can't count — no declared window, or an SDK without a tokenizer — there is no
    /// pre-count and S1's behaviour stands (the runtime's own overflow handling decides).
    @Test func noCountMeansNoPreCount() async throws {
        let noWindow = FakeBackend([.text("ok"), .finish(.stop)], modelType: "qwen3", promptTokenCount: 999999)
        #expect(try #require(full(await call(hello, chatEnvironment(backend: noWindow)))).status == 200)

        let noCount = FakeBackend([.text("ok"), .finish(.stop)], modelType: "qwen3", contextLimit: 10)
        #expect(try #require(full(await call(hello, chatEnvironment(backend: noCount)))).status == 200)

        // …and a stream still streams.
        let streaming = FakeBackend([.text("ok"), .finish(.stop)], modelType: "qwen3")
        let request = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"x"}],"stream":true}"#
        guard case .stream = await call(request, chatEnvironment(backend: streaming)) else {
            Issue.record("expected a stream"); return
        }
    }

    // MARK: - S2-3 reasoning

    /// Non-streaming: the thinking half comes back as `message.reasoning_content`, the answer as
    /// `content`, and nothing is mixed.
    @Test func reasoningIsEncodedNonStreaming() async throws {
        let backend = FakeBackend([
            .reasoning("I should add 2 and 2."), .text("The answer is 4."), .finish(.stop),
        ], modelType: "qwen3")
        let request = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"2+2?"}],"reasoning_effort":"high"}"#
        let response = try #require(full(await call(request, chatEnvironment(backend: backend))))
        let message = try #require(((jsonObject(response.body)?["choices"] as? [[String: Any]])?.first)?["message"] as? [String: Any])
        #expect(message["reasoning_content"] as? String == "I should add 2 and 2.")
        #expect(message["content"] as? String == "The answer is 4.")
    }

    /// Streaming: `delta.reasoning_content` chunks before the answer chunks.
    @Test func reasoningIsEncodedStreaming() async throws {
        let backend = FakeBackend([
            .reasoning("Thinking."), .text("Answer."), .finish(.stop),
        ], modelType: "qwen3")
        let request = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"x"}],"stream":true,"reasoning_effort":"high"}"#
        let streamed = try #require(await collect(await call(request, chatEnvironment(backend: backend))))
        let events = streamed.text.components(separatedBy: "\n\n").filter { $0.hasPrefix("data: {") }
        func delta(_ event: String) -> [String: Any]? {
            guard let chunk = jsonObject(Data(event.dropFirst(6).utf8)) else { return nil }
            return ((chunk["choices"] as? [[String: Any]])?.first)?["delta"] as? [String: Any]
        }
        #expect(events.compactMap { delta($0)?["reasoning_content"] as? String } == ["Thinking."])
        #expect(events.compactMap { delta($0)?["content"] as? String }.joined() == "Answer.")
    }

    /// The log records that thinking was on — the flag, nothing else (rule 13).
    @Test func theLogNotesThinkingButHoldsNoThinking() async throws {
        let backend = FakeBackend([.reasoning("SECRET-REASONING-TEXT"), .text("ok"), .finish(.stop)],
                                  modelType: "qwen3")
        let log = RequestLog()
        let request = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"x"}],"reasoning_effort":"high"}"#
        _ = await call(request, chatEnvironment(backend: backend, log: log))
        let entry = try #require(log.entries.last)
        #expect(entry.reasoningOn)
        #expect(!String(describing: log.entries).contains("SECRET-REASONING-TEXT"))
    }

    /// A family whose template has no thinking: the control is **ignored**, never a 400, and the log
    /// says so (decision 2).
    @Test func aFamilyWithoutThinkingIgnoresTheControl() async throws {
        let backend = FakeBackend([.text("ok"), .finish(.stop)], modelType: "llama")
        let log = RequestLog()
        let request = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"x"}],"reasoning_effort":"high"}"#
        let response = try #require(full(await call(request, chatEnvironment(backend: backend, log: log))))
        #expect(response.status == 200)
        #expect(log.entries.last?.reasoningOn == false)

        // …and a thinking request to a *thinking* family with the control absent stays off too.
        let off = RequestLog()
        _ = await call(#"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"x"}]}"#,
                       chatEnvironment(backend: backend, log: off))
        #expect(off.entries.last?.reasoningOn == false)
    }

    @Test func twoToolCallsKeepTheirIndices() async throws {
        let backend = FakeBackend([
            .toolCall(index: 0, id: "call_a", name: "first", arguments: "{}"),
            .toolCall(index: 1, id: "call_b", name: "second", arguments: "{}"),
            .finish(.toolCalls),
        ], modelType: "qwen3")
        let tools = #""tools":[{"type":"function","function":{"name":"first"}},{"type":"function","function":{"name":"second"}}]"#

        // Non-streaming: two entries, in order.
        let plain = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"x"}],\#(tools)}"#
        let response = try #require(full(await call(plain, chatEnvironment(backend: backend))))
        let message = try #require(((jsonObject(response.body)?["choices"] as? [[String: Any]])?.first)?["message"] as? [String: Any])
        #expect((message["tool_calls"] as? [[String: Any]])?.compactMap { $0["id"] as? String } == ["call_a", "call_b"])

        // Streaming: `index` 0 then 1.
        let streaming = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"x"}],"stream":true,\#(tools)}"#
        let streamed = try #require(await collect(await call(streaming, chatEnvironment(backend: backend))))
        let events = streamed.text.components(separatedBy: "\n\n").filter { $0.hasPrefix("data: {") }
        let indices = events.compactMap { event -> Int? in
            guard let chunk = jsonObject(Data(event.dropFirst(6).utf8)) else { return nil }
            let delta = ((chunk["choices"] as? [[String: Any]])?.first)?["delta"] as? [String: Any]
            return (delta?["tool_calls"] as? [[String: Any]])?.first?["index"] as? Int
        }
        #expect(indices == [0, 1])
    }

    /// Rule 13: the log records **how many** tool calls, never a name, an argument or a result.
    @Test func theLogCountsToolCallsAndNothingElse() async throws {
        let backend = FakeBackend([
            .toolCall(index: 0, id: "call_a", name: "SECRET-TOOL-NAME", arguments: #"{"secret":"ARG-VALUE"}"#),
            .toolCall(index: 1, id: "call_b", name: "b", arguments: "{}"),
            .finish(.toolCalls),
        ], modelType: "qwen3")
        let log = RequestLog()
        let request = #"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"SECRET-PROMPT-TEXT"}],"tools":[{"type":"function","function":{"name":"f"}}]}"#
        _ = await call(request, chatEnvironment(backend: backend, log: log))
        let entry = try #require(log.entries.last)
        #expect(entry.toolCalls == 2)
        let dump = String(describing: log.entries)
        #expect(!dump.contains("SECRET-TOOL-NAME") && !dump.contains("ARG-VALUE") && !dump.contains("SECRET-PROMPT-TEXT"))
    }

    @Test func theRequestLogHoldsMetadataAndNeverContent() async throws {
        let backend = FakeBackend([.text("SECRET-REPLY-TEXT"), .usage(prompt: 4, completion: 3), .finish(.stop)])
        let log = RequestLog()
        _ = await call(hello, chatEnvironment(backend: backend, log: log))
        let entry = try #require(log.entries.last)
        #expect(entry.userAgent == "test-agent/1.0" && entry.model == testModel && entry.status == 200)
        #expect(entry.promptTokens == 4 && entry.completionTokens == 3 && entry.durationSeconds >= 0)
        let dump = String(describing: log.entries)
        #expect(!dump.contains("SECRET-PROMPT-TEXT") && !dump.contains("SECRET-REPLY-TEXT"))
    }

    @Test func refusedRequestsAreLoggedToo() async throws {
        let log = RequestLog()
        _ = await call(hello, chatEnvironment(backend: FakeBackend([]), served: [], log: log))
        #expect(log.entries.last?.status == 404)
    }
}

// MARK: - Wire (real sockets, fake backend)

@Suite(.serialized)
@MainActor
struct ChatCompletionsWireTests {
    @Test func tokensArriveProgressivelyThenDone() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let backend = FakeBackend([.text("one "), .wait(0.4), .text("two "), .wait(0.4), .text("three"),
                                   .usage(prompt: 1, completion: 3), .finish(.stop)])
        let server = LocalServer(settings: ServeSettings(servedModelIDs: [testModel]), defaults: nil, listenPort: 0,
                                 environment: chatEnvironment(backend: backend))
        await server.start()
        defer { Task { await server.stop() } }
        guard case .running(let port) = server.status else { Issue.record("not running: \(server.status)"); return }

        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(#"{"model":"mlx-community/Qwen3-4B-4bit","messages":[{"role":"user","content":"hi"}],"stream":true}"#.utf8)

        let started = Date()
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        var arrivals: [(Double, String)] = []
        for try await line in bytes.lines where line.hasPrefix("data: ") {
            arrivals.append((Date().timeIntervalSince(started), String(line.dropFirst(6))))
        }
        #expect(arrivals.last?.1 == "[DONE]")
        let contentTimes = arrivals.filter { $0.1.contains("\"content\":\"") && !$0.1.contains("\"content\":\"\"") }.map(\.0)
        #expect(contentTimes.count == 3)
        #expect((contentTimes.last ?? 0) - (contentTimes.first ?? 0) > 0.5)   // spread over time, not one lump
    }

    @Test func aNonStreamingRequestOverTheWire() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let backend = FakeBackend([.text("hi there"), .usage(prompt: 2, completion: 2), .finish(.stop)])
        let server = LocalServer(settings: ServeSettings(servedModelIDs: [testModel]), defaults: nil, listenPort: 0,
                                 environment: chatEnvironment(backend: backend))
        await server.start()
        defer { Task { await server.stop() } }
        guard case .running(let port) = server.status else { Issue.record("not running"); return }

        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.httpBody = Data(hello.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let choice = (jsonObject(data)?["choices"] as? [[String: Any]])?.first
        #expect((choice?["message"] as? [String: Any])?["content"] as? String == "hi there")
    }
}

// MARK: - Real model (skipped unless MLXUI_TEST_LLM_DIR points at an installed MLX LLM)

/// Run through the Xcode agent: the sandboxed test host can't read `~/.cache` / another app's
/// container (memory: gallery-parse-tree-golden-red-on-kc5). Give it the model folder with
/// `TEST_RUNNER_MLXUI_TEST_LLM_DIR=<path>`.
struct MLXChatBackendRealModelTests {
    private static let modelDir = ProcessInfo.processInfo.environment["MLXUI_TEST_LLM_DIR"]
    /// A `qwen3_5` model folder for the tokenizer-only prompt-shape probe (S2-6). Weights are never
    /// loaded, so this can point at the 27B checkpoints the 19 GB Mac can't run.
    static let qwen35Dir = ProcessInfo.processInfo.environment["MLXUI_TEST_QWEN35_TOKENIZER_DIR"]
    /// A **Llama 3.1** model folder, for S2-7's raw-turn recording.
    /// `TEST_RUNNER_MLXUI_TEST_LLAMA_DIR=<path>`.
    static let llamaDir = ProcessInfo.processInfo.environment["MLXUI_TEST_LLAMA_DIR"]

    @Test(.enabled(if: MLXChatBackendRealModelTests.modelDir != nil))
    func streamsATokenAtATimeAndKeepsMarkupIntact() async throws {
        let dir = URL(fileURLWithPath: try #require(Self.modelDir))
        let backend = MLXChatBackend(directory: dir, pool: ModelContainerPool(budgetBytes: 40 << 30))
        var request = ServeRequest(model: "m", messages: [
            ServeMessage(role: .user, text: "Reply with exactly this text and nothing else: <div><span>hi</span></div>")])
        request.temperature = 0
        request.maxTokens = 40

        var deltas: [String] = []
        var usage: ServeUsage?
        var finish: ServeFinishReason?
        for try await event in backend.generate(request) {
            switch event {
            case .textDelta(let piece): deltas.append(piece)
            case .usage(let value): usage = value
            case .finish(let reason): finish = reason
            default: break
            }
        }
        let text = deltas.joined()
        #expect(deltas.count > 1)                                       // progressive, not one lump
        #expect(text.contains("<div><span>"))                           // TCP-1, on a real tokenizer
        #expect(!text.contains("<div<span"))
        #expect((usage?.completionTokens ?? 0) > 0 && (usage?.promptTokens ?? 0) > 0)
        #expect(finish != nil)
    }

    /// S2-2 on a real checkpoint: the model calls `get_weather`, and the turn that carries the call
    /// plus its result back through the template renders and answers.
    ///
    /// Two generations, because that is the round trip a harness makes. The first pins the parse
    /// (`.json` on Qwen3-8B) and the second pins the structured-history path — the `tool_calls`
    /// array and the `tool_call_id` that `Chat.Message` cannot carry, which goes to the template as
    /// raw dictionaries.
    @Test(.enabled(if: MLXChatBackendRealModelTests.modelDir != nil))
    func aToolCallAndItsResultRoundTrip() async throws {
        let dir = URL(fileURLWithPath: try #require(Self.modelDir))
        let backend = MLXChatBackend(directory: dir, pool: ModelContainerPool(budgetBytes: 40 << 30))
        let toolJSON = #"{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}"#
        let tools = [ServeToolSpec(name: "get_weather", rawJSON: toolJSON)]
        func request(_ messages: [ServeMessage]) -> ServeRequest {
            var request = ServeRequest(model: "m", messages: messages)
            request.tools = tools
            request.temperature = 0
            request.maxTokens = 160
            return request
        }

        // Turn 1 — the ask. The model should call the tool.
        var calls: [ServeToolCall] = []
        var text = ""
        var finish: ServeFinishReason?
        for try await event in backend.generate(request([
            ServeMessage(role: .user, text: "What is the weather in Paris? Use the get_weather tool.")
        ])) {
            switch event {
            case .textDelta(let piece): text += piece
            case .toolCallDelta(let delta): ChatCompletionsRoute.append(delta, to: &calls)
            case .finish(let reason): finish = reason
            default: break
            }
        }
        print("S2-2 MEASURE tool call: finish=\(String(describing: finish)) calls=\(calls.map(\.name)) text chars=\(text.count)")
        #expect(finish == .toolCalls)
        #expect(calls.first?.name == "get_weather")
        #expect(calls.first?.argumentsJSON.contains("Paris") == true)
        let call = try #require(calls.first)

        // Turn 2 — the result goes back through the template, structured.
        var answer = ""
        for try await event in backend.generate(request([
            ServeMessage(role: .user, text: "What is the weather in Paris? Use the get_weather tool."),
            ServeMessage(role: .assistant, text: "", toolCalls: [call]),
            ServeMessage(role: .tool, text: "18 degrees and sunny", toolCallID: call.id, toolName: "get_weather"),
        ])) {
            if case .textDelta(let piece) = event { answer += piece }
        }
        #expect(!answer.isEmpty)
    }

    /// S2-6 (R23) on a real checkpoint: an oversized prompt is a `400` from the **real** tokenizer
    /// count, before the head — and the count the route uses is the same number the generation
    /// path would build.
    ///
    /// The equality half is the fidelity pin: `countPromptTokens` renders with the tokenizer alone
    /// (no weights), while generation renders through the container's processor. If those two ever
    /// disagree, the pre-count would refuse prompts that fit or wave through prompts that don't.
    @Test(.enabled(if: MLXChatBackendRealModelTests.modelDir != nil))
    func thePreCountMatchesWhatGenerationWouldSend() async throws {
        let dir = URL(fileURLWithPath: try #require(Self.modelDir))
        let backend = MLXChatBackend(directory: dir, pool: ModelContainerPool(budgetBytes: 40 << 30))
        #expect(backend.contextLimit == 40960)                  // qwen3's own config.json

        var request = ServeRequest(model: "m", messages: [
            ServeMessage(role: .system, text: "Be brief."),
            ServeMessage(role: .user, text: String(repeating: "Hello there. ", count: 40)),
        ])
        request.tools = [ServeToolSpec(name: "get_weather", rawJSON: #"{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}"#)]

        // A history that carries **tool traffic** as well: that is the path where the template is
        // handed raw dictionaries, and the one a count could most easily get wrong.
        request.messages.append(ServeMessage(role: .assistant, text: "", toolCalls: [
            ServeToolCall(id: "call_1", name: "get_weather", argumentsJSON: #"{"location":"Paris"}"#)]))
        request.messages.append(ServeMessage(role: .tool, text: "18 degrees", toolCallID: "call_1",
                                            toolName: "get_weather"))
        request.messages.append(ServeMessage(role: .user, text: "And tomorrow?"))

        // S2-6 requirement 5: measure what the pre-count costs the head, cold and warm. Nothing
        // loads weights, so this is the tokenizer read (cached after the first) plus the render.
        let coldStart = Date()
        let counted = try #require(await backend.countPromptTokens(request))
        let cold = Date().timeIntervalSince(coldStart)
        let warmStart = Date()
        let again = try #require(await backend.countPromptTokens(request))
        let warm = Date().timeIntervalSince(warmStart)
        #expect(counted == again)
        #expect(counted > 0)

        // The container path (what generation actually sends), rendered by the same backend helpers.
        let container = try await ModelContainerPool(budgetBytes: 40 << 30)
            .container(directory: dir, architecture: nil)
        let expected = try await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(prompt: .messages(MLXChatBackend.templateMessages(request.messages)),
                                 tools: backend.effectiveTools(request),
                                 additionalContext: backend.templateContext(request)))
            return input.text.tokens.size
        }
        print("S2-6 MEASURE prompt tokens (with tool history): tokenizer-only=\(counted) container=\(expected)")
        print("S2-6 MEASURE pre-count cost (no weights): cold \(Int(cold * 1000)) ms, warm \(Int(warm * 1000)) ms")
        #expect(counted == expected)

        // An oversized prompt is refused before anything is generated.
        var huge = ServeRequest(model: "m", messages: [
            ServeMessage(role: .user, text: String(repeating: "This is a long sentence. ", count: 8000)),
        ])
        huge.maxTokens = 64
        let hugeCount = try #require(await backend.countPromptTokens(huge))
        print("S2-6 MEASURE oversized prompt tokens=\(hugeCount) limit=\(backend.contextLimit ?? -1)")
        #expect(hugeCount > 40960)
    }

    /// …and the same prompt over the **real route**, with the real backend: a plain `400` and not a
    /// stream, so no head and no SSE byte went out (S2-6's whole point).
    @Test(.enabled(if: MLXChatBackendRealModelTests.modelDir != nil))
    func anOversizedPromptIs400OverTheRealRoute() async throws {
        let dir = URL(fileURLWithPath: try #require(Self.modelDir))
        let backend = MLXChatBackend(directory: dir, pool: ModelContainerPool(budgetBytes: 40 << 30))
        let id = "mlx-community/Qwen3-8B-4bit"
        let environment = ServeEnvironment(
            servedModelIDs: { [id] },
            installed: { InstalledModelIndex(entries: [
                .init(id: "mlx-community--Qwen3-8B-4bit", hfModelId: id, kind: .llm, ramGB: 4, downloadSizeGB: 2.67)]) },
            appleFoundationReadiness: { nil },
            created: 1,
            backendFor: { $0 == id ? backend : nil })

        let body = Data(#"{"model":"\#(id)","stream":true,"messages":[{"role":"user","content":"\#(String(repeating: "This is a long sentence. ", count: 8000))"}]}"#.utf8)
        let head = ServeRequestHead(method: "POST", path: "/v1/chat/completions", host: "127.0.0.1:1212", origin: nil,
                                    transferEncoding: nil, contentLength: String(body.count), userAgent: "t")
        let reply = await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212),
                                          environment: environment, body: { body })
        guard case .full(let response) = reply else {
            Issue.record("expected a plain 400 — a stream would mean a 200 head went out"); return
        }
        #expect(response.status == 400)
        let error = try #require(try JSONSerialization.jsonObject(with: response.body) as? [String: Any])
        let inner = try #require(error["error"] as? [String: Any])
        #expect(inner["code"] as? String == "context_length_exceeded")
        let message = try #require(inner["message"] as? String)
        print("S2-6 MEASURE oversized route: \(message)")
        #expect(message.contains("40960") && message.contains("480"))
    }

    /// **S2-8** — the refusal the detector claims to recognise, against Llama's **real** template: the
    /// type name it matches on and the raise text it reads out of the value are both exercised here,
    /// not against a stand-in.
    @Test(.enabled(if: MLXChatBackendRealModelTests.llamaDir != nil))
    func theRealLlamaTemplateRefusalIsRecognised() async throws {
        let dir = URL(fileURLWithPath: try #require(Self.llamaDir))
        let tokenizer = try await HFTokenizerLoader().load(from: dir)
        let packed: [[String: any Sendable]] = [
            ["role": "user", "content": "go"],
            ["role": "assistant", "content": "",
             "tool_calls": [
                ["id": "c1", "type": "function", "function": ["name": "read", "arguments": "{}"]],
                ["id": "c2", "type": "function", "function": ["name": "write", "arguments": "{}"]],
             ] as [[String: any Sendable]]],
        ]
        do {
            _ = try tokenizer.applyChatTemplate(messages: packed, tools: nil, additionalContext: nil)
            Issue.record("expected Llama's template to refuse two calls in one assistant turn")
        } catch {
            print("S2-8 MEASURE real refusal: type=\(type(of: error)) "
                  + "text=\(String(describing: ChatTemplateFailure.raisedText(from: error)))")
            #expect(ChatTemplateFailure.isTemplateRefusal(error))
            let raised = try #require(ChatTemplateFailure.raisedText(from: error))
            #expect(raised.contains("single tool-calls"))
            let mapped = try #require(ChatTemplateFailure.serveError(for: error))
            #expect(mapped.status == 400 && mapped.code == "template_error")
            #expect(mapped.message.contains(raised))
        }
    }

    /// **S2-8's fix, end to end on the real model**: the SV-6 turn — one assistant message carrying
    /// two calls, replayed by a client — used to be a `500 generation_failed` (the template refused
    /// the packed history). With the history split it is served normally.
    @Test(.enabled(if: MLXChatBackendRealModelTests.llamaDir != nil))
    func aPackedLlamaHistoryIsServedAfterTheSplit() async throws {
        let dir = URL(fileURLWithPath: try #require(Self.llamaDir))
        let backend = MLXChatBackend(directory: dir, pool: ModelContainerPool(budgetBytes: 40 << 30))
        let id = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
        let environment = ServeEnvironment(
            servedModelIDs: { [id] },
            installed: { InstalledModelIndex(entries: [
                .init(id: "mlx-community--Meta-Llama-3.1-8B-Instruct-4bit", hfModelId: id,
                      kind: .llm, ramGB: 6.75, downloadSizeGB: 4.5)]) },
            appleFoundationReadiness: { nil }, created: 1,
            backendFor: { $0 == id ? backend : nil })

        let payload: [String: Any] = [
            "model": id, "max_tokens": 64,
            "messages": [
                ["role": "user", "content": "Read hello.txt, then create notes.md."],
                ["role": "assistant", "content": NSNull(),
                 "tool_calls": [
                    ["id": "c1", "type": "function",
                     "function": ["name": "read", "arguments": #"{"filePath": "hello.txt"}"#]],
                    ["id": "c2", "type": "function",
                     "function": ["name": "write", "arguments": #"{"content": "x", "filePath": "notes.md"}"#]],
                 ]],
                ["role": "tool", "tool_call_id": "c1", "content": "Hello from SV-6"],
                ["role": "tool", "tool_call_id": "c2", "content": "ok"],
            ],
            "tools": openCodeTools,
        ]
        let body = try JSONSerialization.data(withJSONObject: payload)
        let head = ServeRequestHead(method: "POST", path: "/v1/chat/completions", host: "127.0.0.1:1212",
                                    origin: nil, transferEncoding: nil,
                                    contentLength: String(body.count), userAgent: "t")
        let reply = await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212),
                                          environment: environment, body: { body })
        guard case .full(let response) = reply else {
            Issue.record("expected a whole response, not a stream"); return
        }
        let text = String(decoding: response.body, as: UTF8.self)
        print("S2-8 MEASURE packed history: status=\(response.status) body=\(text.prefix(240))")
        #expect(response.status == 200)                     // not a 400 refusal, not a 500 failure
        #expect(!text.contains("template_error"))
        #expect(!text.contains("generation_failed"))
    }

    /// **S2-7 requirement 3** — the raw text of one real Llama-3.1-8B tool turn (SV-6's own prompt),
    /// and what the scanner makes of it. The verbatim text is **printed, never logged** (rule 13) and
    /// not asserted verbatim — the model phrases freely run to run — while the two properties SV-6
    /// needs are asserted: the turn yields calls, and no call object leaks into the content.
    @Test(.enabled(if: MLXChatBackendRealModelTests.llamaDir != nil))
    func aRealLlamaToolTurnIsReadAsCalls() async throws {
        let dir = URL(fileURLWithPath: try #require(Self.llamaDir))
        let pool = ModelContainerPool(budgetBytes: 40 << 30)
        let container = try await pool.container(directory: dir, architecture: nil)
        let prompt = "Read hello.txt, then create a file called notes.md whose first line is the "
            + "first line of hello.txt, then list the files in this folder."

        let raw = try await container.perform { context -> String in
            let input = try await context.processor.prepare(
                input: UserInput(chat: [Chat.Message(role: .user, content: prompt)],
                                 tools: openCodeTools))
            let stream = try MLXLMCommon.generateTokens(
                input: input,
                parameters: GenerateParameters(maxTokens: 400, temperature: 0, topP: 1),
                context: context)
            var ids: [Int] = []
            for await generation in stream { if case .token(let id) = generation { ids.append(id) } }
            return context.tokenizer.decode(tokenIds: ids)
        }

        print("S2-7 MEASURE llama raw turn: \(raw.debugDescription)")
        var scanner = ServeToolCallScanner(format: .llama3, tools: openCodeTools)
        var content = scanner.push(raw)
        content += scanner.finish()
        let calls = scanner.takeCalls()
        print("S2-7 MEASURE llama calls=\(calls.map(\.function.name)) content=\(content.debugDescription)")
        #expect(!calls.isEmpty)                                  // the turn is a call, not prose
        #expect(!content.contains("\"parameters\""))              // …and no call object leaked
    }

    /// The review's open item (S2-3b), closed with the tokenizer alone: a `qwen3_5` checkpoint's
    /// generation prompt **ends `<think>`** when thinking is on and does not when it is off — which
    /// is what `startsInsideThink` is computed from. No weights are loaded: only `tokenizer.json`
    /// and the chat template, so a 27B checkpoint is safe to point this at.
    ///
    /// `TEST_RUNNER_MLXUI_TEST_QWEN35_TOKENIZER_DIR=<path to any qwen3_5 model folder>`.
    @Test(.enabled(if: MLXChatBackendRealModelTests.qwen35Dir != nil))
    func aQwen35PromptOpensTheThinkBlockOnlyWhenThinkingIsOn() async throws {
        let dir = URL(fileURLWithPath: try #require(Self.qwen35Dir))
        let tokenizer = try await HFTokenizerLoader().load(from: dir)
        let messages: [[String: any Sendable]] = [["role": "user", "content": "What is 2 + 2?"]]

        func tail(thinking: Bool) throws -> String {
            let ids = try tokenizer.applyChatTemplate(messages: messages, tools: nil,
                                                      additionalContext: ["enable_thinking": thinking])
            return tokenizer.decode(tokenIds: Array(ids.suffix(MLXChatBackend.promptTailTokens)))
        }
        let on = try tail(thinking: true)
        let off = try tail(thinking: false)
        print("S2-6 MEASURE qwen3_5 prompt tail (on)=\(on.debugDescription) (off)=\(off.debugDescription)")
        #expect(MLXChatBackend.promptOpensThinkBlock(decodedTail: on))          // starts inside the block
        #expect(!MLXChatBackend.promptOpensThinkBlock(decodedTail: off))        // pre-closed empty block
    }

    /// S2-3 on a real checkpoint, both ways: with thinking asked for, the thinking comes back as
    /// `.reasoningDelta` and the answer as content with no `<think>` marker in either; without it,
    /// there is no reasoning at all (S1's behaviour). Two runs each cover the non-streaming
    /// (accumulate every event) and streaming (per-event) consumers.
    @Test(.enabled(if: MLXChatBackendRealModelTests.modelDir != nil))
    func thinkingOnAndOffOnARealModel() async throws {
        let dir = URL(fileURLWithPath: try #require(Self.modelDir))
        let backend = MLXChatBackend(directory: dir, pool: ModelContainerPool(budgetBytes: 40 << 30))
        func request(_ reasoning: Bool?) -> ServeRequest {
            var request = ServeRequest(model: "m", messages: [
                ServeMessage(role: .user, text: "What is 2 + 2? Think it through, then answer.")])
            request.temperature = 0
            request.maxTokens = 512
            request.reasoning = reasoning
            return request
        }
        func halves(_ request: ServeRequest) async throws -> (reasoning: String, content: String, finish: ServeFinishReason?) {
            var reasoning = "", content = ""
            var finish: ServeFinishReason?
            for try await event in backend.generate(request) {
                switch event {
                case .reasoningDelta(let piece): reasoning += piece
                case .textDelta(let piece): content += piece
                case .finish(let reason): finish = reason
                default: break
                }
            }
            return (reasoning, content, finish)
        }

        let on = try await halves(request(true))
        print("S2-3 MEASURE thinking on: reasoning chars=\(on.reasoning.count) content chars=\(on.content.count) finish=\(String(describing: on.finish))")
        #expect(!on.reasoning.isEmpty)                                  // the model did think
        #expect(!on.reasoning.contains("<think>") && !on.reasoning.contains("</think>"))
        #expect(!on.content.contains("<think>") && !on.content.contains("</think>"))
        if on.finish == .stop { #expect(!on.content.isEmpty) }          // a clean finish means an answer
        else { #expect(on.content.isEmpty) }                            // length: cut mid-think

        let off = try await halves(request(nil))
        print("S2-3 MEASURE thinking off: reasoning chars=\(off.reasoning.count) content chars=\(off.content.count) finish=\(String(describing: off.finish))")
        #expect(off.reasoning.isEmpty)                                  // S1: nothing is labelled reasoning
        #expect(!off.content.isEmpty)
    }
}
