import Testing
import Foundation
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
        case fail
    }

    private let lock = NSLock()
    private var _requests: [ServeRequest] = []
    private var _cancelled = 0
    private var _active = 0
    private var _maxActive = 0
    let script: [Step]

    init(_ script: [Step]) { self.script = script }

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
                    case .fail: continuation.finish(throwing: ServeError(status: 500, message: "boom", code: "boom")); return
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
            .init(id: "mlx-community--Qwen3-4B-4bit", hfModelId: testModel, kind: .llm, ramGB: 3)]) },
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
        #expect(r.stop == ["END"] && r.stream && r.includeUsage && !r.toolsIgnored)
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

    @Test func toolsAreIgnoredAndFlagged() throws {
        let r = try decode(#"{"model":"m","messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"auto"}"#)
        #expect(r.toolsIgnored)
        #expect(r.tools.isEmpty && r.toolChoice == nil)          // S2's fields stay unset
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
        let r = try decode(Self.toolConversation)
        #expect(r.messages.map(\.role) == [.user, .assistant, .user, .user, .user, .assistant])
        #expect(r.messages[1].text == #"[called read_file with {"path":"a.txt"}]"# + "\n[called list_dir with {}]")
        #expect(r.messages[2].text == "[tool result for read_file: hello]")        // name found via tool_call_id
        #expect(r.messages[3].text == "[tool result for explicit: xy]")            // `name` wins; parts joined
        #expect(r.messages[4].text == "[tool result for tool: z]")                 // unknown id → "tool"
        #expect(r.messages[5].text == "Done.\n[called f with {}]")                 // text kept, call appended
        #expect(!r.toolsIgnored)                                                   // only a `tools` field sets that
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
        #expect(request.toolsIgnored)
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
}
