import Testing
import Foundation
import FoundationModels
@testable import MLXUI

// S1-4 — the Apple Foundation Models backend: snapshot→delta, multi-turn assembly, error and
// readiness mapping, the route, and a gated real-AFM test.

/// A scripted stand-in for the macOS-26 seam: no device, no Apple Intelligence.
nonisolated final class FakeChatStreamer: AppleFoundationChatStreaming, @unchecked Sendable {
    enum Step: Sendable {
        case snapshot(String)
        case usage(Int, Int)
        case wait(Double)
        case fail(AppleFoundationChatError)
    }
    struct Call: Sendable, Equatable {
        var instructions: String
        var prompt: String
        var temperature: Double?
        var maximumResponseTokens: Int?
    }

    private let lock = NSLock()
    private var _calls: [Call] = []
    private var _cancelled = 0
    let script: [Step]

    init(_ script: [Step]) { self.script = script }

    var calls: [Call] { lock.lock(); defer { lock.unlock() }; return _calls }
    var cancelled: Int { lock.lock(); defer { lock.unlock() }; return _cancelled }

    func streamChat(instructions: String, prompt: String,
                    temperature: Double?, maximumResponseTokens: Int?) -> AsyncThrowingStream<AppleFoundationChatEvent, Error> {
        lock.lock()
        _calls.append(Call(instructions: instructions, prompt: prompt,
                           temperature: temperature, maximumResponseTokens: maximumResponseTokens))
        lock.unlock()
        return AsyncThrowingStream { continuation in
            let task = Task {
                for step in script {
                    if Task.isCancelled { break }
                    switch step {
                    case .snapshot(let text): continuation.yield(.snapshot(text))
                    case .usage(let p, let c): continuation.yield(.usage(promptTokens: p, completionTokens: c))
                    case .wait(let seconds): try? await Task.sleep(for: .seconds(seconds))
                    case .fail(let error): continuation.finish(throwing: error); return
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { [self] reason in
                if case .cancelled = reason { lock.lock(); _cancelled += 1; lock.unlock() }
                task.cancel()
            }
        }
    }
}

private func request(_ messages: [ServeMessage], temperature: Double? = nil, maxTokens: Int? = nil,
                     stop: [String] = []) -> ServeRequest {
    var r = ServeRequest(model: "apple-foundation", messages: messages)
    r.temperature = temperature
    r.maxTokens = maxTokens
    r.stop = stop
    return r
}

private func events(_ backend: AFMBackend, _ request: ServeRequest) async throws -> [ServeEvent] {
    var all: [ServeEvent] = []
    for try await event in backend.generate(request) { all.append(event) }
    return all
}

private func text(_ events: [ServeEvent]) -> String {
    events.compactMap { if case .textDelta(let t) = $0 { return t } else { return nil } }.joined()
}

private func ready() -> @Sendable () -> Readiness? { { .ready } }

// MARK: - Snapshot → delta

struct SnapshotDeltaConverterTests {
    @Test func cumulativeSnapshotsBecomeDeltas() {
        var converter = SnapshotDeltaConverter()
        #expect(["He", "Hello", "Hello", "Hello world"].map { converter.push($0) } == ["He", "llo", "", " world"])
    }

    @Test func aRepeatedSnapshotEmitsNothing() {
        var converter = SnapshotDeltaConverter()
        _ = converter.push("abc")
        #expect(converter.push("abc") == "")
    }

    @Test func aSnapshotThatRewritesEarlierTextIsHeldNotRepeated() {
        var converter = SnapshotDeltaConverter()
        #expect(converter.push("Hello") == "Hello")
        #expect(converter.push("Hey there") == "")             // not an extension of what was sent
        #expect(converter.push("Hello there") == " there")     // a later one that is
    }

    @Test func emojiAndCombiningCharactersSurvive() {
        var converter = SnapshotDeltaConverter()
        #expect(converter.push("a🙂") == "a🙂")
        #expect(converter.push("a🙂 é") == " é")
    }
}

// MARK: - Backend

struct AFMBackendTests {
    @Test func streamsDeltasAndEndsWithUsageThenFinish() async throws {
        let streamer = FakeChatStreamer([.snapshot("Hel"), .snapshot("Hello"), .snapshot("Hello"), .usage(12, 3)])
        let out = try await events(AFMBackend(streamer: streamer, readiness: ready()),
                                   request([ServeMessage(role: .user, text: "hi")]))
        #expect(out.compactMap { if case .textDelta(let t) = $0 { return t } else { return nil } } == ["Hel", "lo"])
        #expect(Array(out.suffix(2)) == [.usage(ServeUsage(promptTokens: 12, completionTokens: 3)), .finish(.stop)])
    }

    @Test func assemblesInstructionsAndFoldsEarlierTurnsIntoThePrompt() async throws {
        let streamer = FakeChatStreamer([.snapshot("ok")])
        let r = request([
            ServeMessage(role: .system, text: "Be brief."),
            ServeMessage(role: .user, text: "first"),
            ServeMessage(role: .assistant, text: "answer one"),
            ServeMessage(role: .system, text: "Use metric."),             // a mid-conversation system message
            ServeMessage(role: .user, text: "second"),
        ], temperature: 0.4, maxTokens: 50)
        _ = try await events(AFMBackend(streamer: streamer, readiness: ready()), r)
        let call = try #require(streamer.calls.first)
        #expect(call.instructions == "Be brief.\n\nUse metric.")           // system text, in order
        #expect(call.prompt == "Conversation so far:\nUser: first\nAssistant: answer one\n\nReply to the latest message from the user:\nUser: second")
        #expect(call.temperature == 0.4 && call.maximumResponseTokens == 50)
    }

    @Test func aSingleTurnPromptIsTheMessageVerbatim() async throws {
        let streamer = FakeChatStreamer([.snapshot("x")])
        _ = try await events(AFMBackend(streamer: streamer, readiness: ready()), request([ServeMessage(role: .user, text: "just this")]))
        #expect(streamer.calls.first?.prompt == "just this")
        #expect(streamer.calls.first?.instructions == "")
    }

    @Test func topPAndSeedAreIgnoredNotErrors() async throws {
        var r = request([ServeMessage(role: .user, text: "hi")])
        r.topP = 0.3; r.seed = 9
        let out = try await events(AFMBackend(streamer: FakeChatStreamer([.snapshot("x")]), readiness: ready()), r)
        #expect(text(out) == "x")
    }

    @Test func anUnsetTemperatureIsLeftToTheModelAndMaxTokensDefaultsTo4096() async throws {
        let streamer = FakeChatStreamer([.snapshot("x")])
        _ = try await events(AFMBackend(streamer: streamer, readiness: ready()), request([ServeMessage(role: .user, text: "hi")]))
        #expect(streamer.calls.first?.temperature == nil)
        #expect(streamer.calls.first?.maximumResponseTokens == 4096)
    }

    @Test func aConversationMustEndWithAUserTurn() async throws {
        let out = try await events(AFMBackend(streamer: FakeChatStreamer([]), readiness: ready()),
                                   request([ServeMessage(role: .user, text: "q"), ServeMessage(role: .assistant, text: "a")]))
        #expect(out == [.error(ServeError(status: 400, message: "The last message must come from the user.", code: "invalid_messages"))])
    }

    @Test func aStopSequenceCutsTheReplyAndCancelsTheGeneration() async throws {
        let streamer = FakeChatStreamer([.snapshot("one "), .snapshot("one two"), .snapshot("one two STOP three"), .wait(30),
                                         .snapshot("never")])
        let out = try await events(AFMBackend(streamer: streamer, readiness: ready()),
                                   request([ServeMessage(role: .user, text: "hi")], stop: ["STOP"]))
        #expect(text(out) == "one two ")
        #expect(out.last == .finish(.stop))
        try await Task.sleep(for: .milliseconds(200))
        #expect(streamer.cancelled == 1)
    }

    @Test func withoutOSTokenCountsUsageIsEstimatedAndALongReplyIsLength() async throws {
        let out = try await events(AFMBackend(streamer: FakeChatStreamer([.snapshot(String(repeating: "a", count: 40))]), readiness: ready()),
                                   request([ServeMessage(role: .user, text: "abcd")], maxTokens: 10))
        #expect(out.contains(.usage(ServeUsage(promptTokens: 1, completionTokens: 10))))     // 4 chars → 1, 40 chars → 10
        #expect(out.last == .finish(.length))
    }

    // MARK: Errors and readiness

    @Test func contextWindowExceededIs400WithTheLimit() async throws {
        let out = try await events(AFMBackend(streamer: FakeChatStreamer([.fail(.contextWindowExceeded(limit: 8192, tokenCount: 40060))]), readiness: ready()),
                                   request([ServeMessage(role: .user, text: "hi")]))
        guard case .error(let error)? = out.last else { Issue.record("expected an error"); return }
        #expect(error.status == 400 && error.code == "context_length_exceeded")
        #expect(error.message.contains("8192") && error.message.contains("40060"))
    }

    @Test func contextWindowExceededWithoutATokenCountStillNamesTheLimit() {
        let error = AFMBackend.serveError(for: .contextWindowExceeded(limit: 8192, tokenCount: nil))
        #expect(error.status == 400 && error.message.contains("8192"))
    }

    @Test func aGuardrailViolationIs400AppleIntelligenceDeclined() async throws {
        let out = try await events(AFMBackend(streamer: FakeChatStreamer([.fail(.guardrailViolation)]), readiness: ready()),
                                   request([ServeMessage(role: .user, text: "hi")]))
        #expect(out == [.error(ServeError(status: 400, message: "Apple Intelligence declined this request.", code: "content_filter"))])
    }

    @Test func anyOtherFailureIsAGeneric500() async throws {
        let out = try await events(AFMBackend(streamer: FakeChatStreamer([.fail(.failed)]), readiness: ready()),
                                   request([ServeMessage(role: .user, text: "hi")]))
        guard case .error(let error)? = out.last else { Issue.record("expected an error"); return }
        #expect(error.status == 500)
    }

    @Test(arguments: [
        (Readiness.needsSetup(reason: "Turn on Apple Intelligence in System Settings", action: .enableAppleIntelligence),
         "Turn on Apple Intelligence in System Settings."),
        (Readiness.needsSetup(reason: "macOS is still downloading the model", action: nil), "macOS is still downloading the model."),
        (Readiness.unavailable(reason: "This Mac can't run Apple Intelligence"), "This Mac can't run Apple Intelligence."),
    ])
    func notReadyIs503WithTheReadinessSentence(state: Readiness, sentence: String) async throws {
        let streamer = FakeChatStreamer([.snapshot("never")])
        let out = try await events(AFMBackend(streamer: streamer, readiness: { state }), request([ServeMessage(role: .user, text: "hi")]))
        #expect(out == [.error(ServeError(status: 503, message: sentence, type: "server_error", code: "model_not_ready"))])
        #expect(streamer.calls.isEmpty)                                      // the model was never asked
    }

    @Test func absentBelowMacOS26Is404() async throws {
        let streamer = FakeChatStreamer([.snapshot("never")])
        let out = try await events(AFMBackend(streamer: streamer, readiness: { nil }), request([ServeMessage(role: .user, text: "hi")]))
        guard case .error(let error)? = out.last else { Issue.record("expected an error"); return }
        #expect(error.status == 404)
        #expect(streamer.calls.isEmpty)
    }

    @Test func droppingTheStreamCancelsTheGeneration() async throws {
        let streamer = FakeChatStreamer([.snapshot("a"), .wait(30), .snapshot("ab")])
        let backend = AFMBackend(streamer: streamer, readiness: ready())
        let consumer = Task { for try await _ in backend.generate(request([ServeMessage(role: .user, text: "hi")])) {} }
        try await Task.sleep(for: .milliseconds(200))
        consumer.cancel()
        _ = await consumer.result
        try await Task.sleep(for: .milliseconds(300))
        #expect(streamer.cancelled == 1)
    }
}

// MARK: - Route

struct AFMRouteTests {
    private static let body = #"{"model":"apple-foundation","messages":[{"role":"user","content":"hi"}]}"#

    private func environment(readiness: Readiness?, streamer: FakeChatStreamer?, served: Set<String> = ["apple-foundation"],
                             log: RequestLog = RequestLog()) -> ServeEnvironment {
        ServeEnvironment(
            servedModelIDs: { served },
            installed: { InstalledModelIndex(entries: []) },
            appleFoundationReadiness: { readiness },
            created: 1,
            backendFor: { id in
                guard id == "apple-foundation", let streamer else { return nil }
                return AFMBackend(streamer: streamer, readiness: { readiness })
            },
            requestLog: log)
    }

    private func call(_ env: ServeEnvironment, body: String = AFMRouteTests.body) async -> ServeReply {
        let data = Data(body.utf8)
        let head = ServeRequestHead(method: "POST", path: "/v1/chat/completions", host: "127.0.0.1:1212", origin: nil,
                                    transferEncoding: nil, contentLength: String(data.count), userAgent: "t")
        return await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212), environment: env, body: { data })
    }

    private func response(_ reply: ServeReply) -> ServeResponse? { if case .full(let r) = reply { return r } else { return nil } }

    @Test func answersWhenServedAndReady() async throws {
        let streamer = FakeChatStreamer([.snapshot("Hel"), .snapshot("Hello"), .usage(5, 2)])
        let r = try #require(response(await call(environment(readiness: .ready, streamer: streamer))))
        #expect(r.status == 200)
        let object = try #require(try JSONSerialization.jsonObject(with: r.body) as? [String: Any])
        let choice = try #require((object["choices"] as? [[String: Any]])?.first)
        #expect((choice["message"] as? [String: Any])?["content"] as? String == "Hello")
        #expect((object["usage"] as? [String: Int])?["total_tokens"] == 7)
    }

    @Test func servedButNotReadyIs503WithTheSentenceAndNeverCallsTheModel() async throws {
        let streamer = FakeChatStreamer([.snapshot("never")])
        let state = Readiness.needsSetup(reason: "Turn on Apple Intelligence in System Settings", action: .enableAppleIntelligence)
        let log = RequestLog()
        let r = try #require(response(await call(environment(readiness: state, streamer: streamer, log: log))))
        #expect(r.status == 503 && r.headers["Retry-After"] == "2")
        let error = try #require((try JSONSerialization.jsonObject(with: r.body) as? [String: Any])?["error"] as? [String: Any])
        #expect(error["message"] as? String == "Turn on Apple Intelligence in System Settings.")
        #expect(streamer.calls.isEmpty)
        #expect(log.entries.last?.status == 503)
    }

    @Test func notServedIs404EvenWhenReady() async throws {
        let r = try #require(response(await call(environment(readiness: .ready, streamer: FakeChatStreamer([]), served: []))))
        #expect(r.status == 404)
    }

    @Test func absentBelowMacOS26Is404() async throws {
        let r = try #require(response(await call(environment(readiness: nil, streamer: nil))))
        #expect(r.status == 404)
    }

    @Test func aContextOverflowIs400WithTheLimit() async throws {
        let streamer = FakeChatStreamer([.fail(.contextWindowExceeded(limit: 8192, tokenCount: 9000))])
        let r = try #require(response(await call(environment(readiness: .ready, streamer: streamer))))
        #expect(r.status == 400)
        #expect(String(decoding: r.body, as: UTF8.self).contains("8192"))
    }

    @Test func toolsAreIgnoredAndToolHistoryIsFlattenedAndMerged() async throws {
        let streamer = FakeChatStreamer([.snapshot("ok")])
        let log = RequestLog()
        let body = #"""
        {"model":"apple-foundation","tools":[{"type":"function","function":{"name":"read_file"}}],
         "messages":[{"role":"user","content":"Read a.txt"},
          {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{}"}}]},
          {"role":"tool","tool_call_id":"c1","content":"FILE"},
          {"role":"user","content":"summarise"}]}
        """#
        let r = try #require(response(await call(environment(readiness: .ready, streamer: streamer, log: log), body: body)))
        #expect(r.status == 200)
        let sent = try #require(streamer.calls.first)
        // R20 flatten + R21 merge: the tool result and "summarise" are one user turn, the last message.
        #expect(sent.prompt == "Conversation so far:\nUser: Read a.txt\nAssistant: [called read_file with {}]\n\nReply to the latest message from the user:\nUser: [tool result for read_file: FILE]\n\nsummarise")
        #expect(log.entries.last?.toolsIgnored == true)
    }

    @Test func streamsOverTheSameSSEPatternWithKeepAliveDuringTheWait() async throws {
        let streamer = FakeChatStreamer([.wait(0.45), .snapshot("Hi"), .snapshot("Hi there")])
        var env = environment(readiness: .ready, streamer: streamer)
        env.keepAliveSeconds = 0.1
        let reply = await call(env, body: #"{"model":"apple-foundation","stream":true,"messages":[{"role":"user","content":"hi"}]}"#)
        guard case .stream(let status, let headers, let body) = reply else { Issue.record("expected a stream"); return }
        #expect(status == 200 && headers["Content-Type"] == "text/event-stream")
        var bytes: [UInt8] = []
        for await chunk in body { bytes += chunk }
        let text = String(decoding: bytes, as: UTF8.self)
        #expect(text.hasSuffix("data: [DONE]\n\n"))
        let beforeFirstData = text[..<(text.range(of: "data: ")?.lowerBound ?? text.endIndex)]
        #expect(beforeFirstData.components(separatedBy: ": keep-alive\n\n").count - 1 >= 2)
        let contents = text.components(separatedBy: "\n\n").compactMap { event -> String? in
            guard event.hasPrefix("data: {"), let data = event.dropFirst(6).data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let delta = (object["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any],
                  let content = delta["content"] as? String, !content.isEmpty else { return nil }
            return content
        }
        #expect(contents == ["Hi", " there"])                       // deltas, not repeated snapshots
    }
}

// MARK: - The real error types → the seam's errors (macOS 26+)

struct AppleFoundationErrorMapperTests {
    @Test func the26xContextErrorMapsWithTheModelsOwnContextSize() throws {
        guard #available(macOS 26, *) else { return }
        let error = LanguageModelSession.GenerationError.exceededContextWindowSize(.init(debugDescription: "too long"))
        #expect(AppleFoundationErrorMapper.map(error) == .contextWindowExceeded(limit: SystemLanguageModel.default.contextSize, tokenCount: nil))
    }

    @Test func the26xGuardrailAndLanguageErrorsMap() throws {
        guard #available(macOS 26, *) else { return }
        #expect(AppleFoundationErrorMapper.map(LanguageModelSession.GenerationError.guardrailViolation(.init(debugDescription: "x"))) == .guardrailViolation)
        #expect(AppleFoundationErrorMapper.map(LanguageModelSession.GenerationError.unsupportedLanguageOrLocale(.init(debugDescription: "x"))) == .unsupportedLanguage)
        #expect(AppleFoundationErrorMapper.map(LanguageModelSession.GenerationError.rateLimited(.init(debugDescription: "x"))) == .failed)
    }

    @Test func the27ContextErrorCarriesItsOwnLimitAndCount() throws {
        guard #available(macOS 27, *) else { return }
        let error = LanguageModelError.contextSizeExceeded(.init(contextSize: 8192, tokenCount: 40060, debugDescription: "x"))
        #expect(AppleFoundationErrorMapper.map(error) == .contextWindowExceeded(limit: 8192, tokenCount: 40060))
    }

    @Test func anUnrelatedErrorIsFailed() throws {
        guard #available(macOS 26, *) else { return }
        #expect(AppleFoundationErrorMapper.map(URLError(.badURL)) == .failed)
    }

    @Test func theChatStreamerFactoryHonoursItsOverrideAndIsAbsentByDefault() {
        let saved = (AppleFoundationAvailability.useRealSystem, AppleFoundationAvailability.chatStreamerOverride)
        defer { (AppleFoundationAvailability.useRealSystem, AppleFoundationAvailability.chatStreamerOverride) = saved }
        AppleFoundationAvailability.useRealSystem = false
        AppleFoundationAvailability.chatStreamerOverride = nil
        #expect(AppleFoundationAvailability.makeChatStreamer() == nil)       // mock-by-default (AFM-FOLLOWUP-1)
        AppleFoundationAvailability.chatStreamerOverride = FakeChatStreamer([])
        #expect(AppleFoundationAvailability.makeChatStreamer() != nil)
    }
}

// MARK: - Real Apple Foundation Models (skipped unless Apple Intelligence is ready on this Mac)

/// Runs on a Mac with Apple Intelligence on; skips cleanly otherwise (macOS < 26, Apple
/// Intelligence off, model still downloading). No environment variable needed.
struct AppleFoundationRealModelTests {
    private static let ready: Bool = {
        guard #available(macOS 26, *) else { return false }
        return SystemLanguageModelAvailabilityChecker().readiness == .ready
    }()

    @Test(.enabled(if: AppleFoundationRealModelTests.ready), .timeLimit(.minutes(2)))
    func streamsASnapshotAtATimeAndTheBackendTurnsThemIntoDeltas() async throws {
        guard #available(macOS 26, *) else { return }
        let backend = AFMBackend(streamer: AppleFoundationModelExecutor(), readiness: { .ready })
        let r = request([ServeMessage(role: .system, text: "Answer in one short sentence."),
                         ServeMessage(role: .user, text: "My name is Sam."),
                         ServeMessage(role: .assistant, text: "Nice to meet you, Sam."),
                         ServeMessage(role: .user, text: "What is my name?")], temperature: 0)
        let out = try await events(backend, r)
        let reply = text(out)
        #expect(!reply.isEmpty)
        #expect(reply.localizedCaseInsensitiveContains("sam"))                 // the earlier turns reached the model, not just the last one
        #expect(out.last == .finish(.stop) || out.last == .finish(.length))
        guard case .usage(let usage)? = out.dropLast().last else { Issue.record("expected usage"); return }
        #expect(usage.promptTokens > 0 && usage.completionTokens > 0)
    }

    @Test(.enabled(if: AppleFoundationRealModelTests.ready), .timeLimit(.minutes(2)))
    func aConversationBeyondTheContextWindowIs400NamingTheLimit() async throws {
        guard #available(macOS 26, *) else { return }
        let backend = AFMBackend(streamer: AppleFoundationModelExecutor(), readiness: { .ready })
        let big = String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 4000)
        let out = try await events(backend, request([ServeMessage(role: .user, text: big)]))
        guard case .error(let error)? = out.last else { Issue.record("expected an error, got \(out.count) events"); return }
        #expect(error.status == 400 && error.code == "context_length_exceeded")
        #expect(error.message.contains("\(SystemLanguageModel.default.contextSize)"))
    }
}
