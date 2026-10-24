import Foundation
import MLX
import MLXGuidedGeneration
import Testing

@testable import MLXUI

// WW4-1-3 — the structured-output wiring that needs no weights: the grant's lease rules (ruling AS), the pipeline's ordering
// (compile before any queue place) and every exit path, the AX allow-list, and the pure ending/finish rules. The real-model
// half is `ServeStructuredRealModelTests` (owner-run).

private enum Fake {
    static let characters: [Character] = (32 ... 126).map { Character(UnicodeScalar($0)) } + ["\n", "\t"]
    static let eos = characters.count
    static func vocabulary() throws -> ServeGrammarVocabulary {
        try ServeGrammarVocabulary(key: "fake-backend#eos=\(eos)", vocab: characters.map(String.init) + ["<eos>"], vocabType: .raw,
                                   eosTokenID: eos, whitespaceTokenIDs: [32, 95, 96])
    }
    static let schema = ServeGrammarFormat.jsonSchema(#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}"#)
}

private let model = "mlx-community/gemma-4-e4b-it-4bit"

/// A backend with a real grammar cache and a scripted generation.
private final class StructuredStub: ServeBackend, @unchecked Sendable {
    enum Run: Sendable { case cleanFinish, fail, hang }
    let cache: ServeGrammarCache
    let run: Run
    private let lock = NSLock()
    private var _prepared = 0, _generated = 0
    var queueAtPrepare: (running: Int, waiting: Int)?
    let queue: ServeQueue?
    var prepared: Int { lock.lock(); defer { lock.unlock() }; return _prepared }
    var generated: Int { lock.lock(); defer { lock.unlock() }; return _generated }

    init(cache: ServeGrammarCache, run: Run, queue: ServeQueue? = nil) {
        self.cache = cache; self.run = run; self.queue = queue
    }

    var modelType: String? { "gemma4" }

    func prepareStructure(_ request: ServeRequest) async throws -> ServeGrammarGrant? {
        guard request.structure.isConstrained, let format = request.structure.grammarFormat else { return nil }
        if let queue { queueAtPrepare = (await queue.runningCount, await queue.waitingCount) }
        lock.lock(); _prepared += 1; lock.unlock()
        let vocabulary = try await cache.vocabulary(key: "fake-backend#eos=\(Fake.eos)") { try Fake.vocabulary() }
        let lease = try await cache.lease(vocabulary: vocabulary, format: format)
        return ServeGrammarGrant(cache: cache, lease: lease, vocabulary: vocabulary, format: format)
    }

    func generate(_ request: ServeRequest) -> AsyncThrowingStream<ServeEvent, Error> { generate(request, grammar: nil) }

    func generate(_ request: ServeRequest, grammar: ServeGrammarGrant?) -> AsyncThrowingStream<ServeEvent, Error> {
        lock.lock(); _generated += 1; lock.unlock()
        let run = run
        return AsyncThrowingStream { continuation in
            let task = Task {
                switch run {
                case .cleanFinish:
                    _ = grammar?.makeProcessor()
                    await grammar?.finish(clean: true)
                    continuation.yield(.textDelta("{}"))
                    continuation.yield(.finish(.stop))
                    continuation.finish()
                case .fail:
                    continuation.finish(throwing: ServeError(status: 500, message: "boom", code: "boom"))
                case .hang:
                    try? await Task.sleep(for: .seconds(30))
                    continuation.finish()
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

private func environment(_ backend: ServeBackend, queue: ServeQueue = ServeQueue()) -> ServeEnvironment {
    ServeEnvironment(
        servedModelIDs: { [model] },
        installed: { InstalledModelIndex(entries: [.init(id: "mlx-community--gemma-4-e4b-it-4bit", hfModelId: model, kind: .llm, ramGB: 3, downloadSizeGB: 2)]) },
        appleFoundationReadiness: { nil }, created: 1_700_000_000,
        backendFor: { _ in backend }, queue: queue, requestLog: RequestLog(), keepAliveSeconds: 1.0)
}

private func tracker(_ env: ServeEnvironment) -> RequestTracker {
    RequestTracker(head: ServeRequestHead(method: "POST", path: "/v1/chat/completions", host: "127.0.0.1:1212", origin: nil,
                                          transferEncoding: nil, contentLength: "2", userAgent: nil),
                   started: Date(), environment: env)
}

private func structuredRequest(reasoning: Bool? = nil) -> ServeRequest {
    var request = ServeRequest(model: model, messages: [ServeMessage(role: .user, text: "hi")])
    request.structure = .jsonSchema(name: "t", schema: #"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}"#, strict: true)
    request.reasoning = reasoning
    return request
}

@Suite(.serialized)
struct ServeGrammarGrantTests {
    private static func grant(_ cache: ServeGrammarCache) async throws -> ServeGrammarGrant {
        let vocabulary = try Fake.vocabulary()
        let lease = try await cache.lease(vocabulary: vocabulary, format: Fake.schema)
        return ServeGrammarGrant(cache: cache, lease: lease, vocabulary: vocabulary, format: Fake.schema)
    }

    @Test func aCleanFinishResetsTheConstraintAndParksItIdle() async throws {
        let cache = ServeGrammarCache()
        let grant = try await Self.grant(cache)
        let processor = grant.makeProcessor()
        #expect(processor === grant.makeProcessor(), "one processor per grant")
        #expect(await grant.finish(clean: true))
        #expect(await cache.idleConstraints == 1)
        #expect(await cache.resets == 1)
    }

    @Test func aFinishThatIsNotCleanDropsTheConstraint() async throws {
        let cache = ServeGrammarCache()
        let grant = try await Self.grant(cache)
        #expect(await grant.finish(clean: false))
        #expect(await cache.idleConstraints == 0)
        #expect(await cache.resets == 0)
    }

    @Test func finishIsIdempotentSoALaterNotCleanCallCannotUndoACleanOne() async throws {
        let cache = ServeGrammarCache()
        let grant = try await Self.grant(cache)
        #expect(await grant.finish(clean: true))
        #expect(await grant.finish(clean: false) == false)
        #expect(await grant.finish(clean: true) == false)
        #expect(await cache.idleConstraints == 1)
        #expect(await cache.resets == 1)
    }

    @Test func aReusedConstraintStartsFromTheBeginningAgain() async throws {
        let cache = ServeGrammarCache()
        let first = try await Self.grant(cache)
        let processor = first.makeProcessor()
        for character in #"{"a""# { processor.didSample(token: MLXArray(Int32(Fake.characters.firstIndex(of: character)!))) }
        #expect(processor.committedTokens == 4)
        await first.finish(clean: true)
        let second = try await Self.grant(cache)                          // takes the idle one
        #expect(await cache.compiles == 1, "the second lease reused the first constraint")
        #expect(second.makeProcessor().committedTokens == 0)
        let mask = try second.makeProcessor().process(logits: MLXArray.zeros([1, Fake.characters.count + 1]))
        #expect(mask.shape == [1, Fake.characters.count + 1])
        await second.finish(clean: false)
    }

    @Test func aGrantDroppedWithoutFinishingIsReleasedNotReusable() async throws {
        let cache = ServeGrammarCache()
        do { _ = try await Self.grant(cache) }                            // forgotten on purpose
        try await Task.sleep(for: .milliseconds(200))
        #expect(await cache.idleConstraints == 0)
        #expect(await cache.resets == 0)
    }

    @Test func aMatcherFailureNeverGoesBackIdleEvenIfTheCallerSaysClean() async throws {
        let cache = ServeGrammarCache()
        let grant = try await Self.grant(cache)
        let processor = grant.makeProcessor()
        processor.didSample(token: MLXArray(Int32(Fake.characters.firstIndex(of: "x")!)))     // `x` can't open the document
        #expect(processor.failure != nil)
        await grant.finish(clean: true)
        #expect(await cache.idleConstraints == 0)
    }
}

struct ServeStructuredEndingTests {
    @Test func theFinishReasonComesFromTheGrammar() {
        func ending(_ terminated: Bool, failed: Bool = false, client: Bool = false) -> (ServeStructuredOutcome, ServeFinishReason) {
            let e = MLXChatBackend.structuredEnding(terminated: terminated, matcherFailed: failed, stoppedByClient: client)
            return (e.outcome, e.finish)
        }
        #expect(ending(true) == (.complete, .stop))
        #expect(ending(false) == (.incomplete, .length), "a cut reply is a length stop, never a quiet stop")
        #expect(ending(false, client: true) == (.incomplete, .stop), "the client's own stop sequence is its stop")
        #expect(ending(true, failed: true) == (.incomplete, .length), "a terminated matcher that also failed is not complete")
        #expect(ending(true, client: true) == (.complete, .stop))
    }

    @Test func aConstraintIsReusableOnlyWhenNothingCanStillBeUsingIt() {
        #expect(MLXChatBackend.grammarIsReusable(stoppedByClient: false, matcherFailed: false))
        #expect(!MLXChatBackend.grammarIsReusable(stoppedByClient: true, matcherFailed: false))
        #expect(!MLXChatBackend.grammarIsReusable(stoppedByClient: false, matcherFailed: true))
    }

    @Test func theAllowListIsGemma4UntilAFamilyIsCheckedOnRealWeights() {
        #expect(ServeStructuredOutputFamilies.allows(modelType: "gemma4"))
        for other in ["qwen3", "qwen3_5", "llama", "gemma3", "gemma3_text", "mistral", ""] {
            #expect(!ServeStructuredOutputFamilies.allows(modelType: other), "\(other) is not verified")
        }
    }

    @Test func aBackendThatCannotConstrainRefusesAJsonFormatNamingTheModel() async throws {
        let backend = StructuredStub(cache: ServeGrammarCache(), run: .cleanFinish)
        // The protocol default (what AFM and every test double without an override gets):
        struct Plain: ServeBackend { func generate(_ request: ServeRequest) -> AsyncThrowingStream<ServeEvent, Error> { AsyncThrowingStream { $0.finish() } } }
        _ = backend
        do {
            _ = try await Plain().prepareStructure(structuredRequest())
            Issue.record("should have refused")
        } catch let error as ServeError {
            #expect(error.status == 400 && error.code == "structured_output_unsupported")
            #expect(error.message.contains(model))
        }
        #expect(try await Plain().prepareStructure(ServeRequest(model: model, messages: [])) == nil, "no structure: nothing to prepare")
    }
}

@Suite(.serialized)
struct ServeStructuredPipelineTests {
    @Test func theGrammarIsCompiledBeforeAnyQueuePlaceIsTaken() async throws {
        let queue = ServeQueue()
        let backend = StructuredStub(cache: ServeGrammarCache(), run: .cleanFinish, queue: queue)
        let env = environment(backend, queue: queue)
        let result = await ServePipeline.prepare(structuredRequest(), tracker: tracker(env), environment: env)
        guard case .success(let prepared) = result else { Issue.record("prepare failed: \(result)"); return }
        #expect(backend.queueAtPrepare?.running == 0 && backend.queueAtPrepare?.waiting == 0,
                "prepareStructure ran before reserve: no place existed yet")
        #expect(prepared.grammar != nil)
        let places = await queue.waitingCount + queue.runningCount
        #expect(places == 1, "…and the place was taken after")
        await queue.release(prepared.ticket)
        await prepared.grammar?.finish(clean: true)
    }

    @Test func aJsonFormatForcesThinkingOff() async throws {
        let backend = StructuredStub(cache: ServeGrammarCache(), run: .cleanFinish)
        let env = environment(backend)
        let result = await ServePipeline.prepare(structuredRequest(reasoning: true), tracker: tracker(env), environment: env)
        guard case .success(let prepared) = result else { Issue.record("prepare failed"); return }
        #expect(prepared.request.reasoning == false)
        await env.queue.release(prepared.ticket)
        await prepared.grammar?.finish(clean: true)
    }

    @Test func aRefusedQueuePlaceGivesTheConstraintBackIdle() async throws {
        let cache = ServeGrammarCache()
        let queue = ServeQueue(globalCap: 1, perModelDepth: 0)
        let blocker = await queue.reserve(model: "someone-else")              // running: the slot is taken, depth 0
        #expect(blocker != nil)
        let backend = StructuredStub(cache: cache, run: .cleanFinish)
        let env = environment(backend, queue: queue)
        let result = await ServePipeline.prepare(structuredRequest(), tracker: tracker(env), environment: env)
        guard case .failure(let failure) = result else { Issue.record("expected a 503"); return }
        #expect(failure.error.status == 503)
        #expect(await cache.idleConstraints == 1, "compiled, never used: parked for the next request")
    }

    @Test func aCleanGenerationLeavesTheConstraintReusable() async throws {
        let cache = ServeGrammarCache()
        let backend = StructuredStub(cache: cache, run: .cleanFinish)
        let env = environment(backend)
        let t = tracker(env)
        guard case .success(let prepared) = await ServePipeline.prepare(structuredRequest(), tracker: t, environment: env) else { Issue.record("prepare"); return }
        guard case .success(let collected) = await ServePipeline.collect(prepared, tracker: t, environment: env) else { Issue.record("collect"); return }
        #expect(collected.text == "{}")
        #expect(await cache.idleConstraints == 1, "the backend's clean finish stands; the pipeline's own finish is a no-op")
        #expect(await cache.resets == 1)
    }

    @Test func aFailedGenerationDropsTheConstraint() async throws {
        let cache = ServeGrammarCache()
        let backend = StructuredStub(cache: cache, run: .fail)
        let env = environment(backend)
        let t = tracker(env)
        guard case .success(let prepared) = await ServePipeline.prepare(structuredRequest(), tracker: t, environment: env) else { Issue.record("prepare"); return }
        guard case .failure = await ServePipeline.collect(prepared, tracker: t, environment: env) else { Issue.record("expected failure"); return }
        #expect(await cache.idleConstraints == 0)
        #expect(await cache.resets == 0)
        #expect(await env.queue.runningCount == 0, "the queue place was returned too")
    }

    @Test func aRequestCancelledWhileWaitingForItsTurnReleasesTheConstraintUnreusable() async throws {
        let cache = ServeGrammarCache()
        let queue = ServeQueue(globalCap: 1, perModelDepth: 4)
        let holder = await queue.reserve(model: "someone-else")                // occupies the one slot
        let backend = StructuredStub(cache: cache, run: .cleanFinish)
        let env = environment(backend, queue: queue)
        let t = tracker(env)
        guard case .success(let prepared) = await ServePipeline.prepare(structuredRequest(), tracker: t, environment: env) else { Issue.record("prepare"); return }
        let waiting = Task { await ServePipeline.collect(prepared, tracker: t, environment: env) }
        try await Task.sleep(for: .milliseconds(100))
        waiting.cancel()
        _ = await waiting.value
        #expect(backend.generated == 0, "never reached the backend")
        #expect(await cache.idleConstraints == 0)
        #expect(await cache.resets == 0)
        if let holder { await queue.release(holder) }
        #expect(await queue.waitingCount == 0)
    }

    @Test func aRequestCancelledDuringTheCompileTakesNoQueuePlaceAndLeavesNoLease() async throws {
        let cache = ServeGrammarCache()
        let queue = ServeQueue()
        let backend = StructuredStub(cache: cache, run: .cleanFinish, queue: queue)
        let env = environment(backend, queue: queue)
        // Occupy the compile thread so this request's compile queues behind it.
        let gate = DispatchSemaphore(value: 0), started = DispatchSemaphore(value: 0)
        let blocker = Task { try await ServeGrammarCompileThread.shared.run { started.signal(); gate.wait() } }
        await Task.detached { started.wait() }.value
        let request = Task { await ServePipeline.prepare(structuredRequest(), tracker: tracker(env), environment: env) }
        try await Task.sleep(for: .milliseconds(150))
        request.cancel()
        gate.signal()
        _ = try await blocker.value
        let result = await request.value
        guard case .failure(let failure) = result else { Issue.record("expected the request to fail as cancelled"); return }
        #expect(failure.error.status == 499)
        let (waiting, running) = (await queue.waitingCount, await queue.runningCount)
        #expect(waiting == 0 && running == 0, "no queue place was ever taken")
        #expect(backend.generated == 0)
        #expect(await cache.idleConstraints == 0, "the compile never started, so nothing exists to park")
    }

    @Test func aNonStructuredRequestNeverTouchesTheGrammarPath() async throws {
        let backend = StructuredStub(cache: ServeGrammarCache(), run: .cleanFinish)
        let env = environment(backend)
        let t = tracker(env)
        var plain = structuredRequest()
        plain.structure = .none
        guard case .success(let prepared) = await ServePipeline.prepare(plain, tracker: t, environment: env) else { Issue.record("prepare"); return }
        #expect(prepared.grammar == nil)
        #expect(backend.prepared == 0)
        await env.queue.release(prepared.ticket)
    }
}

/// The comma the real model left out (journal 2026-573): a document grammar-"complete" with `"tags": [...]` followed by `"steps"`
/// and no comma. These feed the same text one CHARACTER at a time to the real xgrammar, to say whether the grammar itself is lenient.
struct ServeMissingCommaProbeTests {
    private static let schema = ServeStructuredRealModelTests.schema
    private static let bad = "{\n  \"title\": \"x\",\n  \"priority\": \"medium\",\n  \"tags\": [\"a\"]\n    \"steps\": []\n}"
    private static let good = "{\n  \"title\": \"x\",\n  \"priority\": \"medium\",\n  \"tags\": [\"a\"],\n    \"steps\": []\n}"

    private static func rejectsAt(_ text: String) async throws -> Int? {
        let vocabulary = try Fake.vocabulary()
        let constraint = try await ServeGrammarCache().lease(vocabulary: vocabulary, format: .jsonSchema(schema)).constraint
        for (position, character) in text.enumerated() {
            guard let id = Fake.characters.firstIndex(of: character) else { return position }
            let mask = try constraint.computeMask()
            let allowed = !mask.needsApply || (mask.mask[id / 32] >> Int32(id % 32)) & 1 == 1
            if !allowed { return position }
            _ = try constraint.commitToken(Int32(id))
        }
        return nil
    }

    @Test func theGoodDocumentIsAcceptedCharacterByCharacter() async throws {
        #expect(try await Self.rejectsAt(Self.good) == nil)
    }

    @Test func theCommaLessDocumentIsRejectedCharacterByCharacter() async throws {
        let position = try await Self.rejectsAt(Self.bad)
        print("PROBE comma-less document rejected at character \(String(describing: position))")
        #expect(position != nil, "xgrammar's grammar itself accepts a missing comma")
    }
}

/// The detokenizer bug the real-model check found (journal 2026-573): a decode that merges a space into a following comma
/// (swift-transformers' `clean_up_tokenization_spaces`) made the window shrink, and the delta lost the comma.
struct IncrementalDetokenizerCleanupTests {
    /// Each id is a string piece; the decode joins them and then applies the " ," → "," clean-up, as the tokenizer does.
    private static let pieces = ["\"]", "\n", "    ", ",\"", "steps", "\":"]
    private static func decode(_ ids: [Int]) -> String {
        ids.map { pieces[$0] }.joined().replacingOccurrences(of: " ,", with: ",")
    }

    @Test func aCommaAfterWhitespaceIsNotSwallowedWhenTheDecodeCleansUpTheSpace() {
        var detokenizer = IncrementalDetokenizer(decode: Self.decode)
        var out = ""
        for id in 0 ..< Self.pieces.count { out += detokenizer.push(id) }
        out += detokenizer.flush()
        #expect(out.contains(",\"steps"), "the comma survived: \(out.debugDescription)")
        #expect(out.replacingOccurrences(of: " ", with: "") == "\"]\n,\"steps\":", "every non-space scalar of the stream is emitted once")
    }

    @Test func anOrdinaryStreamIsUnchanged() {
        var detokenizer = IncrementalDetokenizer(decode: { $0.map { "<\($0)>" }.joined() })
        var out = ""
        for id in [1, 2, 3] { out += detokenizer.push(id) }
        #expect(out == "<1><2><3>")
    }
}
