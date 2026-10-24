import Foundation
import MLX
import MLXGuidedGeneration
import MLXLMCommon
import Testing

@testable import MLXUI

// WW4-1-3 — the owner-run real-model checks (KICKOFF 13b), through the real `MLXChatBackend`, one family per invocation:
//
//   TEST_RUNNER_MLXUI_TEST_STRUCT_DIR=<installed checkpoint folder>  (the sandboxed host can't read it, so the run overrides
//   the sandbox on the command line: CODE_SIGN_ENTITLEMENTS=MLXUI/MLXUI-Direct.entitlements ENABLE_APP_SANDBOX=NO)
//
// Prints `STRUCT <label> …` lines for the journal. A family joins `ServeStructuredOutputFamilies.verified` only after every
// test here passes on its own weights. The allow-list gate is lifted for the process with `MLXUI_TEST_STRUCT_CANDIDATE=1`
// (a family not yet on the list), exactly as `PromptCacheRealModelTests` does for the prompt cache.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["MLXUI_TEST_STRUCT_DIR"] != nil))
struct ServeStructuredRealModelTests {
    static let schema = ##"""
    {"type":"object","properties":{
      "title":{"type":"string"},
      "priority":{"enum":["low","medium","high"]},
      "tags":{"type":"array","items":{"type":"string"}},
      "steps":{"type":"array","items":{"type":"object","properties":{"action":{"type":"string"},"done":{"type":"boolean"}},"required":["action","done"]}}},
     "required":["title","priority","tags","steps"]}
    """##

    struct Result {
        var text = ""
        var reasoning = ""
        var finish: ServeFinishReason?
        var usage: ServeUsage?
        var error: ServeError?
        var seconds = 0.0
        var firstContentSeconds: Double?
    }

    private static func make() throws -> (URL, MLXChatBackend, ServeGrammarCache) {
        let dir = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["MLXUI_TEST_STRUCT_DIR"]))
        var backend = MLXChatBackend(directory: dir, pool: ModelContainerPool(budgetBytes: 40 << 30))
        let cache = ServeGrammarCache()
        backend.grammarCache = cache
        if ProcessInfo.processInfo.environment["MLXUI_TEST_STRUCT_CANDIDATE"] == "1" {
            ServeStructuredOutputFamilies.overrideForTesting = ServeStructuredOutputFamilies.verified.union([backend.checkpointModelType])
        }
        return (dir, backend, cache)
    }

    private static func request(_ prompt: String, structured: Bool, temperature: Double = 0.7, maxTokens: Int = 600,
                                reasoning: Bool? = nil, seed: Int? = nil) -> ServeRequest {
        var request = ServeRequest(model: "m", messages: [ServeMessage(role: .user, text: prompt)])
        request.temperature = temperature
        request.maxTokens = maxTokens
        request.reasoning = reasoning
        request.seed = seed
        if structured { request.structure = .jsonSchema(name: "task", schema: schema, strict: true) }
        return request
    }

    /// prepare → generate → finish, the way the pipeline does it (the backend finishes the grant on a clean end).
    private static func run(_ request: ServeRequest, backend: MLXChatBackend, cancelAfterFirst: Bool = false) async throws -> Result {
        var result = Result()
        let grammar = try await backend.prepareStructure(request)
        let start = ContinuousClock.now
        let task = Task { () -> Result in
            var result = Result()
            for try await event in backend.generate(request, grammar: grammar) {
                switch event {
                case .textDelta(let piece):
                    if result.firstContentSeconds == nil { result.firstContentSeconds = MLXChatBackend.seconds(ContinuousClock.now - start) }
                    result.text += piece
                case .reasoningDelta(let piece): result.reasoning += piece
                case .usage(let usage): result.usage = usage
                case .finish(let reason): result.finish = reason
                case .error(let error): result.error = error
                default: break
                }
                if cancelAfterFirst, !result.text.isEmpty { throw CancellationError() }
            }
            return result
        }
        do { result = try await task.value } catch is CancellationError { result.finish = nil }
        await grammar?.finish(clean: false)                       // what the pipeline does after the backend returns
        result.seconds = MLXChatBackend.seconds(ContinuousClock.now - start)
        return result
    }

    private static func isValidTask(_ text: String) -> Bool {
        guard let object = (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any],
              object["title"] is String,
              let priority = object["priority"] as? String, ["low", "medium", "high"].contains(priority),
              let tags = object["tags"] as? [Any], tags.allSatisfy({ $0 is String }),
              let steps = object["steps"] as? [[String: Any]],
              steps.allSatisfy({ $0["action"] is String && $0["done"] is Bool }) else { return false }
        return true
    }

    // MARK: 1. the stop token

    @Test func theStopTokenIsReadFromTheInstalledTokenizerAndIsOneTheLoadedModelStopsOn() async throws {
        let (dir, backend, _) = try Self.make()
        let tokenizer = try await HFTokenizerLoader().load(from: dir)
        let stops = ServeGrammarEOS.stopIDs(directory: dir, tokenizer: tokenizer)
        let pick = try #require(ServeGrammarEOS.pick(directory: dir, tokenizer: tokenizer))
        let token = tokenizer.convertIdToToken(pick) ?? "nil"
        print("STRUCT stop model=\(dir.lastPathComponent) type=\(backend.checkpointModelType) stopIDs=\(stops.sorted()) tokenizerEOS=\(String(describing: tokenizer.eosTokenId)) pick=\(pick) token=\(token)")
        let container = try await backend.pool.container(directory: dir)
        let loaded = await container.perform { context in
            (context.configuration.eosTokenIds,
             ServeGrammarEOS.pickIsStopToken(pick, configuration: context.configuration, tokenizer: context.tokenizer))
        }
        print("STRUCT stop loaded configuration.eosTokenIds=\(loaded.0.sorted()) pickIsStopToken=\(loaded.1)")
        #expect(loaded.1, "the pick must be in the loaded model's own stop set")
        #expect(ServeGrammarEOS.turnEnders.contains(token), "\(token) is not a known turn-ender spelling")
    }

    // MARK: 2. ten replies parse

    @Test func tenSchemaRepliesParseAndEachEndsComplete() async throws {
        let (_, backend, cache) = try Self.make()
        let prompts = ["Plan a birthday party.", "Write a task for fixing a leaking tap.", "Create a to-do for moving flat.",
                       "A task about learning to swim.", "Plan a week of meals.", "Task: renew a passport.",
                       "Organise a team offsite.", "Prepare a conference talk.", "Repaint the kitchen.", "Set up a home network."]
        var valid = 0
        let rounds = Int(ProcessInfo.processInfo.environment["MLXUI_TEST_STRUCT_ROUNDS"] ?? "") ?? 1
        let all = (0 ..< rounds).flatMap { round in prompts.enumerated().map { ($0.offset + round * 10, $0.element) } }
        for (index, prompt) in all {
            let result = try await Self.run(Self.request(prompt, structured: true, maxTokens: 1500, seed: index), backend: backend)
            let ok = Self.isValidTask(result.text)
            if ok { valid += 1 }
            if !ok { print("STRUCT parse #\(index) INVALID TEXT: \(result.text.debugDescription)") }
            print("STRUCT parse #\(index) valid=\(ok) finish=\(String(describing: result.finish)) outcome=\(String(describing: result.usage?.structuredOutcome)) tokens=\(result.usage?.completionTokens ?? -1) error=\(String(describing: result.error?.code))")
            #expect(result.finish == .stop && result.usage?.structuredOutcome == .complete, "#\(index) did not end complete")
            #expect(result.reasoning.isEmpty)
        }
        print("STRUCT parse \(valid)/\(all.count)")
        #expect(valid == all.count)
        let compiles = await cache.compiles, idle = await cache.idleConstraints, resets = await cache.resets
        print("STRUCT leases compiles=\(compiles) resets=\(resets) idle=\(idle)")
        #expect(compiles == 1, "every request, one compile: the constraint was reused")
        #expect(idle == 1)
    }

    // MARK: 3. cache-on equals cache-off

    @Test func promptCacheOnEqualsPromptCacheOffUnderTheGrammar() async throws {
        let (_, backend, _) = try Self.make()
        let facts = (0..<60).map { "Fact \($0): the depot in district \($0 * 7 % 53) opens at \(6 + $0 % 5) on day \($0 % 7)." }.joined(separator: "\n")
        let prompt = facts + "\n\nUsing only the facts above, write one task about visiting the depot in district 14."
        func once(cache: Bool) async throws -> Result {
            PromptCache.overrideForTesting = cache
            defer { PromptCache.overrideForTesting = nil }
            return try await Self.run(Self.request(prompt, structured: true, temperature: 0), backend: backend)
        }
        let saved = PromptCacheAllowlist.overrideForTesting
        PromptCacheAllowlist.overrideForTesting = [backend.checkpointModelType]       // gemma4 isn't on the table; this is the evidence
        defer { PromptCacheAllowlist.overrideForTesting = saved }
        let off = try await once(cache: false)
        let onFirst = try await once(cache: true)
        let onSecond = try await once(cache: true)                                     // the hit
        print("STRUCT cache off.tokens=\(off.usage?.completionTokens ?? -1) onFirst.cached=\(onFirst.usage?.cachedTokens ?? -1) onSecond.cached=\(onSecond.usage?.cachedTokens ?? -1)")
        print("STRUCT cache identical off==onFirst:\(off.text == onFirst.text) off==onSecond:\(off.text == onSecond.text)")
        #expect(Self.isValidTask(off.text))
        #expect(off.text == onFirst.text)
        #expect(off.text == onSecond.text)
        #expect((onSecond.usage?.cachedTokens ?? 0) > 0, "the second cache-on run should have hit")
    }

    // MARK: 4. throughput

    @Test func tokensPerSecondGuidedAgainstUnguided() async throws {
        let (_, backend, _) = try Self.make()
        let prompt = "Write a task about preparing a garden for winter, as JSON with title, priority, tags and steps."
        _ = try await Self.run(Self.request(prompt, structured: false, temperature: 0, maxTokens: 20), backend: backend)    // warm the model
        var guided: [Double] = [], plain: [Double] = []
        for _ in 0..<3 {
            let g = try await Self.run(Self.request(prompt, structured: true, temperature: 0), backend: backend)
            let p = try await Self.run(Self.request(prompt, structured: false, temperature: 0, maxTokens: g.usage?.completionTokens ?? 200), backend: backend)
            func rate(_ r: Result) -> Double {
                let decode = r.seconds - (r.firstContentSeconds ?? 0)
                return decode > 0 ? Double(max(0, (r.usage?.completionTokens ?? 0) - 1)) / decode : 0
            }
            guided.append(rate(g)); plain.append(rate(p))
            print("STRUCT speed guided.tok=\(g.usage?.completionTokens ?? -1) guided.ttft=\(String(format: "%.2f", g.firstContentSeconds ?? -1)) plain.tok=\(p.usage?.completionTokens ?? -1) plain.ttft=\(String(format: "%.2f", p.firstContentSeconds ?? -1))")
        }
        let g = guided.sorted()[1], p = plain.sorted()[1]
        print("STRUCT speed median guided=\(String(format: "%.1f", g)) tok/s unguided=\(String(format: "%.1f", p)) tok/s ratio=\(String(format: "%.2f", p > 0 ? g / p : 0))")
        #expect(g > 0 && p > 0)
    }

    // MARK: 5. leases on the real paths, and thinking

    @Test func aCancelledGenerationDropsItsConstraintAndACleanOneKeepsIt() async throws {
        let (_, backend, cache) = try Self.make()
        let clean = try await Self.run(Self.request("A task about tidying a desk.", structured: true), backend: backend)
        #expect(clean.finish == .stop)
        let afterClean = (await cache.idleConstraints, await cache.resets)
        print("STRUCT lease clean idle=\(afterClean.0) resets=\(afterClean.1)")
        #expect(afterClean == (1, 1))
        // Cancel mid-generation: the long prompt guarantees more than one delta.
        let cancelled = try await Self.run(Self.request("A very long task with fifty detailed steps about building a shed.", structured: true, maxTokens: 900),
                                           backend: backend, cancelAfterFirst: true)
        _ = cancelled
        let afterCancel = (await cache.idleConstraints, await cache.resets)
        print("STRUCT lease cancelled idle=\(afterCancel.0) resets=\(afterCancel.1)")
        #expect(afterCancel.1 == 1, "a cancelled generation is not reset-and-reused")
        #expect(afterCancel.0 == 0, "the idle constraint was taken by the cancelled request and dropped")
        // The next request compiles a fresh one and is fine.
        let next = try await Self.run(Self.request("A task about washing a car.", structured: true), backend: backend)
        #expect(Self.isValidTask(next.text))
    }

    @Test func aStructuredReplyIsNeverAThinkingReplyAndALengthCutIsReportedIncomplete() async throws {
        let (_, backend, _) = try Self.make()
        let thinking = try await Self.run(Self.request("A task about baking bread.", structured: true, reasoning: true), backend: backend)
        #expect(thinking.reasoning.isEmpty)
        #expect(Self.isValidTask(thinking.text))
        // 12 tokens is refused up front (the smallest reply for this schema is 13: `max_tokens_too_small_for_schema`); 20 passes
        // that check and is still far short of a whole task.
        do {
            _ = try await backend.prepareStructure(Self.request("x", structured: true, maxTokens: 12))
            Issue.record("12 tokens should be refused")
        } catch let error as ServeError { #expect(error.code == "max_tokens_too_small_for_schema") }
        let cut = try await Self.run(Self.request("A task with many steps about baking bread.", structured: true, maxTokens: 20), backend: backend)
        print("STRUCT cut finish=\(String(describing: cut.finish)) outcome=\(String(describing: cut.usage?.structuredOutcome)) text=\(cut.text)")
        #expect(cut.finish == .length)
        #expect(cut.usage?.structuredOutcome == .incomplete)
    }

    // MARK: 6. the soft-zone closing bias (commit 3)

    @Test func theClosingBiasFinishesMoreRepliesInsideATightLimitAndNeverBreaksOne() async throws {
        let (_, backend, _) = try Self.make()
        let prompts = ["Plan a birthday party.", "Write a task for fixing a leaking tap.", "Create a to-do for moving flat.",
                       "A task about learning to swim.", "Plan a week of meals.", "Task: renew a passport.",
                       "Organise a team offsite.", "Prepare a conference talk."]
        var completeWith = 0, completeWithout = 0, brokenWith = 0
        for (index, prompt) in prompts.enumerated() {
            for enabled in [true, false] {
                var b = backend
                b.closingBiasEnabled = enabled
                let result = try await Self.run(Self.request(prompt, structured: true, temperature: 0, maxTokens: 130, seed: index), backend: b)
                let complete = result.usage?.structuredOutcome == .complete
                if enabled { completeWith += complete ? 1 : 0 } else { completeWithout += complete ? 1 : 0 }
                if enabled, complete, !Self.isValidTask(result.text) { brokenWith += 1 }
                print("STRUCT bias #\(index) enabled=\(enabled) outcome=\(String(describing: result.usage?.structuredOutcome)) tokens=\(result.usage?.completionTokens ?? -1) valid=\(Self.isValidTask(result.text))")
            }
        }
        print("STRUCT bias complete with=\(completeWith)/8 without=\(completeWithout)/8 brokenWhenComplete=\(brokenWith)")
        #expect(brokenWith == 0, "a reply the bias closed must still be a valid instance")
        #expect(completeWith >= completeWithout)
    }

    // MARK: diagnostics (opt in: MLXUI_TEST_STRUCT_DIAGNOSE=n runs up to n requests, stops at the first invalid reply)

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MLXUI_TEST_STRUCT_DIAGNOSE"] != nil))
    func diagnoseAnInvalidReply() async throws {
        let (dir, backend, _) = try Self.make()
        let limit = Int(ProcessInfo.processInfo.environment["MLXUI_TEST_STRUCT_DIAGNOSE"] ?? "") ?? 40
        let tokenizer = try await HFTokenizerLoader().load(from: dir)
        let eos = try #require(ServeGrammarEOS.pick(directory: dir, tokenizer: tokenizer))
        let vocabulary = try ServeGrammarVocabulary(directory: dir, tokenizer: tokenizer, eosTokenID: eos)
        let extracted = TokenizerVocabExtractor.extractForGrammar(from: tokenizer)
        let prompts = ["Task: renew a passport.", "Plan a birthday party.", "Set up a home network.", "Prepare a conference talk."]
        for index in 0 ..< limit {
            let request = Self.request(prompts[index % prompts.count], structured: true, seed: index)
            let grammar = try #require(try await backend.prepareStructure(request))
            let processor = grammar.makeProcessor()
            var text = ""
            for try await event in backend.generate(request, grammar: grammar) { if case .textDelta(let piece) = event { text += piece } }
            let ids = processor.committedHistory.map(Int.init)
            await grammar.finish(clean: false)
            guard !Self.isValidTask(text) else { continue }
            print("STRUCT DIAG invalid at request \(index); \(ids.count) tokens; failure=\(String(describing: processor.failure)) terminated=\(processor.isTerminated)")
            print("STRUCT DIAG text=\(text.debugDescription)")
            // Compare each committed token's decoded text with what xgrammar was told the token is.
            var joined = ""
            for (position, id) in ids.enumerated() {
                let decoded = tokenizer.decode(tokenIds: [id])
                let told = id < extracted.vocab.count ? extracted.vocab[id] : "?"
                joined += decoded
                if decoded != told { print("STRUCT DIAG mismatch at \(position): id=\(id) decoded=\(decoded.debugDescription) xgrammar=\(told.debugDescription)") }
            }
            print("STRUCT DIAG joined==text:\(joined == text) tokens=\(ids)")
            // Replay through a fresh matcher.
            let fresh = try ServeGrammarCache.compile(vocabulary: vocabulary, format: .jsonSchema(Self.schema))
            for (position, id) in ids.enumerated() {
                let mask = try fresh.computeMask()
                let ok = mask.mask.count * 32 > id && (mask.mask[id / 32] >> Int32(id % 32)) & 1 == 1
                if !ok || !mask.needsApply { print("STRUCT DIAG replay step \(position) id=\(id) \(tokenizer.decode(tokenIds: [id]).debugDescription): allowedByMask=\(ok) needsApply=\(mask.needsApply)") }
                _ = try fresh.commitToken(Int32(id))
            }
            return
        }
        print("STRUCT DIAG no invalid reply in \(limit) requests")
    }
}

/// The token sequence of the one reply the grammar let through without a comma (journal 2026-573), replayed against the real
/// vocabulary with no weights: where does the token-level matcher disagree with the character-level one?
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["MLXUI_TEST_STRUCT_DIR"] != nil))
struct ServeStructuredTokenProbeTests {
    static let ids: [Int] = [236782, 107, 138, 236775, 3250, 1083, 623, 34553, 5827, 496, 6863, 10981, 236787, 562, 61774, 15308, 827, 107, 138, 236775, 972, 236747, 160687, 1083, 623, 28602, 827, 107, 138, 236775, 9088, 236751, 1083, 11058, 120592, 827, 623, 45789, 827, 623, 7843, 236772, 1071, 1935, 107, 140, 2098, 25438, 1083, 870, 107, 140, 236782, 107, 142, 236775, 552, 34667, 1083, 623, 47770, 236743, 236770, 236787, 22113, 833, 55302, 568, 818, 50573, 22730, 107, 142, 236775, 236753, 811, 1083, 2416, 107, 140, 1263, 107, 140, 236782, 107, 142, 236775, 552, 34667, 1083, 623, 47770, 236743, 236778, 236787, 56972, 40383, 568, 818, 8689, 22730, 107, 142, 236775, 236753, 811, 1083, 2416, 107, 140, 1263, 107, 140, 236782, 107, 142, 236775, 552, 34667, 1083, 623, 47770, 236743, 236800, 236787, 35265, 833, 33582, 568, 818, 3764, 236772, 28898, 22730, 107, 142, 236775, 236753, 811, 1083, 2416, 107, 140, 1263, 107, 140, 236782, 107, 142, 236775, 552, 34667, 1083, 623, 47770, 236743, 236812, 236787, 62840, 833, 12096, 568, 818, 40675, 22730, 107, 142, 236775, 236753, 811, 1083, 2416, 107, 140, 236783, 107, 138, 236842, 107, 236783, 106]

    @Test func replayTheCommaLessReply() async throws {
        let dir = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["MLXUI_TEST_STRUCT_DIR"]))
        let tokenizer = try await HFTokenizerLoader().load(from: dir)
        let eos = try #require(ServeGrammarEOS.pick(directory: dir, tokenizer: tokenizer))
        let vocabulary = try ServeGrammarVocabulary(directory: dir, tokenizer: tokenizer, eosTokenID: eos)
        let extracted = TokenizerVocabExtractor.extractForGrammar(from: tokenizer)
        print("PROBE vocabType=\(extracted.vocabType) vocab=\(extracted.vocab.count)")
        let constraint = try ServeGrammarCache.compile(vocabulary: vocabulary, format: .jsonSchema(ServeStructuredRealModelTests.schema))
        for (position, id) in Self.ids.enumerated() {
            let mask = try constraint.computeMask()
            let allowed = !mask.needsApply || (mask.mask[id / 32] >> Int32(id % 32)) & 1 == 1
            if (40 ... 52).contains(position) {
                print("PROBE step \(position) id=\(id) decoded=\(tokenizer.decode(tokenIds: [id]).debugDescription) xgrammarString=\(extracted.vocab[id].debugDescription) allowed=\(allowed) needsApply=\(mask.needsApply)")
            }
            _ = try constraint.commitToken(Int32(id))
        }
        // Which single tokens does the matcher allow right after the closing bracket of "tags"?
        let fresh = try ServeGrammarCache.compile(vocabulary: vocabulary, format: .jsonSchema(ServeStructuredRealModelTests.schema))
        for id in Self.ids.prefix(46) { _ = try fresh.commitToken(Int32(id)) }
        let mask = try fresh.computeMask()
        var allowedIDs: [Int] = []
        for id in 0 ..< extracted.vocab.count where (mask.mask[id / 32] >> Int32(id % 32)) & 1 == 1 { allowedIDs.append(id) }
        print("PROBE after the tags array closed: \(allowedIDs.count) tokens allowed; first 40: \(allowedIDs.prefix(40).map { "\($0):\(extracted.vocab[$0].debugDescription)" })")
        print("PROBE is 107 (newline) allowed there? \(allowedIDs.contains(107)); 2098 allowed? \(allowedIDs.contains(2098))")
    }
}
