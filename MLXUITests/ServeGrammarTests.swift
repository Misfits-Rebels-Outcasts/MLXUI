import Foundation
import MLX
import MLXGuidedGeneration
import Testing

@testable import MLXUI

// WW4-1-1 — the grammar pieces, on a fake one-character-per-token vocabulary and the REAL xgrammar (no weights).
// Design note: RSI/notes/claude-code/wwdc26/ww4-design.md. Nothing here touches the server.

/// ids 0…94 are the printable ASCII characters 32…126, then `\n`, `\t`, then `<eos>`.
private enum FakeVocab {
    static let characters: [Character] = (32 ... 126).map { Character(UnicodeScalar($0)) } + ["\n", "\t"]
    static let eos = characters.count
    static let strings: [String] = characters.map(String.init) + ["<eos>"]
    static let ids: [Character: Int] = Dictionary(uniqueKeysWithValues: characters.enumerated().map { ($1, $0) })
    static let whitespace: Set<Int> = [ids[" "]!, ids["\n"]!, ids["\t"]!]
    static var count: Int { strings.count }

    static func vocabulary(key: String = "fake#eos=\(eos)") throws -> ServeGrammarVocabulary {
        try ServeGrammarVocabulary(key: key, vocab: strings, vocabType: .raw, eosTokenID: eos, whitespaceTokenIDs: whitespace)
    }

    static func allowed(_ mask: MaskResult, _ id: Int) -> Bool {
        guard mask.needsApply else { return true }
        let word = UInt32(bitPattern: mask.mask[id / 32])
        return (word >> UInt32(id % 32)) & 1 == 1
    }
}

private final class Box<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: T
    init(_ value: T) { stored = value }
    var value: T {
        get { lock.lock(); defer { lock.unlock() }; return stored }
        set { lock.lock(); stored = newValue; lock.unlock() }
    }
}

@Suite(.serialized)
struct ServeGrammarTests {
    private static let personSchema = #"""
    {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}
    """#

    private static func constraint(_ format: ServeGrammarFormat, vocabulary: ServeGrammarVocabulary) async throws -> GrammarConstraint {
        try await ServeGrammarCache().lease(vocabulary: vocabulary, format: format).constraint
    }

    private static func processor(_ format: ServeGrammarFormat, vocabulary: ServeGrammarVocabulary) async throws -> ServeGrammarProcessor {
        ServeGrammarProcessor(constraint: try await constraint(format, vocabulary: vocabulary), vocabulary: vocabulary, format: format)
    }

    /// Feed `text` one character at a time; `false` at the first character the grammar's mask forbids.
    private static func feed(_ text: String, into constraint: GrammarConstraint) throws -> Bool {
        for character in text {
            guard let id = FakeVocab.ids[character] else { return false }
            let mask = try constraint.computeMask()
            guard FakeVocab.allowed(mask, id) else { return false }
            _ = try constraint.commitToken(Int32(id))
        }
        return true
    }

    private static func canStop(_ constraint: GrammarConstraint) throws -> Bool {
        FakeVocab.allowed(try constraint.computeMask(), FakeVocab.eos)
    }

    /// The ids the processor lets through for the next token (logit stays 0), over `width` logits.
    private static func allowedIDs(_ processor: ServeGrammarProcessor, width: Int = FakeVocab.count) throws -> Set<Int> {
        let out = try MLXErrorGuard.read(processor.process(logits: MLXArray.zeros([1, width])), as: Float.self)
        return Set(out.indices.filter { out[$0] > -1000 })
    }

    private static func sample(_ processor: ServeGrammarProcessor, _ text: String) {
        for character in text { processor.didSample(token: MLXArray([Int32(FakeVocab.ids[character]!)])) }
    }

    // MARK: - mask correctness

    @Test func theMaskAtTheStartAllowsOnlyAnOpenBraceAndWhitespace() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let processor = try await Self.processor(.jsonSchema(Self.personSchema), vocabulary: vocabulary)
        let allowed = try Self.allowedIDs(processor)
        #expect(allowed.contains(FakeVocab.ids["{"]!))
        #expect(!allowed.contains(FakeVocab.ids["["]!))
        #expect(!allowed.contains(FakeVocab.ids["a"]!))
        #expect(!allowed.contains(FakeVocab.ids["1"]!))
        #expect(!allowed.contains(FakeVocab.eos), "the stop token is not allowed before the document is complete")
    }

    @Test func theSchemaTypesTheValues() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let processor = try await Self.processor(.jsonSchema(Self.personSchema), vocabulary: vocabulary)
        Self.sample(processor, #"{"name":"Ada","age":"#)
        let allowed = try Self.allowedIDs(processor)
        #expect(allowed.contains(FakeVocab.ids["4"]!), "an integer's digits are allowed")
        #expect(!allowed.contains(FakeVocab.ids["\""]!), "a string is not an integer")
        #expect(!allowed.contains(FakeVocab.ids["a"]!))
        #expect(processor.failure == nil)
    }

    @Test func aCompleteDocumentAllowsOnlyTheStopTokenAndThenTerminates() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let processor = try await Self.processor(.jsonSchema(Self.personSchema), vocabulary: vocabulary)
        Self.sample(processor, #"{"name":"Ada","age":36}"#)
        #expect(!processor.isTerminated, "complete, but the stop token has not been accepted yet")
        #expect(try Self.allowedIDs(processor).contains(FakeVocab.eos))
        processor.didSample(token: MLXArray([Int32(FakeVocab.eos)]))
        #expect(processor.isTerminated)
        #expect(processor.failure == nil)
        // After termination the mask admits the stop token and nothing else, so a loop that doesn't stop still can't emit text.
        #expect(try Self.allowedIDs(processor) == [FakeVocab.eos])
        #expect(processor.committedTokens == #"{"name":"Ada","age":36}"#.count + 1)
    }

    @Test func aCommitTheMatcherRefusesFailsTheProcessorAndForcesStop() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let processor = try await Self.processor(.jsonSchema(Self.personSchema), vocabulary: vocabulary)
        Self.sample(processor, "[")            // not in the mask — only reachable if the sampler ignored it
        #expect(processor.failure != nil)
        #expect(!processor.isTerminated)
        #expect(try Self.allowedIDs(processor) == [FakeVocab.eos])
    }

    @Test func logitsWiderThanTheVocabularyAreMaskedPastIt() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let processor = try await Self.processor(.jsonSchema(Self.personSchema), vocabulary: vocabulary)
        let allowed = try Self.allowedIDs(processor, width: FakeVocab.count + 40)        // a padded embedding matrix
        #expect(allowed.allSatisfy { $0 < FakeVocab.count })
        #expect(allowed.contains(FakeVocab.ids["{"]!))
    }

    @Test func logitsNarrowerThanTheVocabularyAreCut() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let processor = try await Self.processor(.jsonSchema(Self.personSchema), vocabulary: vocabulary)
        let allowed = try Self.allowedIDs(processor, width: 50)
        #expect(allowed.allSatisfy { $0 < 50 })
    }

    @Test func theMaskKeepsTheLogitsDtype() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let processor = try await Self.processor(.jsonSchema(Self.personSchema), vocabulary: vocabulary)
        let out = processor.process(logits: MLXArray.zeros([1, FakeVocab.count], dtype: .bfloat16))
        #expect(out.dtype == .bfloat16)
        #expect(out.shape == [1, FakeVocab.count])
    }

    // MARK: - json_object

    @Test func jsonObjectAcceptsANonEmptyObject() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let accepted = [
            #"{"a": 1, "b": [true, null, "x"], "c": {"d": 2.5e3}}"#,
            #"{"name":"Ada"}"#,
            "{}",
        ]
        for text in accepted {
            let constraint = try await Self.constraint(.jsonObject, vocabulary: vocabulary)
            #expect(try Self.feed(text, into: constraint), "json_object must accept \(text)")
            #expect(try Self.canStop(constraint), "…and be able to stop after it")
        }
    }

    @Test func jsonObjectRefusesWhatIsNotAnObject() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        for text in ["[1]", #""text""#, "42", "true", "null"] {
            let constraint = try await Self.constraint(.jsonObject, vocabulary: vocabulary)
            #expect(try !Self.feed(text, into: constraint), "json_object must refuse a root of \(text)")
        }
    }

    @Test func aBareObjectSchemaAlsoAcceptsMembersSoTheExplicitFormIsBeltAndBraces() async throws {
        // Measured (WW4-1-1): the design note feared strict_mode made `{"type":"object"}` mean `{}` only. It does not.
        let vocabulary = try FakeVocab.vocabulary()
        let constraint = try await Self.constraint(.jsonSchema(#"{"type":"object"}"#), vocabulary: vocabulary)
        #expect(try Self.feed(#"{"a":1}"#, into: constraint))
    }

    @Test func aDeclaredPropertyListIsClosedAndOrdered() async throws {
        // xgrammar emits properties in the schema's order, and strict_mode forbids undeclared ones.
        let vocabulary = try FakeVocab.vocabulary()
        let wrongOrder = try await Self.constraint(.jsonSchema(Self.personSchema), vocabulary: vocabulary)
        #expect(try !Self.feed(#"{"age":1"#, into: wrongOrder), "age before name")
        let extra = try await Self.constraint(.jsonSchema(Self.personSchema), vocabulary: vocabulary)
        #expect(try !Self.feed(#"{"name":"x","age":1,"zip":1}"#, into: extra), "no undeclared zip")
    }

    // MARK: - leases (fork is a stub in the vendored xgrammar), reset and the cache

    @Test func theVendoredXGrammarCannotForkSoTheDesignDoesNotDependOnIt() async throws {
        // Documents WHY the cache leases and rolls back instead of cloning (design note "clone()" is not available).
        let vocabulary = try FakeVocab.vocabulary()
        let constraint = try await Self.constraint(.jsonObject, vocabulary: vocabulary)
        do { _ = try constraint.clone(); Issue.record("xg_matcher_fork unexpectedly works: the cache can clone again — revisit ServeGrammarCache") }
        catch GrammarError.forkFailed(let message) { #expect(message.contains("Fork() not available")) }
    }

    @Test func twoLeasesAtOnceAreIndependentMatchers() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let cache = ServeGrammarCache()
        let first = try await cache.lease(vocabulary: vocabulary, format: .jsonSchema(Self.personSchema)).constraint
        let second = try await cache.lease(vocabulary: vocabulary, format: .jsonSchema(Self.personSchema)).constraint
        #expect(try Self.feed(#"{"name":"#, into: first))
        #expect(FakeVocab.allowed(try second.computeMask(), FakeVocab.ids["{"]!), "`second` is still at the start")
        #expect(try Self.feed(#"{"name":"x","age":7}"#, into: second))
        #expect(try Self.feed(#""x","age":1}"#, into: first), "…and advancing `second` did not move `first`")
        #expect(await cache.compiles == 2, "nothing was idle, so each lease compiled")
    }

    @Test func aReleasedConstraintIsRolledBackToTheStartAndReused() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let cache = ServeGrammarCache()
        let format = ServeGrammarFormat.jsonSchema(Self.personSchema)
        let lease = try await cache.lease(vocabulary: vocabulary, format: format)
        let processor = ServeGrammarProcessor(constraint: lease.constraint, vocabulary: vocabulary, format: format)
        Self.sample(processor, #"{"name":"Ada","age":36}"#)
        processor.didSample(token: MLXArray([Int32(FakeVocab.eos)]))
        #expect(processor.isTerminated, "released after the stop token was accepted — the hardest case to roll back")
        await cache.release(lease, committedTokens: processor.committedTokens, reusable: processor.failure == nil)
        #expect(await cache.idleConstraints == 1)
        #expect(await cache.resets == 1)

        let again = try await cache.lease(vocabulary: vocabulary, format: format)
        #expect(await cache.compiles == 1, "the second request paid no compile")
        let mask = try again.constraint.computeMask()
        #expect(FakeVocab.allowed(mask, FakeVocab.ids["{"]!))
        #expect(!FakeVocab.allowed(mask, FakeVocab.eos))
        #expect(!FakeVocab.allowed(mask, FakeVocab.ids["a"]!))
        // …and it is fully usable: a different document runs to the end.
        #expect(try Self.feed(#"{"name":"Bo","age":1}"#, into: again.constraint))
        #expect(try Self.canStop(again.constraint))
    }

    @Test func aReleaseMidDocumentAlsoResets() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let cache = ServeGrammarCache()
        let format = ServeGrammarFormat.jsonObject
        let lease = try await cache.lease(vocabulary: vocabulary, format: format)
        let processor = ServeGrammarProcessor(constraint: lease.constraint, vocabulary: vocabulary, format: format)
        Self.sample(processor, #"{"a": [1, 2"#)            // a max_tokens cut: no stop token
        await cache.release(lease, committedTokens: processor.committedTokens)
        let again = try await cache.lease(vocabulary: vocabulary, format: format)
        #expect(await cache.compiles == 1)
        #expect(try Self.feed("{}", into: again.constraint))
    }

    @Test func aFailedOrUnreleasedConstraintIsNotReused() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let cache = ServeGrammarCache()
        let format = ServeGrammarFormat.jsonObject
        let lease = try await cache.lease(vocabulary: vocabulary, format: format)
        await cache.release(lease, committedTokens: 0, reusable: false)
        #expect(await cache.idleConstraints == 0)
        _ = try await cache.lease(vocabulary: vocabulary, format: format)
        #expect(await cache.compiles == 2)
    }

    @Test func aProcessorCopyReplaysTheHistoryIntoAnIndependentMatcher() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let original = try await Self.processor(.jsonSchema(Self.personSchema), vocabulary: vocabulary)
        Self.sample(original, #"{"name":"#)
        let copy = original.copy()
        #expect(copy.committedTokens == original.committedTokens)
        Self.sample(copy, #""Bo","age":1}"#)
        #expect(!(original.isTerminated))
        #expect(try Self.allowedIDs(original).contains(FakeVocab.ids["\""]!), "the original is still at the opening quote")
        #expect(try Self.allowedIDs(copy).contains(FakeVocab.eos), "the copy ran to the end of the document")
        #expect(copy.failure == nil)
        #expect(original.failure == nil)
    }

    @Test func theIdleCacheIsKeyedBySchemaAndEvictsTheLeastRecentlyUsedKey() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let cache = ServeGrammarCache(vocabularyCapacity: 4, keyCapacity: 2, idlePerKey: 2)
        let a = ServeGrammarFormat.jsonSchema(#"{"type":"object","properties":{"a":{"type":"integer"}}}"#)
        let b = ServeGrammarFormat.jsonSchema(#"{"type":"object","properties":{"b":{"type":"integer"}}}"#)
        let c = ServeGrammarFormat.jsonSchema(#"{"type":"object","properties":{"c":{"type":"integer"}}}"#)
        for format in [a, b, a, c] {                       // a is touched again before c arrives, so c evicts b
            let lease = try await cache.lease(vocabulary: vocabulary, format: format)
            await cache.release(lease, committedTokens: 0)
        }
        #expect(await cache.compiles == 3, "a, b and c each compiled once; the second a was idle")
        #expect(await cache.cachedKeys == 2)
        _ = try await cache.lease(vocabulary: vocabulary, format: a)
        #expect(await cache.compiles == 3, "a survived")
        _ = try await cache.lease(vocabulary: vocabulary, format: b)
        #expect(await cache.compiles == 4, "b was evicted and is compiled again")
    }

    @Test func idleConstraintsPerKeyAreCapped() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let cache = ServeGrammarCache(vocabularyCapacity: 4, keyCapacity: 4, idlePerKey: 1)
        let one = try await cache.lease(vocabulary: vocabulary, format: .jsonObject)
        let two = try await cache.lease(vocabulary: vocabulary, format: .jsonObject)
        await cache.release(one, committedTokens: 0)
        await cache.release(two, committedTokens: 0)
        #expect(await cache.idleConstraints == 1)
    }

    @Test func theVocabularyIsBuiltOncePerKeyAndEvictionDropsItsConstraints() async throws {
        let cache = ServeGrammarCache(vocabularyCapacity: 1)
        let builds = Box(0)
        let first = try await cache.vocabulary(key: "m1") { builds.value += 1; return try FakeVocab.vocabulary(key: "m1") }
        _ = try await cache.vocabulary(key: "m1") { builds.value += 1; return try FakeVocab.vocabulary(key: "m1") }
        #expect(builds.value == 1)
        let lease = try await cache.lease(vocabulary: first, format: .jsonObject)
        await cache.release(lease, committedTokens: 0)
        #expect(await cache.idleConstraints == 1)
        _ = try await cache.vocabulary(key: "m2") { builds.value += 1; return try FakeVocab.vocabulary(key: "m2") }    // evicts m1
        #expect(await cache.cachedVocabularies == 1)
        #expect(await cache.idleConstraints == 0, "a constraint compiled against an evicted vocabulary is dropped with it")
    }

    @Test func aSchemaXGrammarRejectsIsAnInvalidSchemaError() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let cache = ServeGrammarCache()
        for bad in ["not json", #"{"type": 42}"#] {
            do {
                _ = try await cache.lease(vocabulary: vocabulary, format: .jsonSchema(bad))
                Issue.record("\(bad) should not compile")
            } catch GrammarError.invalidJSONSchema(let message) {
                #expect(!message.isEmpty)
            }
        }
        let idle = await cache.idleConstraints, compiled = await cache.compiles
        #expect(idle == 0 && compiled == 0, "a failed compile caches and counts nothing")
    }

    // MARK: - whitespace

    @Test func threeWhitespaceTokensInARowPenaliseWhitespaceAfterwards() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let processor = try await Self.processor(.jsonObject, vocabulary: vocabulary)
        let space = FakeVocab.ids[" "]!, quote = FakeVocab.ids["\""]!
        func logits() throws -> [Float] {
            try MLXErrorGuard.read(processor.process(logits: MLXArray.zeros([1, FakeVocab.count])), as: Float.self)
        }
        Self.sample(processor, "{")                   // whitespace is allowed after '{' (not before it: xgrammar refuses a leading space)
        #expect(try logits()[space] == 0)
        Self.sample(processor, "  ")
        #expect(try logits()[space] == 0, "two is not a run")
        Self.sample(processor, " ")
        let after = try logits()
        #expect(after[space] == ServeGrammarProcessor.whitespacePenalty)
        #expect(after[quote] == 0, "only whitespace is penalised; the structural token is untouched")
        #expect(processor.failure == nil)
    }

    @Test func aNonWhitespaceTokenResetsTheRun() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let processor = try await Self.processor(.jsonObject, vocabulary: vocabulary)
        Self.sample(processor, "{  \"  ")           // 2 spaces, reset by the quote, 2 spaces inside the key string
        let out = try MLXErrorGuard.read(processor.process(logits: MLXArray.zeros([1, FakeVocab.count])), as: Float.self)
        #expect(out[FakeVocab.ids[" "]!] == 0)
        #expect(processor.failure == nil)
    }

    @Test func leadingWhitespaceBeforeTheDocumentIsRefusedByTheGrammar() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let constraint = try await Self.constraint(.jsonObject, vocabulary: vocabulary)
        #expect(try !Self.feed(" {}", into: constraint))
    }

    // MARK: - threads (the reason the MLXArray is built inside process)

    @Test func theMaskSurvivesTheIteratorChangingThreadsBetweenSamplingAndProcessing() async throws {
        let vocabulary = try FakeVocab.vocabulary()
        let processor = try await Self.processor(.jsonSchema(Self.personSchema), vocabulary: vocabulary)
        let text = Array(#"{"name":"Ada","age":36}"#)
        let failures = Box<[String]>([])
        let seen = Box<[Set<Int>]>([])
        // Alternate: the token is sampled (and read) on one thread, the next mask is built and evaluated on another — what
        // `generateTokens` does between the prefill's first token (caller's thread) and the rest (GenerationWorker's).
        for (index, character) in text.enumerated() {
            let id = FakeVocab.ids[character]!
            try await Self.onFreshThread { processor.didSample(token: MLXArray([Int32(id)])) }
            try await Self.onFreshThread {
                do {
                    let out = try MLXErrorGuard.read(processor.process(logits: MLXArray.zeros([1, FakeVocab.count])), as: Float.self)
                    seen.value.append(Set(out.indices.filter { out[$0] > -1000 }))
                } catch { failures.value.append("step \(index): \(error)") }
            }
        }
        #expect(failures.value.isEmpty, "\(failures.value)")
        #expect(seen.value.count == text.count)
        #expect(seen.value.last?.contains(FakeVocab.eos) == true, "the mask after the last character allows the stop token")
        #expect(processor.failure == nil)
    }

    private static func onFreshThread(_ body: @escaping @Sendable () -> Void) async throws {
        await withCheckedContinuation { continuation in
            let thread = Thread { body(); continuation.resume() }
            thread.stackSize = 4 << 20
            thread.start()
        }
    }
}

// MARK: - The stop token pick

struct ServeGrammarEOSTests {
    private static func pick(_ stops: Set<Int>, tokenizerEOS: Int?, _ names: [Int: String]) -> Int? {
        ServeGrammarEOS.pick(stopIDs: stops, tokenizerEOS: tokenizerEOS, tokenString: { names[$0] })
    }

    @Test func gemma4EndsOnTheTurnEnderNotTheBaseEOS() {
        // generation_config.json of mlx-community/gemma-4-e4b-it-4bit: eos_token_id [1, 106, 50]; tokenizer eos_token `<eos>`.
        #expect(Self.pick([1, 106, 50], tokenizerEOS: 1, [1: "<eos>", 106: "<turn|>", 50: "<|tool_response>"]) == 106)
    }

    @Test func gemma3EndsOnEndOfTurn() {
        #expect(Self.pick([1, 106], tokenizerEOS: 1, [1: "<eos>", 106: "<end_of_turn>"]) == 106)
    }

    @Test func qwenEndsOnImEnd() {
        #expect(Self.pick([151645, 151643], tokenizerEOS: 151645, [151645: "<|im_end|>", 151643: "<|endoftext|>"]) == 151645)
    }

    @Test func llama3EndsOnEotNotEom() {
        #expect(Self.pick([128001, 128008, 128009], tokenizerEOS: 128001,
                          [128001: "<|end_of_text|>", 128008: "<|eom_id|>", 128009: "<|eot_id|>"]) == 128009)
    }

    @Test func aFamilyWithNoKnownTurnEnderUsesTheTokenizersEOSThenTheSmallestID() {
        #expect(Self.pick([2, 7], tokenizerEOS: 7, [2: "</s>", 7: "<eos>"]) == 7)
        #expect(Self.pick([9, 4], tokenizerEOS: nil, [9: "a", 4: "b"]) == 4)
        #expect(Self.pick([9, 4], tokenizerEOS: 99, [9: "a", 4: "b"]) == 4, "a tokenizer EOS outside the stop set is not usable")
    }

    @Test func anEmptyStopSetHasNoPick() {
        #expect(Self.pick([], tokenizerEOS: 1, [:]) == nil)
    }
}

// MARK: - WW4-1-3 commit 1: the compile thread (ruling AV) and recursive `$ref` (ruling AW)

struct ServeGrammarCompileThreadTests {
    @Test func compilesRunOnTheNamedThreadWithTheLargeStack() async throws {
        let (name, onMain, stack) = try await ServeGrammarCompileThread.shared.run {
            (Thread.current.name ?? "", Thread.isMainThread, Thread.current.stackSize)
        }
        #expect(name == "ServeGrammarCompileThread")
        #expect(!onMain)
        #expect(stack >= ServeGrammarCompileThread.stackBytes)
    }

    @Test func aSchemaPastTheOldCrashDepthCompilesThroughTheLeasePath() async throws {
        // 201 containers: the old test-host thread died with SIGBUS at 101 (journal 2026-572).
        let depth = 100
        let schema = String(repeating: #"{"type":"object","properties":{"c":"#, count: depth) + #"{"type":"string"}"#
            + String(repeating: #"},"required":["c"]}"#, count: depth)
        let lease = try await ServeGrammarCache().lease(vocabulary: try FakeVocab.vocabulary(), format: .jsonSchema(schema))
        #expect(!(try lease.constraint.computeMask().isTerminated))
    }

    @Test func aJobStillQueuedWhenItsCallerIsCancelledNeverRuns() async throws {
        let thread = ServeGrammarCompileThread.shared
        let gate = DispatchSemaphore(value: 0)
        let started = DispatchSemaphore(value: 0)
        let ran = RanFlag()
        let blocker = Task { try await thread.run { started.signal(); gate.wait() } }
        await Task.detached { started.wait() }.value        // the blocker occupies the thread
        let queued = Task { try await thread.run { ran.set() } }
        try await Task.sleep(for: .milliseconds(50))
        queued.cancel()
        gate.signal()
        _ = try await blocker.value
        do { _ = try await queued.value; Issue.record("the cancelled job should throw") } catch is CancellationError {}
        #expect(!ran.value)
    }

    @Test func aRequestCancelledDuringACompileLeavesTheConstraintIdleAndLeasesNothing() async throws {
        let cache = ServeGrammarCache()
        let vocabulary = try FakeVocab.vocabulary()
        let thread = ServeGrammarCompileThread.shared
        let gate = DispatchSemaphore(value: 0), started = DispatchSemaphore(value: 0)
        let blocker = Task { try await thread.run { started.signal(); gate.wait() } }
        await Task.detached { started.wait() }.value
        let request = Task { try await cache.lease(vocabulary: vocabulary, format: .jsonSchema(#"{"type":"object"}"#)) }
        try await Task.sleep(for: .milliseconds(50))
        request.cancel()
        gate.signal()
        _ = try await blocker.value
        do { _ = try await request.value; Issue.record("a cancelled lease should throw") } catch is CancellationError {}
        let idle = await cache.idleConstraints
        #expect(idle == 0, "cancelled before the compile ran: nothing compiled, nothing idle")
    }

    private final class RanFlag: @unchecked Sendable {
        private let lock = NSLock(); private var flag = false
        func set() { lock.lock(); flag = true; lock.unlock() }
        var value: Bool { lock.lock(); defer { lock.unlock() }; return flag }
    }
}

struct ServeRecursiveRefTests {
    private static let tree = ##"{"type":"object","properties":{"name":{"type":"string"},"children":{"type":"array","items":{"$ref":"#"}}},"required":["name","children"]}"##
    private static let defs = ##"{"$defs":{"node":{"type":"object","properties":{"next":{"anyOf":[{"$ref":"#/$defs/node"},{"type":"null"}]}},"required":["next"]}},"$ref":"#/$defs/node"}"##

    private static func accepts(_ schema: String, _ document: String) async throws -> Bool {
        let constraint = try await ServeGrammarCache().lease(vocabulary: try FakeVocab.vocabulary(), format: .jsonSchema(schema)).constraint
        for character in document {
            guard let id = FakeVocab.ids[character], FakeVocab.allowed(try constraint.computeMask(), id) else { return false }
            _ = try constraint.commitToken(Int32(id))
        }
        return FakeVocab.allowed(try constraint.computeMask(), FakeVocab.eos)
    }

    @Test func aSelfReferentialTreePassesTheCheckAndTheWalkTerminates() {
        #expect(ServeSchemaCheck.check(Self.tree) == .ok)
        #expect(ServeSchemaCheck.check(Self.defs) == .ok)
        #expect((ServeSchemaCheck.estimatedCompileMs(Self.tree) ?? .infinity) < Double(ServeSchemaCheck.Bounds.estimatedCompileMs))
    }

    @Test func aSelfReferentialTreeCompilesAndGuidesNestedDocuments() async throws {
        let start = ContinuousClock.now
        #expect(try await Self.accepts(Self.tree, #"{"name":"a","children":[{"name":"b","children":[{"name":"c","children":[]}]}]}"#))
        #expect(ContinuousClock.now - start < .seconds(1))
        #expect(!(try await Self.accepts(Self.tree, #"{"name":"a","children":[{"name":1,"children":[]}]}"#)))
    }

    @Test func aMutuallyRecursiveDefsSchemaCompilesAndGuides() async throws {
        #expect(try await Self.accepts(Self.defs, #"{"next":{"next":{"next":null}}}"#))
        #expect(!(try await Self.accepts(Self.defs, #"{"next":{"nxt":null}}"#)))
    }
}
