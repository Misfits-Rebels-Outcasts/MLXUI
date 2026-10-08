import Testing
import Foundation
import MLX
import MLXLMCommon
import Tokenizers
@testable import MLXUI

// PC-2 — real-model equivalence (ruling B1, lengthened): cache-on against cache-off at
// `temperature: 0`, answers of ≥ 256 tokens, through the real `MLXChatBackend`.
//
// Enabled only when `MLXUI_TEST_PC_DIR` names an installed checkpoint (pass it as
// `TEST_RUNNER_MLXUI_TEST_PC_DIR=<folder>`); run one family per invocation — the owner's 19 GB Mac
// can't hold two (F17). The sandboxed test host can't read arbitrary folders, so the gated runs
// override the sandbox on the command line (named in the journal). A skipped test doesn't count.
//
// The cache is switched with `PromptCache.overrideForTesting`; `AppState.disablePromptCache` stays
// `true` in the tree.
@Suite(.serialized)
struct PromptCacheRealModelTests {
    static let modelDir = ProcessInfo.processInfo.environment["MLXUI_TEST_PC_DIR"]
    static let enabled = modelDir != nil

    // MARK: harness

    struct Result {
        var text = ""
        var reasoning = ""
        var calls: [ServeToolCall] = []
        var usage: ServeUsage?
        var outcome: PromptCacheOutcome?
    }

    private final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private var value: PromptCacheOutcome?
        func set(_ v: PromptCacheOutcome) { lock.lock(); value = v; lock.unlock() }
        func get() -> PromptCacheOutcome? { lock.lock(); defer { lock.unlock() }; return value }
    }

    private static func run(_ request: ServeRequest, backend: MLXChatBackend, cache: Bool) async throws -> Result {
        PromptCache.overrideForTesting = cache
        defer { PromptCache.overrideForTesting = nil }
        let box = Box()
        var backend = backend
        backend.promptCacheObserver = { box.set($0) }
        var result = Result()
        for try await event in backend.generate(request) {
            switch event {
            case .textDelta(let piece): result.text += piece
            case .reasoningDelta(let piece): result.reasoning += piece
            case .toolCallDelta(let delta): ChatCompletionsRoute.append(delta, to: &result.calls)
            case .usage(let usage): result.usage = usage
            case .error(let error): Issue.record("stream error: \(error)")
            default: break
            }
        }
        result.outcome = box.get()
        return result
    }

    private static func facts(_ lines: Int) -> String {
        (0..<lines).map { i in
            "Fact \(i): the harbour of town \(i * 7 % 101) opens at \(6 + i % 5) and closes at \(17 + i % 4) on day \(i % 7) of every week, and the ferry master is called Mr \(["Alder", "Birch", "Cedar", "Dunn", "Elm"][i % 5])\(i)."
        }.joined(separator: "\n")
    }

    private static let factsText = "You are a precise assistant. Use only the facts below.\n" + facts(50)
    /// `gemma2`'s template has no system role; in production S2-1's assembly folds the system text
    /// into the first user turn before the backend sees it. This mirrors that for the one family.
    nonisolated(unsafe) static var foldSystem = false
    private static var system: ServeMessage { ServeMessage(role: .system, text: factsText) }
    /// The conversation's opening: a system message, or (no system role) the facts inside the user turn.
    private static func opening(_ userText: String) -> [ServeMessage] {
        foldSystem ? [ServeMessage(role: .user, text: factsText + "\n\n" + userText)]
                   : [system, ServeMessage(role: .user, text: userText)]
    }
    private static let weatherTool = ServeToolSpec(name: "get_weather", rawJSON: #"{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}"#)

    private static func request(_ messages: [ServeMessage], tools: [ServeToolSpec] = [], thinking: Bool? = nil,
                                maxTokens: Int = 320) -> ServeRequest {
        var request = ServeRequest(model: "m", messages: messages)
        request.temperature = 0
        request.maxTokens = maxTokens
        request.tools = tools
        request.reasoning = thinking
        return request
    }

    /// B1: identical text, or — only where the cache-off run's top-two logit gap at the first
    /// differing step is ≤ 1.0 — a logged near-tie. Anything else is red.
    private static func assertEquivalent(_ off: Result, _ on: Result, label: String, request: ServeRequest,
                                         pool: ModelContainerPool, dir: URL) async throws {
        let offAll = off.reasoning + "\u{1}" + off.text + "\u{1}" + off.calls.map { $0.name + $0.argumentsJSON }.joined()
        let onAll = on.reasoning + "\u{1}" + on.text + "\u{1}" + on.calls.map { $0.name + $0.argumentsJSON }.joined()
        if offAll == onAll { print("PCTEST \(label): identical (\(off.usage?.completionTokens ?? -1) completion tokens)"); return }
        let (step, gap, offTok, onTok) = try await divergence(off: off.reasoning + off.text, on: on.reasoning + on.text,
                                                              request: request, pool: pool, dir: dir)
        print("PCTEST \(label): DIVERGED at token step \(step), cache-off top-two gap \(gap), off=\(offTok) on=\(onTok)")
        #expect(gap <= 1.0, "\(label): diverged with a top-two gap of \(gap) > 1.0 at step \(step)")
    }

    /// The first token position where the two texts differ, and the cache-off logit gap there
    /// (teacher-forced on the cache-off text). Diagnostic only.
    private static func divergence(off: String, on: String, request: ServeRequest, pool: ModelContainerPool,
                                   dir: URL) async throws -> (Int, Float, String, String) {
        let container = try await pool.container(directory: dir)
        let modelType = ModelContainerPool.modelTypeString(directory: dir, architecture: nil)
        let messages = request.messages
        let tools = request.renderedTools.compactMap(\.object)
        let thinking = ReasoningTemplateTable.thinkingOn(requested: request.reasoning, forModelType: modelType)
        return try await container.perform { (context: ModelContext) async throws -> (Int, Float, String, String) in
            let input = try await context.processor.prepare(input: UserInput(
                prompt: .messages(MLXChatBackend.templateMessages(messages)), tools: tools.isEmpty ? nil : tools,
                additionalContext: ["enable_thinking": thinking]))
            let prompt = input.text.tokens.asArray(Int.self)
            let offIDs = context.tokenizer.encode(text: off, addSpecialTokens: false)
            let onIDs = context.tokenizer.encode(text: on, addSpecialTokens: false)
            var d = 0
            while d < offIDs.count, d < onIDs.count, offIDs[d] == onIDs[d] { d += 1 }
            let forced = prompt + Array(offIDs[0..<min(d, offIDs.count)])
            let cache = context.model.newCache(parameters: nil)
            try PromptCheckpointing.prefill(forced, from: 0, to: forced.count - 1, cache: cache, context: context, step: 512)
            let last = MLXArray([forced[forced.count - 1]])[.newAxis, 0...]
            let logits = context.model(LMInput.Text(tokens: last), cache: cache, state: nil).logits[0, -1, 0...].asType(.float32)
            let ranked = logits.asArray(Float.self).sorted(by: >)
            let gap = ranked.count > 1 ? ranked[0] - ranked[1] : Float.infinity
            return (d, gap,
                    d < offIDs.count ? context.tokenizer.decode(tokenIds: [offIDs[d]]) : "<end>",
                    d < onIDs.count ? context.tokenizer.decode(tokenIds: [onIDs[d]]) : "<end>")
        }
    }

    /// `MLXUI_TEST_PC_CANDIDATE=1` runs a family that is **not yet allowlisted** through the cache
    /// (its type is added to the table for this process only) — that is how a family earns its place.
    private static func make() throws -> (URL, ModelContainerPool, MLXChatBackend) {
        let dir = URL(fileURLWithPath: try #require(modelDir))
        let pool = ModelContainerPool(budgetBytes: 40 << 30)
        let backend = MLXChatBackend(directory: dir, pool: pool)
        foldSystem = backend.checkpointModelType == "gemma2"
        if ProcessInfo.processInfo.environment["MLXUI_TEST_PC_CANDIDATE"] == "1" {
            PromptCacheAllowlist.overrideForTesting = PromptCacheAllowlist.tested.union([backend.checkpointModelType])
        }
        return (dir, pool, backend)
    }

    private static func assertLongEnough(_ r: Result, _ label: String) {
        #expect((r.usage?.completionTokens ?? 0) >= 256, "\(label): only \(r.usage?.completionTokens ?? 0) completion tokens — B1 needs ≥ 256")
    }

    private static let askForList = "Using only the facts above, write a numbered list of at least 40 items, one sentence each, about the harbours and their ferry masters."

    // MARK: the cases

    /// A two-turn conversation: the first turn stores S and T; the second reuses ≥ T₁; both equal cache-off.
    @Test(.enabled(if: PromptCacheRealModelTests.enabled))
    func aTwoTurnConversationReusesTheFirstTurnAndMatchesCacheOff() async throws {
        let (dir, pool, backend) = try Self.make()
        await PromptCacheShared.store.clear()
        print("PCTEST model_type=\(backend.checkpointModelType) allowlisted=\(PromptCacheAllowlist.allows(backend.checkpointModelType))")
        let turn1 = Self.request(Self.opening(Self.askForList))
        let on1 = try await Self.run(turn1, backend: backend, cache: true)
        #expect(on1.outcome?.bypass == nil)
        #expect(on1.outcome?.reusedTokens == 0)                                  // nothing stored yet
        #expect(!(on1.outcome?.storedAt.isEmpty ?? true))                        // …but it checkpointed
        print("PCTEST turn1 stored checkpoints \(on1.outcome?.storedAt ?? []) of \(on1.outcome?.promptTokens ?? 0) prompt tokens")
        let off1 = try await Self.run(turn1, backend: backend, cache: false)
        #expect(off1.outcome?.bypass == .disabled)
        Self.assertLongEnough(on1, "turn 1")
        try await Self.assertEquivalent(off1, on1, label: "turn 1 (miss, checkpointed prefill)", request: turn1, pool: pool, dir: dir)

        let turn2 = Self.request(turn1.messages + [ServeMessage(role: .assistant, text: on1.text),
                                                  ServeMessage(role: .user, text: "Now list 30 more items, continuing the numbering.")])
        let on2 = try await Self.run(turn2, backend: backend, cache: true)
        let t1 = on1.outcome?.storedAt.max() ?? .max
        print("PCTEST turn2 reused \(on2.outcome?.reusedTokens ?? -1) of \(on2.outcome?.promptTokens ?? -1); T1=\(t1)")
        #expect((on2.outcome?.reusedTokens ?? 0) >= t1)
        let off2 = try await Self.run(turn2, backend: backend, cache: false)
        Self.assertLongEnough(on2, "turn 2")
        try await Self.assertEquivalent(off2, on2, label: "turn 2 (hit)", request: turn2, pool: pool, dir: dir)
        #expect(on2.usage?.promptTokens == off2.usage?.promptTokens)             // usage still counts the whole prompt
    }

    /// The raw `.messages` render path: an assistant `tool_calls` turn and a `tool` result in the history.
    @Test(.enabled(if: PromptCacheRealModelTests.enabled))
    func aToolCallHistoryConversationMatchesCacheOff() async throws {
        let (dir, pool, backend) = try Self.make()
        guard backend.acceptsTools else { print("PCTEST tool-history: family does not accept tools — case not applicable"); return }
        await PromptCacheShared.store.clear()
        let call = ServeToolCall(id: "call_1", name: "get_weather", argumentsJSON: #"{"location":"Paris"}"#)
        let ask = "What is the weather in Paris? Use the get_weather tool, then write a detailed numbered packing list of at least 40 items for that weather, one full sentence each."
        let turnA = Self.request(Self.opening(ask) + [
            ServeMessage(role: .assistant, text: "", toolCalls: [call]),
            ServeMessage(role: .tool, text: "18 degrees and sunny", toolCallID: "call_1", toolName: "get_weather")],
            tools: [Self.weatherTool])
        #expect(Self.runsRawPath(turnA, backend))
        let onA = try await Self.run(turnA, backend: backend, cache: true)
        #expect(onA.outcome?.bypass == nil)
        print("PCTEST tool-history turn A stored \(onA.outcome?.storedAt ?? []) of \(onA.outcome?.promptTokens ?? 0)")
        #expect(!(onA.outcome?.storedAt.isEmpty ?? true))
        let offA = try await Self.run(turnA, backend: backend, cache: false)
        Self.assertLongEnough(onA, "tool-history A")
        try await Self.assertEquivalent(offA, onA, label: "tool-history A", request: turnA, pool: pool, dir: dir)

        let turnB = Self.request(turnA.messages + [ServeMessage(role: .assistant, text: onA.text),
                                                  ServeMessage(role: .user, text: "Continue the packing list with at least 40 more items, numbered from where you stopped, two full sentences each.")],
                                 tools: [Self.weatherTool], maxTokens: 400)
        let onB = try await Self.run(turnB, backend: backend, cache: true)
        let tA = onA.outcome?.storedAt.max() ?? .max
        print("PCTEST tool-history turn B reused \(onB.outcome?.reusedTokens ?? -1); T_A=\(tA)")
        #expect((onB.outcome?.reusedTokens ?? 0) >= tA)
        let offB = try await Self.run(turnB, backend: backend, cache: false)
        Self.assertLongEnough(onB, "tool-history B")
        try await Self.assertEquivalent(offB, onB, label: "tool-history B (hit)", request: turnB, pool: pool, dir: dir)
    }

    private static func runsRawPath(_ request: ServeRequest, _ backend: MLXChatBackend) -> Bool {
        MLXChatBackend.hasToolHistory(request.messages)
    }

    /// Thinking on (`qwen3`, `qwen3_5`): the reasoning and the answer both match cache-off.
    @Test(.enabled(if: PromptCacheRealModelTests.enabled))
    func aThinkingOnConversationMatchesCacheOff() async throws {
        let (dir, pool, backend) = try Self.make()
        guard ["qwen3", "qwen3_5"].contains(backend.checkpointModelType) else {
            print("PCTEST thinking-on: not a thinking family (\(backend.checkpointModelType)) — case not applicable"); return
        }
        await PromptCacheShared.store.clear()
        let turn1 = Self.request(Self.opening(Self.askForList), thinking: true, maxTokens: 700)
        let on1 = try await Self.run(turn1, backend: backend, cache: true)
        #expect(on1.outcome?.bypass == nil)
        let off1 = try await Self.run(turn1, backend: backend, cache: false)
        #expect(!on1.reasoning.isEmpty, "thinking was requested but no reasoning came back")
        Self.assertLongEnough(on1, "thinking turn 1")
        try await Self.assertEquivalent(off1, on1, label: "thinking turn 1", request: turn1, pool: pool, dir: dir)

        let turn2 = Self.request(turn1.messages + [ServeMessage(role: .assistant, text: on1.text),
                                                  ServeMessage(role: .user, text: "Now list 30 more items, continuing the numbering.")],
                                 thinking: true, maxTokens: 700)
        let on2 = try await Self.run(turn2, backend: backend, cache: true)
        print("PCTEST thinking turn 2 reused \(on2.outcome?.reusedTokens ?? -1) of \(on2.outcome?.promptTokens ?? -1)")
        #expect((on2.outcome?.reusedTokens ?? 0) >= (on1.outcome?.storedAt.min() ?? .max))
        let off2 = try await Self.run(turn2, backend: backend, cache: false)
        try await Self.assertEquivalent(off2, on2, label: "thinking turn 2 (hit)", request: turn2, pool: pool, dir: dir)
    }

    /// A model off the allowlist never touches the store: no lookup, no insert, no new snapshots.
    @Test(.enabled(if: PromptCacheRealModelTests.enabled))
    func aNonAllowlistedModelNeverTouchesTheStore() async throws {
        let (_, _, backend) = try Self.make()
        await PromptCacheShared.store.clear()
        PromptCacheAllowlist.overrideForTesting = ["some_other_type"]          // this checkpoint is now "untested"
        defer { PromptCacheAllowlist.overrideForTesting = nil }
        let turn = Self.request(Self.opening("Say hello in one short sentence."), maxTokens: 24)
        let first = try await Self.run(turn, backend: backend, cache: true)
        let second = try await Self.run(turn, backend: backend, cache: true)
        #expect(first.outcome?.bypass == .modelNotTested)
        #expect(second.outcome?.bypass == .modelNotTested)
        #expect(second.outcome?.reusedTokens == 0)
        #expect(await PromptCacheShared.store.count == 0)
        #expect(await PromptCacheShared.store.totalBytes == 0)
        #expect(first.text == second.text)
    }

    /// The seed bypass and a cancelled request that stores nothing from its unfinished segment.
    @Test(.enabled(if: PromptCacheRealModelTests.enabled))
    func aSeededRequestBypassesAndACancelledPrefillLeavesTheStoreAsItWas() async throws {
        let (dir, pool, backend) = try Self.make()
        await PromptCacheShared.store.clear()
        var seeded = Self.request(Self.opening("Say hello in one short sentence."), maxTokens: 24)
        seeded.seed = 7
        let r = try await Self.run(seeded, backend: backend, cache: true)
        #expect(r.outcome?.bypass == .seeded)
        #expect(await PromptCacheShared.store.count == 0)

        // Cancel while the first segment is still prefilling: the weights are already loaded (below),
        // and a ~2K-token prefill takes longer than the 300 ms we wait on every family we test.
        _ = try await pool.container(directory: dir)
        PromptCache.overrideForTesting = true
        defer { PromptCache.overrideForTesting = nil }
        let long = Self.request(Self.opening(Self.askForList))
        let task = Task { () -> Int in
            var n = 0
            do { for try await _ in backend.generate(long) { n += 1 } } catch {}
            return n
        }
        try await Task.sleep(for: .milliseconds(300))
        task.cancel()
        let events = await task.value
        let stored = await PromptCacheShared.store.count
        print("PCTEST cancelled mid-prefill: \(events) events, store holds \(stored) snapshots")
        #expect(events == 0)                                                    // nothing was generated
        #expect(stored == 0)                                                    // nothing from the unfinished segment
    }
}
