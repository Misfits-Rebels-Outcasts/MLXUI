import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXUI

/// MM-S addendum — a **diagnostic, not a gate**: what MiMo V2.6 9B actually produced in the two
/// `PromptCacheRealModelTests` cases whose *content* assertions failed on 2026-10-19
/// (`aThinkingOnConversationMatchesCacheOff`: no reasoning came back; `markupSurvivesACachedHit`: no
/// complete tag in the reply), with the cache-on/cache-off comparison already fine in both.
///
/// The requests are the same as those tests' (same facts block, same prompt text, same `maxTokens`),
/// run **cache off** once each through the real `MLXChatBackend`, and every observation is a `print`:
/// the reply as routed (`text` / `reasoning`, as `debugDescription`), whether either holds a literal
/// `<think>` / `</think>`, the decoded prompt tail and the `startsInsideThink` the backend derives
/// from it. The only assertions are "the stream did not error".
///
/// Owner-run, gated on `MLXUI_TEST_MIMO_DIR` (like `MiMoRealLoadTests`). Nothing here is product code.
struct MiMoCacheDiagnosticTests {
    static let dir = ProcessInfo.processInfo.environment["MLXUI_TEST_MIMO_DIR"]

    // The facts block of `PromptCacheRealModelTests` (private there), reproduced byte for byte.
    private static func facts(_ lines: Int) -> String {
        (0..<lines).map { i in
            "Fact \(i): the harbour of town \(i * 7 % 101) opens at \(6 + i % 5) and closes at \(17 + i % 4) on day \(i % 7) of every week, and the ferry master is called Mr \(["Alder", "Birch", "Cedar", "Dunn", "Elm"][i % 5])\(i)."
        }.joined(separator: "\n")
    }
    private static let factsText = "You are a precise assistant. Use only the facts below.\n" + facts(50)
    private static let askForList = "Using only the facts above, write a numbered list of at least 40 items, one sentence each, about the harbours and their ferry masters."

    private struct Observed {
        var text = ""
        var reasoning = ""
        var completionTokens: Int?
        var errors: [String] = []
    }

    private static func run(_ messages: [ServeMessage], thinking: Bool?, maxTokens: Int,
                            backend: MLXChatBackend) async throws -> Observed {
        PromptCache.overrideForTesting = false          // the cache verdict is settled; isolate the content
        defer { PromptCache.overrideForTesting = nil }
        var request = ServeRequest(model: "m", messages: messages)
        request.temperature = 0
        request.maxTokens = maxTokens
        request.reasoning = thinking
        var seen = Observed()
        for try await event in backend.generate(request) {
            switch event {
            case .textDelta(let piece): seen.text += piece
            case .reasoningDelta(let piece): seen.reasoning += piece
            case .usage(let usage): seen.completionTokens = usage.completionTokens
            case .error(let error): seen.errors.append("\(error)")
            default: break
            }
        }
        return seen
    }

    /// The rendered prompt's decoded tail and the mode the backend derives from it, using the same
    /// template context the backend sends (`enable_thinking: <thinkingOn>`).
    private static func promptTail(_ messages: [ServeMessage], thinking: Bool?, dir: URL, pool: ModelContainerPool)
        async throws -> (tail: String, startsInsideThink: Bool, enableThinking: Bool) {
        let container = try await pool.container(directory: dir)
        let modelType = ModelContainerPool.modelTypeString(directory: dir, architecture: nil)
        let enable = ReasoningTemplateTable.thinkingOn(requested: thinking, forModelType: modelType)
        return try await container.perform { (context: ModelContext) async throws -> (String, Bool, Bool) in
            let input = try await context.processor.prepare(input: UserInput(
                prompt: .messages(MLXChatBackend.templateMessages(messages)), tools: nil,
                additionalContext: ["enable_thinking": enable]))
            let ids = input.text.tokens.asArray(Int.self)
            let tail = context.tokenizer.decode(tokenIds: Array(ids.suffix(16)))
            return (tail, MLXChatBackend.promptOpensThinkBlock(decodedTail: tail), enable)
        }
    }

    private static func report(_ label: String, _ seen: Observed, tail: (tail: String, startsInsideThink: Bool, enableThinking: Bool)) {
        func head(_ s: String, _ n: Int = 300) -> String { String(s.prefix(n)).debugDescription }
        func end(_ s: String, _ n: Int = 200) -> String { String(s.suffix(n)).debugDescription }
        print("MMDIAG[\(label)] enable_thinking=\(tail.enableThinking) startsInsideThink=\(tail.startsInsideThink) promptTail=\(tail.tail.debugDescription)")
        print("MMDIAG[\(label)] completionTokens=\(seen.completionTokens.map(String.init) ?? "nil") errors=\(seen.errors)")
        print("MMDIAG[\(label)] reasoning.count=\(seen.reasoning.count) text.count=\(seen.text.count)")
        print("MMDIAG[\(label)] text has `<think>`=\(seen.text.contains("<think>")) `</think>`=\(seen.text.contains("</think>")); reasoning has `<think>`=\(seen.reasoning.contains("<think>")) `</think>`=\(seen.reasoning.contains("</think>"))")
        print("MMDIAG[\(label)] reasoning.head=\(head(seen.reasoning)) reasoning.tail=\(end(seen.reasoning))")
        print("MMDIAG[\(label)] text.head=\(head(seen.text)) text.tail=\(end(seen.text))")
        print("MMDIAG[\(label)] completeTags(text)=\(PromptCacheMarkup.completeTags(in: seen.text)) completeTags(reasoning)=\(PromptCacheMarkup.completeTags(in: seen.reasoning))")
    }

    /// `aThinkingOnConversationMatchesCacheOff`'s first request: facts system message, the list prompt,
    /// thinking requested, `maxTokens: 700`.
    @Test(.enabled(if: dir != nil)) func thinkingOnFirstTurn() async throws {
        let url = URL(fileURLWithPath: try #require(Self.dir))
        let pool = ModelContainerPool(budgetBytes: 40 << 30)
        let backend = MLXChatBackend(directory: url, pool: pool)
        let messages = [ServeMessage(role: .system, text: Self.factsText), ServeMessage(role: .user, text: Self.askForList)]
        let tail = try await Self.promptTail(messages, thinking: true, dir: url, pool: pool)
        let seen = try await Self.run(messages, thinking: true, maxTokens: 700, backend: backend)
        Self.report("thinking-on", seen, tail: tail)
        #expect(seen.errors.isEmpty)
    }

    /// `markupSurvivesACachedHit`'s request: facts system message, "Reply with exactly this text…",
    /// thinking not requested (so `enable_thinking: false` and a pre-closed `<think></think>`),
    /// `maxTokens: 40`. A second run asks with thinking requested, to see the other prompt shape.
    @Test(.enabled(if: dir != nil)) func markupReply() async throws {
        let url = URL(fileURLWithPath: try #require(Self.dir))
        let pool = ModelContainerPool(budgetBytes: 40 << 30)
        let backend = MLXChatBackend(directory: url, pool: pool)
        let ask = "Reply with exactly this text and nothing else: <div><span>hi</span></div>"
        let messages = [ServeMessage(role: .system, text: Self.factsText), ServeMessage(role: .user, text: ask)]
        for (label, thinking) in [("markup/thinking-off", nil as Bool?), ("markup/thinking-on", true as Bool?)] {
            let tail = try await Self.promptTail(messages, thinking: thinking, dir: url, pool: pool)
            let seen = try await Self.run(messages, thinking: thinking, maxTokens: label.hasSuffix("on") ? 400 : 40, backend: backend)
            Self.report(label, seen, tail: tail)
            #expect(seen.errors.isEmpty)
        }
    }
}
