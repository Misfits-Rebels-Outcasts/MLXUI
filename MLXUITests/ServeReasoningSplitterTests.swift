import Foundation
import Testing
@testable import MLXUI

// S2-3 — the reasoning split, on raw text, byte-exact.
//
// Two generation-prompt shapes have to work and both are tested here: `qwen3_5*` templates end the
// prompt with `<think>\n` (the output starts *inside* the block), and `qwen3` ends at the assistant
// header (the model emits `<think>` itself). The splitter is not told which — it decides by looking
// — so these tests pin that behaviour rather than a flag.

struct ServeReasoningSplitterTests {

    /// Feed every string to one splitter, collecting both halves.
    ///
    /// `startsInsideThink` is what the prompt says (S2-3b): `true` for a `qwen3_5*` prompt that ends
    /// `<think>\n`, `false` for a `qwen3` prompt that ends at the assistant header. Most of these
    /// tests are about the *body* of the split, so they use a prompt-opened one; the two shapes have
    /// their own tests below.
    private func run(_ chunks: [String], startsInsideThink: Bool = true) -> (reasoning: String, content: String) {
        var splitter = ServeReasoningSplitter(startsInsideThink: startsInsideThink)
        var reasoning = "", content = ""
        for chunk in chunks {
            let halves = splitter.push(chunk)
            reasoning += halves.reasoning
            content += halves.content
        }
        let tail = splitter.finish()
        return (reasoning + tail.reasoning, content + tail.content)
    }

    // MARK: - The two prompt shapes

    /// **Model-emitted** (`qwen3`): the output opens with `<think>`, which is not part of either
    /// half.
    @Test func theModelEmittedShapeSwallowsTheOpeningTag() {
        let result = run(["<think>\nI should add them.\n</think>\n\n", "The answer is 4."],
                         startsInsideThink: false)
        #expect(result.reasoning == "\nI should add them.\n")
        #expect(result.content == "The answer is 4.")
    }

    /// **The prompt decides** (S2-3b, review call 2): with a prompt that did not open a block, an
    /// output with no `<think>` is the whole answer — no reasoning at all.
    @Test func aPromptThatDidNotOpenABlockTreatsTheOutputAsAnswer() {
        let result = run(["The answer is 4."], startsInsideThink: false)
        #expect(result.reasoning.isEmpty)
        #expect(result.content == "The answer is 4.")
    }

    /// …and the same output under a prompt that *did* open one is thinking, because that is where
    /// the model is continuing from.
    @Test func aPromptThatOpenedABlockTreatsTheOutputAsThinking() {
        let result = run(["The answer is 4."], startsInsideThink: true)
        #expect(result.reasoning == "The answer is 4.")
        #expect(result.content.isEmpty)
    }

    /// A prompt that opened the block, and an output that starts at the closing tag: empty thinking
    /// and the answer (S2-3b, review call 2b).
    @Test func aPromptOpenedBlockWhoseOutputStartsAtTheClosingTag() {
        let result = run(["</think>\n\nDirect answer."], startsInsideThink: true)
        #expect(result.reasoning.isEmpty)
        #expect(result.content == "Direct answer.")
    }

    /// **Prompt-opened** (`qwen3_5*`): the prompt already wrote `<think>\n`, so the output starts
    /// inside the block with only `</think>` to find.
    @Test func thePromptOpenedShapeFindsOnlyTheClosingTag() {
        let result = run(["Let me think about it.\n</think>", "\n\nIt is 4."])
        #expect(result.reasoning == "Let me think about it.\n")
        #expect(result.content == "It is 4.")
    }

    /// A template that always writes `<think>` (Qwen3-Thinking-2507 style) can produce an output
    /// that starts *at* `</think>`: empty thinking, then the answer.
    @Test func anOutputThatStartsAtTheClosingTagHasEmptyReasoning() {
        let result = run(["</think>\n\nDirect answer."])
        #expect(result.reasoning.isEmpty)
        #expect(result.content == "Direct answer.")
    }

    /// A partial `<think` at the opening is held, then recognised.
    @Test func aPartialOpeningTagIsHeldUntilItCompletes() {
        var splitter = ServeReasoningSplitter(startsInsideThink: false)
        let held = splitter.push("<thi")
        #expect(held.reasoning.isEmpty && held.content.isEmpty)
        let halves = splitter.push("nk>thinking")
        #expect(halves.reasoning == "thinking")
        #expect(halves.content.isEmpty)
    }

    // MARK: - Boundaries

    /// `</th` + `ink>` — the closing tag split across deltas.
    @Test func aClosingTagSplitAcrossDeltasIsNotLeaked() {
        let result = run(["reasoning here</th", "ink>", "answer"])
        #expect(result.reasoning == "reasoning here")
        #expect(result.content == "answer")
        #expect(!result.reasoning.contains("<"))
        #expect(!result.content.contains("ink"))
    }

    /// Every byte survives, in both halves, when the same text is fed one character at a time.
    @Test func bothHalvesAreByteExactCharacterWise() {
        let reasoning = "Let me check <div><span>hi</span></div> first. "
        let content = "Done: <div><span>ok</span></div>."
        let result = run((reasoning + "</think>\n\n" + content).map(String.init))
        #expect(result.reasoning == reasoning)
        #expect(result.content == content)
    }

    /// An unclosed block at end of stream is **all** reasoning and no content — a `max_tokens` cut
    /// mid-think (S2-3 decision 7).
    @Test func anUnclosedBlockAtEndOfStreamIsAllReasoning() {
        let result = run(["I am still thinking about", " the problem and"])
        #expect(result.reasoning == "I am still thinking about the problem and")
        #expect(result.content.isEmpty)
    }

    /// The blank line the templates leave after `</think>` never becomes content, however it is
    /// chunked.
    @Test func whitespaceAfterTheClosingTagIsNeverContent() {
        #expect(run(["<think>x</think>\n\nanswer"]).content == "answer")
        #expect(run(["<think>x</think>", "\n\nanswer"]).content == "answer")
        #expect(run(["<think>x</think>", "\n", "\n", "answer"]).content == "answer")
        #expect(run(["<think>x</think>\n\n"]).content.isEmpty)
    }

    /// Only the *first* closing tag ends the thinking: a later one is ordinary text.
    @Test func reasoningEndsAtTheFirstClosingTag() {
        let result = run(["<think>a</think>b</think>c"], startsInsideThink: false)
        #expect(result.reasoning == "a")
        #expect(result.content == "b</think>c")
    }

    /// A turn that is only whitespace is neither half (the templates sometimes leave a stray
    /// newline before the answer).
    @Test func aWhitespaceOnlyTurnIsNeitherHalf() {
        let result = run(["\n"], startsInsideThink: false)
        #expect(result.reasoning.isEmpty && result.content.isEmpty)
    }

    /// What a tool call written *inside* the thinking looks like to the splitter: it is reasoning,
    /// and the answer that follows is the content. (The scanner never sees the reasoning half —
    /// that is `MLXChatBackend`'s wiring, pinned in `ServeReasoningTests`.)
    @Test func aToolCallInsideTheThinkingIsReasoning() {
        let result = run(["<think>\nI could call <tool_call>{\"name\": \"f\"}</tool_call> here.\n</think>\n\n",
                          "The answer is 4."], startsInsideThink: false)
        #expect(result.reasoning.contains("<tool_call>"))
        #expect(result.content == "The answer is 4.")
    }

    /// S2-3b, review call 5: only the newline(s) after `</think>` are dropped. An answer that starts
    /// with indentation keeps it.
    @Test func indentationAfterTheClosingTagSurvives() {
        #expect(run(["<think>x</think>\n\n  indented"], startsInsideThink: false).content == "  indented")
        #expect(run(["<think>x</think>\n\r\n  indented"], startsInsideThink: false).content == "  indented")
        #expect(run(["<think>x</think>\n\n\ttabbed"], startsInsideThink: false).content == "\ttabbed")
        // A space with no newline is content too, in the prompt-opened shape as well.
        #expect(run(["</think> kept"], startsInsideThink: true).content == " kept")
    }
}
