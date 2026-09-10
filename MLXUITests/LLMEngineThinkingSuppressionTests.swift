import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

/// DA-11 (`RSI/DelegateDeciderBacklog.md`) — thinking suppression for the flow-row LLM
/// primitives, ported from `catflow-mlx` `engines/llm.py:324-354` (`_render_chat_prompt`'s
/// `enable_thinking=False` + `_strip_think_block`). A reasoning model (Qwen3 8B) spends a
/// 32-token yes/no gate inside `<think>` and never emits a tag → F004 on every one of the
/// 11 gallery rows that name it on a tag-firing task.
struct LLMEngineThinkingSuppressionTests {

    // MARK: - _strip_think_block (hermetic — always runs)

    @Test func stripsOneLeadingClosedThinkBlock() {
        #expect(LLMEngine.stripThinkBlock("<think>\nreasoning here\n</think>\n\nyes") == "yes")
        #expect(LLMEngine.stripThinkBlock("  <think>x</think>  no") == "no")
    }

    @Test func closedThinkBlockSpansNewlines() {
        // DOTALL: the whole multi-line trace is one block.
        let leaked = "<think>\nline one\nline two\nline three\n</think>\nCafé Rey"
        #expect(LLMEngine.stripThinkBlock(leaked) == "Café Rey")
    }

    @Test func leavesAnUnclosedThinkBlockAlone() {
        // The reference's explicit caveat: an unclosed block (ran past max_tokens) is left
        // as-is rather than guessed at. With maxTokens: 32 on the gate this is the common case.
        let unclosed = "<think>\nOkay, I need to work out whether there is another record in the"
        #expect(LLMEngine.stripThinkBlock(unclosed) == unclosed)
    }

    @Test func leavesAThinkTagThatIsNotAtTheStartAlone() {
        #expect(LLMEngine.stripThinkBlock("yes <think>after</think>") == "yes <think>after</think>")
    }

    @Test func noThinkBlockIsUntouched() {
        #expect(LLMEngine.stripThinkBlock("Dashboard,High") == "Dashboard,High")
        #expect(LLMEngine.stripThinkBlock("") == "")
    }

    /// The correctness point, not just F004: a *leaked but closed* trace that mentions the
    /// wrong tag ("yes") before the real answer ("no") would make `extractTag` fire the wrong
    /// edge. Stripping the block first is what keeps the tag honest.
    @Test func strippingKeepsTheFiredTagHonest() {
        let leaked = "<think>\nHmm, is there another? Yes, wait — actually no.\n</think>\nno"
        #expect(DeciderFrame.extractTag(from: leaked, tags: ["yes", "no"]) == "yes")   // the bug
        #expect(DeciderFrame.extractTag(from: LLMEngine.stripThinkBlock(leaked),
                                        tags: ["yes", "no"]) == "no")                  // the fix
    }

    // MARK: - the rendered prompt (opt-in: needs a Qwen3-family tokenizer installed)

    /// Candidate install locations for `mlx-community--Qwen3-8B-4bit` across both editions'
    /// sandbox containers and the un-sandboxed path.
    private static func installedQwen3Dir() -> URL? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let rel = "Application Support/AI Browser/models/mlx-community--Qwen3-8B-4bit"
        return [
            home.appending(path: "Library/Containers/net.connectcode.mlxui/Data/Library/\(rel)"),
            home.appending(path: "Library/Containers/ConnectCode.PipelineStudio/Data/Library/\(rel)"),
            home.appending(path: "Library/\(rel)"),
        ].first { FileManager.default.fileExists(atPath: $0.appending(path: "tokenizer.json").path) }
    }

    /// This one loads a real `tokenizer.json` + compiles the Jinja template (no model weights).
    /// It runs whenever `mlx-community--Qwen3-8B-4bit` is installed and self-skips otherwise —
    /// the whole point of DA-11 is that this render behaves a specific way, so it earns the
    /// couple of seconds when the model is present.
    @Test
    func qwen3TemplateHonoursEnableThinkingFalse() async throws {
        guard let dir = Self.installedQwen3Dir() else { return }   // not installed here
        let tokenizer = try await HFTokenizerLoader().load(from: dir)
        let messages: [[String: any Sendable]] = [["role": "user", "content": "Is there another record? Answer yes or no."]]

        let withFlag = tokenizer.decode(
            tokenIds: try tokenizer.applyChatTemplate(
                messages: messages, tools: nil, additionalContext: LLMEngine.noThinkingContext),
            skipSpecialTokens: false)
        let plain = tokenizer.decode(
            tokenIds: try tokenizer.applyChatTemplate(
                messages: messages, tools: nil, additionalContext: nil),
            skipSpecialTokens: false)

        // Qwen3's template, with `enable_thinking is false`, pre-closes the think block so the
        // model answers directly; without the flag it leaves the assistant turn open to think.
        #expect(withFlag.contains("</think>"),
                "enable_thinking:false should pre-close <think>; rendered:\n\(withFlag)")
        #expect(!plain.contains("</think>"),
                "the default render must leave thinking open; rendered:\n\(plain)")
        #expect(withFlag != plain)
    }
}
