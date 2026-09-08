import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

/// TCP-1 (`RSI/backlog.md`) — plain chat replies were silently losing `>` before `<`: any
/// reply containing HTML / XML / SVG / JSX. Cause is the defect diagnosed in `2026-230` —
/// mlx-swift-lm's `ToolCallProcessor` drops the text before a `<` that partially matches
/// `<tool_call>` — and `ChatSession`'s generate path runs that scanner **unconditionally**,
/// even for a turn with no tools.
///
/// **Fix:** `AgentSession` routes a turn with **no tools armed** through a raw token stream
/// (`MLXLMCommon.generateTokens` + `NaiveStreamingDetokenizer`, no `ToolCallProcessor` in the
/// path — `rawStreamResponse`). A tool-armed turn keeps `ChatSession`, because tool-call
/// parsing genuinely needs the scanner. `LLMEngine` (flow rows, never tool-calling) takes the
/// full-decode path from OCP-FIX-1-2's playbook.
///
/// What this file pins: the **routing rule**. The two facts it rests on are already goldened
/// elsewhere on real model vocab, so they are not re-tested here:
/// - `NaiveStreamingDetokenizer` / `decode(tokenIds:)` are lossless on `><`-dense markup —
///   `OCRDetokenizationIsLossless` (GLM-OCR's real tokenizer: 17/17 boundaries kept).
/// - `ToolCallProcessor` is what drops them — `ToolCallProcessorUpstreamBehavior` (`.json`
///   and `.glm4`, the exact `"><"` chunk).
@Suite struct TCP1ChatMarkupTests {

    nonisolated struct StubTool: AgentTool {
        let name = "stub"
        let toolDescription = "no-op"
        let parameters: [ToolParameter] = []
        func execute(arguments: [String: JSONValue]) async throws -> String { "" }
    }

    @Test func rawPathChosenExactlyWhenNoToolsArmed() {
        // No tools → raw token stream, no ToolCallProcessor → markup survives.
        #expect(AgentSession.usesRawTextPath(ToolRegistry([])) == true)
        // Tools armed → ChatSession (needs the scanner to parse tool calls).
        #expect(AgentSession.usesRawTextPath(ToolRegistry([StubTool()])) == false)
    }
}
