import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

/// OCP-FIX-1-1 (`RSI/DelegateOCRPromptBacklog.md` §2) — diagnosis of the malformed markup in
/// GLM-OCR output that the spike (`RSI/journal/2026-228`) recorded: every `>` sitting
/// **immediately before an opening tag** is dropped (`</td><td>` → `</td<td>`), while a `>`
/// before a closing tag (`</td></tr>`) survives.
///
/// **Verdict: hypothesis 3 — the `>` is emitted and detokenized correctly, then lost
/// downstream.** Not the model, not the BPE vocab. The loss is in mlx-swift-lm's
/// `MLXLMCommon/Tool/ToolCallProcessor.swift` (`processTaggedChunk`): it splits a streamed
/// chunk at the first `<`, and when the `<…` tail is a *partial* prefix of the tool-call
/// start tag (`<tool_call>` — so `<`, `<t`, `<to`, …) it buffers that tail and does
/// `return nil`, silently discarding `leadingToken` (everything before the `<`, which
/// includes the preceding `>`). `</…` breaks the partial match at the second character, so it
/// is flushed whole and never loses its `>` — exactly the selectivity the spike saw.
///
/// This is confirmed at the **vocabulary level**, no model run required: GLM-OCR's
/// `tokenizer.json` has an atomic token `'><'` (id 2582) — and `'"><'` (4279) — that the
/// model emits between two adjacent HTML tags. Detokenized, that token is the chunk `"><"`;
/// fed to `ToolCallProcessor` the `>` half is dropped. The "safe" variants `'></'` (2631) /
/// `'"></'` (3741) survive because `</` fails the partial match. `'<tool_call>'` is itself an
/// atomic token (59269), so the tag can never actually arrive in fragments for this model and
/// the partial-match buffering serves no purpose here.
///
/// Every OCR/VLM/LLM run in the app goes through this path (`VLMEngine.generate` →
/// `MLXLMCommon.generate` → `TextToolTokenLoopHandler`), with `format` =
/// `context.configuration.toolCallFormat ?? .json`. Both `.json` and `.glm4` use the
/// `<tool_call>` start tag, so the defect is not GLM-specific — it is merely invisible for
/// models whose output is plain text / markdown (olmOCR-2, dots.ocr) and pervasive for
/// GLM-OCR because it emits HTML tables (`<table>`, `<thead>`, `<tr>`, `<td>` — adjacent
/// `>`+`<` everywhere).
///
/// These tests reproduce the mechanism with **no model weights**. They assert the *current
/// buggy* behavior on purpose; OCP-FIX-1-2 fixes at the layer the evidence names and replaces
/// them with a golden on the corrected output.
@Suite struct GLMOCRMarkupDropDiagnosisTests {

    /// Feed a `ToolCallProcessor` the way `TextToolTokenLoopHandler.onToken` does — one
    /// detokenized chunk at a time — and accumulate the text it yields for display.
    private func streamThrough(_ processor: ToolCallProcessor, chunks: [String]) -> String {
        var out = ""
        for chunk in chunks {
            if let text = processor.processChunk(chunk) { out += text }
        }
        processor.processEOS()
        return out
    }

    /// The exact real-world trigger: GLM-OCR emits the atomic token `'><'` (id 2582) between
    /// `…</td` and `td>…` — that token carries the `>` that closes one cell and the `<` that
    /// opens the next. Detokenized it is the chunk `"><"`, and its `>` is dropped.
    @Test func realGLMTokenGreaterThenLessDropsTheGreater_glm4() {
        let got = streamThrough(ToolCallProcessor(format: .glm4),
                                chunks: ["</td", "><", "td>$168.00"])
        #expect(got == "</td<td>$168.00")          // buggy: the `>` from the `"><"` chunk is gone
        #expect(got != "</td><td>$168.00")         // fails once OCP-FIX-1-2 lands — swap for the golden
    }

    /// `.json` is the default (`toolCallFormat ?? .json`) and uses the same `<tool_call>` tag,
    /// so a model that never declares a format is affected identically.
    @Test func realGLMTokenGreaterThenLessDropsTheGreater_jsonDefault() {
        let got = streamThrough(ToolCallProcessor(format: .json),
                                chunks: ["</thead", "><", "tbody>"])
        #expect(got == "</thead<tbody>")
    }

    /// The `'"><'` token (id 4279) — same story, two characters lost (`">`).
    @Test func realGLMTokenQuoteGreaterLessDropsQuoteGreater() {
        let got = streamThrough(ToolCallProcessor(format: .glm4),
                                chunks: ["<table class=\"table table-bordered", "\"><", "thead>"])
        #expect(got == "<table class=\"table table-bordered<thead>")
    }

    /// The selectivity the spike observed: the `'></'` token (id 2631) survives because `</`
    /// fails the partial match at char 2, so the chunk is flushed whole.
    @Test func closingTagBoundaryKeepsItsGreater() {
        let got = streamThrough(ToolCallProcessor(format: .glm4),
                                chunks: ["<td>a", "></", "tr>"])
        #expect(got == "<td>a></tr>")
    }

    /// A chunk split mid-tag-name (`…"><t` | `head>`) loses the leadingToken too — it is not
    /// only the atomic `'><'` token; any chunk whose first `<` is followed by `t`/`o`/… does it.
    @Test func chunkSplitInsideOpeningTagNameAlsoDrops() {
        let got = streamThrough(ToolCallProcessor(format: .glm4),
                                chunks: ["\"><t", "head><tr>"])
        #expect(got == "<thead><tr>")
        #expect(got != "\"><thead><tr>")
    }

    /// Control: the whole run of adjacent tags in a single chunk (no boundary at `<t…`) is
    /// untouched — the bug needs the chunk boundary.
    @Test func adjacentTagsInOneChunkAreUntouched() {
        let got = streamThrough(ToolCallProcessor(format: .glm4),
                                chunks: ["\"><thead><tr><td>Category</td></tr></thead>"])
        #expect(got == "\"><thead><tr><td>Category</td></tr></thead>")
    }
}
