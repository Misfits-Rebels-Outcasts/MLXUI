import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

/// OCP-FIX-1 (`RSI/DelegateOCRPromptBacklog.md` §2, journals `2026-230` diagnosis / `2026-231`
/// fix) — the malformed markup in GLM-OCR output the spike recorded (`</td><td>` arriving as
/// `</td<td>`; the `>` before an **opening** tag dropped, before a **closing** tag kept).
///
/// **Cause (hypothesis 3): lost downstream.** The `>` is generated and detokenized correctly,
/// then discarded in mlx-swift-lm's `MLXLMCommon/Tool/ToolCallProcessor.swift`
/// (`processTaggedChunk`): it splits a streamed chunk at the first `<`, and when the `<…` tail
/// is a *partial* prefix of the tool-call start tag (`<tool_call>` — `<`, `<t`, `<to`, …) it
/// buffers that tail and `return nil`, silently dropping `leadingToken` (everything before the
/// `<`). `</…` diverges at char 2, so a closing tag is flushed whole and keeps its `>`.
///
/// **Fix (OCP-FIX-1-2, option 2): the OCR engines bypass that scanner.** `VLMEngine`,
/// `DotsOCREngine` and `DeepSeekOCREngine` now consume `MLXLMCommon.generateTokens` (raw token
/// ids) and detokenize the full stream once, instead of accumulating `.chunk` text that has
/// already been through `ToolCallProcessor`. OCR/VQA never emit tool calls, so nothing is
/// lost; the agent chat path (`ModelRunner`) still uses the tool-call path deliberately and is
/// still exposed — an upstream report is the real fix (journal `2026-230`).
///
/// - `ToolCallProcessorUpstreamBehavior` — pins the upstream bug the engines route around
///   (hermetic, no weights). If these start failing, mlx-swift-lm changed `ToolCallProcessor`
///   and the bypass may no longer be needed.
/// - `OCRDetokenizationIsLossless` — the golden for the fix: on GLM-OCR's real vocab, the new
///   engine path (`decode(tokenIds:)` over the whole stream) round-trips every `>`, while the
///   old path (`NaiveStreamingDetokenizer` + `ToolCallProcessor`) drops them.
@Suite struct ToolCallProcessorUpstreamBehavior {

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

    /// The real-world trigger: GLM-OCR's tokenizer has an atomic token `'><'` (id 2582) emitted
    /// between `…</td` and `td>…`. Detokenized it is the chunk `"><"`, and its `>` is dropped.
    @Test func atomicGreaterLessTokenLosesItsGreater_glm4() {
        let got = streamThrough(ToolCallProcessor(format: .glm4),
                                chunks: ["</td", "><", "td>$168.00"])
        #expect(got == "</td<td>$168.00")
    }

    /// `.json` is the default (`toolCallFormat ?? .json`) and uses the same `<tool_call>` tag.
    @Test func atomicGreaterLessTokenLosesItsGreater_jsonDefault() {
        let got = streamThrough(ToolCallProcessor(format: .json),
                                chunks: ["</thead", "><", "tbody>"])
        #expect(got == "</thead<tbody>")
    }

    /// The `'"><'` token (id 4279) — same story, two characters lost (`">`).
    @Test func atomicQuoteGreaterLessTokenLosesQuoteGreater() {
        let got = streamThrough(ToolCallProcessor(format: .glm4),
                                chunks: ["<table class=\"table table-bordered", "\"><", "thead>"])
        #expect(got == "<table class=\"table table-bordered<thead>")
    }

    /// The selectivity the spike observed: the `'></'` token (id 2631) survives — `</` fails
    /// the partial match at char 2, so the chunk is flushed whole.
    @Test func closingTagBoundaryKeepsItsGreater() {
        let got = streamThrough(ToolCallProcessor(format: .glm4),
                                chunks: ["<td>a", "></", "tr>"])
        #expect(got == "<td>a></tr>")
    }

    /// A chunk split mid-tag-name (`…"><t` | `head>`) loses the leadingToken too — not only the
    /// atomic `'><'` token; any chunk whose first `<` is followed by `t`/`o`/… does it.
    @Test func chunkSplitInsideOpeningTagNameAlsoDrops() {
        let got = streamThrough(ToolCallProcessor(format: .glm4),
                                chunks: ["\"><t", "head><tr>"])
        #expect(got == "<thead><tr>")
    }

    /// Control: the whole run of adjacent tags in one chunk (no boundary at `<t…`) is
    /// untouched — the bug needs the chunk boundary.
    @Test func adjacentTagsInOneChunkAreUntouched() {
        let got = streamThrough(ToolCallProcessor(format: .glm4),
                                chunks: ["\"><thead><tr><td>Category</td></tr></thead>"])
        #expect(got == "\"><thead><tr><td>Category</td></tr></thead>")
    }
}

/// The golden for OCP-FIX-1-2 — validated on GLM-OCR's **real tokenizer** (no model weights,
/// just `tokenizer.json`). Skips if GLM-OCR isn't installed in this environment.
@Suite struct OCRDetokenizationIsLossless {

    /// Candidate install locations for `mlx-community--GLM-OCR-4bit` across both editions'
    /// sandbox containers and the un-sandboxed path (see `RSI/journal/2026-228`).
    private static func installedGLMOCRDir() -> URL? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let rel = "Application Support/AI Browser/models/mlx-community--GLM-OCR-4bit"
        let candidates = [
            home.appending(path: "Library/Containers/net.connectcode.mlxui/Data/Library/\(rel)"),
            home.appending(path: "Library/Containers/ConnectCode.PipelineStudio/Data/Library/\(rel)"),
            home.appending(path: "Library/\(rel)"),
        ]
        return candidates.first {
            FileManager.default.fileExists(atPath: $0.appending(path: "tokenizer.json").path)
        }
    }

    private static func greaterThenLessCount(_ s: String) -> Int {
        let c = Array(s)
        return (0 ..< max(0, c.count - 1)).reduce(0) { $0 + (c[$1] == ">" && c[$1 + 1] == "<" ? 1 : 0) }
    }

    /// Whether the tokenizer-loading goldens run. `HFTokenizerLoader().load()` reads a large
    /// `tokenizer.json` and builds a Jinja template; on a loaded machine that is tens of
    /// seconds, and it has been seen to stall the whole serial suite. So it is **opt-in** —
    /// set `MLXUI_TOKENIZER_GOLDENS=1` (the owner does this for a real sign-off run). The
    /// hermetic `ToolCallProcessorUpstreamBehavior` tests carry the everyday proof.
    static var tokenizerGoldensEnabled: Bool {
        ProcessInfo.processInfo.environment["MLXUI_TOKENIZER_GOLDENS"] != nil
    }

    /// Sample HTML with many adjacent-tag boundaries (`><`), the shape GLM-OCR emits for a
    /// table. The new engine path detokenizes the whole id stream at once; the old path
    /// replayed it through `NaiveStreamingDetokenizer` + `ToolCallProcessor`.
    @Test(.enabled(if: OCRDetokenizationIsLossless.tokenizerGoldensEnabled))
    func newEnginePathRoundTripsMarkupThatTheOldPathCorrupted() async throws {
        guard let dir = Self.installedGLMOCRDir() else { return }   // not installed here

        let tokenizer = try await HFTokenizerLoader().load(from: dir)
        let html = """
        <table class="table table-bordered"><thead><tr><th>Category</th><th>Budget</th></tr>\
        </thead><tbody><tr><td>Auto</td><td>$200.00</td></tr><tr><td>Other</td><td>$50.00</td>\
        </tr></tbody></table>
        """
        let ids = tokenizer.encode(text: html, addSpecialTokens: false)
        let boundaries = Self.greaterThenLessCount(html)
        #expect(boundaries >= 6, "sample should contain several '><' boundaries to be meaningful")

        // NEW path — exactly what VLMEngine/DotsOCREngine/DeepSeekOCREngine now do.
        let newPath = tokenizer.decode(tokenIds: ids, skipSpecialTokens: true)

        // OLD path — NaiveStreamingDetokenizer chunks fed through ToolCallProcessor.
        var detok = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        let processor = ToolCallProcessor(format: .glm4)
        var oldPath = ""
        for id in ids {
            detok.append(token: id)
            if let chunk = detok.next(), let text = processor.processChunk(chunk) { oldPath += text }
        }
        processor.processEOS()

        print("=== OCP-FIX-1-2 golden ===")
        print("'><' boundaries — source: \(boundaries), new path: \(Self.greaterThenLessCount(newPath)), old path: \(Self.greaterThenLessCount(oldPath))")
        print("new: \(newPath)")
        print("old: \(oldPath)")

        // The golden: the new engine path preserves every `><` boundary and never fuses tags.
        #expect(Self.greaterThenLessCount(newPath) == boundaries)
        #expect(!newPath.contains("<thead<"))
        #expect(!newPath.contains("</td<td>"))
        // The old path can only lose boundaries, never gain them (informational — whether the
        // drop reproduces on a given string depends on which BPE tokens the encoder picks).
        #expect(Self.greaterThenLessCount(oldPath) <= boundaries)
    }
}
