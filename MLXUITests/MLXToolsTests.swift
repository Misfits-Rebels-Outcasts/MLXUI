import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// AG4a — in-app MLX tools (`embed_text`, `summarize`) + the `InstalledModelIndex` resolver.
/// Pure static helpers hold the logic and are tested synchronously; real inference (engine
/// loads weights, Apple-Silicon-only) and live disk/bundle reads are not exercised here.
struct MLXToolsTests {

    // MARK: InstalledModelIndex (model resolution)

    private func entry(_ id: String, _ kind: RunnerKind, _ ram: Double) -> InstalledModelIndex.Entry {
        .init(id: id, kind: kind, ramGB: ram)
    }

    @Test func indexPicksSmallestInstalledOfKind() {
        let index = InstalledModelIndex.make(
            catalog: [
                entry("big-llm", .llm, 16),
                entry("small-llm", .llm, 4),
                entry("embed", .embedding, 1),
            ],
            installedIDs: ["big-llm", "small-llm", "embed"])
        #expect(index.best(kind: .llm) == "small-llm")
        #expect(index.best(kind: .embedding) == "embed")
    }

    @Test func indexReturnsNilWhenNoneOfKindInstalled() {
        let index = InstalledModelIndex.make(
            catalog: [entry("llm", .llm, 4), entry("embed", .embedding, 1)],
            installedIDs: ["llm"])  // embed is in catalog but not installed
        #expect(index.best(kind: .embedding) == nil)
        #expect(index.best(kind: .asr) == nil)
    }

    @Test func indexKeepsOnlyInstalled() {
        let index = InstalledModelIndex.make(
            catalog: [entry("a", .llm, 2), entry("b", .llm, 8)],
            installedIDs: ["b"])
        #expect(index.best(kind: .llm) == "b")  // "a" not installed, so "b" wins despite being larger
    }

    // MARK: cosine similarity

    @Test func cosineOfIdenticalIsOne() {
        #expect(abs(EmbedTextTool.cosine([1, 2, 3], [1, 2, 3]) - 1) < 1e-5)
    }

    @Test func cosineOfOrthogonalIsZero() {
        #expect(abs(EmbedTextTool.cosine([1, 0], [0, 1])) < 1e-5)
    }

    @Test func cosineOfOppositeIsMinusOne() {
        #expect(abs(EmbedTextTool.cosine([1, 0], [-1, 0]) - -1) < 1e-5)
    }

    @Test func cosineHandlesDegenerateInput() {
        #expect(EmbedTextTool.cosine([1, 2], [1, 2, 3]) == 0)  // length mismatch
        #expect(EmbedTextTool.cosine([], []) == 0)             // empty
        #expect(EmbedTextTool.cosine([0, 0], [1, 1]) == 0)     // zero vector
    }

    // MARK: formatting

    @Test func formatsSimilarityAndSingle() {
        let sim = EmbedTextTool.formatSimilarity(0.8234, model: "m", dim: 384)
        #expect(sim.contains("0.823"))
        #expect(sim.contains("384-dim"))
        #expect(sim.contains("m"))

        let single = EmbedTextTool.formatSingle(dim: 768, model: "nomic")
        #expect(single.contains("768-dimensional"))
        #expect(single.contains("nomic"))
    }

    // MARK: summarize prompt / token budget

    @Test func summarizePromptEmbedsTextAndLimit() {
        let withLimit = SummarizeTool.prompt(for: "hello world", maxWords: 20)
        #expect(withLimit.contains("hello world"))
        #expect(withLimit.contains("at most 20 words"))

        let noLimit = SummarizeTool.prompt(for: "hello world", maxWords: nil)
        #expect(noLimit.contains("hello world"))
        #expect(!noLimit.contains("at most"))
    }

    @Test func summarizeTokenBudgetIsClamped() {
        #expect(SummarizeTool.maxTokens(forMaxWords: nil) == 256)
        #expect(SummarizeTool.maxTokens(forMaxWords: 0) == 256)   // non-positive → default
        #expect(SummarizeTool.maxTokens(forMaxWords: 100) == 200) // 2 tokens/word
        #expect(SummarizeTool.maxTokens(forMaxWords: 5) == 32)    // floor
        #expect(SummarizeTool.maxTokens(forMaxWords: 10_000) == 1024) // ceiling
    }

    // MARK: protocol conformance

    @Test func mlxToolsAreReadOnly() {
        #expect(EmbedTextTool().requiresApproval == false)
        #expect(SummarizeTool().requiresApproval == false)
    }

    @Test func mlxToolsAreSpecced() {
        let embed = EmbedTextTool().toolSpec["function"] as? [String: any Sendable]
        let embedParams = embed?["parameters"] as? [String: any Sendable]
        #expect(embedParams?["required"] as? [String] == ["text"])  // compare_to is optional

        let summarize = SummarizeTool().toolSpec["function"] as? [String: any Sendable]
        let summarizeParams = summarize?["parameters"] as? [String: any Sendable]
        #expect(summarizeParams?["required"] as? [String] == ["text"])  // max_words is optional
    }
}
