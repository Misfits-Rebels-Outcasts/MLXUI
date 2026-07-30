import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// AG4b — `semantic_search` + the on-disk `SemanticIndex`. Ranking, document parsing, result
/// formatting, and Codable round-trip are pure and tested synchronously; the embedding call and
/// live disk persistence are not exercised here.
struct SemanticSearchToolTests {

    private func entry(_ text: String, _ vector: [Float]) -> SemanticIndex.Entry {
        .init(text: text, vector: vector)
    }

    // MARK: document parsing

    @Test func documentsSplitTrimAndDropEmpties() {
        let raw = "  first  \n\nsecond\n   \nthird\n"
        #expect(SemanticIndex.documents(from: raw) == ["first", "second", "third"])
    }

    @Test func documentsFromBlankIsEmpty() {
        #expect(SemanticIndex.documents(from: "   \n \n").isEmpty)
    }

    // MARK: ranking

    @Test func searchRanksByCosineDescending() {
        let index = SemanticIndex(modelID: "m", entries: [
            entry("orthogonal", [0, 1]),
            entry("identical", [1, 0]),
            entry("opposite", [-1, 0]),
        ])
        let results = index.search(queryVector: [1, 0], topK: 3)
        #expect(results.map(\.text) == ["identical", "orthogonal", "opposite"])
        #expect(abs(results[0].score - 1) < 1e-5)
        #expect(abs(results[1].score) < 1e-5)
        #expect(abs(results[2].score - -1) < 1e-5)
    }

    @Test func searchRespectsTopK() {
        let index = SemanticIndex(modelID: "m", entries: [
            entry("a", [1, 0]), entry("b", [0, 1]), entry("c", [1, 1]),
        ])
        #expect(index.search(queryVector: [1, 0], topK: 1).count == 1)
        #expect(index.search(queryVector: [1, 0], topK: 10).count == 3)  // clamps to available
        #expect(index.search(queryVector: [1, 0], topK: 0).isEmpty)
    }

    // MARK: persistence round-trip (Codable, no disk)

    @Test func indexRoundTripsThroughJSON() throws {
        let index = SemanticIndex(modelID: "all-MiniLM", entries: [
            entry("hello", [0.1, 0.2, 0.3]),
            entry("world", [0.4, 0.5, 0.6]),
        ])
        let data = try JSONEncoder().encode(index)
        let decoded = try JSONDecoder().decode(SemanticIndex.self, from: data)
        #expect(decoded == index)
    }

    // MARK: formatting

    @Test func formatsResultsNumberedWithScores() {
        let out = SemanticSearchTool.formatResults([
            (text: "first doc", score: 0.912),
            (text: "second doc", score: 0.4),
        ])
        #expect(out.contains("1. [0.912] first doc"))
        #expect(out.contains("2. [0.400] second doc"))
    }

    @Test func formatsEmptyResults() {
        #expect(SemanticSearchTool.formatResults([]) == "No matching documents.")
    }

    // MARK: protocol conformance

    @Test func toolIsReadOnlyAndSpecced() {
        #expect(SemanticSearchTool().requiresApproval == false)
        let fn = SemanticSearchTool().toolSpec["function"] as? [String: any Sendable]
        let params = fn?["parameters"] as? [String: any Sendable]
        #expect(params?["required"] as? [String] == ["query"])  // documents + top_k optional
    }
}
