import Testing
import Foundation
@testable import MLXUI

/// MoC-4-3 (`RSI/DelegateMoCBacklog.md`) — `Modules/Rerank/`: engine, SDK, registration.
struct RerankModuleTests {

    // MARK: - RerankPrompt: pinned byte-for-byte against the Python constants

    /// The test that catches a paraphrase — every constant compared against the exact text
    /// from the model card's own reference implementation
    /// (`mlx-community/Qwen3-Reranker-0.6B-4bit`'s README), which the backlog identifies as
    /// itself `catflow-mlx/src/catflow/engines/rerank.py`'s `_CAUSAL_YESNO_INSTRUCT` /
    /// `_CAUSAL_YESNO_PREFIX` / `_CAUSAL_YESNO_SUFFIX`.
    @Test func instructIsPinnedVerbatim() {
        #expect(RerankPrompt.instruct == "Given a web search query, retrieve relevant passages that answer the query")
    }

    @Test func prefixIsPinnedVerbatim() {
        #expect(RerankPrompt.prefix == "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    }

    @Test func suffixIsPinnedVerbatim() {
        #expect(RerankPrompt.suffix == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    @Test func contentAssemblesInstructQueryDocumentInOrder() {
        let content = RerankPrompt.content(query: "what is MLX", document: "MLX is an array framework.")
        #expect(content == "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: what is MLX\n<Document>: MLX is an array framework.")
    }

    // MARK: - Resolution: registry.bestModule(for:) resolves a .rerank entry to RerankSDK

    private func rerankEntry() -> ModelEntry {
        makeEntry(
            id: "mlx-community--Qwen3-Reranker-0.6B-4bit",
            family: "Qwen3-Reranker",
            displayName: "Qwen3-Reranker-0.6B",
            modelType: .rerank,
            source: .mlx,
            ramGB: 0.52,
            downloadSizeGB: 0.35
        )
    }

    @Test func rerankHasNoSupportGap() {
        #expect(ModelSupport.unsupportedReason(for: rerankEntry()) == nil)
    }

    @Test func rerankRoutesToRunnableKind() {
        #expect(rerankEntry().runnerKind == .rerank)
    }

    @MainActor
    @Test func rerankResolvesToRerankSDK() throws {
        let registry = ModelRegistry()
        for module in installedModules {
            module.register(into: registry)
        }
        let resolved = try #require(registry.bestModule(for: rerankEntry()))
        #expect(resolved.descriptor.id == "rerank")
        #expect(resolved.sdk.claim(rerankEntry()) == .exact)
    }

    @Test func rerankSDKDeclinesNonRerankModels() {
        #expect(RerankSDK().claim(makeEntry(modelType: .llm, source: .mlx)) == .no)
        #expect(RerankSDK().claim(makeEntry(modelType: .rerank, source: .research)) == .no)
    }

    @Test func rerankMakeStageProducesTextToTextStage() throws {
        let stage = try RerankSDK().makeStage(for: rerankEntry(), config: .default)
        #expect(stage.accepts == .text)
        #expect(stage.produces == .text)
    }

    // MARK: - RerankScoringStage: the query/candidate split, testable without weights

    @Test func scoringStageSplitsQueryAndCandidateOnTheFirstNewlineOnly() async throws {
        var received: (query: String, candidate: String)?
        let stage = RerankScoringStage(id: "test", name: "Test") { combined in
            let parts = combined.split(separator: "\n", maxSplits: 1, omittingEmptySubsequences: false)
            received = (String(parts[0]), String(parts[1]))
            return "0.5"
        }
        // The candidate itself carries a newline — must survive intact in the second half.
        let result = try await stage.run(.text("what is MLX\nline one\nline two"), progress: { _ in })
        #expect(received?.query == "what is MLX")
        #expect(received?.candidate == "line one\nline two")
        guard case .text(let scoreText) = result else {
            Issue.record("expected .text, got \(result)")
            return
        }
        #expect(scoreText == "0.5")
    }

    @Test func scoringStageRejectsNonTextInput() async {
        let stage = RerankScoringStage(id: "test", name: "Test") { _ in "0" }
        await #expect(throws: StageError.self) {
            _ = try await stage.run(.audio(AudioBuffer(samples: [], sampleRate: 16000)), progress: { _ in })
        }
    }
}
