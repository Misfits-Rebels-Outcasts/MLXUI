import Foundation

/// `text → text` stage backed by the Qwen3-Reranker scorer. The scorer is injectable so the
/// stage's plumbing is unit-testable without loading a model — mirrors `LLMStage`'s shape
/// (MoC-4-3, `RSI/DelegateMoCBacklog.md`).
nonisolated struct RerankScoringStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }

    private let score: @Sendable (String) async throws -> String

    init(id: String, name: String, score: @escaping @Sendable (String) async throws -> String) {
        self.id = id
        self.name = name
        self.score = score
    }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .text)
        guard case let .text(combined) = input else {
            throw StageError.kindMismatch(expected: .text, got: input.kind)
        }
        progress(0.1)
        let result = try await score(combined)
        progress(1.0)
        return .text(result)
    }
}

/// `ModelSDK` for the Qwen3-Reranker causal-LM scorer. Claims every `.rerank` entry backed
/// by `source == .mlx` — mirrors `ChatSDK`'s shape.
nonisolated struct RerankSDK: ModelSDK {
    let id = "rerank"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .rerank, model.source == .mlx else { return .no }
        return .exact
    }

    /// The seam `RealExecutor.runModel`'s `engines.rerank.` branch relies on (MoC-3-3): the
    /// input is `"\(query)\n\(candidate)"`, split on the **first** newline only, so a
    /// candidate's own internal newlines survive intact in the second half. This is safe
    /// only because `RealExecutor.stageConfig`'s `engines.rerank.` branch refuses a query
    /// containing a newline before a `RerankStage` is ever built (MoC-FIX-2) — a `.cat`
    /// row's settings string is single-line in the *source file*, but `FlowSettings`
    /// unescapes a literal `\n` inside a quoted value, so "a query never contains one" is
    /// not actually guaranteed by the parser and must not be assumed here.
    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        let dir = RerankEngine.modelDirectory(for: model.id)
        return RerankScoringStage(id: model.id, name: model.displayName) { combined in
            let parts = combined.split(separator: "\n", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2 else {
                throw StageError.engineFailure(
                    stage: "Rerank",
                    underlying: RerankEngineError.malformedScoringInput)
            }
            let query = String(parts[0])
            let document = String(parts[1])
            let score = try await RerankEngine.score(query: query, document: document, modelDir: dir)
            return String(score)
        }
    }
}
