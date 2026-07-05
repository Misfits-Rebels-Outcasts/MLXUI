import Foundation

/// `text → embedding` stage backed by `MLXEmbedders` (via `EmbeddingEngine`). Produces a
/// `Media.embedding` (a batch of vectors — here a batch of one). The embedder is injectable
/// so the stage's logic is unit-testable without a model — the same seam the other stages use.
nonisolated struct EmbeddingStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .text }
    var produces: MediaKind { .embedding }

    /// text → L2-normalized vectors.
    private let embed: @Sendable (String) async throws -> [[Float]]

    init(
        id: String,
        name: String,
        embed: @escaping @Sendable (String) async throws -> [[Float]]
    ) {
        self.id = id
        self.name = name
        self.embed = embed
    }

    func run(
        _ input: Media,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> Media {
        try require(input, .text)
        guard case let .text(text) = input else {
            throw StageError.kindMismatch(expected: .text, got: input.kind)
        }
        progress(0.1)
        let vectors = try await embed(text)
        progress(1.0)
        return .embedding(vectors)
    }
}

extension EmbeddingStage {
    /// Stage backed by the real MLX embedder, loading the installed model directory.
    init(modelID: String) {
        let directory = EmbeddingEngine.installedModelDirectory(id: modelID)
        self.init(id: "embedding.\(modelID)", name: "Embedding") { text in
            try await EmbeddingEngine.embed([text], modelDirectory: directory)
        }
    }
}
