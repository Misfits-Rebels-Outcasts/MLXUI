import Foundation
import CoreGraphics

/// `image → text` stage backed by `MLXVLM` (via `VLMEngine`). The prompt is fixed at
/// construction (carried in `StageConfig.prompt`, or supplied per-run by the run view),
/// so the stage still satisfies `PipelineStage.run(_ input:)`'s single-`Media` contract.
/// The generator is injectable so the stage's logic is unit-testable without a model —
/// the same seam `ASRStage`/`LLMStage` use.
nonisolated struct VLMStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .image }
    var produces: MediaKind { .text }

    /// The question asked about every image this stage runs.
    let prompt: String
    private let generate: @Sendable (CGImage, String) async throws -> String

    /// Designated init with an injectable generator (tests pass a mock).
    init(
        id: String,
        name: String,
        prompt: String,
        generate: @escaping @Sendable (CGImage, String) async throws -> String
    ) {
        self.id = id
        self.name = name
        self.prompt = prompt
        self.generate = generate
    }

    func run(
        _ input: Media,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> Media {
        try require(input, .image)
        guard case let .image(media) = input else {
            throw StageError.kindMismatch(expected: .image, got: input.kind)
        }
        progress(0.1)
        let text = try await generate(media.cgImage, prompt)
        progress(1.0)
        return .text(text)
    }
}

extension VLMStage {
    /// Stage backed by the real MLX VLM engine, loading the installed model directory.
    init(modelID: String, prompt: String, maxTokens: Int = 512) {
        let dir = VLMEngine.installedModelDirectory(id: modelID)
        self.init(id: "vlm.\(modelID)", name: "VLM (\(modelID))", prompt: prompt) { image, prompt in
            try await VLMEngine.generate(prompt: prompt, image: image, modelDir: dir, maxTokens: maxTokens)
        }
    }
}
