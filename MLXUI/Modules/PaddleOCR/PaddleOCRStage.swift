import Foundation
import CoreGraphics

/// `image → text` stage backed by the vendored PaddleOCR-VL pipeline (via `PaddleOCREngine`).
/// Mirrors `VLMStage`: the generator is injectable so the stage's contract is unit-testable
/// without a model (the same seam `ASRStage`/`LLMStage`/`VLMStage` use). PaddleOCR has no
/// per-run prompt — the transcription task is fixed in the engine — so it satisfies
/// `PipelineStage.run(_:)`'s single-`Media` contract directly.
nonisolated struct PaddleOCRStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .image }
    var produces: MediaKind { .text }

    private let generate: @Sendable (CGImage) async throws -> String

    /// Designated init with an injectable generator (tests pass a mock).
    init(
        id: String,
        name: String,
        generate: @escaping @Sendable (CGImage) async throws -> String
    ) {
        self.id = id
        self.name = name
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
        let text = try await generate(media.cgImage)
        progress(1.0)
        return .text(text)
    }
}

extension PaddleOCRStage {
    /// Stage backed by the real PaddleOCR-VL engine, loading the installed model directory.
    init(modelID: String, maxTokens: Int = 2048) {
        let dir = PaddleOCREngine.installedModelDirectory(id: modelID)
        self.init(id: "paddleocr.\(modelID)", name: "PaddleOCR-VL (\(modelID))") { image in
            try await PaddleOCREngine.generate(image: image, modelDir: dir, maxTokens: maxTokens)
        }
    }
}
