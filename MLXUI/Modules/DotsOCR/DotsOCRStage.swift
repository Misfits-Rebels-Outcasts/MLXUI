import Foundation
import CoreGraphics

/// `image → text` stage for **dots.ocr / dots.mocr** (SUP-3). Mirrors `PaddleOCRStage`/`VLMStage`:
/// the generator is injectable so the stage's contract is unit-testable without a model. The real
/// engine registers a custom `dots_ocr` model + processor into MLXVLM and runs the shared VLM
/// generate path (`DotsOCREngine`).
nonisolated struct DotsOCRStage: PipelineStage {
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

extension DotsOCRStage {
    /// Stage backed by the real dots.ocr engine, loading the installed model directory.
    init(modelID: String, maxTokens: Int = 2048) {
        let dir = DotsOCREngine.installedModelDirectory(id: modelID)
        self.init(id: "dots-ocr.\(modelID)", name: "dots.ocr (\(modelID))") { image in
            try await DotsOCREngine.generate(image: image, modelDir: dir, maxTokens: maxTokens)
        }
    }
}
