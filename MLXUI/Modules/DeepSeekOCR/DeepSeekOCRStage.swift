import Foundation
import CoreGraphics

/// `image → text` stage for **DeepSeek-OCR-2** (SUP-2). Mirrors `DotsOCRStage`/`PaddleOCRStage`: the
/// generator is injectable so the stage's contract is unit-testable without a model. The real engine
/// — a custom `deepseekocr_2` model (SAM ViT → Qwen2-0.5B encoder → DeepSeek-V2 MoE LM) registered
/// into MLXVLM (see `RSI/plan-deepseek-ocr.md`) — lands in a later slice; until then
/// `DeepSeekOCRSDK.makeStage` supplies a generator that errors gracefully, and the model still shows
/// the `ModelSupport` "unsupported" banner. (The injectable seam also leaves room for the Approach-B
/// fallback if the tiling processor can't be expressed through `LMInput`.)
nonisolated struct DeepSeekOCRStage: PipelineStage {
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

extension DeepSeekOCRStage {
    /// Stage backed by the real DeepSeek-OCR-2 engine, loading the installed model directory.
    init(modelID: String, maxTokens: Int = 2048) {
        let dir = DeepSeekOCREngine.installedModelDirectory(id: modelID)
        self.init(id: "deepseek-ocr.\(modelID)", name: "DeepSeek-OCR (\(modelID))") { image in
            try await DeepSeekOCREngine.generate(image: image, modelDir: dir, maxTokens: maxTokens)
        }
    }
}
