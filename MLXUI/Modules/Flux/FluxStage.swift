import Foundation
import CoreGraphics

/// `text → image` stage for FLUX.1-family diffusion (AM4/AM5). Accepts a text prompt and
/// produces a generated `CGImage`. The generator is injectable so the stage's contract is
/// unit-testable without a model (the seam `ASRStage`/`VLMStage`/`DeepSeekOCRStage` use).
/// Both `MediaKind` cases already exist — no `Core` change was needed for a new modality.
nonisolated struct FluxStage: PipelineStage {
    let id: String
    let name: String
    /// Fixed PRNG seed for reproducible output (nil = random). Set by the run view.
    let seed: UInt64?
    var accepts: MediaKind { .text }
    var produces: MediaKind { .image }

    private let generate: @Sendable (String, UInt64?, (Double) -> Void) async throws -> CGImage

    /// Designated init with an injectable generator (tests pass a mock).
    init(
        id: String,
        name: String,
        seed: UInt64? = nil,
        generate: @escaping @Sendable (String, UInt64?, (Double) -> Void) async throws -> CGImage
    ) {
        self.id = id
        self.name = name
        self.seed = seed
        self.generate = generate
    }

    func run(
        _ input: Media,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> Media {
        try require(input, .text)
        guard case let .text(prompt) = input else {
            throw StageError.kindMismatch(expected: .text, got: input.kind)
        }
        progress(0.1)
        let image = try await generate(prompt, seed, progress)
        progress(1.0)
        return .image(ImageMedia(cgImage: image))
    }
}

extension FluxStage {
    /// Stage for a catalog model id, backed by the real FLUX engine (`FluxEngine`, AM4h).
    init(modelID: String, seed: UInt64? = nil) {
        self.init(id: "flux.\(modelID)", name: "FLUX (\(modelID))", seed: seed) { prompt, seed, progress in
            try await FluxEngine.generate(prompt: prompt, modelID: modelID, seed: seed, progress: progress)
        }
    }
}
