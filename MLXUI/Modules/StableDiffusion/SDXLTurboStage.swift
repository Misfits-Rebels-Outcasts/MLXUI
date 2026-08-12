import Foundation
import CoreGraphics

/// `text → image` stage for SDXL-Turbo (SD-MOD1). Accepts a text prompt and produces a generated
/// `CGImage`. The generator is injectable so the stage's contract is unit-testable without a model
/// (the same seam `FluxStage`/`ASRStage` use).
nonisolated struct SDXLTurboStage: PipelineStage {
    let id: String
    let name: String
    /// Fixed PRNG seed for reproducible output (nil = random). Set by the run view.
    let seed: UInt64?
    var accepts: MediaKind { .text }
    var produces: MediaKind { .image }

    private let generate: @Sendable (String, UInt64?, @Sendable (Double) -> Void) async throws -> CGImage

    /// Designated init with an injectable generator (tests pass a mock).
    init(
        id: String,
        name: String,
        seed: UInt64? = nil,
        generate: @escaping @Sendable (String, UInt64?, @Sendable (Double) -> Void) async throws -> CGImage
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

extension SDXLTurboStage {
    /// Stage for a catalog model id, backed by the real SDXL-Turbo engine (`SDXLTurboEngine`).
    init(modelID: String, seed: UInt64? = nil) {
        self.init(id: "sdxl-turbo.\(modelID)", name: "SDXL-Turbo (\(modelID))", seed: seed) { prompt, seed, progress in
            try await SDXLTurboEngine.generate(prompt: prompt, modelID: modelID, seed: seed, progress: progress)
        }
    }
}
