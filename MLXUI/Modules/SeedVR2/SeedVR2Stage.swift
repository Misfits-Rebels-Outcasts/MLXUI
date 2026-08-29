import CoreGraphics
import Foundation

/// `image → image` stage for SeedVR2 3B single-step super-resolution (SV-AM4).
/// Accepts a `.image` (the user's LR photo), bicubic-upsamples it to the target
/// size, denoises in latent space via the int8 transformer, and returns the HR output.
nonisolated struct SeedVR2Stage: PipelineStage {
    let id:   String
    let name: String
    var accepts:  MediaKind { .image }
    var produces: MediaKind { .image }

    let scale: Int
    private let runUpscale: @Sendable (CGImage, @Sendable (Double) -> Void) async throws -> CGImage

    init(
        id:   String,
        name: String,
        scale: Int = 2,
        run: @escaping @Sendable (CGImage, @Sendable (Double) -> Void) async throws -> CGImage
    ) {
        self.id   = id
        self.name = name
        self.scale = scale
        self.runUpscale = run
    }

    func run(
        _ input: Media,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> Media {
        try require(input, .image)
        guard case let .image(img) = input else {
            throw StageError.kindMismatch(expected: .image, got: input.kind)
        }
        progress(0.05)
        let hr = try await runUpscale(img.cgImage, progress)
        progress(1.0)
        return .image(ImageMedia(cgImage: hr))
    }
}

extension SeedVR2Stage {
    init(modelID: String, scale: Int = 2) {
        self.init(
            id:    "seedvr2.\(modelID)",
            name:  "SeedVR2 3B (\(modelID))",
            scale: scale
        ) { cgImage, progress in
            try await SeedVR2Engine.upscale(
                image:    cgImage,
                scale:    scale,
                modelID:  modelID,
                progress: progress)
        }
    }
}
