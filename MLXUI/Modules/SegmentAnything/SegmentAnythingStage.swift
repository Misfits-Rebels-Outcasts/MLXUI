import CoreGraphics
import Foundation
import MLX

/// `image → image` stage for SAM3 image segmentation (SA-AM4).
///
/// Accepts a `.image` (the user's photo), encodes it through the SAM3 ViT backbone,
/// decodes with a default center-point foreground prompt, and returns `.image`
/// containing the highest-IoU mask rendered as a binary overlay.
///
/// Full interactive prompting (click to place points, box draw) lives in SA-AM5
/// (`SegmentAnythingRunView`).
nonisolated struct SegmentAnythingStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .image }
    var produces: MediaKind { .image }

    private let runSegmentation:
        @Sendable (CGImage) async throws -> CGImage

    /// Designated init — injectable for unit tests.
    init(
        id: String,
        name: String,
        run: @escaping @Sendable (CGImage) async throws -> CGImage
    ) {
        self.id = id
        self.name = name
        self.runSegmentation = run
    }

    func run(
        _ input: Media,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> Media {
        try require(input, .image)
        guard case let .image(img) = input else {
            throw StageError.kindMismatch(expected: .image, got: input.kind)
        }
        progress(0.1)
        let masked = try await runSegmentation(img.cgImage)
        progress(1.0)
        return .image(ImageMedia(cgImage: masked))
    }
}

extension SegmentAnythingStage {
    /// Stage for a catalog model ID, backed by the real `SegmentAnythingEngine`.
    /// Uses a default center-point foreground prompt (x=0.5, y=0.5).
    init(modelID: String) {
        self.init(
            id: "segmentation.\(modelID)",
            name: "SAM3 (\(modelID))"
        ) { cgImage in
            let embedding = try await SegmentAnythingEngine.encodeImage(cgImage, modelID: modelID)
            let (masks, iouScores) = try await SegmentAnythingEngine.decodeMasks(
                embedding: embedding,
                points: [[0.5, 0.5]],
                labels: [1],
                modelID: modelID)
            return SegmentAnythingEngine.renderTopMask(masks: masks, iouScores: iouScores, sourceImage: cgImage)
        }
    }
}

