import CoreGraphics
import Foundation
import MLX

/// `image → image` stage for SAM3 image segmentation (SA-AM4, CFM-R15-1).
///
/// Accepts a `.image` (the user's photo), encodes it through the SAM3 ViT backbone,
/// decodes with a default center-point foreground prompt, and returns `.image`
/// containing the highest-IoU mask rendered as a **two-valued binary mask** (white =
/// mask, black = background, nothing of the source photo survives — the flow boundary
/// `tools/media.py::save_mask` expects).
///
/// The centre point is a **deliberate MLXUI choice, not the reference's behaviour**: the
/// reference (`engines/diffusion.py::_sam_load_and_segment`) runs an automatic mask
/// generator over a dense point grid and ORs every object's mask into one map. Same
/// contract (`image → image`, a binary mask), different picture — recorded in
/// `SPEC_QUESTIONS.md` (SPEC-Q210) alongside Q207–Q209.
///
/// Full interactive prompting (click to place points, box draw) lives in SA-AM5
/// (`SegmentAnythingRunView`), which keeps the red-tinted overlay rendering.
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
    /// Uses a default center-point foreground prompt (x=0.5, y=0.5) — a deliberate MLXUI
    /// choice per SPEC-Q210 — and emits the binary mask through `renderBinaryMask`
    /// (CFM-R15-1), never the run view's red overlay.
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
            guard let mask = SegmentAnythingEngine.renderBinaryMask(
                masks: masks, iouScores: iouScores,
                width: cgImage.width, height: cgImage.height) else {
                throw SegmentAnythingError.maskRenderFailed
            }
            return mask
        }
    }
}

