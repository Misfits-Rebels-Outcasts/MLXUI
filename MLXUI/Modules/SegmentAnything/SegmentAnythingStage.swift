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
            return renderTopMask(masks: masks, iouScores: iouScores, sourceImage: cgImage)
        }
    }
}

// MARK: - Mask rendering

/// Render the highest-IoU mask as a red-tinted overlay on the source image.
private func renderTopMask(
    masks: MLXArray, iouScores: MLXArray, sourceImage: CGImage
) -> CGImage {
    // iouScores: [1, 3] — pick the mask index with the best score
    let scores = iouScores[0].asArray(Float.self)
    let bestIdx = scores.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0

    let (h, w) = (masks.dim(1), masks.dim(2))
    // masks: [1, H', W', 3] — extract the best mask logits
    let logits = masks[0, 0..., 0..., bestIdx]         // [H', W']
    let binary  = (logits .> 0).asArray(Float.self)    // 0.0 or 1.0

    let W = sourceImage.width, H = sourceImage.height
    guard let ctx = CGContext(
        data: nil, width: W, height: H,
        bitsPerComponent: 8, bytesPerRow: W * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return sourceImage }

    // Draw original image
    ctx.draw(sourceImage, in: CGRect(x: 0, y: 0, width: W, height: H))

    // Overlay mask in translucent red
    ctx.setBlendMode(.normal)
    for py in 0 ..< H {
        for px in 0 ..< W {
            let mx = min(Int(Float(px) / Float(W) * Float(w)), w - 1)
            let my = min(Int(Float(py) / Float(H) * Float(h)), h - 1)
            if binary[my * w + mx] > 0.5 {
                ctx.setFillColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 0.4)
                ctx.fill(CGRect(x: px, y: H - 1 - py, width: 1, height: 1))
            }
        }
    }
    return ctx.makeImage() ?? sourceImage
}
