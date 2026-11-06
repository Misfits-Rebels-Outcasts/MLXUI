import Foundation
import CoreGraphics

/// `image → image` stage for Depth Anything V2 Small: one photo in, an 8-bit grayscale relative-depth map out
/// at the same pixel size (near = bright). The row takes no settings.
nonisolated struct DepthAnythingStage: PipelineStage {
    let id: String
    let name: String
    /// The installed folder (`config.json` + `model.safetensors`).
    let modelDirectory: URL

    var accepts: MediaKind { .image }
    var produces: MediaKind { .image }

    init(modelDirectory: URL) {
        self.modelDirectory = modelDirectory
        self.id = "depth.\(modelDirectory.lastPathComponent)"
        self.name = "Depth Anything V2 Small"
    }

    init(modelID: String) {
        self.init(modelDirectory: DepthAnythingEngine.installedModelDirectory(id: modelID))
    }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .image)
        guard case let .image(media) = input else {
            throw StageError.kindMismatch(expected: .image, got: input.kind)
        }
        // The input limits are refused before any model work (owner decision 5).
        if let refusal = DepthAnythingInput.refusal(width: media.cgImage.width, height: media.cgImage.height) {
            throw DepthAnythingError.badInput(refusal)
        }
        progress(0.05)
        let result = try DepthAnythingEngine.estimateDepth(media.cgImage, modelDirectory: modelDirectory)
        progress(1.0)
        return .image(ImageMedia(cgImage: result.grayImage))
    }
}
