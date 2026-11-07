import Foundation
import CoreGraphics
import MLX

/// Loads the installed Depth Anything V2 Small checkpoint and turns one image into its 8-bit grayscale depth
/// map and PNG. Everything here is synchronous MLX work; callers run it off the main actor.
nonisolated enum DepthAnythingEngine {

    struct Result: @unchecked Sendable {
        /// The grayscale depth map at the input image's pixel size (near = bright).
        let grayImage: CGImage
        /// The same map as the 8-bit grayscale PNG `Save PNG` writes.
        let pngData: Data
        let width: Int
        let height: Int
    }

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout.
    static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }

    static func load(directory: URL) throws -> DepthAnythingModel.Loaded {
        try DepthAnythingModel.load(directory: directory)
    }

    /// Estimate depth for one image. Refuses the input limits before any model work.
    static func estimateDepth(_ image: CGImage, modelDirectory: URL) throws -> Result {
        let prepared = try DepthAnythingInput.prepare(image: image)
        // MC3-11: the footprint above the 99 MB of weights is MLX's own buffer cache (freed activation buffers it keeps for
        // reuse), not decode / preprocess / PNG work. Release it once the model is done and again on the way out, so the
        // upsample and the PNG do not sit on top of it and the Run screen does not keep it afterwards.
        defer { Memory.clearCache() }
        let loaded = try load(directory: modelDirectory)
        let input = try DepthAnythingPreprocessor.preprocess(image: prepared.image)
        let prediction = loaded.model.predictedDepth(input)
        eval(prediction)
        Memory.clearCache()
        let gray = try DepthAnythingPostprocessor.grayscale(
            prediction: prediction, height: prepared.height, width: prepared.width)
        let grayImage = try DepthAnythingPostprocessor.image(gray: gray, width: prepared.width, height: prepared.height)
        guard let pngData = PNGEncoder.pngData(from: grayImage) else {
            throw DepthAnythingError.badInput("could not encode the depth map as PNG")
        }
        return Result(grayImage: grayImage, pngData: pngData, width: prepared.width, height: prepared.height)
    }
}
