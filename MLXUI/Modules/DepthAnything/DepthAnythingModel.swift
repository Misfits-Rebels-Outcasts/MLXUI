import Foundation
import MLX
import MLXNN

/// `DepthAnythingDepthEstimationHead`: 64→32 conv, resize to the patch grid × 14, 32→32 conv + ReLU,
/// 1×1 conv to one channel, ReLU (relative depth), scale by `max_depth` (1 here). Trap 5: the resize inside
/// the head is bilinear with `align_corners=True` and to an exact size, not a factor.
nonisolated final class DepthAnythingDepthHead: Module {
    let conv1: Conv2d
    let conv2: Conv2d
    let conv3: Conv2d

    private let headInIndex: Int
    private let patchSize: Int
    private let maxDepth: Float

    init(config: DepthAnythingConfig) {
        self.conv1 = Conv2d(inputChannels: config.fusionHiddenSize,
                            outputChannels: config.fusionHiddenSize / 2,
                            kernelSize: 3, stride: 1, padding: 1, bias: true)
        self.conv2 = Conv2d(inputChannels: config.fusionHiddenSize / 2,
                            outputChannels: config.headHiddenSize,
                            kernelSize: 3, stride: 1, padding: 1, bias: true)
        self.conv3 = Conv2d(inputChannels: config.headHiddenSize,
                            outputChannels: 1,
                            kernelSize: 1, stride: 1, padding: 0, bias: true)
        self.headInIndex = config.headInIndex
        self.patchSize = config.patchSize
        self.maxDepth = config.maxDepth
        super.init()
    }

    /// `fused`: the neck's list of fused maps, smallest first. Returns `[1, patchHeight * 14, patchWidth * 14]`.
    func callAsFunction(_ fused: [MLXArray], patchHeight: Int, patchWidth: Int) -> MLXArray {
        let index = headInIndex < 0 ? fused.count + headInIndex : headInIndex
        var predicted = conv1(fused[index])
        predicted = DepthAnythingOps.resizeBilinear(predicted,
                                                    outH: patchHeight * patchSize,
                                                    outW: patchWidth * patchSize,
                                                    alignCorners: true)
        predicted = relu(conv2(predicted))
        predicted = conv3(predicted)
        predicted = relu(predicted) * maxDepth
        return predicted.reshaped(predicted.dim(0), predicted.dim(1), predicted.dim(2))
    }
}

/// The full Depth Anything V2 Small network: DINOv2 ViT-S/14 backbone, DPT neck, depth head.
///
/// Loads the **published** `model.safetensors` (fp32) from an installed folder and converts the PyTorch
/// layouts in the loader; nothing on disk is rewritten and nothing is downloaded here.
nonisolated final class DepthAnythingModel: Module {

    struct Loaded: @unchecked Sendable {
        let model: DepthAnythingModel
        let config: DepthAnythingConfig
    }

    let backbone: DINOv2Backbone
    let neck: DPTNeck
    let head: DepthAnythingDepthHead

    private let patchSize: Int

    init(config: DepthAnythingConfig) {
        self.backbone = DINOv2Backbone(config: config)
        self.neck = DPTNeck(config: config)
        self.head = DepthAnythingDepthHead(config: config)
        self.patchSize = config.patchSize
        super.init()
    }

    /// `pixelValues`: `[1, H, W, 3]` float32, preprocessed. Returns the four backbone feature maps
    /// (`[1, patchTokens + 1, 384]`, CLS first, after the backbone layer norm).
    func backboneFeatures(_ pixelValues: MLXArray) -> [MLXArray] { backbone(pixelValues) }

    /// The depth prediction, `[1, patchHeight * 14, patchWidth * 14]` float32.
    func predictedDepth(_ pixelValues: MLXArray) -> MLXArray {
        let features = backbone(pixelValues)
        let patchHeight = pixelValues.dim(1) / patchSize
        let patchWidth = pixelValues.dim(2) / patchSize
        let fused = neck(features, patchHeight: patchHeight, patchWidth: patchWidth)
        return head(fused, patchHeight: patchHeight, patchWidth: patchWidth)
    }

    /// Strict load from an installed folder: every tensor in the file must be used and every parameter set.
    static func load(directory: URL) throws -> Loaded {
        let config = try DepthAnythingWeights.loadConfig(directory: directory)
        let weights = try DepthAnythingWeights.loadArrays(directory: directory)
        let model = DepthAnythingModel(config: config)
        try DepthAnythingWeights.install(weights: weights, prefix: "", into: model, config: config)
        return Loaded(model: model, config: config)
    }

    /// Strict load from an already-read checkpoint.
    static func load(weights: [String: MLXArray], config: DepthAnythingConfig) throws -> DepthAnythingModel {
        let model = DepthAnythingModel(config: config)
        try DepthAnythingWeights.install(weights: weights, prefix: "", into: model, config: config)
        return model
    }

    /// The published checkpoint's tensor count; a port that drops a tensor fails the strict load.
    static let checkpointTensorCount = 287
}
