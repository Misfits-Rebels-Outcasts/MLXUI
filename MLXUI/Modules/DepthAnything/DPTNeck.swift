import Foundation
import MLX
import MLXNN

/// One reassemble layer's resize stage. PyTorch uses three different modules here (a transposed
/// convolution for factor > 1, the identity for factor 1, a strided convolution for factor < 1); because a
/// single Swift class cannot declare three different property types under the same checkpoint key, the
/// weight and bias live here directly and the mode picks the op. Keys stay `…resize.weight` / `…resize.bias`.
nonisolated final class DepthAnythingReassembleResize: Module {
    nonisolated enum Mode: Sendable, Equatable {
        case transposed(kernel: Int, stride: Int)
        case identity
        case stridedConv(kernel: Int, stride: Int, padding: Int)
    }

    let weight: MLXArray?
    let bias: MLXArray?
    let mode: Mode

    init(channels: Int, factor: Float) {
        if factor > 1 {
            let kernel = Int(factor)
            self.weight = MLXArray.zeros([channels, kernel, kernel, channels])
            self.bias = MLXArray.zeros([channels])
            self.mode = .transposed(kernel: kernel, stride: kernel)
        } else if factor == 1 {
            self.weight = nil
            self.bias = nil
            self.mode = .identity
        } else {
            let kernel = 3
            let stride = Int((1 / factor).rounded())
            self.weight = MLXArray.zeros([channels, kernel, kernel, channels])
            self.bias = MLXArray.zeros([channels])
            self.mode = .stridedConv(kernel: kernel, stride: stride, padding: 1)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let weight, let bias else { return x }
        switch mode {
        case let .transposed(_, stride):
            return convTransposed2d(x, weight, stride: IntOrPair((stride, stride)), padding: IntOrPair((0, 0))) + bias
        case .identity:
            return x
        case let .stridedConv(_, stride, padding):
            return conv2d(x, weight, stride: IntOrPair((stride, stride)), padding: IntOrPair((padding, padding))) + bias
        }
    }
}

/// `DepthAnythingReassembleLayer`: 1 x 1 projection to the neck width, then the up/down resize.
nonisolated final class DepthAnythingReassembleLayer: Module {
    let projection: Conv2d
    let resize: DepthAnythingReassembleResize

    init(config: DepthAnythingConfig, channels: Int, factor: Float) {
        self.projection = Conv2d(inputChannels: config.reassembleHiddenSize,
                                 outputChannels: channels,
                                 kernelSize: 1,
                                 stride: 1,
                                 padding: 0,
                                 bias: true)
        self.resize = DepthAnythingReassembleResize(channels: channels, factor: factor)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { resize(projection(x)) }
}

/// `DepthAnythingReassembleStage`: patch tokens (CLS dropped) → image-like maps at four resolutions.
nonisolated final class DepthAnythingReassembleStage: Module {
    let layers: [DepthAnythingReassembleLayer]

    init(config: DepthAnythingConfig) {
        self.layers = zip(config.neckHiddenSizes, config.reassembleFactors).map { channels, factor in
            DepthAnythingReassembleLayer(config: config, channels: channels, factor: factor)
        }
        super.init()
    }

    func callAsFunction(_ hiddenStates: [MLXArray], patchHeight: Int, patchWidth: Int) -> [MLXArray] {
        var out: [MLXArray] = []
        for (index, hidden) in hiddenStates.enumerated() {
            let batch = hidden.dim(0)
            let channels = hidden.dim(2)
            let patches = hidden[0..., 1..., 0...]
            let grid = patches.reshaped(batch, patchHeight, patchWidth, channels)
            out.append(layers[index](grid))
        }
        return out
    }
}

/// `DepthAnythingPreActResidualLayer`: `conv2(relu(conv1(relu(x)))) + x`.
nonisolated final class DepthAnythingPreActResidualLayer: Module {
    let convolution1: Conv2d
    let convolution2: Conv2d

    init(features: Int) {
        self.convolution1 = Conv2d(inputChannels: features, outputChannels: features,
                                   kernelSize: 3, stride: 1, padding: 1, bias: true)
        self.convolution2 = Conv2d(inputChannels: features, outputChannels: features,
                                   kernelSize: 3, stride: 1, padding: 1, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = convolution2(relu(convolution1(relu(x))))
        return hidden + x
    }
}

/// `DepthAnythingFeatureFusionLayer` — Trap 5: the residual mismatch resize is bilinear with
/// `align_corners=False`, the fusion output resize is bilinear with `align_corners=True`.
nonisolated final class DepthAnythingFeatureFusionLayer: Module {
    let projection: Conv2d
    let residualLayer1: DepthAnythingPreActResidualLayer
    let residualLayer2: DepthAnythingPreActResidualLayer

    init(features: Int) {
        self.projection = Conv2d(inputChannels: features, outputChannels: features,
                                 kernelSize: 1, stride: 1, padding: 0, bias: true)
        self.residualLayer1 = DepthAnythingPreActResidualLayer(features: features)
        self.residualLayer2 = DepthAnythingPreActResidualLayer(features: features)
        super.init()
    }

    func callAsFunction(_ hiddenState: MLXArray, residual: MLXArray?,
                        size: (height: Int, width: Int)?) -> MLXArray {
        var hidden = hiddenState
        if var residual {
            if residual.shape != hidden.shape {
                residual = DepthAnythingOps.resizeBilinear(residual, outH: hidden.dim(1), outW: hidden.dim(2),
                                                           alignCorners: false)
            }
            hidden = hidden + residualLayer1(residual)
        }
        hidden = residualLayer2(hidden)
        let outHeight = size?.height ?? hidden.dim(1) * 2
        let outWidth = size?.width ?? hidden.dim(2) * 2
        hidden = DepthAnythingOps.resizeBilinear(hidden, outH: outHeight, outW: outWidth, alignCorners: true)
        return projection(hidden)
    }
}

/// `DepthAnythingFeatureFusionStage` — smallest feature first, each layer resized to the next level's size.
nonisolated final class DepthAnythingFeatureFusionStage: Module {
    let layers: [DepthAnythingFeatureFusionLayer]

    init(config: DepthAnythingConfig) {
        self.layers = (0 ..< config.neckHiddenSizes.count).map { _ in
            DepthAnythingFeatureFusionLayer(features: config.fusionHiddenSize)
        }
        super.init()
    }

    func callAsFunction(_ features: [MLXArray]) -> [MLXArray] {
        let reversed = features.reversed().map { $0 }
        var fused: [MLXArray] = []
        var previous: MLXArray?
        for (index, hidden) in reversed.enumerated() {
            let size: (height: Int, width: Int)? = index != reversed.count - 1
                ? (reversed[index + 1].dim(1), reversed[index + 1].dim(2))
                : nil
            let result: MLXArray
            if let previous {
                result = layers[index](previous, residual: hidden, size: size)
            } else {
                result = layers[index](hidden, residual: nil, size: size)
            }
            previous = result
            fused.append(result)
        }
        return fused
    }
}

/// `DepthAnythingNeck`: reassemble → per-level 3 x 3 projection to 64 → fusion.
nonisolated final class DPTNeck: Module {
    let reassembleStage: DepthAnythingReassembleStage
    let convs: [Conv2d]
    let fusionStage: DepthAnythingFeatureFusionStage

    init(config: DepthAnythingConfig) {
        self.reassembleStage = DepthAnythingReassembleStage(config: config)
        self.convs = config.neckHiddenSizes.map { channels in
            Conv2d(inputChannels: channels, outputChannels: config.fusionHiddenSize,
                   kernelSize: 3, stride: 1, padding: 1, bias: false)
        }
        self.fusionStage = DepthAnythingFeatureFusionStage(config: config)
        super.init()
    }

    func callAsFunction(_ features: [MLXArray], patchHeight: Int, patchWidth: Int) -> [MLXArray] {
        let reassembled = reassembleStage(features, patchHeight: patchHeight, patchWidth: patchWidth)
        let projected = zip(reassembled, convs).map { feature, conv in conv(feature) }
        return fusionStage(projected)
    }
}
