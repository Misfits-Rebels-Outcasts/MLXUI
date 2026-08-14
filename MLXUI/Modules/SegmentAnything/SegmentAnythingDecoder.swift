import Foundation
import MLX
import MLXNN

// MARK: - Prompt Encoder

/// Encodes point prompts into token embeddings.
/// Weight keys: `tracker_model.prompt_encoder.point_embed.*`, `no_mask_embed.*`
nonisolated final class SAM3PromptEncoder: Module {
    let pointEmbed: Embedding     // [2, promptDim]
    let noMaskEmbed: Embedding    // [1, promptDim]
    let promptDim: Int

    init(promptDim: Int = 256) {
        self.promptDim = promptDim
        pointEmbed  = Embedding(embeddingCount: 2, dimensions: promptDim)
        noMaskEmbed = Embedding(embeddingCount: 1, dimensions: promptDim)
        super.init()
    }

    /// Returns [1, N+1, promptDim]
    func callAsFunction(points: MLXArray, labels: MLXArray) -> MLXArray {
        let ptTokens  = pointEmbed(labels) + encodePoints(points)
        let noMaskTok = noMaskEmbed(MLXArray([Int32(0)]))
        return concatenated([ptTokens, noMaskTok], axis: 0).expandedDimensions(axis: 0)
    }

    private func encodePoints(_ points: MLXArray) -> MLXArray {
        let half  = promptDim / 2
        // Build frequency array as [Float] then wrap in MLXArray
        let freqData = (0 ..< half).map { Float($0) * Float.pi / Float(half) }
        let freqs = MLXArray(freqData)   // [half]
        let px = points[0..., 0].expandedDimensions(axis: -1) * freqs
        let py = points[0..., 1].expandedDimensions(axis: -1) * freqs
        return concatenated([sin(px), sin(py)], axis: -1)   // [N, promptDim]
    }
}

// MARK: - Pixel Decoder

/// Convolutional upsampling decoder.
/// Weight keys: `detector_model.mask_decoder.pixel_decoder.conv_layers.N.*`,
///              `mask_embedder.*`, `instance_projection.*`
nonisolated final class SAM3PixelDecoder: Module {
    let convLayers: [Conv2d]
    let maskEmbedder: Linear
    let instanceProjection: Linear
    let numMasks: Int

    init(inChannels: Int = 256, numMasks: Int = 3) {
        self.numMasks = numMasks
        var cs: [Conv2d] = []
        var c = inChannels
        for _ in 0 ..< 3 {
            let out = max(c / 2, 32)
            cs.append(Conv2d(inputChannels: c, outputChannels: out,
                             kernelSize: .init(3), padding: .init(1)))
            c = out
        }
        convLayers         = cs
        maskEmbedder       = Linear(c, numMasks)
        instanceProjection = Linear(inChannels, inChannels)
        super.init()
    }

    /// [B, H, W, C] → [B, H×8, W×8, numMasks]  (3 × 2× nearest-neighbour upsample)
    func callAsFunction(_ imageEmbed: MLXArray, promptTokens: MLXArray) -> MLXArray {
        let q = instanceProjection(promptTokens.mean(axis: 1))   // [B, D]
        var h = imageEmbed + q.reshaped([q.dim(0), 1, 1, q.dim(1)])
        for conv in convLayers {
            h = MLXNN.relu(conv(upsample2x(h)))
        }
        return maskEmbedder(h)
    }

    /// Nearest-neighbour 2× spatial upsample using `tiled`.
    private func upsample2x(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        // Insert dummy axes, tile ×2 along each spatial dim, then collapse.
        let expanded = x.reshaped([b, h, 1, w, 1, c])
        let tiled    = MLX.tiled(expanded, repetitions: [1, 1, 2, 1, 2, 1])
        return tiled.reshaped([b, h * 2, w * 2, c])
    }
}

// MARK: - IoU Predictor

/// Weight keys: `detector_model.mask_decoder.iou_predictor.layers.N.*`
nonisolated final class SAM3IoUPredictor: Module {
    let layers: [Linear]

    init(inChannels: Int = 256, numMasks: Int = 3) {
        layers = [Linear(inChannels, inChannels), Linear(inChannels, numMasks)]
        super.init()
    }

    func callAsFunction(_ imageEmbed: MLXArray) -> MLXArray {
        var h = imageEmbed.mean(axes: [1, 2])
        for (i, layer) in layers.enumerated() {
            h = layer(h)
            if i < layers.count - 1 { h = MLXNN.relu(h) }
        }
        return sigmoid(h)
    }
}
