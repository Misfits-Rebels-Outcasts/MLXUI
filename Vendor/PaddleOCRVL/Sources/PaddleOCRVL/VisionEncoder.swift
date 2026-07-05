import Foundation
import MLX
import MLXNN
import MLXFast

public class VisionMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    public init(hiddenSize: Int, intermediateSize: Int) {
        self._fc1.wrappedValue = Linear(hiddenSize, intermediateSize)
        self._fc2.wrappedValue = Linear(intermediateSize, hiddenSize)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // PaddleOCR-VL's SigLIP uses precise (erf) GELU, not the tanh approximation.
        fc2(gelu(fc1(x)))
    }
}

/// Rotate the last dim's halves: `[-x2, x1]` (Qwen2-VL / SigLIP 2D-RoPE convention).
private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let d = x.dim(-1)
    let x1 = x[.ellipsis, 0 ..< (d / 2)]
    let x2 = x[.ellipsis, (d / 2) ..< d]
    return concatenated([-x2, x1], axis: -1)
}

public class VisionAttention: Module {
    let numHeads: Int
    let scale: Float
    let headDim: Int

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    public init(config: PaddleOCRVLVisionConfig) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = pow(Float(headDim), -0.5)

        self._qkv.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 3, bias: true)
        self._proj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)

        super.init()
    }

    /// `cos`/`sin` are `[seqLen, headDim]` 2D-RoPE tables (nil to disable RoPE).
    public func callAsFunction(_ hiddenStates: MLXArray, cos: MLXArray?, sin: MLXArray?) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let seqLen = hiddenStates.dim(1)

        var qkvOut = qkv(hiddenStates)
        qkvOut = qkvOut.reshaped(batchSize, seqLen, 3, numHeads, headDim)
        qkvOut = qkvOut.transposed(2, 0, 3, 1, 4)  // [3, B, heads, seq, headDim]

        var q = qkvOut[0]
        var k = qkvOut[1]
        let v = qkvOut[2]

        if let cos, let sin {
            // Broadcast [seq, headDim] → [1, 1, seq, headDim].
            let c = cos.reshaped(1, 1, seqLen, headDim)
            let s = sin.reshaped(1, 1, seqLen, headDim)
            q = q * c + rotateHalf(q) * s
            k = k * c + rotateHalf(k) * s
        }

        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none
        )

        let output = attnOutput
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, seqLen, -1)

        return proj(output)
    }
}

public class VisionEncoderLayer: Module {
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "self_attn") var attn: VisionAttention
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: VisionMLP

    public init(config: PaddleOCRVLVisionConfig) {
        self._layerNorm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._attn.wrappedValue = VisionAttention(config: config)
        self._layerNorm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._mlp.wrappedValue = VisionMLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
    }

    // Pre-LN: x + attn(ln1(x)); x + mlp(ln2(x)).
    public func callAsFunction(_ hiddenStates: MLXArray, cos: MLXArray?, sin: MLXArray?) -> MLXArray {
        var h = hiddenStates + attn(layerNorm1(hiddenStates), cos: cos, sin: sin)
        h = h + mlp(layerNorm2(h))
        return h
    }
}

public class PatchEmbedding: Module {
    @ModuleInfo(key: "proj") public var projection: Conv2d

    let patchSize: Int
    let hiddenSize: Int

    public init(config: PaddleOCRVLVisionConfig) {
        self.patchSize = config.patchSize
        self.hiddenSize = config.hiddenSize

        self._projection.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrPair(config.patchSize),
            stride: IntOrPair(config.patchSize)
        )
    }

    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        var patches = projection(pixelValues)
        let (b, h, w, c) = (patches.dim(0), patches.dim(1), patches.dim(2), patches.dim(3))
        patches = patches.reshaped(b, h * w, c)
        return patches
    }
}

public class NaViTVisionEncoder: Module {
    @ModuleInfo(key: "patch_embed") public var patchEmbed: PatchEmbedding
    @ModuleInfo(key: "layers") var layers: [VisionEncoderLayer]
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

    /// Learned absolute position table `[numPositions, hidden]` (dequantized at load; no cls token).
    var positionEmbedding: MLXArray?
    /// Kept for API compatibility; PaddleOCR-VL 1.5 has no class token.
    var classEmbedding: MLXArray?

    let config: PaddleOCRVLVisionConfig
    let patchSize: Int
    let hiddenSize: Int
    let numHeads: Int
    let headDim: Int
    // 2D-RoPE inverse frequencies for a `headDim/2`-wide rotation (half for height, half for width).
    let ropeInvFreq: [Float]

    public init(config: PaddleOCRVLVisionConfig) {
        self.config = config
        self.patchSize = config.patchSize
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads

        // RoPE rotates headDim total; each spatial axis contributes headDim/2, whose half-count of
        // frequencies is headDim/4. inv_freq[j] = 1 / theta^(2j / (headDim/2)).
        let rotaryDim = headDim / 2
        let theta: Float = 10_000
        var invFreq: [Float] = []
        var j = 0
        while j < rotaryDim {
            invFreq.append(1.0 / pow(theta, Float(j) / Float(rotaryDim)))
            j += 2
        }
        self.ropeInvFreq = invFreq

        self._patchEmbed.wrappedValue = PatchEmbedding(config: config)

        var visionLayers: [VisionEncoderLayer] = []
        for _ in 0 ..< config.numHiddenLayers {
            visionLayers.append(VisionEncoderLayer(config: config))
        }
        self._layers.wrappedValue = visionLayers

        self._postLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)

        super.init()
    }

    /// Bilinearly interpolate the learned position table from its base √N×√N grid to `height×width`.
    private func interpolatedPositionEmbedding(height: Int, width: Int) -> MLXArray {
        guard let pos = positionEmbedding else {
            return MLXArray.zeros([1, height * width, hiddenSize])
        }
        let numPositions = pos.dim(0)
        let baseSize = Int(Double(numPositions).squareRoot().rounded())
        if height == baseSize && width == baseSize {
            return pos.reshaped(1, numPositions, hiddenSize)
        }
        let grid = pos.reshaped(1, baseSize, baseSize, hiddenSize)  // NHWC
        // `Upsample` truncates output size from a fractional scale, so overshoot by ~1 then crop
        // to the exact target grid — avoids an off-by-one shape mismatch against the patches.
        let scaleH = (Float(height) + 1) / Float(baseSize)
        let scaleW = (Float(width) + 1) / Float(baseSize)
        let upsample = Upsample(scaleFactor: [scaleH, scaleW], mode: .linear(alignCorners: false))
        var resized = upsample(grid)
        resized = resized[0..., 0 ..< height, 0 ..< width, 0...]
        return resized.reshaped(1, height * width, hiddenSize)
    }

    /// Build 2D-RoPE `cos`/`sin` tables `[seq, headDim]` for a row-major `height×width` grid.
    private func ropeTables(height: Int, width: Int) -> (cos: MLXArray, sin: MLXArray) {
        let seq = height * width
        var hIds = [Float](repeating: 0, count: seq)
        var wIds = [Float](repeating: 0, count: seq)
        for i in 0 ..< seq {
            hIds[i] = Float(i / width)
            wIds[i] = Float(i % width)
        }
        let invFreq = MLXArray(ropeInvFreq).reshaped(1, ropeInvFreq.count)  // [1, headDim/4]
        let hPos = MLXArray(hIds).reshaped(seq, 1)
        let wPos = MLXArray(wIds).reshaped(seq, 1)
        let freqsH = hPos * invFreq   // [seq, headDim/4]
        let freqsW = wPos * invFreq   // [seq, headDim/4]
        var emb = concatenated([freqsH, freqsW], axis: 1)  // [seq, headDim/2]
        emb = concatenated([emb, emb], axis: 1)            // [seq, headDim]
        return (cos(emb), sin(emb))
    }

    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let height = pixelValues.dim(1) / patchSize
        let width = pixelValues.dim(2) / patchSize

        var hiddenStates = patchEmbed(pixelValues)
        hiddenStates = hiddenStates + interpolatedPositionEmbedding(height: height, width: width)

        let (cosT, sinT) = ropeTables(height: height, width: width)

        for layer in layers {
            hiddenStates = layer(hiddenStates, cos: cosT, sin: sinT)
        }

        return postLayerNorm(hiddenStates)
    }

    /// Full patch sequence (no cls token to drop): `[B, gridH*gridW, hidden]`.
    public func getImageFeatures(_ pixelValues: MLXArray) -> MLXArray {
        callAsFunction(pixelValues)
    }
}
