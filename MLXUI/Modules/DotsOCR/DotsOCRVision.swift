import Foundation
import MLX
import MLXNN
import MLXFast

/// The `dots_vit` NaViT vision tower for dots.ocr / dots.mocr — a from-scratch Swift/MLX port of
/// `Blaizzy/mlx-vlm`'s `mlx_vlm/models/dots_ocr/vision.py`. Structurally a Qwen2-VL vision tower
/// (patch-embed → RoPE blocks → PatchMerger) but with a Conv2d+RMSNorm patch embed, SwiGLU FFN,
/// RMSNorm blocks, an optional post-trunk norm, and a **LayerNorm** merger `ln_q`.
///
/// **Single-frame assumption:** dots.ocr is a document-OCR model driven with one image, so the
/// Python `cu_seqlens` block-diagonal attention degenerates to full self-attention over the whole
/// patch sequence (one frame → one block). We therefore run plain full attention (mask `.none`),
/// which is numerically identical for a single frame. (Multi-image/video would need the block mask.)
///
/// All classes are `nonisolated` because the app target defaults to MainActor isolation while MLX's
/// `Module` is nonisolated (ModernBERT lesson, journal 2026-40/41).

// MARK: - Math helpers (static so they stay off the main actor)

private enum DotsVisionMath {
    /// Rotates the second half of the last dim to the front, negated (RoPE `rotate_half`).
    nonisolated static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let parts = split(x, parts: 2, axis: -1)
        return concatenated([-parts[1], parts[0]], axis: -1)
    }

    /// `apply_rotary_pos_emb_vision`: tile cos/sin ×2 across the head dim and rotate.
    /// `tensor` is `(seq, heads, headDim)`; `freqs` is `(seq, headDim/2)`.
    nonisolated static func applyRope(_ tensor: MLXArray, freqs: MLXArray) -> MLXArray {
        var cosF = cos(freqs)
        var sinF = sin(freqs)
        cosF = expandedDimensions(cosF, axis: 1)
        cosF = tiled(cosF, repetitions: [1, 1, 2])
        cosF = expandedDimensions(cosF, axis: 0)
        sinF = expandedDimensions(sinF, axis: 1)
        sinF = tiled(sinF, repetitions: [1, 1, 2])
        sinF = expandedDimensions(sinF, axis: 0)
        let out = (tensor * cosF) + (rotateHalf(tensor) * sinF)
        return out.asType(tensor.dtype)
    }
}

// MARK: - Rotary embedding

nonisolated final class DotsVisionRotaryEmbedding: Module {
    let dim: Int
    let theta: Float

    init(dim: Int, theta: Float = 10_000) {
        self.dim = dim
        self.theta = theta
        super.init()
    }

    func callAsFunction(_ sequenceLength: Int) -> MLXArray {
        let exponent = MLXArray(Array(stride(from: 0, to: dim, by: 2))).asType(.float32) / Float(dim)
        let invFreq = 1.0 / pow(MLXArray(theta), exponent)          // (dim/2)
        let seq = MLXArray(0 ..< sequenceLength).asType(.float32)   // (seqlen)
        return outer(seq, invFreq)                                  // (seqlen, dim/2)
    }
}

// MARK: - Patch embed (Conv2d + RMSNorm)

nonisolated final class DotsPatchEmbed: Module {
    let numChannels: Int
    let patchSize: Int
    let temporalPatchSize: Int
    let embedDim: Int

    @ModuleInfo(key: "proj") var proj: Conv2d
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: DotsOCRConfig.Vision) {
        self.numChannels = config.numChannels
        self.patchSize = config.patchSize
        self.temporalPatchSize = config.temporalPatchSize
        self.embedDim = config.embedDim
        self._proj.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.embedDim,
            kernelSize: IntOrPair(config.patchSize),
            stride: IntOrPair(config.patchSize),
            bias: true)
        self._norm.wrappedValue = RMSNorm(dimensions: config.embedDim, eps: config.rmsNormEps)
        super.init()
    }

    /// `x` is the pre-patchified pixel tensor `(numPatches, C * T * P * P)`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var p = x.reshaped(-1, numChannels, temporalPatchSize, patchSize, patchSize)
        p = p[0..., 0..., 0]                    // select the single temporal slice → (N, C, P, P)
        p = p.transposed(0, 2, 3, 1)            // NHWC for MLX Conv2d → (N, P, P, C)
        p = proj(p).reshaped(-1, embedDim)      // (N, 1, 1, embed) → (N, embed)
        return norm(p)
    }
}

nonisolated final class DotsViTPreprocessor: Module {
    @ModuleInfo(key: "patchifier") var patchifier: DotsPatchEmbed

    init(_ config: DotsOCRConfig.Vision) {
        self._patchifier.wrappedValue = DotsPatchEmbed(config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { patchifier(x) }
}

// MARK: - Attention / MLP / block

nonisolated final class DotsVisionAttention: Module {
    let numHeads: Int
    let scale: Float

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    init(_ config: DotsOCRConfig.Vision) {
        self.numHeads = config.numAttentionHeads
        let headDim = config.embedDim / config.numAttentionHeads
        self.scale = pow(Float(headDim), -0.5)
        self._qkv.wrappedValue = Linear(config.embedDim, config.embedDim * 3, bias: config.useBias)
        self._proj.wrappedValue = Linear(config.embedDim, config.embedDim, bias: config.useBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, freqs: MLXArray) -> MLXArray {
        let seqLen = x.dim(0)
        let s = split(qkv(x), parts: 3, axis: -1)
        var q = s[0].reshaped(seqLen, numHeads, -1)
        var k = s[1].reshaped(seqLen, numHeads, -1)
        var v = s[2].reshaped(seqLen, numHeads, -1)

        q = DotsVisionMath.applyRope(q, freqs: freqs)
        k = DotsVisionMath.applyRope(k, freqs: freqs)

        q = q.reshaped(1, seqLen, numHeads, -1).transposed(0, 2, 1, 3)
        k = k.reshaped(1, seqLen, numHeads, -1).transposed(0, 2, 1, 3)
        v = v.reshaped(1, seqLen, numHeads, -1).transposed(0, 2, 1, 3)

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none)
            .transposed(0, 2, 1, 3)
            .reshaped(seqLen, -1)
        return proj(out)
    }
}

nonisolated final class DotsSwiGLUFFN: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "fc3") var fc3: Linear

    init(_ config: DotsOCRConfig.Vision) {
        self._fc1.wrappedValue = Linear(config.embedDim, config.intermediateSize, bias: config.useBias)
        self._fc2.wrappedValue = Linear(config.intermediateSize, config.embedDim, bias: config.useBias)
        self._fc3.wrappedValue = Linear(config.embedDim, config.intermediateSize, bias: config.useBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(silu(fc1(x)) * fc3(x)) }
}

nonisolated final class DotsVisionBlock: Module {
    @ModuleInfo(key: "attn") var attn: DotsVisionAttention
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: DotsSwiGLUFFN

    init(_ config: DotsOCRConfig.Vision) {
        self._attn.wrappedValue = DotsVisionAttention(config)
        self._norm1.wrappedValue = RMSNorm(dimensions: config.embedDim, eps: config.rmsNormEps)
        self._norm2.wrappedValue = RMSNorm(dimensions: config.embedDim, eps: config.rmsNormEps)
        self._mlp.wrappedValue = DotsSwiGLUFFN(config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, freqs: MLXArray) -> MLXArray {
        var h = x + attn(norm1(x), freqs: freqs)
        h = h + mlp(norm2(h))
        return h
    }
}

// MARK: - PatchMerger (LayerNorm ln_q + GELU MLP)

nonisolated final class DotsPatchMerger: Module {
    let hiddenSize: Int
    @ModuleInfo(key: "ln_q") var lnQ: LayerNorm
    @ModuleInfo var mlp: (Linear, GELU, Linear)

    init(dim: Int, contextDim: Int, spatialMergeSize: Int) {
        self.hiddenSize = contextDim * (spatialMergeSize * spatialMergeSize)
        self._lnQ.wrappedValue = LayerNorm(dimensions: contextDim, eps: 1e-6)
        self.mlp = (Linear(hiddenSize, hiddenSize), GELU(), Linear(hiddenSize, dim))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = lnQ(x).reshaped(-1, hiddenSize)
        y = mlp.0(y)
        y = mlp.1(y)
        y = mlp.2(y)
        return y
    }
}

// MARK: - Vision model

nonisolated final class DotsVisionModel: Module {
    let spatialMergeSize: Int
    let postNorm: Bool

    @ModuleInfo(key: "patch_embed") var patchEmbed: DotsViTPreprocessor
    @ModuleInfo(key: "rotary_pos_emb") var rotaryPosEmb: DotsVisionRotaryEmbedding
    @ModuleInfo(key: "blocks") var blocks: [DotsVisionBlock]
    @ModuleInfo(key: "post_trunk_norm") var postTrunkNorm: RMSNorm
    @ModuleInfo(key: "merger") var merger: DotsPatchMerger

    init(_ config: DotsOCRConfig.Vision) {
        self.spatialMergeSize = config.spatialMergeSize
        self.postNorm = config.postNorm

        self._patchEmbed.wrappedValue = DotsViTPreprocessor(config)
        let headDim = config.embedDim / config.numAttentionHeads
        self._rotaryPosEmb.wrappedValue = DotsVisionRotaryEmbedding(dim: headDim / 2, theta: 10_000)
        self._blocks.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in DotsVisionBlock(config) }
        self._postTrunkNorm.wrappedValue = RMSNorm(dimensions: config.embedDim, eps: config.rmsNormEps)
        self._merger.wrappedValue = DotsPatchMerger(
            dim: config.hiddenSize, contextDim: config.embedDim, spatialMergeSize: config.spatialMergeSize)
        super.init()
    }

    /// Per-patch rotary frequencies in merge-block order (mirrors `get_pos_ids_by_grid` + `rot_pos_emb`).
    private func rotaryFrequencies(_ gridTHW: [(t: Int, h: Int, w: Int)]) -> MLXArray {
        let m = spatialMergeSize
        var positionIds = [MLXArray]()
        for (t, h, w) in gridTHW {
            var hpos = expandedDimensions(MLXArray(0 ..< h), axis: 1)   // (h,1)
            hpos = repeated(hpos, count: w, axis: 1)                    // (h,w)
            hpos = hpos.reshaped(h / m, m, w / m, m).transposed(0, 2, 1, 3).flattened()

            var wpos = expandedDimensions(MLXArray(0 ..< w), axis: 0)   // (1,w)
            wpos = repeated(wpos, count: h, axis: 0)                    // (h,w)
            wpos = wpos.reshaped(h / m, m, w / m, m).transposed(0, 2, 1, 3).flattened()

            let stackedPos = stacked([hpos, wpos], axis: -1)           // (h*w, 2)
            positionIds.append(tiled(stackedPos, repetitions: [t, 1]))
        }
        let indices = concatenated(positionIds, axis: 0)                // (seq, 2)
        let maxGrid = gridTHW.map { max($0.h, $0.w) }.max() ?? 0
        let full = rotaryPosEmb(maxGrid)                                // (maxGrid, rotDim)
        let emb = full[indices]                                         // (seq, 2, rotDim)
        return emb.reshaped(indices.dim(0), -1)                         // (seq, 2*rotDim)
    }

    /// `pixelValues`: `(numPatches, C*T*P*P)`. Returns merged image tokens `(numPatches/merge², hidden)`.
    func callAsFunction(_ pixelValues: MLXArray, gridTHW: [(t: Int, h: Int, w: Int)]) -> MLXArray {
        var h = patchEmbed(pixelValues)
        let freqs = rotaryFrequencies(gridTHW)
        for block in blocks {
            h = block(h, freqs: freqs)
        }
        if postNorm {
            h = postTrunkNorm(h)
        }
        return merger(h)
    }

    // MARK: Weight sanitize (conv-weight transpose + drop position_ids)

    /// Mirrors `vision.py`'s `sanitize`: transpose the PyTorch conv weight `(out,in,kH,kW)` to MLX
    /// `(out,kH,kW,in)` unless it's already MLX-shaped, and drop any `position_ids` buffers. Operates
    /// on the full (already key-remapped) weight dict — keys are matched by suffix.
    nonisolated static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        for (key, value) in weights {
            if key.contains("position_ids") { continue }
            if key.hasSuffix("patch_embed.patchifier.proj.weight") {
                out[key] = isMLXConvShape(value) ? value : value.transposed(0, 2, 3, 1)
            } else {
                out[key] = value
            }
        }
        return out
    }

    /// `check_array_shape`: true when the conv weight is already in MLX `(out,kH,kW,in)` layout.
    nonisolated static func isMLXConvShape(_ a: MLXArray) -> Bool {
        guard a.ndim == 4 else { return false }
        let outC = a.dim(0), kH = a.dim(1), kW = a.dim(2)
        return outC >= kH && outC >= kW && kH == kW
    }
}
