import Foundation
import MLX
import MLXNN
import MLXFast

/// The SDXL UNet (`UNet2DConditionModel`) for `stabilityai/sdxl-turbo`, ported idiom-for-idiom
/// from `ml-explore/mlx-examples/stable_diffusion`'s `unet.py` + diffusers v0.24.0
/// (`unet_2d_condition.py` / `embeddings.py`). Structure verified against the actual checkpoint
/// safetensors header (SD-ENG3).
///
/// Checkpoint facts (from the header):
/// - fp16 weights → plain `Conv2d`/`Linear` (no 4-bit quant); **PyTorch conv layout** `(out,in,kH,kW)`
///   → run through `SDWeights.sanitize` before `update(parameters:)`;
/// - channels `[320, 640, 1280]`, cross-attn 2048, `use_linear_projection` (Transformer2D uses
///   Linear proj_in/proj_out), resnets carry a `time_emb_proj` temb MLP;
/// - `transformer_layers_per_block [1, 2, 10]`: down blocks use `[i]` (down.1 → 2, down.2 → 10),
///   **mid uses the last (10)**, up blocks use the reversed list (up.0 → 10, up.1 → 2);
/// - block types: `down_block_types [DownBlock2D, CrossAttnDownBlock2D, CrossAttnDownBlock2D]`
///   (down.0 has NO attention), `up_block_types [CrossAttnUpBlock2D, CrossAttnUpBlock2D, UpBlock2D]`
///   (up.2 has NO attention);
/// - added conditioning (`addition_embed_type "text_time"`): the diffusers
///   `add_embedding.linear_1` input is `2816` = **text_encoder_2 pooled (1280)** +
///   **6 time_ids × 256** (`get_timestep_embedding`, `flip_sin_to_cos`, `downscale_freq_shift 0`).
///   `time_ids = [orig_h, orig_w, crop_h, crop_w, target_h, target_w]` flattened before the proj.
///
/// SDXL-Turbo runs **no CFG** (guidance 0) → single batch, one forward pass; no negative prompt.

/// Config for the SDXL UNet. `SDUNetConfig.sdxl` is the real model; the shape test builds a tiny
/// variant with small channels (the full model is ~2.6B params — too heavy for a unit gate).
nonisolated struct SDUNetConfig: Sendable {
    let inChannels: Int
    let outChannels: Int
    let blockOutChannels: [Int]
    let layersPerBlock: [Int]
    /// Per down block (`[1, 2, 10]`): down blocks use `[i]`, mid uses the last, up blocks use the
    /// reversed list (matching the checkpoint).
    let transformerLayersPerBlock: [Int]
    let crossAttentionDim: Int
    let attentionHeadDim: Int
    /// Sinusoidal time-projection dim for the denoise timestep (= block_out_channels[0] for SDXL).
    let timeEmbedDim: Int
    /// Sinusoidal dim for each of the 6 time ids in the added conditioning.
    let addTimeEmbedDim: Int
    /// Pooled text-encoder-2 dim (the added-conditioning text half, 1280 for SDXL).
    let textEmbedDim: Int
    let normGroups: Int
    let normEps: Float

    var tembChannels: Int { blockOutChannels[0] * 4 }
    /// diffusers `addition_time_embed_dim * len(time_ids) + text_encoder_projection_dim` = 2816.
    var additionEmbedDim: Int { textEmbedDim + addTimeEmbedDim * 6 }
    /// Attention heads per block = channels / head dim (SDXL: 5, 10, 20).
    func heads(for channels: Int) -> Int { channels / attentionHeadDim }

    static let sdxl = SDUNetConfig(
        inChannels: 4, outChannels: 4, blockOutChannels: [320, 640, 1280],
        layersPerBlock: [2, 2, 2], transformerLayersPerBlock: [1, 2, 10],
        crossAttentionDim: 2048, attentionHeadDim: 64,
        timeEmbedDim: 320, addTimeEmbedDim: 256, textEmbedDim: 1280,
        normGroups: 32, normEps: 1e-5)
}

/// The full SDXL UNet. Input latent is **NHWC** `(B, H, W, 4)` (MLX conv layout); output is the
/// 4-channel noise prediction, same shape.
nonisolated final class SDUNet: Module {
    let config: SDUNetConfig

    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "time_embedding") var timeEmbedding: SDTimestepEmbed
    @ModuleInfo(key: "add_embedding") var addEmbedding: SDTimestepEmbed
    @ModuleInfo(key: "down_blocks") var downBlocks: [SDDownBlock]
    @ModuleInfo(key: "mid_block") var midBlock: SDMidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [SDUpBlock]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(config: SDUNetConfig) {
        self.config = config
        let c = config
        self._convIn.wrappedValue = Conv2d(inputChannels: c.inChannels, outputChannels: c.blockOutChannels[0], kernelSize: 3, padding: 1)
        self._timeEmbedding.wrappedValue = SDTimestepEmbed(inputDim: c.timeEmbedDim, timeEmbedDim: c.tembChannels)
        self._addEmbedding.wrappedValue = SDTimestepEmbed(inputDim: c.additionEmbedDim, timeEmbedDim: c.tembChannels)

        let downCount = c.blockOutChannels.count
        let topChannels = c.blockOutChannels[downCount - 1]

        // Down path + the residual channel stack the up path will pop skips from. Each entry is
        // the OUTPUT channel count of a state the down path pushes (resnet outputs + downsample),
        // matching the runtime residual stack exactly.
        var downs: [SDDownBlock] = []
        var prevOut = c.blockOutChannels[0]
        var residualChannels: [Int] = [c.blockOutChannels[0]]   // conv_in output
        for i in 0 ..< downCount {
            let out = c.blockOutChannels[i]
            let layers = c.layersPerBlock[i]
            var resnetIns = [prevOut] + Array(repeating: out, count: layers - 1)
            downs.append(SDDownBlock(
                resnetInChannels: resnetIns, outChannels: out, tembChannels: c.tembChannels,
                hasCrossAttention: i > 0, transformerLayers: c.transformerLayersPerBlock[i],
                crossAttentionDim: c.crossAttentionDim, headDim: c.attentionHeadDim,
                normGroups: c.normGroups, normEps: c.normEps,
                hasDownsample: i < downCount - 1))
            residualChannels.append(contentsOf: Array(repeating: out, count: layers))   // resnets
            if i < downCount - 1 { residualChannels.append(out) }                        // downsampler
            prevOut = out
        }
        self._downBlocks.wrappedValue = downs

        self._midBlock.wrappedValue = SDMidBlock(
            channels: topChannels, tembChannels: c.tembChannels,
            transformerLayers: c.transformerLayersPerBlock[downCount - 1],
            crossAttentionDim: c.crossAttentionDim, headDim: c.attentionHeadDim,
            normGroups: c.normGroups, normEps: c.normEps)

        // Up path: reversed channels; resnet input = base + the skip popped off the stack.
        var ups: [SDUpBlock] = []
        var skipIndex = residualChannels.count
        for i in 0 ..< downCount {
            let out = c.blockOutChannels[downCount - 1 - i]
            let layers = c.layersPerBlock[downCount - 1 - i] + 1
            var resnetIns: [Int] = []
            for j in 0 ..< layers {
                skipIndex -= 1
                let base = (j == 0) ? (i == 0 ? topChannels : c.blockOutChannels[downCount - i]) : out
                resnetIns.append(base + residualChannels[skipIndex])
            }
            ups.append(SDUpBlock(
                resnetInChannels: resnetIns, outChannels: out, tembChannels: c.tembChannels,
                hasCrossAttention: i < downCount - 1, transformerLayers: c.transformerLayersPerBlock[downCount - 1 - i],
                crossAttentionDim: c.crossAttentionDim, headDim: c.attentionHeadDim,
                normGroups: c.normGroups, normEps: c.normEps,
                hasUpsample: i < downCount - 1))
        }
        self._upBlocks.wrappedValue = ups

        self._convNormOut.wrappedValue = GroupNorm(groupCount: c.normGroups, dimensions: c.blockOutChannels[0], eps: c.normEps, affine: true, pytorchCompatible: true)
        self._convOut.wrappedValue = Conv2d(inputChannels: c.blockOutChannels[0], outputChannels: c.outChannels, kernelSize: 3, padding: 1)
        super.init()
    }

    /// diffusers `get_timestep_embedding` (v0.24.0): `exponent = -log(max_period)·arange(half)/half`
    /// (downscale_freq_shift 0), `[sin, cos]`, optionally swap the halves. No weights.
    static func timestepEmbedding(_ t: MLXArray, dim: Int, flipSinToCos: Bool, maxPeriod: Float = 10_000) -> MLXArray {
        let half = dim / 2
        let exponent = -log(maxPeriod) * (MLXArray.arange(half).asType(.float32) / Float(half))
        let emb = exp(exponent)                                   // (half,)
        let x = t.expandedDimensions(axis: -1) * emb.expandedDimensions(axis: 0)
        var out = concatenated([sin(x), cos(x)], axis: -1)
        if flipSinToCos {
            out = concatenated([out[.ellipsis, half ..< dim], out[.ellipsis, 0 ..< half]], axis: -1)
        }
        return out
    }

    /// One unconditional forward pass (SDXL-Turbo: no CFG, no negative batch).
    /// - `sample`: NHWC latent `(B, H, W, inChannels)`
    /// - `timestep`: scalar or `(B,)` float denoise timestep
    /// - `encoderHiddenStates`: `(B, seq, crossAttentionDim)` — concat of both CLIP last-hidden states
    /// - `textEmbeds`: `(B, textEmbedDim)` — pooled text_encoder_2 output (added conditioning)
    /// - `timeIDs`: `(B, 6)` — `[orig_h, orig_w, crop_h, crop_w, target_h, target_w]`
    func callAsFunction(
        sample: MLXArray,
        timestep: MLXArray,
        encoderHiddenStates: MLXArray,
        textEmbeds: MLXArray,
        timeIDs: MLXArray
    ) -> MLXArray {
        let c = config
        let batch = sample.dim(0)

        var ts = timestep
        if ts.ndim == 0 { ts = ts.expandedDimensions(axis: 0) }
        if ts.dim(0) != batch { ts = broadcast(ts.reshaped([1]), to: [batch]) }

        var temb = timeEmbedding(Self.timestepEmbedding(ts.asType(.float32), dim: c.timeEmbedDim, flipSinToCos: true).asType(sample.dtype))

        // Added text-time conditioning: 6 time ids → 256-dim each → (B, 1536); concat pooled text → (B, 2816).
        let timeFlat = timeIDs.reshaped([-1]).asType(.float32)
        let timeEmb = Self.timestepEmbedding(timeFlat, dim: c.addTimeEmbedDim, flipSinToCos: true)
            .reshaped([batch, -1])
            .asType(sample.dtype)
        let addEmb = concatenated([textEmbeds, timeEmb], axis: -1)
        temb = temb + addEmbedding(addEmb)

        var hidden = convIn(sample)
        var residuals: [MLXArray] = [hidden]
        for block in downBlocks {
            (hidden, residuals) = block(hidden, temb: temb, context: encoderHiddenStates, residuals: residuals)
        }
        hidden = midBlock(hidden, temb: temb, context: encoderHiddenStates)
        for block in upBlocks {
            hidden = block(hidden, temb: temb, context: encoderHiddenStates, residuals: &residuals)
        }
        hidden = silu(convNormOut(hidden))
        return convOut(hidden)
    }
}

/// `TimestepEmbedding`: Linear → silu → Linear (the `time_embedding` and `add_embedding` MLPs).
nonisolated final class SDTimestepEmbed: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inputDim: Int, timeEmbedDim: Int) {
        self._linear1.wrappedValue = Linear(inputDim, timeEmbedDim, bias: true)
        self._linear2.wrappedValue = Linear(timeEmbedDim, timeEmbedDim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

/// `ResnetBlock2D` with time-embedding modulation: GroupNorm → silu → 3×3 conv (+temb) →
/// GroupNorm → silu → 3×3 conv, plus a 1×1 shortcut when the channel count changes. NHWC.
nonisolated final class SDResnetBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "time_emb_proj") var timeEmbProj: Linear
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

    init(inChannels: Int, outChannels: Int, tembChannels: Int, normGroups: Int, normEps: Float) {
        self._norm1.wrappedValue = GroupNorm(groupCount: normGroups, dimensions: inChannels, eps: normEps, affine: true, pytorchCompatible: true)
        self._conv1.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self._timeEmbProj.wrappedValue = Linear(tembChannels, outChannels, bias: true)
        self._norm2.wrappedValue = GroupNorm(groupCount: normGroups, dimensions: outChannels, eps: normEps, affine: true, pytorchCompatible: true)
        self._conv2.wrappedValue = Conv2d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self._convShortcut.wrappedValue = inChannels != outChannels
            ? Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1)
            : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray, temb: MLXArray) -> MLXArray {
        let t = timeEmbProj(silu(temb))
        var hidden = silu(norm1(x))
        hidden = conv1(hidden)
        hidden = hidden + t.expandedDimensions(axis: 1).expandedDimensions(axis: 1)
        hidden = silu(norm2(hidden))
        hidden = conv2(hidden)
        if let convShortcut {
            return convShortcut(x) + hidden
        }
        return x + hidden
    }
}

/// `Transformer2DModel` (use_linear_projection): GroupNorm → proj_in → transformer blocks →
/// proj_out → residual add. `norm`/`proj_in`/`proj_out` keyed directly under the attention module.
nonisolated final class SDTransformer2D: Module {
    @ModuleInfo(key: "norm") var norm: GroupNorm
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [SDTransformerBlock]
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(channels: Int, transformerLayers: Int, crossAttentionDim: Int, heads: Int, headDim: Int, normGroups: Int, normEps: Float) {
        self._norm.wrappedValue = GroupNorm(groupCount: normGroups, dimensions: channels, eps: normEps, affine: true, pytorchCompatible: true)
        self._projIn.wrappedValue = Linear(channels, channels, bias: true)
        self._transformerBlocks.wrappedValue = (0 ..< transformerLayers).map { _ in
            SDTransformerBlock(channels: channels, heads: heads, headDim: headDim, crossAttentionDim: crossAttentionDim, normEps: normEps)
        }
        self._projOut.wrappedValue = Linear(channels, channels, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
        let input = x
        var hidden = norm(x).reshaped([b, h * w, c])
        hidden = projIn(hidden)
        for block in transformerBlocks { hidden = block(hidden, context: context) }
        hidden = projOut(hidden).reshaped([b, h, w, c])
        return hidden + input
    }
}

/// `BasicTransformerBlock`: self-attn → cross-attn → GEGLU feed-forward, each with LayerNorm
/// pre-norm and residual adds.
nonisolated final class SDTransformerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn1") var attn1: SDAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn2") var attn2: SDAttention
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    @ModuleInfo(key: "ff") var ff: SDFeedForward

    init(channels: Int, heads: Int, headDim: Int, crossAttentionDim: Int, normEps: Float) {
        self._norm1.wrappedValue = LayerNorm(dimensions: channels, eps: normEps, affine: true, bias: true)
        self._attn1.wrappedValue = SDAttention(channels: channels, keyDim: channels, heads: heads, headDim: headDim)
        self._norm2.wrappedValue = LayerNorm(dimensions: channels, eps: normEps, affine: true, bias: true)
        self._attn2.wrappedValue = SDAttention(channels: channels, keyDim: crossAttentionDim, heads: heads, headDim: headDim)
        self._norm3.wrappedValue = LayerNorm(dimensions: channels, eps: normEps, affine: true, bias: true)
        self._ff.wrappedValue = SDFeedForward(channels: channels)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        var hidden = x + attn1(norm1(x))
        hidden = hidden + attn2(norm2(hidden), context: context)
        hidden = hidden + ff(norm3(hidden))
        return hidden
    }
}

/// `Attention` (diffusers): q from `channels`, k/v projected from `keyDim` (self-attn: keyDim ==
/// channels; cross-attn: keyDim == 2048). Reshaped to `heads × headDim`, SDPA, back out.
nonisolated final class SDAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]

    let heads: Int
    let headDim: Int

    init(channels: Int, keyDim: Int, heads: Int, headDim: Int) {
        self.heads = heads
        self.headDim = headDim
        self._toQ.wrappedValue = Linear(channels, channels, bias: true)
        self._toK.wrappedValue = Linear(keyDim, channels, bias: true)
        self._toV.wrappedValue = Linear(keyDim, channels, bias: true)
        self._toOut.wrappedValue = [Linear(channels, channels, bias: true)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray? = nil) -> MLXArray {
        let b = x.dim(0), seq = x.dim(1)
        let ctx = context ?? x
        let query = toQ(x).reshaped([b, seq, heads, headDim]).transposed(0, 2, 1, 3)
        let key = toK(ctx).reshaped([b, -1, heads, headDim]).transposed(0, 2, 1, 3)
        let value = toV(ctx).reshaped([b, -1, heads, headDim]).transposed(0, 2, 1, 3)
        let scale = 1.0 / sqrt(Float(headDim))
        let attn = scaledDotProductAttention(queries: query, keys: key, values: value, scale: scale, mask: nil)
        let out = attn.transposed(0, 2, 1, 3).reshaped([b, seq, heads * headDim])
        return toOut[0](out)
    }
}

/// diffusers `FeedForward` with GEGLU. The checkpoint keys are `ff.net.{0,2}` (ModuleList
/// `[GEGLU, Dropout, Linear]` — the dropout `net.1` has no weights); `SDWeights.sanitize` remaps
/// them to non-numeric children (`ff.net.geglu.proj` / `ff.net.out`) so `net` unflattens as a
/// dict instead of an array-with-a-hole.
nonisolated final class SDFeedForward: Module {
    @ModuleInfo(key: "net") var net: SDFeedForwardNet

    init(channels: Int) {
        self._net.wrappedValue = SDFeedForwardNet(channels: channels)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        net(x)
    }
}

nonisolated final class SDFeedForwardNet: Module {
    @ModuleInfo(key: "geglu") var geglu: SDGEGLU
    @ModuleInfo(key: "out") var out: Linear

    init(channels: Int) {
        self._geglu.wrappedValue = SDGEGLU(channels: channels)
        self._out.wrappedValue = Linear(4 * channels, channels, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        out(geglu(x))
    }
}

nonisolated final class SDGEGLU: Module {
    @ModuleInfo(key: "proj") var proj: Linear

    init(channels: Int) {
        self._proj.wrappedValue = Linear(channels, 2 * (4 * channels), bias: true)
        super.init()
    }

    /// `x * gelu(gate)` over the two halves of the 2×inner projection.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = proj(x)
        let inner = h.dim(-1) / 2
        let value = h[.ellipsis, 0 ..< inner]
        let gate = h[.ellipsis, inner ..< 2 * inner]
        return value * gelu(gate)
    }
}

/// `DownBlock2D` / `CrossAttnDownBlock2D`: resnets (with optional attention after each), then an
/// optional stride-2 downsample. Returns the block output + the residual states for the up path.
nonisolated final class SDDownBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [SDResnetBlock]
    @ModuleInfo(key: "attentions") var attentions: [SDTransformer2D]
    @ModuleInfo(key: "downsamplers") var downsamplers: [SDDownsample]

    init(resnetInChannels: [Int], outChannels: Int, tembChannels: Int, hasCrossAttention: Bool,
         transformerLayers: Int, crossAttentionDim: Int, headDim: Int,
         normGroups: Int, normEps: Float, hasDownsample: Bool) {
        self._resnets.wrappedValue = resnetInChannels.map {
            SDResnetBlock(inChannels: $0, outChannels: outChannels, tembChannels: tembChannels, normGroups: normGroups, normEps: normEps)
        }
        self._attentions.wrappedValue = hasCrossAttention
            ? resnetInChannels.map { _ in
                SDTransformer2D(channels: outChannels, transformerLayers: transformerLayers,
                                crossAttentionDim: crossAttentionDim, heads: outChannels / headDim,
                                headDim: headDim, normGroups: normGroups, normEps: normEps)
            }
            : []
        self._downsamplers.wrappedValue = hasDownsample ? [SDDownsample(channels: outChannels)] : []
        super.init()
    }

    func callAsFunction(_ x: MLXArray, temb: MLXArray, context: MLXArray, residuals: [MLXArray])
        -> (MLXArray, [MLXArray]) {
        var hidden = x
        var states = residuals
        for i in 0 ..< resnets.count {
            hidden = resnets[i](hidden, temb: temb)
            if !attentions.isEmpty { hidden = attentions[i](hidden, context: context) }
            states.append(hidden)
        }
        if let downsampler = downsamplers.first {
            hidden = downsampler(hidden)
            states.append(hidden)
        }
        return (hidden, states)
    }
}

/// `CrossAttnUpBlock2D` / `UpBlock2D`: 3 resnets that each concatenate a skip popped off the
/// residual stack, optional attention after each, then an optional nearest-2× upsample.
nonisolated final class SDUpBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [SDResnetBlock]
    @ModuleInfo(key: "attentions") var attentions: [SDTransformer2D]
    @ModuleInfo(key: "upsamplers") var upsamplers: [SDUpSample]

    init(resnetInChannels: [Int], outChannels: Int, tembChannels: Int, hasCrossAttention: Bool,
         transformerLayers: Int, crossAttentionDim: Int, headDim: Int,
         normGroups: Int, normEps: Float, hasUpsample: Bool) {
        self._resnets.wrappedValue = resnetInChannels.map {
            SDResnetBlock(inChannels: $0, outChannels: outChannels, tembChannels: tembChannels, normGroups: normGroups, normEps: normEps)
        }
        self._attentions.wrappedValue = hasCrossAttention
            ? resnetInChannels.map { _ in
                SDTransformer2D(channels: outChannels, transformerLayers: transformerLayers,
                                crossAttentionDim: crossAttentionDim, heads: outChannels / headDim,
                                headDim: headDim, normGroups: normGroups, normEps: normEps)
            }
            : []
        self._upsamplers.wrappedValue = hasUpsample ? [SDUpSample(channels: outChannels)] : []
        super.init()
    }

    func callAsFunction(_ x: MLXArray, temb: MLXArray, context: MLXArray, residuals: inout [MLXArray]) -> MLXArray {
        var hidden = x
        for i in 0 ..< resnets.count {
            let skip = residuals.removeLast()
            hidden = concatenated([hidden, skip], axis: -1)
            hidden = resnets[i](hidden, temb: temb)
            if !attentions.isEmpty { hidden = attentions[i](hidden, context: context) }
        }
        if let upsampler = upsamplers.first {
            hidden = upsampler(hidden)
        }
        return hidden
    }
}

nonisolated final class SDDownsample: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(channels: Int) {
        self._conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 2, padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(x)
    }
}

nonisolated final class SDUpSample: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(channels: Int) {
        self._conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let up = MLXNN.Upsample(scaleFactor: 2.0, mode: .nearest)(x)
        return conv(up)
    }
}

/// `UNetMidBlock2DCrossAttn`: resnet → attention → resnet.
nonisolated final class SDMidBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [SDResnetBlock]
    @ModuleInfo(key: "attentions") var attentions: [SDTransformer2D]

    init(channels: Int, tembChannels: Int, transformerLayers: Int, crossAttentionDim: Int,
         headDim: Int, normGroups: Int, normEps: Float) {
        self._resnets.wrappedValue = [
            SDResnetBlock(inChannels: channels, outChannels: channels, tembChannels: tembChannels, normGroups: normGroups, normEps: normEps),
            SDResnetBlock(inChannels: channels, outChannels: channels, tembChannels: tembChannels, normGroups: normGroups, normEps: normEps),
        ]
        self._attentions.wrappedValue = [
            SDTransformer2D(channels: channels, transformerLayers: transformerLayers,
                            crossAttentionDim: crossAttentionDim, heads: channels / headDim,
                            headDim: headDim, normGroups: normGroups, normEps: normEps),
        ]
        super.init()
    }

    func callAsFunction(_ x: MLXArray, temb: MLXArray, context: MLXArray) -> MLXArray {
        var hidden = resnets[0](x, temb: temb)
        hidden = attentions[0](hidden, context: context)
        hidden = resnets[1](hidden, temb: temb)
        return hidden
    }
}
