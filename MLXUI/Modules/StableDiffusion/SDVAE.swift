import Foundation
import MLX
import MLXNN
import MLXFast

/// The SDXL VAE **decoder** (decode-only; encoder/quant_conv unused at inference), ported from
/// `stabilityai/sdxl-turbo`'s `vae/diffusion_pytorch_model.fp16.safetensors`. Structurally
/// identical to `FluxVAE`'s decoder (mid resnet+attn+resnet; 4 up blocks 512/512/256/128) with
/// three SDXL differences: **4-channel** latent (Flux = 16), a top-level `post_quant_conv`
/// 1×1 conv between scaling and the decoder, and scale `1/0.13025` (the repo's
/// `vae/config.json` `scaling_factor` — NOT 0.18215).
///
/// Checkpoint facts (from the safetensors header, SD-ENG2):
/// - all weights plain **F16** → `Conv2d`/`Linear`, not the 4-bit `QuantizedLinear` FLUX's VAE uses;
/// - keys are single-segment `@ModuleInfo` segments: `post_quant_conv`, `decoder.conv_in`,
///   `decoder.mid_block.{resnets,attentions}`, `decoder.up_blocks.{i}.{resnets,upsamplers}`,
///   `decoder.conv_norm_out`, `decoder.conv_out`;
/// - PyTorch conv weights are `(out,in,kH,kW)` — must be transposed to MLX `(out,kH,kW,in)`
///   via `SDWeights.sanitize` before `update(parameters:)` (see the DeepSeek-OCR precedent).
///
/// The VAE operates in **NHWC** internally, matching MLX `Conv2d`.
nonisolated final class SDVAE: Module {
    /// The repo's `vae/config.json` `scaling_factor`.
    static let scaleFactor: Float = 0.13025

    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d
    @ModuleInfo(key: "decoder") var decoder: SDVAEDecoder

    override init() {
        self._postQuantConv.wrappedValue = Conv2d(inputChannels: 4, outputChannels: 4, kernelSize: 1)
        self._decoder.wrappedValue = SDVAEDecoder()
        super.init()
    }

    /// Full decode: latent `(B, 4, H, W)` → image `(B, 3, H·8, W·8)`. Scale → `post_quant_conv`
    /// → decoder (NHWC internally) → NCHW.
    func decode(_ latents: MLXArray) -> MLXArray {
        let scaled = latents / Self.scaleFactor
        let nhwc = scaled.transposed(0, 2, 3, 1)
        let post = postQuantConv(nhwc)
        let out = decoder(post)
        return out.transposed(0, 3, 1, 2)
    }
}

/// `Decoder`: conv_in → mid_block → up_blocks → conv_norm_out → conv_out.
nonisolated final class SDVAEDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid_block") var midBlock: SDVAEMidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [SDVAEUpBlock]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    override init() {
        self._convIn.wrappedValue = Conv2d(inputChannels: 4, outputChannels: 512, kernelSize: 3, padding: 1)
        self._midBlock.wrappedValue = SDVAEMidBlock()
        self._upBlocks.wrappedValue = [
            SDVAEUpBlock(resnet: (512, 512, 512), hasUpsampler: true),
            SDVAEUpBlock(resnet: (512, 512, 512), hasUpsampler: true),
            SDVAEUpBlock(resnet: (512, 512, 256), hasUpsampler: true),
            SDVAEUpBlock(resnet: (256, 256, 128), hasUpsampler: false),
        ]
        self._convNormOut.wrappedValue = GroupNorm(groupCount: 32, dimensions: 128, eps: 1e-6, affine: true, pytorchCompatible: true)
        self._convOut.wrappedValue = Conv2d(inputChannels: 128, outputChannels: 3, kernelSize: 3, padding: 1)
        super.init()
    }

    func callAsFunction(_ latents: MLXArray) -> MLXArray {
        var hidden = convIn(latents)
        hidden = midBlock(hidden)
        for up in upBlocks { hidden = up(hidden) }
        hidden = silu(convNormOut(hidden))
        return convOut(hidden)
    }
}

/// `UnetMidBlock`: resnet → attention → resnet (all 512-dim).
nonisolated final class SDVAEMidBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [SDVAEResnet]
    @ModuleInfo(key: "attentions") var attentions: [SDVAEAttention]

    override init() {
        self._resnets.wrappedValue = [
            SDVAEResnet(channels: 512, channels: 512),
            SDVAEResnet(channels: 512, channels: 512),
        ]
        self._attentions.wrappedValue = [SDVAEAttention(channels: 512)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        resnets[1](attentions[0](resnets[0](x)))
    }
}

/// One decoder up block: 3 resnets (1×1 shortcut on the first when the channel count changes),
/// then an optional 2× nearest upsampler.
nonisolated final class SDVAEUpBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [SDVAEResnet]
    @ModuleInfo(key: "upsamplers") var upsamplers: [SDVAEUpSampler]

    init(resnet: (in: Int, mid: Int, out: Int), hasUpsampler: Bool) {
        self._resnets.wrappedValue = [
            SDVAEResnet(channels: resnet.in, channels: resnet.out, shortcut: resnet.in != resnet.out),
            SDVAEResnet(channels: resnet.out, channels: resnet.out),
            SDVAEResnet(channels: resnet.out, channels: resnet.out),
        ]
        self._upsamplers.wrappedValue = hasUpsampler
            ? [SDVAEUpSampler(channels: resnet.out)]
            : []
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = resnets[2](resnets[1](resnets[0](x)))
        if let upsampler = upsamplers.first {
            hidden = upsampler(hidden)
        }
        return hidden
    }
}

nonisolated final class SDVAEUpSampler: Module {
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

/// `ResnetBlock2D`: GroupNorm(32) → silu → 3×3 conv → GroupNorm(32) → silu → 3×3 conv, plus a
/// 1×1 shortcut when the channel count changes. NHWC.
nonisolated final class SDVAEResnet: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

    init(channels inChannels: Int, channels outChannels: Int, shortcut: Bool = false) {
        self._norm1.wrappedValue = GroupNorm(groupCount: 32, dimensions: inChannels, eps: 1e-6, affine: true, pytorchCompatible: true)
        self._conv1.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self._norm2.wrappedValue = GroupNorm(groupCount: 32, dimensions: outChannels, eps: 1e-6, affine: true, pytorchCompatible: true)
        self._conv2.wrappedValue = Conv2d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self._convShortcut.wrappedValue = shortcut
            ? Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1)
            : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = silu(norm1(x))
        hidden = conv1(hidden)
        hidden = silu(norm2(hidden))
        hidden = conv2(hidden)
        if let convShortcut {
            return convShortcut(x) + hidden
        }
        return x + hidden
    }
}

/// `Attention` in the VAE mid block: GroupNorm → q/k/v (Linear, 512-dim) → SDPA → out. NHWC.
nonisolated final class SDVAEAttention: Module {
    @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]

    let channels: Int

    init(channels: Int) {
        self.channels = channels
        self._groupNorm.wrappedValue = GroupNorm(groupCount: 32, dimensions: channels, eps: 1e-6, affine: true, pytorchCompatible: true)
        func lin() -> Linear { Linear(channels, channels, bias: true) }
        self._toQ.wrappedValue = lin()
        self._toK.wrappedValue = lin()
        self._toV.wrappedValue = lin()
        self._toOut.wrappedValue = [lin()]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), h = x.dim(1), w = x.dim(2)
        let y = groupNorm(x)
        let q = toQ(y).reshaped([b, -1, 1, channels]).transposed(0, 2, 1, 3)
        let k = toK(y).reshaped([b, -1, 1, channels]).transposed(0, 2, 1, 3)
        let v = toV(y).reshaped([b, -1, 1, channels]).transposed(0, 2, 1, 3)
        let scale = 1.0 / sqrt(Float(channels))
        var out = scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        out = out.transposed(0, 2, 1, 3).reshaped([b, h, w, channels])
        out = toOut[0](out)
        return x + out
    }
}

/// Conv-weight sanitize shared by the conv-bearing SDXL modules (VAE, UNet): the diffusers
/// checkpoint stores PyTorch layout `(out, in, kH, kW)`, MLX wants `(out, kH, kW, in)`.
/// Linear/Embedding/LayerNorm layouts are identical in both, so only 4-D tensors are touched
/// (mirrors `DeepSeekOCRSAM.sanitize` / `DotsOCRVision.sanitize`).
nonisolated enum SDWeights {
    /// Transpose PyTorch conv weights `(out,in,kH,kW)` → MLX `(out,kH,kW,in)` unless they're
    /// already MLX-shaped, and remap the diffusers `FeedForward` keys so `ModuleParameters.unflattened`
    /// doesn't build an array with a hole. Applied to the raw safetensors dict before
    /// `update(parameters:)`.
    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        for (key, value) in weights {
            var k = key
            // diffusers `FeedForward.net` is a ModuleList [GEGLU, Dropout, Linear] — the dropout
            // (`net.1`) has no weights, leaving a numeric-key gap (`ff.net.0.proj` / `ff.net.2`)
            // that `ModuleParameters.unflattened` turns into an array `[GEGLU, none, Linear]`. The
            // module tree can't match that, so rename to non-numeric children → `net` unflattens
            // as a dict (same fix the Python reference does with `linear1`/`linear2`).
            if k.contains("ff.net.0.proj") {
                k = k.replacingOccurrences(of: "ff.net.0.proj", with: "ff.net.geglu.proj")
            } else if k.contains("ff.net.2") {
                k = k.replacingOccurrences(of: "ff.net.2", with: "ff.net.out")
            }
            if value.ndim == 4, !isMLXConvShape(value) {
                out[k] = value.transposed(0, 2, 3, 1)
            } else {
                out[k] = value
            }
        }
        return out
    }

    /// `check_array_shape`: true when a 4-D weight is already in MLX `(out,kH,kW,in)` layout —
    /// `out` ≥ both kernel dims and a square kernel (PyTorch `(out,in,kH,kW)` has `in` in slot 1).
    static func isMLXConvShape(_ a: MLXArray) -> Bool {
        guard a.ndim == 4 else { return false }
        let outC = a.dim(0), kH = a.dim(1), kW = a.dim(2)
        return outC >= kH && outC >= kW && kH == kW
    }
}
