import Foundation
import MLX
import MLXNN

/// The FLUX VAE **decoder** (decode-only; the encoder is unused at inference), ported from mflux
/// (`flux_vae/`) in the checkpoint's key format (AM4g). Keys: `decoder.{conv_in, mid_block,
/// up_blocks.{0..3}, conv_norm_out, conv_out}`.
///
/// Checkpoint facts:
/// - the Convs are **plain BF16 `Conv2d`** — mflux's `quantization_predicate` skips Conv2d;
/// - the **attention Linears ARE 4-bit quantized** → `QuantizedLinear`;
/// - block out-channels: conv_in 512; mid 512; up0/up1 512; up2 512→256; up3 256→128; conv_out →3.
///
/// The VAE operates in **NHWC** (channels-last) internally, matching MLX `Conv2d`.
/// Scale/shift: `1/0.3611`, `+0.1159`.
nonisolated final class FluxVAE: Module {

    static let scaleFactor: Float = 0.3611
    static let shiftFactor: Float = 0.1159

    @ModuleInfo(key: "decoder") var decoder: FluxVAEDecoder

    override init() {
        self._decoder.wrappedValue = FluxVAEDecoder()
        super.init()
    }

    /// Full decode: latent `(B, 16, H, W)` → image `(B, 3, H·8, W·8)`.
    func decode(_ latents: MLXArray) -> MLXArray {
        let scaled = (latents / Self.scaleFactor) + Self.shiftFactor
        let nhwc = scaled.transposed(0, 2, 3, 1)
        let out = decoder(nhwc)
        return out.transposed(0, 3, 1, 2)
    }
}

/// `Decoder`: conv_in → mid_block → up_blocks → conv_norm_out → conv_out.
/// All `@ModuleInfo` keys are single segments to ensure `update(parameters:)` descends correctly.
nonisolated final class FluxVAEDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: FluxVAEConvIn
    @ModuleInfo(key: "mid_block") var midBlock: FluxVAEMidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [FluxVAEUpBlock]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: FluxVAEConvNormOut
    @ModuleInfo(key: "conv_out") var convOut: FluxVAEConvOut

    override init() {
        self._convIn.wrappedValue = FluxVAEConvIn()
        self._midBlock.wrappedValue = FluxVAEMidBlock()
        self._upBlocks.wrappedValue = [
            FluxVAEUpBlock(resnet: (512, 512, 512), hasUpsampler: true),
            FluxVAEUpBlock(resnet: (512, 512, 512), hasUpsampler: true),
            FluxVAEUpBlock(resnet: (512, 512, 256), hasUpsampler: true),
            FluxVAEUpBlock(resnet: (256, 256, 128), hasUpsampler: false),
        ]
        self._convNormOut.wrappedValue = FluxVAEConvNormOut()
        self._convOut.wrappedValue = FluxVAEConvOut()
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

nonisolated final class FluxVAEConvIn: Module {
    @ModuleInfo(key: "conv2d") var conv: Conv2d

    override init() {
        self._conv.wrappedValue = Conv2d(inputChannels: 16, outputChannels: 512, kernelSize: 3, padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

nonisolated final class FluxVAEConvNormOut: Module {
    @ModuleInfo(key: "norm") var norm: GroupNorm

    override init() {
        self._norm.wrappedValue = GroupNorm(groupCount: 32, dimensions: 128, eps: 1e-6, affine: true, pytorchCompatible: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(x) }
}

nonisolated final class FluxVAEConvOut: Module {
    @ModuleInfo(key: "conv2d") var conv: Conv2d

    override init() {
        self._conv.wrappedValue = Conv2d(inputChannels: 128, outputChannels: 3, kernelSize: 3, padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

/// `UnetMidBlock`: resnet → attention → resnet (all 512-dim).
/// Call order: `resnets[0] → attentions[0] → resnets[1]` (mflux's UnetMidBlock).
nonisolated final class FluxVAEMidBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [FluxVAEResnet]
    @ModuleInfo(key: "attentions") var attentions: [FluxVAEAttention]

    override init() {
        self._resnets.wrappedValue = [
            FluxVAEResnet(channels: 512, channels: 512),
            FluxVAEResnet(channels: 512, channels: 512),
        ]
        self._attentions.wrappedValue = [FluxVAEAttention(channels: 512)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        resnets[1](attentions[0](resnets[0](x)))
    }
}

/// One decoder up block: 3 resnets (with optional 1×1 shortcut on the first), then an optional
/// 2× nearest upsampler.
nonisolated final class FluxVAEUpBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [FluxVAEResnet]
    @ModuleInfo(key: "upsamplers") var upsamplers: [FluxVAEUpSampler]

    init(resnet: (in: Int, mid: Int, out: Int), hasUpsampler: Bool) {
        self._resnets.wrappedValue = [
            FluxVAEResnet(channels: resnet.in, channels: resnet.out, shortcut: resnet.in != resnet.out),
            FluxVAEResnet(channels: resnet.out, channels: resnet.out),
            FluxVAEResnet(channels: resnet.out, channels: resnet.out),
        ]
        self._upsamplers.wrappedValue = hasUpsampler
            ? [FluxVAEUpSampler(channels: resnet.out)]
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

nonisolated final class FluxVAEUpSampler: Module {
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
nonisolated final class FluxVAEResnet: Module {
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

/// `Attention` in the VAE mid block: GroupNorm → q/k/v (quantized Linear, 512-dim) → SDPA →
/// out. NHWC, batch 1, heads 1.
nonisolated final class FluxVAEAttention: Module {
    @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: QuantizedLinear
    @ModuleInfo(key: "to_k") var toK: QuantizedLinear
    @ModuleInfo(key: "to_v") var toV: QuantizedLinear
    @ModuleInfo(key: "to_out") var toOut: [QuantizedLinear]

    let channels: Int

    init(channels: Int) {
        self.channels = channels
        self._groupNorm.wrappedValue = GroupNorm(groupCount: 32, dimensions: channels, eps: 1e-6, affine: true, pytorchCompatible: true)
        func ql() -> QuantizedLinear {
            QuantizedLinear(channels, channels, bias: true, groupSize: 64, bits: 4, mode: .affine)
        }
        self._toQ.wrappedValue = ql()
        self._toK.wrappedValue = ql()
        self._toV.wrappedValue = ql()
        self._toOut.wrappedValue = [ql()]
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
