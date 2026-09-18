import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Precision + norm helpers

private nonisolated enum VAEPrecision { static let dtype: DType = .bfloat16 }

private nonisolated func groupNorm32(_ dims: Int) -> GroupNorm {
    GroupNorm(groupCount: 32, dimensions: dims, eps: 1e-6, affine: true, pytorchCompatible: true)
}

/// GroupNorm over channels (input [B,C,T,H,W]) in fp32, cast back to VAE precision.
private nonisolated func vaeGroupNorm(_ x: MLXArray, _ norm: GroupNorm) -> MLXArray {
    var h = x.transposed(0, 2, 3, 4, 1)   // NDHWC
    h = norm(h.asType(.float32)).asType(VAEPrecision.dtype)
    return h.transposed(0, 4, 1, 2, 3)
}

// MARK: - CausalConv3d

/// Causal (in time) 3-D conv. Weight layout [O, kt, kh, kw, I] (MLX NDHWC).
nonisolated final class SeedVR2CausalConv3d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias")   var bias:   MLXArray
    let k: (Int, Int, Int), s: (Int, Int, Int), p: (Int, Int, Int)
    let causalTemporal: Bool, usePaddingCausal: Bool

    init(_ inCh: Int, _ outCh: Int, kernel: (Int, Int, Int) = (3, 3, 3),
         stride: (Int, Int, Int) = (1, 1, 1), padding: (Int, Int, Int) = (1, 1, 1),
         causalTemporal: Bool = true, usePaddingCausal: Bool = false) {
        self.k = kernel; self.s = stride; self.p = padding
        self.causalTemporal = causalTemporal; self.usePaddingCausal = usePaddingCausal
        self._weight.wrappedValue = MLXArray.zeros([outCh, kernel.0, kernel.1, kernel.2, inCh])
        self._bias.wrappedValue = MLXArray.zeros([outCh])
        super.init()
    }

    func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        var x = xIn  // [B,C,T,H,W]
        var tPad = p.0
        if causalTemporal && k.0 > 1 {
            let causalPad = usePaddingCausal ? 2 * p.0 : k.0 - 1
            if causalPad > 0 {
                let first = x[0..., 0..., 0 ..< 1]
                let pad = repeated(first, count: causalPad, axis: 2)
                x = concatenated([pad, x], axis: 2)
            }
            tPad = 0
        }
        x = x.transposed(0, 2, 3, 4, 1).asType(weight.dtype)   // NDHWC
        var out = convGeneral(x, weight, strides: [s.0, s.1, s.2], padding: [tPad, p.1, p.2])
        out = out + bias
        return out.transposed(0, 4, 1, 2, 3)                    // [B,O,T,H,W]
    }
}

// MARK: - ResnetBlock3D

nonisolated final class SeedVR2Resnet3D: Module {
    @ModuleInfo(key: "norm1")          var norm1:         GroupNorm
    @ModuleInfo(key: "norm2")          var norm2:         GroupNorm
    @ModuleInfo(key: "conv1")          var conv1:         SeedVR2CausalConv3d
    @ModuleInfo(key: "conv2")          var conv2:         SeedVR2CausalConv3d
    @ModuleInfo(key: "conv_shortcut")  var convShortcut:  SeedVR2CausalConv3d?

    init(_ inCh: Int, _ outCh: Int) {
        self._norm1.wrappedValue = groupNorm32(inCh)
        self._norm2.wrappedValue = groupNorm32(outCh)
        self._conv1.wrappedValue = SeedVR2CausalConv3d(inCh, outCh)
        self._conv2.wrappedValue = SeedVR2CausalConv3d(outCh, outCh)
        self._convShortcut.wrappedValue = inCh != outCh
            ? SeedVR2CausalConv3d(inCh, outCh, kernel: (1, 1, 1), padding: (0, 0, 0)) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(MLXNN.silu(vaeGroupNorm(x, norm1)))
        h = conv2(MLXNN.silu(vaeGroupNorm(h, norm2)))
        let residual = convShortcut?(x) ?? x
        return h + residual
    }
}

// MARK: - Attention3D (mid block)

nonisolated final class SeedVR2Attention3D: Module {
    @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
    @ModuleInfo(key: "to_q")       var toQ:       Linear
    @ModuleInfo(key: "to_k")       var toK:       Linear
    @ModuleInfo(key: "to_v")       var toV:       Linear
    @ModuleInfo(key: "to_out")     var toOut:     [Linear]
    let scale: Float

    init(_ channels: Int) {
        self._groupNorm.wrappedValue = groupNorm32(channels)
        self._toQ.wrappedValue = Linear(channels, channels)
        self._toK.wrappedValue = Linear(channels, channels)
        self._toV.wrappedValue = Linear(channels, channels)
        self._toOut.wrappedValue = [Linear(channels, channels)]
        self.scale = powf(Float(channels), -0.5)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, C, T, H, W) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        let residual = x
        var h = x.transposed(0, 2, 1, 3, 4).reshaped([B * T, C, H * W]).transposed(0, 2, 1)  // [B*T, HW, C]
        h = groupNorm(h.asType(.float32)).asType(VAEPrecision.dtype)
        let q = toQ(h).expandedDimensions(axis: 1)
        let k = toK(h).expandedDimensions(axis: 1)
        let v = toV(h).expandedDimensions(axis: 1)
        var o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        o = o.squeezed(axis: 1)
        o = toOut[0](o).transposed(0, 2, 1).reshaped([B, T, C, H, W]).transposed(0, 2, 1, 3, 4)
        return o + residual
    }
}

// MARK: - Downsample / Upsample

nonisolated final class SeedVR2Downsample3D: Module {
    @ModuleInfo(key: "conv") var conv: SeedVR2CausalConv3d
    init(_ channels: Int, spatialOnly: Bool) {
        let (kt, st, pt) = spatialOnly ? (1, 1, 0) : (3, 2, 1)
        self._conv.wrappedValue = SeedVR2CausalConv3d(channels, channels,
            kernel: (kt, 3, 3), stride: (st, 2, 2), padding: (pt, 0, 0))
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = MLX.padded(x, widths: [.init((0, 0)), .init((0, 0)), .init((0, 0)), .init((0, 1)), .init((0, 1))])
        return conv(padded)
    }
}

nonisolated final class SeedVR2Upsample3D: Module {
    @ModuleInfo(key: "conv")          var conv:         SeedVR2CausalConv3d
    @ModuleInfo(key: "upscale_conv")  var upscaleConv:  SeedVR2CausalConv3d
    let sf = 2, tf: Int
    init(_ channels: Int, temporalUp: Bool) {
        self.tf = temporalUp ? 2 : 1
        let total = sf * sf * tf
        self._conv.wrappedValue = SeedVR2CausalConv3d(channels, channels, usePaddingCausal: true)
        self._upscaleConv.wrappedValue = SeedVR2CausalConv3d(channels, channels * total,
            kernel: (1, 1, 1), padding: (0, 0, 0))
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, C, T, H, W) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        var h = upscaleConv(x)
        h = h.reshaped([B, sf, sf, tf, C, T, H, W]).transposed(0, 4, 5, 3, 6, 1, 7, 2)
        h = h.reshaped([B, C, T * tf, H * sf, W * sf])
        if T == 1 && tf > 1 { h = h[0..., 0..., 0 ..< 1] }
        return conv(h)
    }
}

// MARK: - Mid / Down / Up blocks

nonisolated final class SeedVR2MidBlock3D: Module {
    @ModuleInfo(key: "attentions") var attentions: [SeedVR2Attention3D]
    @ModuleInfo(key: "resnets")    var resnets:    [SeedVR2Resnet3D]
    init(_ channels: Int) {
        self._attentions.wrappedValue = [SeedVR2Attention3D(channels)]
        self._resnets.wrappedValue = [SeedVR2Resnet3D(channels, channels), SeedVR2Resnet3D(channels, channels)]
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        resnets[1](attentions[0](resnets[0](x)))
    }
}

nonisolated final class SeedVR2DownBlock3D: Module {
    @ModuleInfo(key: "resnets")      var resnets:      [SeedVR2Resnet3D]
    @ModuleInfo(key: "downsamplers") var downsamplers: [SeedVR2Downsample3D]
    init(_ inCh: Int, _ outCh: Int, numLayers: Int, addDownsample: Bool, temporalDown: Bool) {
        self._resnets.wrappedValue = (0 ..< numLayers).map { SeedVR2Resnet3D($0 == 0 ? inCh : outCh, outCh) }
        self._downsamplers.wrappedValue = addDownsample ? [SeedVR2Downsample3D(outCh, spatialOnly: !temporalDown)] : []
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        for d in downsamplers { h = d(h) }
        return h
    }
}

nonisolated final class SeedVR2UpBlock3D: Module {
    @ModuleInfo(key: "resnets")    var resnets:    [SeedVR2Resnet3D]
    @ModuleInfo(key: "upsamplers") var upsamplers: [SeedVR2Upsample3D]
    init(_ inCh: Int, _ outCh: Int, numLayers: Int, addUpsample: Bool, temporalUp: Bool) {
        self._resnets.wrappedValue = (0 ..< numLayers).map { SeedVR2Resnet3D($0 == 0 ? inCh : outCh, outCh) }
        self._upsamplers.wrappedValue = addUpsample ? [SeedVR2Upsample3D(outCh, temporalUp: temporalUp)] : []
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        for u in upsamplers { h = u(h) }
        return h
    }
}

// MARK: - Encoder / Decoder

nonisolated final class SeedVR2Encoder3D: Module {
    @ModuleInfo(key: "conv_in")       var convIn:       SeedVR2CausalConv3d
    @ModuleInfo(key: "down_blocks")   var downBlocks:   [SeedVR2DownBlock3D]
    @ModuleInfo(key: "mid_block")     var midBlock:     SeedVR2MidBlock3D
    @ModuleInfo(key: "conv_norm_out") var convNormOut:  GroupNorm
    @ModuleInfo(key: "conv_out")      var convOut:      SeedVR2CausalConv3d

    init(inChannels: Int = 3, outChannels: Int = 16,
         blockOut: [Int] = [128, 256, 512, 512], layersPerBlock: Int = 2, temporalDownBlocks: Int = 2) {
        self._convIn.wrappedValue = SeedVR2CausalConv3d(inChannels, blockOut[0])
        var blocks: [SeedVR2DownBlock3D] = []
        var outCh = blockOut[0]
        let n = blockOut.count
        for (i, ch) in blockOut.enumerated() {
            let inCh = outCh; outCh = ch
            let isFinal = i == n - 1
            let temporalDown = (i >= n - temporalDownBlocks - 1) && !isFinal
            blocks.append(SeedVR2DownBlock3D(inCh, outCh, numLayers: layersPerBlock,
                                             addDownsample: !isFinal, temporalDown: temporalDown))
        }
        self._downBlocks.wrappedValue = blocks
        self._midBlock.wrappedValue = SeedVR2MidBlock3D(blockOut[n - 1])
        self._convNormOut.wrappedValue = groupNorm32(blockOut[n - 1])
        self._convOut.wrappedValue = SeedVR2CausalConv3d(blockOut[n - 1], 2 * outChannels)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for b in downBlocks { h = b(h) }
        h = midBlock(h)
        h = MLXNN.silu(vaeGroupNorm(h, convNormOut))
        return convOut(h)   // [B, 2*outCh, T, H, W]
    }
}

nonisolated final class SeedVR2Decoder3D: Module {
    @ModuleInfo(key: "conv_in")       var convIn:       SeedVR2CausalConv3d
    @ModuleInfo(key: "mid_block")     var midBlock:     SeedVR2MidBlock3D
    @ModuleInfo(key: "up_blocks")     var upBlocks:     [SeedVR2UpBlock3D]
    @ModuleInfo(key: "conv_norm_out") var convNormOut:  GroupNorm
    @ModuleInfo(key: "conv_out")      var convOut:      SeedVR2CausalConv3d

    init(inChannels: Int = 16, outChannels: Int = 3,
         blockOut: [Int] = [128, 256, 512, 512], layersPerBlock: Int = 3, temporalUpBlocks: Int = 2) {
        let rev = Array(blockOut.reversed())
        self._convIn.wrappedValue = SeedVR2CausalConv3d(inChannels, rev[0])
        self._midBlock.wrappedValue = SeedVR2MidBlock3D(rev[0])
        var blocks: [SeedVR2UpBlock3D] = []
        var outCh = rev[0]
        let n = rev.count
        for (i, ch) in rev.enumerated() {
            let inCh = outCh; outCh = ch
            let isFinal = i == n - 1
            let temporalUp = i < temporalUpBlocks
            blocks.append(SeedVR2UpBlock3D(inCh, outCh, numLayers: layersPerBlock,
                                           addUpsample: !isFinal, temporalUp: temporalUp))
        }
        self._upBlocks.wrappedValue = blocks
        self._convNormOut.wrappedValue = groupNorm32(rev[n - 1])
        self._convOut.wrappedValue = SeedVR2CausalConv3d(rev[n - 1], outChannels)
        super.init()
    }

    func callAsFunction(_ z: MLXArray) -> MLXArray {
        var h = convIn(z)
        h = midBlock(h)
        for b in upBlocks { h = b(h) }
        h = MLXNN.silu(vaeGroupNorm(h, convNormOut))
        return convOut(h)
    }
}

// MARK: - SeedVR2VAE

nonisolated final class SeedVR2VAE: Module {
    @ModuleInfo(key: "encoder") var encoder: SeedVR2Encoder3D
    @ModuleInfo(key: "decoder") var decoder: SeedVR2Decoder3D
    let scalingFactor: Float = 0.9152
    let latentChannels = 16

    override init() {
        self._encoder.wrappedValue = SeedVR2Encoder3D()
        self._decoder.wrappedValue = SeedVR2Decoder3D()
        super.init()
    }

    /// x [B,3,1,H,W] → latent [B,16,1,H/8,W/8] (mean × scalingFactor).
    func encode(_ x: MLXArray) -> MLXArray {
        let h = encoder(x.asType(.bfloat16))   // [B, 32, T, H, W]
        let mean = h[0..., 0 ..< latentChannels]
        return mean * scalingFactor
    }

    /// z [B,16,1,H,W] → image [B,3,1,H*8,W*8].
    func decode(_ z: MLXArray) -> MLXArray {
        let h = z / scalingFactor
        return decoder(h.asType(.bfloat16))
    }
}
