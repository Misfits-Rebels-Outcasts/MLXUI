import Foundation
import MLX
import MLXNN

// MARK: - CausalConv3d

/// 3-D causal convolution. Pads the temporal axis with (kT-1) zeros at the front so
/// each output frame only attends to past/current input frames.
/// Weights are stored in MLX NDHWC layout [O, kT, kH, kW, I] — no transposition needed.
nonisolated final class SeedVR2CausalConv3d: Module {
    var weight: MLXArray    // [O, kT, kH, kW, I]
    var bias:   MLXArray    // [O]

    let kT: Int
    let kH: Int
    let kW: Int
    let sT: Int
    let sH: Int
    let sW: Int

    init(inCh: Int, outCh: Int, k: Int = 3, s: Int = 1) {
        self.kT = k; self.kH = k; self.kW = k
        self.sT = s; self.sH = s; self.sW = s
        self.weight = MLXRandom.normal([outCh, k, k, k, inCh]) * 0.02
        self.bias   = MLXArray.zeros([outCh])
        super.init()
    }

    init(inCh: Int, outCh: Int, kT: Int, kH: Int, kW: Int, sT: Int = 1, sH: Int = 1, sW: Int = 1) {
        self.kT = kT; self.kH = kH; self.kW = kW
        self.sT = sT; self.sH = sH; self.sW = sW
        self.weight = MLXRandom.normal([outCh, kT, kH, kW, inCh]) * 0.02
        self.bias   = MLXArray.zeros([outCh])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, C, T, H, W] — transpose to NDHWC [B, T, H, W, C]
        let xT = x.transposed(0, 2, 3, 4, 1)
        let pH = (kH - 1) / 2
        let pW = (kW - 1) / 2
        // causal temporal pad: (kT-1) zeros before temporal dim
        let padded = MLX.padded(xT, widths: [.init(0), .init((kT-1, 0)), .init(0), .init(0), .init(0)])
        let out = conv3d(padded, weight, stride: [sT, sH, sW], padding: [0, pH, pW]) + bias
        // result: [B, T', H', W', C] → [B, C, T', H', W']
        return out.transposed(0, 4, 1, 2, 3)
    }
}

// MARK: - GroupNorm helper

private func vaeGN(_ ch: Int, numGroups: Int = 32) -> GroupNorm {
    GroupNorm(groupCount: numGroups, dimensions: ch, eps: 1e-6, affine: true, pytorchCompatible: true)
}

// MARK: - ResnetBlock3D

nonisolated final class SeedVR2Resnet3D: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: SeedVR2CausalConv3d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: SeedVR2CausalConv3d
    @ModuleInfo(key: "shortcut") var shortcut: SeedVR2CausalConv3d?

    init(inCh: Int, outCh: Int) {
        self._norm1.wrappedValue    = vaeGN(inCh)
        self._conv1.wrappedValue    = SeedVR2CausalConv3d(inCh: inCh, outCh: outCh)
        self._norm2.wrappedValue    = vaeGN(outCh)
        self._conv2.wrappedValue    = SeedVR2CausalConv3d(inCh: outCh, outCh: outCh)
        self._shortcut.wrappedValue = inCh != outCh
            ? SeedVR2CausalConv3d(inCh: inCh, outCh: outCh, k: 1) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = norm1(x.transposed(0, 2, 3, 4, 1))    // [B, T, H, W, C]
        h = MLXNN.silu(h).transposed(0, 4, 1, 2, 3)   // back to [B, C, T, H, W]
        h = conv1(h)
        h = norm2(h.transposed(0, 2, 3, 4, 1))
        h = MLXNN.silu(h).transposed(0, 4, 1, 2, 3)
        h = conv2(h)
        let res = shortcut.map { $0(x) } ?? x
        return h + res
    }
}

// MARK: - Attention3D (mid block)

/// Spatial self-attention inside the mid block. Operates channel-first.
nonisolated final class SeedVR2Attention3D: Module {
    @ModuleInfo(key: "norm") var norm: GroupNorm
    @ModuleInfo(key: "toQ")  var toQ:  Linear
    @ModuleInfo(key: "toK")  var toK:  Linear
    @ModuleInfo(key: "toV")  var toV:  Linear
    @ModuleInfo(key: "proj") var proj: Linear

    init(ch: Int) {
        self._norm.wrappedValue = vaeGN(ch)
        self._toQ.wrappedValue  = Linear(ch, ch)
        self._toK.wrappedValue  = Linear(ch, ch)
        self._toV.wrappedValue  = Linear(ch, ch)
        self._proj.wrappedValue = Linear(ch, ch)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, C, T, H, W]
        let (B, C, T, H, W) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        let n = T * H * W

        // GroupNorm (in NDHWC)
        let xT = x.transposed(0, 2, 3, 4, 1)              // [B, T, H, W, C]
        let normed = norm(xT).reshaped([B, n, C])          // [B, n, C]

        let q = toQ(normed)                                // [B, n, C]
        let k = toK(normed)
        let v = toV(normed)

        let scale = 1.0 / sqrt(Float(C))
        let q3 = q.expandedDimensions(axis: 1)             // [B, 1, n, C]
        let k3 = k.expandedDimensions(axis: 1)
        let v3 = v.expandedDimensions(axis: 1)

        let out = scaledDotProductAttention(
            queries: q3, keys: k3, values: v3, scale: scale, mask: nil)  // [B, 1, n, C]
        let flat = out.squeezed(axis: 1)                   // [B, n, C]
        let projected = proj(flat)                         // [B, n, C]
        let res = projected.reshaped([B, T, H, W, C]).transposed(0, 4, 1, 2, 3)
        return res + x
    }
}

// MARK: - MidBlock3D

nonisolated final class SeedVR2MidBlock3D: Module {
    @ModuleInfo(key: "resnet1") var resnet1: SeedVR2Resnet3D
    @ModuleInfo(key: "attn")    var attn:    SeedVR2Attention3D
    @ModuleInfo(key: "resnet2") var resnet2: SeedVR2Resnet3D

    init(ch: Int) {
        self._resnet1.wrappedValue = SeedVR2Resnet3D(inCh: ch, outCh: ch)
        self._attn.wrappedValue    = SeedVR2Attention3D(ch: ch)
        self._resnet2.wrappedValue = SeedVR2Resnet3D(inCh: ch, outCh: ch)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        resnet2(attn(resnet1(x)))
    }
}

// MARK: - DownBlock3D

nonisolated final class SeedVR2DownBlock3D: Module {
    @ModuleInfo(key: "resnets")    var resnets:    [SeedVR2Resnet3D]
    @ModuleInfo(key: "downsample") var downsample: SeedVR2CausalConv3d?

    init(inCh: Int, outCh: Int, numResnets: Int, hasDownsample: Bool) {
        self._resnets.wrappedValue = (0 ..< numResnets).map { i in
            SeedVR2Resnet3D(inCh: i == 0 ? inCh : outCh, outCh: outCh)
        }
        self._downsample.wrappedValue = hasDownsample
            ? SeedVR2CausalConv3d(inCh: outCh, outCh: outCh, kT: 3, kH: 3, kW: 3,
                                   sT: 2, sH: 2, sW: 2) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        if let ds = downsample { h = ds(h) }
        return h
    }
}

// MARK: - UpBlock3D

nonisolated final class SeedVR2UpBlock3D: Module {
    @ModuleInfo(key: "resnets")  var resnets:  [SeedVR2Resnet3D]
    @ModuleInfo(key: "upsample") var upsample: SeedVR2CausalConv3d?

    init(inCh: Int, outCh: Int, numResnets: Int, hasUpsample: Bool) {
        self._resnets.wrappedValue = (0 ..< numResnets).map { i in
            SeedVR2Resnet3D(inCh: i == 0 ? inCh : outCh, outCh: outCh)
        }
        self._upsample.wrappedValue = hasUpsample
            ? SeedVR2CausalConv3d(inCh: outCh, outCh: outCh) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        if let up = upsample {
            // Nearest-neighbor upsample ×2 in T, H, W then smooth with conv
            h = nearestUpsample3D(h)
            h = up(h)
        }
        return h
    }

    /// Nearest-neighbor 2× upsample in all three spatial+temporal dimensions.
    private func nearestUpsample3D(_ x: MLXArray) -> MLXArray {
        let (B, C, T, H, W) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        // Temporal
        let tUp = MLX.tiled(x.reshaped([B, C, T, 1, H, W]), repetitions: [1, 1, 1, 2, 1, 1])
                     .reshaped([B, C, T*2, H, W])
        // Height
        let hUp = MLX.tiled(tUp.reshaped([B, C, T*2, H, 1, W]), repetitions: [1, 1, 1, 1, 2, 1])
                     .reshaped([B, C, T*2, H*2, W])
        // Width
        let wUp = MLX.tiled(hUp.reshaped([B, C, T*2, H*2, W, 1]), repetitions: [1, 1, 1, 1, 1, 2])
                     .reshaped([B, C, T*2, H*2, W*2])
        return wUp
    }
}

// MARK: - Encoder3D

nonisolated final class SeedVR2Encoder3D: Module {
    @ModuleInfo(key: "convIn")     var convIn:     SeedVR2CausalConv3d
    @ModuleInfo(key: "downBlocks") var downBlocks: [SeedVR2DownBlock3D]
    @ModuleInfo(key: "midBlock")   var midBlock:   SeedVR2MidBlock3D
    @ModuleInfo(key: "normOut")    var normOut:    GroupNorm
    @ModuleInfo(key: "convOut")    var convOut:    SeedVR2CausalConv3d

    init(cfg: SeedVR2VAEArchConfig = SeedVR2VAEArchConfig()) {
        let chs = cfg.chMult.map { cfg.baseCh * $0 }  // [128, 256, 512, 512]
        self._convIn.wrappedValue = SeedVR2CausalConv3d(inCh: cfg.inChannels, outCh: chs[0])
        self._downBlocks.wrappedValue = (0 ..< chs.count).map { i in
            let inCh  = i == 0 ? chs[0] : chs[i-1]
            let outCh = chs[i]
            return SeedVR2DownBlock3D(
                inCh: inCh, outCh: outCh,
                numResnets: cfg.numResBlocks,
                hasDownsample: i < chs.count - 1)
        }
        let innerCh = chs.last!
        self._midBlock.wrappedValue = SeedVR2MidBlock3D(ch: innerCh)
        self._normOut.wrappedValue  = vaeGN(innerCh, numGroups: cfg.numGroups)
        self._convOut.wrappedValue  = SeedVR2CausalConv3d(inCh: innerCh, outCh: 2 * cfg.zChannels)
        super.init()
    }

    /// Returns the posterior mean: [B, zChannels, T, H, W].
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for block in downBlocks { h = block(h) }
        h = midBlock(h)
        // normOut + silu (in NDHWC then back)
        let hT = h.transposed(0, 2, 3, 4, 1)
        h = MLXNN.silu(normOut(hT)).transposed(0, 4, 1, 2, 3)
        let moments = convOut(h)            // [B, 2*z, T, H, W]
        let z = moments.dim(1) / 2
        return moments[0..., ..<z, 0..., 0..., 0...]   // mu only
    }
}

// MARK: - Decoder3D

nonisolated final class SeedVR2Decoder3D: Module {
    @ModuleInfo(key: "convIn")   var convIn:   SeedVR2CausalConv3d
    @ModuleInfo(key: "midBlock") var midBlock: SeedVR2MidBlock3D
    @ModuleInfo(key: "upBlocks") var upBlocks: [SeedVR2UpBlock3D]
    @ModuleInfo(key: "normOut")  var normOut:  GroupNorm
    @ModuleInfo(key: "convOut")  var convOut:  SeedVR2CausalConv3d

    init(cfg: SeedVR2VAEArchConfig = SeedVR2VAEArchConfig()) {
        let chs   = cfg.chMult.map { cfg.baseCh * $0 }.reversed() // [512, 512, 256, 128]
        let chsArr = Array(chs)
        let innerCh = chsArr[0]
        self._convIn.wrappedValue   = SeedVR2CausalConv3d(inCh: cfg.zChannels, outCh: innerCh)
        self._midBlock.wrappedValue = SeedVR2MidBlock3D(ch: innerCh)
        self._upBlocks.wrappedValue = (0 ..< chsArr.count).map { i in
            let inCh  = chsArr[i]
            let outCh = i < chsArr.count - 1 ? chsArr[i+1] : chsArr.last!
            return SeedVR2UpBlock3D(
                inCh: inCh, outCh: outCh,
                numResnets: cfg.numResBlocks,
                hasUpsample: i < chsArr.count - 1)
        }
        let outCh = chsArr.last!
        self._normOut.wrappedValue  = vaeGN(outCh, numGroups: cfg.numGroups)
        self._convOut.wrappedValue  = SeedVR2CausalConv3d(inCh: outCh, outCh: cfg.outChannels)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        h = midBlock(h)
        for block in upBlocks { h = block(h) }
        let hT = h.transposed(0, 2, 3, 4, 1)
        h = MLXNN.silu(normOut(hT)).transposed(0, 4, 1, 2, 3)
        return convOut(h)
    }
}

// MARK: - SeedVR2VAE

nonisolated final class SeedVR2VAE: Module {
    @ModuleInfo(key: "encoder")       var encoder:       SeedVR2Encoder3D
    @ModuleInfo(key: "decoder")       var decoder:       SeedVR2Decoder3D
    @ModuleInfo(key: "quantConv")     var quantConv:     SeedVR2CausalConv3d
    @ModuleInfo(key: "postQuantConv") var postQuantConv: SeedVR2CausalConv3d

    let scalingFactor: Float

    init(cfg: SeedVR2VAEArchConfig = SeedVR2VAEArchConfig()) {
        let z = cfg.zChannels
        self._encoder.wrappedValue       = SeedVR2Encoder3D(cfg: cfg)
        self._decoder.wrappedValue       = SeedVR2Decoder3D(cfg: cfg)
        self._quantConv.wrappedValue     = SeedVR2CausalConv3d(inCh: 2*z, outCh: 2*z, k: 1)
        self._postQuantConv.wrappedValue = SeedVR2CausalConv3d(inCh: z, outCh: z, k: 1)
        self.scalingFactor = cfg.scalingFactor
        super.init()
    }

    /// Encode image to latent mu. x: [B, 3, 1, H, W] in [-1, 1].
    func encode(_ x: MLXArray) -> MLXArray {
        let h = encoder(x.asType(.bfloat16))
        return quantConv(
            concatenated([h, MLXArray.zeros(like: h)], axis: 1)
        )[0..., ..<h.dim(1), 0..., 0..., 0...] * scalingFactor
    }

    /// Decode latent to image. z: [B, 16, 1, H, W].
    func decode(_ z: MLXArray) -> MLXArray {
        let h = postQuantConv(z / scalingFactor)
        return decoder(h)
    }
}
