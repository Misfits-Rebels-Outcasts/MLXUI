import Foundation
import MLX
import MLXNN

// MARK: - Config

struct WanVAEConfig: Sendable {
    var baseDim: Int = 96    // smallest channel count
    var zDim:    Int = 16    // latent channel count
    var outCh:   Int = 3     // RGB output
}

// MARK: - Decoder

/// 3-D causal VAE decoder for WAN 2.1. Weights from
/// `Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae/diffusion_pytorch_model.safetensors`.
///
/// Channel progression (base_dim=96):
///  conv_in: zDim→384  mid_block: 384  up[0]: 384→192  up[1]: 192→96  up[2]: 96→96  up[3]: 96→3
nonisolated final class WanVAEDecoder: Module {
    @ModuleInfo(key: "convIn")   var convIn:   WanCausalConv3d
    @ModuleInfo(key: "midBlock") var midBlock: WanVAEMidBlock
    @ModuleInfo(key: "upBlocks") var upBlocks: [WanVAEUpBlock]
    @ModuleInfo(key: "normOut")  var normOut:  WanVAERMSNorm
    @ModuleInfo(key: "convOut")  var convOut:  WanCausalConv3d

    init(config: WanVAEConfig = WanVAEConfig()) {
        let b = config.baseDim   // 96
        let inner = b * 4        // 384
        self._convIn.wrappedValue   = WanCausalConv3d(inCh: config.zDim, outCh: inner, k: 3)
        self._midBlock.wrappedValue = WanVAEMidBlock(ch: inner)
        self._upBlocks.wrappedValue = [
            WanVAEUpBlock(inCh: inner, outCh: inner, resnets: 3, upsample: .timeAndSpace(inCh: inner, outCh: b * 2)),
            WanVAEUpBlock(inCh: b * 2, outCh: inner, resnets: 3, upsample: .timeAndSpace(inCh: inner, outCh: b * 2)),
            WanVAEUpBlock(inCh: b * 2, outCh: b * 2, resnets: 3, upsample: .spaceOnly(inCh: b * 2, outCh: b)),
            WanVAEUpBlock(inCh: b,     outCh: b,     resnets: 3, upsample: .none),
        ]
        self._normOut.wrappedValue  = WanVAERMSNorm(ch: b)
        self._convOut.wrappedValue  = WanCausalConv3d(inCh: b, outCh: config.outCh, k: 3)
        super.init()
    }

    // MARK: Sanitize

    static func sanitize(_ raw: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, val) in raw {
            // Drop encoder keys
            guard !key.hasPrefix("encoder.") && !key.hasPrefix("quant_") && key != "post_quant_conv.weight" && key != "post_quant_conv.bias" else { continue }
            // Strip "decoder." prefix
            let k = key.hasPrefix("decoder.") ? String(key.dropFirst("decoder.".count)) : key
            if let (mapped, v) = remapKey(k, val: val) { out[mapped] = v }
        }
        return out
    }

    // swiftlint:disable:next cyclomatic_complexity function_body_length
    private static func remapKey(_ key: String, val: MLXArray) -> (String, MLXArray)? {
        // Top-level conv in/out
        if key == "conv_in.weight"  { return ("convIn.weight",  conv3dTranspose(val)) }
        if key == "conv_in.bias"    { return ("convIn.bias",    val) }
        if key == "conv_out.weight" { return ("convOut.weight", conv3dTranspose(val)) }
        if key == "conv_out.bias"   { return ("convOut.bias",   val) }
        if key == "norm_out.gamma"  { return ("normOut.gamma",  squeezeGamma(val)) }

        // Mid block attention — fused QKV (to_qkv) and output proj (proj) as 1×1 convs
        if key.hasPrefix("mid_block.attentions.0.") {
            let rest = String(key.dropFirst("mid_block.attentions.0.".count))
            let pfx  = "midBlock.attn"
            switch rest {
            case "norm.gamma":
                return ("\(pfx).norm.gamma",   squeezeGamma(val))
            case "to_qkv.weight":
                return ("\(pfx).toQKV.weight", val.reshaped([val.dim(0), val.dim(1)]))
            case "to_qkv.bias":
                return ("\(pfx).toQKV.bias",   val)
            case "proj.weight":
                return ("\(pfx).toOut.weight", val.reshaped([val.dim(0), val.dim(1)]))
            case "proj.bias":
                return ("\(pfx).toOut.bias",   val)
            default:
                return nil
            }
        }

        // Mid block resnets
        if key.hasPrefix("mid_block.resnets.") {
            let s = String(key.dropFirst("mid_block.resnets.".count))
            guard let (i, rest) = splitIdx(s) else { return nil }
            return remapResnet(rest, val: val, prefix: "midBlock.resnets.\(i)")
        }

        // Up blocks
        if key.hasPrefix("up_blocks.") {
            let s = String(key.dropFirst("up_blocks.".count))
            guard let (bi, rest) = splitIdx(s) else { return nil }
            let bpfx = "upBlocks.\(bi)"

            if rest.hasPrefix("resnets.") {
                let s2 = String(rest.dropFirst("resnets.".count))
                guard let (ri, rrest) = splitIdx(s2) else { return nil }
                return remapResnet(rrest, val: val, prefix: "\(bpfx).resnets.\(ri)")
            }
            if rest.hasPrefix("upsamplers.0.time_conv.") {
                let s2 = String(rest.dropFirst("upsamplers.0.time_conv.".count))
                let pfx = "\(bpfx).upsampler.convT"
                if s2 == "weight" { return ("\(pfx).weight", conv3dTranspose(val)) }
                if s2 == "bias"   { return ("\(pfx).bias",   val) }
            }
            if rest.hasPrefix("upsamplers.0.resample.1.") {
                let s2 = String(rest.dropFirst("upsamplers.0.resample.1.".count))
                let pfx = "\(bpfx).upsampler.convS"
                if s2 == "weight" { return ("\(pfx).weight", conv2dTranspose(val)) }
                if s2 == "bias"   { return ("\(pfx).bias",   val) }
            }
        }
        return nil
    }

    private static func remapResnet(_ rest: String, val: MLXArray, prefix: String) -> (String, MLXArray)? {
        switch rest {
        case "conv1.weight":          return ("\(prefix).conv1.weight", conv3dTranspose(val))
        case "conv1.bias":            return ("\(prefix).conv1.bias",   val)
        case "conv2.weight":          return ("\(prefix).conv2.weight", conv3dTranspose(val))
        case "conv2.bias":            return ("\(prefix).conv2.bias",   val)
        case "norm1.gamma":           return ("\(prefix).norm1.gamma",  squeezeGamma(val))
        case "norm2.gamma":           return ("\(prefix).norm2.gamma",  squeezeGamma(val))
        case "conv_shortcut.weight":  return ("\(prefix).shortcut.weight", conv3dTranspose(val))
        case "conv_shortcut.bias":    return ("\(prefix).shortcut.bias",   val)
        default:                      return nil
        }
    }

    // [O, I, kD, kH, kW] → [O, kD, kH, kW, I]  (5-D → 5-D)
    private static func conv3dTranspose(_ v: MLXArray) -> MLXArray {
        v.ndim == 5 ? v.transposed(0, 2, 3, 4, 1) : v
    }
    // [O, I, kH, kW] → [O, kH, kW, I]  (4-D → 4-D)
    private static func conv2dTranspose(_ v: MLXArray) -> MLXArray {
        v.ndim == 4 ? v.transposed(0, 2, 3, 1) : v
    }
    // [C, 1, 1, 1] or [C, 1, 1] → [C]
    private static func squeezeGamma(_ v: MLXArray) -> MLXArray {
        v.ndim > 1 ? v.reshaped([v.dim(0)]) : v
    }
    private static func splitIdx(_ s: String) -> (Int, String)? {
        guard let dot = s.firstIndex(of: ".") else { return nil }
        guard let i = Int(String(s[s.startIndex ..< dot])) else { return nil }
        return (i, String(s[s.index(after: dot)...]))
    }

    // MARK: Forward

    func callAsFunction(_ z: MLXArray) -> MLXArray {
        var h = convIn(z)
        h = midBlock(h)
        for block in upBlocks { h = block(h) }
        h = silu(normOut(h))
        let raw = convOut(h)
        return minimum(maximum(raw, MLXArray(-1.0)), MLXArray(1.0))
    }
}

// MARK: - Upsampler kind

enum WanVAEUpsampleKind {
    case timeAndSpace(inCh: Int, outCh: Int)
    case spaceOnly(inCh: Int, outCh: Int)
    case none
}

// MARK: - Building blocks

nonisolated final class WanCausalConv3d: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray   // [O, kD, kH, kW, I]
    @ModuleInfo(key: "bias")   var bias:   MLXArray   // [O]
    let kD, kH, kW: Int

    init(inCh: Int, outCh: Int, k: Int = 3, kH: Int? = nil, kW: Int? = nil) {
        self.kD = k
        self.kH = kH ?? k
        self.kW = kW ?? k
        self._weight.wrappedValue = MLXRandom.normal([outCh, self.kD, self.kH, self.kW, inCh]) * 0.02
        self._bias.wrappedValue   = MLXArray.zeros([outCh])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Causal temporal padding: left-pad T by kD-1 so the conv sees no future frames.
        var h = x
        if kD > 1 {
            let pad = kD - 1
            // widths: one IntOrPair per dimension in BTHWC order
            h = MLX.padded(h, widths: [.init(0), .init((pad, 0)), .init(0), .init(0), .init(0)])
        }
        let spatialPad = kH > 1 ? (kH - 1) / 2 : 0
        return conv3d(h, weight, stride: [1, 1, 1], padding: [0, spatialPad, spatialPad]) + bias
    }
}

nonisolated final class WanVAERMSNorm: Module {
    @ModuleInfo(key: "gamma") var gamma: MLXArray   // [C]
    init(ch: Int) {
        self._gamma.wrappedValue = MLXArray.ones([ch])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let ms = pow(x.asType(.float32), 2).mean(axis: -1, keepDims: true)
        return gamma.asType(x.dtype) * (x * rsqrt(ms + 1e-6).asType(x.dtype))
    }
}

nonisolated final class WanVAEResNet: Module {
    @ModuleInfo(key: "norm1")     var norm1:     WanVAERMSNorm
    @ModuleInfo(key: "conv1")     var conv1:     WanCausalConv3d
    @ModuleInfo(key: "norm2")     var norm2:     WanVAERMSNorm
    @ModuleInfo(key: "conv2")     var conv2:     WanCausalConv3d
    @ModuleInfo(key: "shortcut")  var shortcut:  WanCausalConv3d?

    init(inCh: Int, outCh: Int) {
        self._norm1.wrappedValue    = WanVAERMSNorm(ch: inCh)
        self._conv1.wrappedValue    = WanCausalConv3d(inCh: inCh, outCh: outCh, k: 3)
        self._norm2.wrappedValue    = WanVAERMSNorm(ch: outCh)
        self._conv2.wrappedValue    = WanCausalConv3d(inCh: outCh, outCh: outCh, k: 3)
        self._shortcut.wrappedValue = inCh != outCh ? WanCausalConv3d(inCh: inCh, outCh: outCh, k: 1, kH: 1, kW: 1) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(silu(norm1(x)))
        h     = conv2(silu(norm2(h)))
        let res = shortcut.map { $0(x) } ?? x
        return h + res
    }
}

nonisolated final class WanVAEUpsampler: Module {
    @ModuleInfo(key: "convT") var convT: WanCausalConv3d?   // temporal pixel-shuffle
    @ModuleInfo(key: "convS") var convS: WanConv2d          // spatial channel-reduce

    init(kind: WanVAEUpsampleKind) {
        switch kind {
        case let .timeAndSpace(inCh, outCh):
            self._convT.wrappedValue = WanCausalConv3d(inCh: inCh, outCh: inCh * 2, k: 3, kH: 1, kW: 1)
            self._convS.wrappedValue = WanConv2d(inCh: inCh, outCh: outCh, k: 3)
        case let .spaceOnly(inCh, outCh):
            self._convT.wrappedValue = nil
            self._convS.wrappedValue = WanConv2d(inCh: inCh, outCh: outCh, k: 3)
        case .none:
            fatalError("WanVAEUpsampler init called with .none")
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        // Temporal pixel-shuffle ×2 if convT exists
        if let cT = convT {
            let mapped = cT(h)   // [B, T, H, W, 2C]
            h = temporalPixelShuffle(mapped, factor: 2)
        }
        // Nearest-neighbour spatial upsample ×2
        h = nearestUpsample2D(h)
        // Spatial conv (channel reduce)
        h = convS(h)
        return h
    }

    private func temporalPixelShuffle(_ x: MLXArray, factor: Int) -> MLXArray {
        let (B, T, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        let outC = C / factor
        // [B, T, H, W, factor, outC] → [B, T, factor, H, W, outC] → [B, T*factor, H, W, outC]
        return x
            .reshaped([B, T, H, W, factor, outC])
            .transposed(0, 1, 4, 2, 3, 5)
            .reshaped([B, T * factor, H, W, outC])
    }

    private func nearestUpsample2D(_ x: MLXArray) -> MLXArray {
        let (B, T, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        let hUp = MLX.tiled(x.reshaped([B, T, H, 1, W, C]),     repetitions: [1, 1, 1, 2, 1, 1])
                     .reshaped([B, T, H * 2, W, C])
        let hwUp = MLX.tiled(hUp.reshaped([B, T, H * 2, W, 1, C]), repetitions: [1, 1, 1, 1, 2, 1])
                      .reshaped([B, T, H * 2, W * 2, C])
        return hwUp
    }
}

/// Conv2d applied frame-by-frame over a [B, T, H, W, C] volume.
nonisolated final class WanConv2d: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray   // [O, kH, kW, I]  (MLX format)
    @ModuleInfo(key: "bias")   var bias:   MLXArray

    init(inCh: Int, outCh: Int, k: Int = 3) {
        self._weight.wrappedValue = MLXRandom.normal([outCh, k, k, inCh]) * 0.02
        self._bias.wrappedValue   = MLXArray.zeros([outCh])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, T, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        let flat = x.reshaped([B * T, H, W, C])
        let pad  = (weight.dim(1) - 1) / 2
        let out  = conv2d(flat, weight, padding: [pad, pad]) + bias
        return out.reshaped([B, T, out.dim(1), out.dim(2), out.dim(3)])
    }
}

nonisolated final class WanVAEMidBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [WanVAEResNet]
    @ModuleInfo(key: "attn")    var attn:    WanVAEMidAttn

    init(ch: Int) {
        self._resnets.wrappedValue = [WanVAEResNet(inCh: ch, outCh: ch), WanVAEResNet(inCh: ch, outCh: ch)]
        self._attn.wrappedValue    = WanVAEMidAttn(ch: ch)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = resnets[0](x)
        h = attn(h)
        return resnets[1](h)
    }
}

/// Spatial self-attention for the VAE mid-block. Processes each frame independently
/// (temporal dim treated as batch). Uses fused QKV to match the diffusers checkpoint
/// (to_qkv: [3C, C, 1, 1], proj: [C, C, 1, 1]).
nonisolated final class WanVAEMidAttn: Module {
    @ModuleInfo(key: "norm")  var norm:  WanVAERMSNorm
    @ModuleInfo(key: "toQKV") var toQKV: Linear   // [3C, C] fused Q+K+V
    @ModuleInfo(key: "toOut") var toOut: Linear

    init(ch: Int) {
        self._norm.wrappedValue  = WanVAERMSNorm(ch: ch)
        self._toQKV.wrappedValue = Linear(ch, ch * 3)
        self._toOut.wrappedValue = Linear(ch, ch)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, T, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        let flat = norm(x).reshaped([B * T, H * W, C])
        let qkv  = toQKV(flat)   // [BT, HW, 3C]
        let q = qkv[0..., 0..., ..<C]
        let k = qkv[0..., 0..., C ..< 2 * C]
        let v = qkv[0..., 0..., (2 * C)...]
        let scale = Float(1.0 / sqrt(Double(C)))
        let w = softmax((matmul(q, k.transposed(0, 2, 1)) * scale).asType(.float32), axis: -1).asType(x.dtype)
        let out = toOut(matmul(w, v))
        return x + out.reshaped([B, T, H, W, C])
    }
}

nonisolated final class WanVAEUpBlock: Module {
    @ModuleInfo(key: "resnets")   var resnets:   [WanVAEResNet]
    @ModuleInfo(key: "upsampler") var upsampler: WanVAEUpsampler?

    init(inCh: Int, outCh: Int, resnets nRes: Int, upsample: WanVAEUpsampleKind) {
        self._resnets.wrappedValue = (0 ..< nRes).map { i in
            WanVAEResNet(inCh: i == 0 ? inCh : outCh, outCh: outCh)
        }
        switch upsample {
        case .none:
            self._upsampler.wrappedValue = nil
        default:
            self._upsampler.wrappedValue = WanVAEUpsampler(kind: upsample)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        if let up = upsampler { h = up(h) }
        return h
    }
}
