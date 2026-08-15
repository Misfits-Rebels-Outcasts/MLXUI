import Foundation
import MLX
import MLXNN

// MARK: - Config

struct WanDiTConfig: Sendable {
    var dim:      Int = 1536    // 12 heads × 128 head_dim
    var ffnDim:   Int = 8960
    var freqDim:  Int = 256     // sinusoidal timestep embedding dim
    var textDim:  Int = 4096    // T5 output dim
    var outDim:   Int = 16      // latent channel count
    var heads:    Int = 12
    var layers:   Int = 30
    var patchT:   Int = 1
    var patchH:   Int = 2
    var patchW:   Int = 2
    var headDim:  Int { dim / heads }
}

// MARK: - WanDiT  (top-level model)

/// WAN 2.1 T2V 1.3B DiT backbone. Weights from
/// `Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/diffusion_pytorch_model-*.safetensors`.
nonisolated final class WanDiT: Module {
    let c: WanDiTConfig

    @ModuleInfo(key: "patchEmbed")      var patchEmbed:     WanPatchEmbed
    @ModuleInfo(key: "timeEmbed")       var timeEmbed:      WanTimeEmbed
    @ModuleInfo(key: "textEmbed")       var textEmbed:      WanTextEmbed
    @ModuleInfo(key: "blocks")          var blocks:         [WanDiTBlock]
    @ModuleInfo(key: "outNorm")         var outNorm:        LayerNorm
    @ModuleInfo(key: "outShiftTable")   var outShiftTable:  MLXArray   // [2, dim]
    @ModuleInfo(key: "projOut")         var projOut:        Linear

    init(config: WanDiTConfig = WanDiTConfig()) {
        self.c = config
        self._patchEmbed.wrappedValue    = WanPatchEmbed(config: config)
        self._timeEmbed.wrappedValue     = WanTimeEmbed(config: config)
        self._textEmbed.wrappedValue     = WanTextEmbed(config: config)
        self._blocks.wrappedValue        = (0 ..< config.layers).map { _ in WanDiTBlock(config: config) }
        self._outNorm.wrappedValue       = LayerNorm(dimensions: config.dim, eps: 1e-6)
        self._outShiftTable.wrappedValue = MLXArray.zeros([2, config.dim])
        self._projOut.wrappedValue       = Linear(config.dim, config.patchT * config.patchH * config.patchW * config.outDim)
        super.init()
    }

    // MARK: Sanitize

    static func sanitize(_ raw: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, val) in raw {
            if let (k, v) = remap(key, val: val) { out[k] = v }
        }
        return out
    }

    // swiftlint:disable:next cyclomatic_complexity
    private static func remap(_ key: String, val: MLXArray) -> (String, MLXArray)? {
        // Top-level keys
        switch key {
        case "patch_embedding.weight":
            // [O, I, kD, kH, kW] → [O, kD, kH, kW, I]
            return ("patchEmbed.weight", val.transposed(0, 2, 3, 4, 1))
        case "patch_embedding.bias":
            return ("patchEmbed.bias", val)
        case "norm_out.linear.weight":
            return ("outShiftTable", val.reshaped([2, val.dim(0) / 2]))
        case "proj_out.weight":
            return ("projOut.weight", val)
        case "proj_out.bias":
            return ("projOut.bias", val)
        case "condition_embedder.time_proj.weight":
            return ("timeEmbed.proj.weight", val)
        case "condition_embedder.time_proj.bias":
            return ("timeEmbed.proj.bias", val)
        case "condition_embedder.time_embedder.linear_1.weight":
            return ("timeEmbed.fc1.weight", val)
        case "condition_embedder.time_embedder.linear_1.bias":
            return ("timeEmbed.fc1.bias", val)
        case "condition_embedder.time_embedder.linear_2.weight":
            return ("timeEmbed.fc2.weight", val)
        case "condition_embedder.time_embedder.linear_2.bias":
            return ("timeEmbed.fc2.bias", val)
        case "condition_embedder.text_embedder.linear_1.weight":
            return ("textEmbed.fc1.weight", val)
        case "condition_embedder.text_embedder.linear_1.bias":
            return ("textEmbed.fc1.bias", val)
        case "condition_embedder.text_embedder.linear_2.weight":
            return ("textEmbed.fc2.weight", val)
        case "condition_embedder.text_embedder.linear_2.bias":
            return ("textEmbed.fc2.bias", val)
        default: break
        }
        // Per-block keys: "transformer_blocks.{i}.*"
        guard key.hasPrefix("transformer_blocks.") else { return nil }
        let after = String(key.dropFirst("transformer_blocks.".count))
        guard let dot = after.firstIndex(of: ".") else { return nil }
        let i    = String(after[after.startIndex ..< dot])
        let rest = String(after[after.index(after: dot)...])
        let pfx  = "blocks.\(i)"
        switch rest {
        case "scale_shift_table":
            return ("\(pfx).shiftTable", val)
        case "norm2.weight":
            return ("\(pfx).norm.weight", val)
        case "norm2.bias":
            return ("\(pfx).norm.bias", val)
        // Self-attention
        case "attn1.norm_q.weight":  return ("\(pfx).selfAttn.normQ.weight", val)
        case "attn1.norm_k.weight":  return ("\(pfx).selfAttn.normK.weight", val)
        case "attn1.to_q.weight":    return ("\(pfx).selfAttn.toQ.weight", val)
        case "attn1.to_q.bias":      return ("\(pfx).selfAttn.toQ.bias", val)
        case "attn1.to_k.weight":    return ("\(pfx).selfAttn.toK.weight", val)
        case "attn1.to_k.bias":      return ("\(pfx).selfAttn.toK.bias", val)
        case "attn1.to_v.weight":    return ("\(pfx).selfAttn.toV.weight", val)
        case "attn1.to_v.bias":      return ("\(pfx).selfAttn.toV.bias", val)
        case "attn1.to_out.0.weight": return ("\(pfx).selfAttn.toOut.weight", val)
        case "attn1.to_out.0.bias":   return ("\(pfx).selfAttn.toOut.bias", val)
        // Cross-attention
        case "attn2.norm_q.weight":  return ("\(pfx).crossAttn.normQ.weight", val)
        case "attn2.norm_k.weight":  return ("\(pfx).crossAttn.normK.weight", val)
        case "attn2.to_q.weight":    return ("\(pfx).crossAttn.toQ.weight", val)
        case "attn2.to_q.bias":      return ("\(pfx).crossAttn.toQ.bias", val)
        case "attn2.to_k.weight":    return ("\(pfx).crossAttn.toK.weight", val)
        case "attn2.to_k.bias":      return ("\(pfx).crossAttn.toK.bias", val)
        case "attn2.to_v.weight":    return ("\(pfx).crossAttn.toV.weight", val)
        case "attn2.to_v.bias":      return ("\(pfx).crossAttn.toV.bias", val)
        case "attn2.to_out.0.weight": return ("\(pfx).crossAttn.toOut.weight", val)
        case "attn2.to_out.0.bias":   return ("\(pfx).crossAttn.toOut.bias", val)
        // FFN — diffusers uses net.0.proj (gated proj) and net.2 (output)
        case "ff.net.0.proj.weight": return ("\(pfx).ffn.fc1.weight", val)
        case "ff.net.0.proj.bias":   return ("\(pfx).ffn.fc1.bias", val)
        case "ff.net.2.weight":      return ("\(pfx).ffn.fc2.weight", val)
        case "ff.net.2.bias":        return ("\(pfx).ffn.fc2.bias", val)
        default:                     return nil
        }
    }

    // MARK: Forward

    /// - Parameters:
    ///   - x:         Latent input [B, T, H, W, inCh] (e.g. [1, T_lat, 60, 104, 16]).
    ///   - timestep:  Noise level [B] float.
    ///   - textCond:  T5 encoder output [B, Seq, 4096].
    /// - Returns: Noise prediction [B, T, H, W, outCh].
    func callAsFunction(
        _ x:       MLXArray,
        timestep:  MLXArray,
        textCond:  MLXArray
    ) -> MLXArray {
        let B  = x.dim(0)
        let nT = x.dim(1)
        let nH = x.dim(2) / c.patchH
        let nW = x.dim(3) / c.patchW

        // 1. Patch embed: [B, nT, nH, nW, dim]
        var h = patchEmbed(x).reshaped([B, nT * nH * nW, c.dim])

        // 2. Time conditioning [B, 6*dim] and text cross-attention cond [B, Seq, dim]
        let temb = timeEmbed(timestep)         // [B, 6*dim]
        let cross = textEmbed(textCond)        // [B, Seq, dim]

        // 3. Build 3-D RoPE
        let headDim = c.dim / c.heads
        let dHW   = 2 * (headDim / 6)
        let dT    = headDim - 2 * dHW
        let (tPos, hPos, wPos) = wanMakePositions(nT: nT, nH: nH, nW: nW)
        let freqsT = wanBuildFreqs(seqLen: nT,   dim: dT)
        let freqsH = wanBuildFreqs(seqLen: nH,   dim: dHW)
        let freqsW = wanBuildFreqs(seqLen: nW,   dim: dHW)

        // 4. Transformer blocks
        for block in blocks {
            h = block(h, cross: cross, temb: temb,
                      freqsT: freqsT, freqsH: freqsH, freqsW: freqsW,
                      tPos: tPos, hPos: hPos, wPos: wPos)
        }

        // 5. Final norm + output projection
        let shift = outShiftTable[0]
        let scale = outShiftTable[1]
        h = outNorm(h) * (1 + scale) + shift
        h = projOut(h)   // [B, S, pT*pH*pW*outDim]

        // 6. Reshape back to [B, nT, nH*pH, nW*pW, outDim]
        return h.reshaped([B, nT, nH, nW, c.outDim])
    }
}

// MARK: - Sub-modules

nonisolated final class WanPatchEmbed: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray   // [O, kT, kH, kW, I]
    @ModuleInfo(key: "bias")   var bias:   MLXArray   // [O]
    let sT, sH, sW: Int

    init(config: WanDiTConfig) {
        let (kT, kH, kW) = (config.patchT, config.patchH, config.patchW)
        sT = kT; sH = kH; sW = kW
        self._weight.wrappedValue = MLXRandom.normal([config.dim, kT, kH, kW, config.outDim]) * 0.02
        self._bias.wrappedValue   = MLXArray.zeros([config.dim])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, T, H, W, C] → [B, T/sT, H/sH, W/sW, dim]
        conv3d(x, weight, stride: [sT, sH, sW]) + bias
    }
}

nonisolated final class WanTimeEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Linear   // freq_dim → freq_dim
    @ModuleInfo(key: "fc1")  var fc1:  Linear   // freq_dim → dim
    @ModuleInfo(key: "fc2")  var fc2:  Linear   // dim → 6*dim
    let freqDim: Int

    init(config: WanDiTConfig) {
        freqDim = config.freqDim
        self._proj.wrappedValue = Linear(config.freqDim, config.freqDim)
        self._fc1.wrappedValue  = Linear(config.freqDim, config.dim)
        self._fc2.wrappedValue  = Linear(config.dim, 6 * config.dim)
        super.init()
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let sinEmb = sinusoidalEmbed(t, dim: freqDim)
        return fc2(silu(fc1(silu(proj(sinEmb)))))
    }

    private func sinusoidalEmbed(_ t: MLXArray, dim: Int) -> MLXArray {
        let half  = dim / 2
        let freqs = exp(-log(10000.0) * MLXArray(Int32(0) ..< Int32(half)).asType(.float32) / Float(half))
        let args  = t.asType(.float32).expandedDimensions(axis: -1) * freqs.expandedDimensions(axis: 0)
        return MLX.concatenated([cos(args), sin(args)], axis: -1).asType(t.dtype)
    }
}

nonisolated final class WanTextEmbed: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear   // textDim → dim
    @ModuleInfo(key: "fc2") var fc2: Linear   // dim → dim
    init(config: WanDiTConfig) {
        self._fc1.wrappedValue = Linear(config.textDim, config.dim)
        self._fc2.wrappedValue = Linear(config.dim, config.dim)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(silu(fc1(x)))
    }
}

nonisolated final class WanDiTBlock: Module {
    @ModuleInfo(key: "norm")       var norm:       LayerNorm
    @ModuleInfo(key: "shiftTable") var shiftTable: MLXArray    // [6, dim]
    @ModuleInfo(key: "selfAttn")   var selfAttn:   WanDiTSelfAttn
    @ModuleInfo(key: "crossAttn")  var crossAttn:  WanDiTCrossAttn
    @ModuleInfo(key: "ffn")        var ffn:        WanDiTFFN

    init(config: WanDiTConfig) {
        self._norm.wrappedValue       = LayerNorm(dimensions: config.dim, eps: 1e-6)
        self._shiftTable.wrappedValue = MLXArray.zeros([6, config.dim])
        self._selfAttn.wrappedValue   = WanDiTSelfAttn(config: config)
        self._crossAttn.wrappedValue  = WanDiTCrossAttn(config: config)
        self._ffn.wrappedValue        = WanDiTFFN(config: config)
        super.init()
    }

    func callAsFunction(
        _ x:     MLXArray,    // [B, S, dim]
        cross:   MLXArray,    // [B, Seq, dim]
        temb:    MLXArray,    // [B, 6*dim]
        freqsT:  MLXArray, freqsH: MLXArray, freqsW: MLXArray,
        tPos:    MLXArray, hPos:   MLXArray, wPos:   MLXArray
    ) -> MLXArray {
        let B   = x.dim(0)
        let mod = (shiftTable.expandedDimensions(axis: 0) + temb.reshaped([B, 6, -1]))
        // mod: [B, 6, dim] — split into 6 vectors each [B, 1, dim]
        let shiftMSA = mod[0..., 0, 0...]
        let scaleMSA = mod[0..., 1, 0...]
        let gateMSA  = mod[0..., 2, 0...]
        let shiftMLP = mod[0..., 3, 0...]
        let scaleMLP = mod[0..., 4, 0...]
        let gateMLP  = mod[0..., 5, 0...]

        // Self-attention with adaLN
        var xN = norm(x)
        xN = xN * (1 + scaleMSA.expandedDimensions(axis: 1)) + shiftMSA.expandedDimensions(axis: 1)
        var h  = x + gateMSA.expandedDimensions(axis: 1) * selfAttn(xN, freqsT: freqsT, freqsH: freqsH, freqsW: freqsW, tPos: tPos, hPos: hPos, wPos: wPos)

        // Cross-attention (no modulation)
        h = h + crossAttn(norm(h), cross: cross)

        // FFN with adaLN
        xN = norm(h)
        xN = xN * (1 + scaleMLP.expandedDimensions(axis: 1)) + shiftMLP.expandedDimensions(axis: 1)
        h  = h + gateMLP.expandedDimensions(axis: 1) * ffn(xN)
        return h
    }
}

// MARK: - Attention modules

nonisolated final class WanDiTSelfAttn: Module {
    @ModuleInfo(key: "normQ") var normQ: RMSNorm
    @ModuleInfo(key: "normK") var normK: RMSNorm
    @ModuleInfo(key: "toQ")   var toQ:   Linear
    @ModuleInfo(key: "toK")   var toK:   Linear
    @ModuleInfo(key: "toV")   var toV:   Linear
    @ModuleInfo(key: "toOut") var toOut:  Linear
    let nH, hD, D: Int

    init(config: WanDiTConfig) {
        nH = config.heads; hD = config.dim / config.heads; D = config.dim
        self._normQ.wrappedValue = RMSNorm(dimensions: hD)
        self._normK.wrappedValue = RMSNorm(dimensions: hD)
        self._toQ.wrappedValue   = Linear(D, D)
        self._toK.wrappedValue   = Linear(D, D)
        self._toV.wrappedValue   = Linear(D, D)
        self._toOut.wrappedValue = Linear(D, D)
        super.init()
    }

    private func splitH(_ x: MLXArray) -> MLXArray {
        x.reshaped([x.dim(0), -1, nH, hD])  // [B, S, H, hD]
    }
    private func mergeH(_ x: MLXArray) -> MLXArray {
        x.reshaped([x.dim(0), -1, D])
    }

    func callAsFunction(
        _ x: MLXArray,
        freqsT: MLXArray, freqsH: MLXArray, freqsW: MLXArray,
        tPos:   MLXArray, hPos:   MLXArray, wPos:   MLXArray
    ) -> MLXArray {
        var q = splitH(toQ(x))   // [B, S, H, hD]
        var k = splitH(toK(x))
        let v = splitH(toV(x))

        // QK-norm then RoPE
        q = normQ(q)
        k = normK(k)
        q = wanApplyRoPE3D(q, freqsT: freqsT, freqsH: freqsH, freqsW: freqsW, tPos: tPos, hPos: hPos, wPos: wPos)
        k = wanApplyRoPE3D(k, freqsT: freqsT, freqsH: freqsH, freqsW: freqsW, tPos: tPos, hPos: hPos, wPos: wPos)

        // SDPA: [B, H, S, S] — transpose to [B, H, S, hD]
        let qT = q.transposed(0, 2, 1, 3)   // [B, H, S, hD]
        let kT = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)
        let scale = Float(1.0 / sqrt(Double(hD)))
        let w = softmax((matmul(qT, kT.transposed(0, 1, 3, 2)) * scale).asType(.float32), axis: -1).asType(x.dtype)
        return toOut(mergeH(matmul(w, vT).transposed(0, 2, 1, 3)))
    }
}

nonisolated final class WanDiTCrossAttn: Module {
    @ModuleInfo(key: "normQ") var normQ: RMSNorm
    @ModuleInfo(key: "normK") var normK: RMSNorm
    @ModuleInfo(key: "toQ")   var toQ:   Linear
    @ModuleInfo(key: "toK")   var toK:   Linear
    @ModuleInfo(key: "toV")   var toV:   Linear
    @ModuleInfo(key: "toOut") var toOut:  Linear
    let nH, hD, D: Int

    init(config: WanDiTConfig) {
        nH = config.heads; hD = config.dim / config.heads; D = config.dim
        self._normQ.wrappedValue = RMSNorm(dimensions: hD)
        self._normK.wrappedValue = RMSNorm(dimensions: hD)
        self._toQ.wrappedValue   = Linear(D, D)
        self._toK.wrappedValue   = Linear(config.textDim, D)
        self._toV.wrappedValue   = Linear(config.textDim, D)
        self._toOut.wrappedValue = Linear(D, D)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cross: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let q = normQ(toQ(x).reshaped([B, -1, nH, hD])).transposed(0, 2, 1, 3)
        let k = normK(toK(cross).reshaped([B, -1, nH, hD])).transposed(0, 2, 1, 3)
        let v = toV(cross).reshaped([B, -1, nH, hD]).transposed(0, 2, 1, 3)
        let scale = Float(1.0 / sqrt(Double(hD)))
        let w = softmax((matmul(q, k.transposed(0, 1, 3, 2)) * scale).asType(.float32), axis: -1).asType(x.dtype)
        return toOut(matmul(w, v).transposed(0, 2, 1, 3).reshaped([B, -1, D]))
    }
}

nonisolated final class WanDiTFFN: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear   // dim → 2*ffnDim (GEGLU)
    @ModuleInfo(key: "fc2") var fc2: Linear   // ffnDim → dim
    init(config: WanDiTConfig) {
        self._fc1.wrappedValue = Linear(config.dim, 2 * config.ffnDim)
        self._fc2.wrappedValue = Linear(config.ffnDim, config.dim)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g     = fc1(x)
        let half  = g.dim(-1) / 2
        let gate  = g[.ellipsis, 0 ..< half]
        let value = g[.ellipsis, half...]
        return fc2(gelu(gate) * value)
    }
}
