import Foundation
import MLX
import MLXNN

// MARK: - Config

nonisolated struct WanDiTConfig: Sendable {
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
#if DEBUG
    static var dbgDone = false   // print internals once
#endif

    @ModuleInfo(key: "patchEmbed")      var patchEmbed:     WanPatchEmbed
    @ModuleInfo(key: "timeEmbed")       var timeEmbed:      WanTimeEmbed
    @ModuleInfo(key: "textEmbed")       var textEmbed:      WanTextEmbed
    @ModuleInfo(key: "timeProj")        var timeProj:       Linear          // condition_embedder.time_proj — shared dim → 6·dim adaLN expansion
    @ModuleInfo(key: "blocks")          var blocks:         [WanDiTBlock]
    @ModuleInfo(key: "outNorm")         var outNorm:        LayerNorm
    @ModuleInfo(key: "outShiftTable")   var outShiftTable:  MLXArray   // [2, dim]
    @ModuleInfo(key: "projOut")         var projOut:        Linear

    init(config: WanDiTConfig = WanDiTConfig()) {
        self.c = config
        self._patchEmbed.wrappedValue    = WanPatchEmbed(config: config)
        self._timeEmbed.wrappedValue     = WanTimeEmbed(config: config)
        self._textEmbed.wrappedValue     = WanTextEmbed(config: config)
        self._timeProj.wrappedValue      = Linear(config.dim, 6 * config.dim)
        self._blocks.wrappedValue        = (0 ..< config.layers).map { _ in WanDiTBlock(config: config) }
        self._outNorm.wrappedValue       = LayerNorm(dimensions: config.dim, eps: 1e-6, affine: false)
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
        case "scale_shift_table":
            return ("outShiftTable", val)
        case "proj_out.weight":
            return ("projOut.weight", val)
        case "proj_out.bias":
            return ("projOut.bias", val)
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
        case "condition_embedder.time_proj.weight":
            return ("timeProj.weight", val)
        case "condition_embedder.time_proj.bias":
            return ("timeProj.bias", val)
        default: break
        }
        // Per-block keys: "blocks.{i}.*"
        guard key.hasPrefix("blocks.") else { return nil }
        let after = String(key.dropFirst("blocks.".count))
        guard let dot = after.firstIndex(of: ".") else { return nil }
        let i    = String(after[after.startIndex ..< dot])
        let rest = String(after[after.index(after: dot)...])
        let pfx  = "blocks.\(i)"
        switch rest {
        case "scale_shift_table":
            return ("\(pfx).shiftTable", val)
        case "norm2.weight":
            return ("\(pfx).norm2.weight", val)
        case "norm2.bias":
            return ("\(pfx).norm2.bias", val)
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
        case "ffn.net.0.proj.weight": return ("\(pfx).ffn.fc1.weight", val)
        case "ffn.net.0.proj.bias":   return ("\(pfx).ffn.fc1.bias", val)
        case "ffn.net.2.weight":      return ("\(pfx).ffn.fc2.weight", val)
        case "ffn.net.2.bias":        return ("\(pfx).ffn.fc2.bias", val)
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

        // 2. Conditioning (matching diffusers WanTransformer3DModel / original Wan-AI):
        //    - textEmbed (2-layer GELU-tanh MLP 4096→1536): applied to the FULL T5 sequence → cross-attn
        //      (P8.3 — the checkpoint's text_proj Linear is an unused leftover key; do not use it)
        //    - timeEmbed: sinusoidal→fc1+silu+fc2 → temb [B, dim=1536]
        //    - timeProj (shared Linear 1536→6·1536): timeProj(silu(temb)) → [B, 6, dim] adaLN
        //      modulation fed to every block (P8.1 — there is no per-block time_embedder in the checkpoint)
        let cross   = textEmbed(textCond)                     // [B, Seq, 1536] — full-sequence MLP for cross-attn
        let timeVec = timeEmbed(timestep)                     // [B, dim=1536] — base time embedding (temb)
        let mods    = timeProj(silu(timeVec)).reshaped([B, 6, c.dim])   // [B, 6, dim] — shared adaLN modulation

        // 3. Build 3-D RoPE
        let headDim = c.dim / c.heads
        let dHW   = 2 * (headDim / 6)
        let dT    = headDim - 2 * dHW
        let (tPos, hPos, wPos) = wanMakePositions(nT: nT, nH: nH, nW: nW)
        let freqsT = wanBuildFreqs(seqLen: nT,   dim: dT)
        let freqsH = wanBuildFreqs(seqLen: nH,   dim: dHW)
        let freqsW = wanBuildFreqs(seqLen: nW,   dim: dHW)

#if DEBUG
        // [DIT-DBG] First-call diagnostics — temb, gate, and h-before-blocks
        let dbgThisCall = !WanDiT.dbgDone
        if dbgThisCall {
            WanDiT.dbgDone = true
            func std(_ a: MLXArray) -> Float {
                let f = a.asType(.float32); eval(f)
                return sqrt(pow(f - f.mean(), 2).mean()).item(Float.self)
            }
            func mn(_ a: MLXArray) -> Float { a.asType(.float32).mean().item(Float.self) }

            // block0 modulation = shared timeProj(silu(temb)) → [B, 6, dim]
            let b0temb = mods.asType(.float32); eval(b0temb)
            print("[DIT-DBG] block0 temb (timeProj mods): mean=\(String(format:"%.4f",mn(b0temb))) std=\(String(format:"%.4f",std(b0temb))) min=\(String(format:"%.4f",b0temb.min().item(Float.self))) max=\(String(format:"%.4f",b0temb.max().item(Float.self)))")

            let st0rows = blocks[0].shiftTable.asType(.float32).reshaped([-1, c.dim])  // [6, dim]
            let gateBias = st0rows[2]
            print("[DIT-DBG] block0 shiftTable gateBias: mean=\(String(format:"%.4f",mn(gateBias))) std=\(String(format:"%.4f",std(gateBias)))")

            let tembGate = mods[0, 2]
            let fullGate = (gateBias + tembGate).asType(.float32); eval(fullGate)
            print("[DIT-DBG] block0 gateMSA (bias+temb): mean=\(String(format:"%.4f",mn(fullGate))) std=\(String(format:"%.4f",std(fullGate)))")

            print("[DIT-DBG] h after patchEmbed: std=\(String(format:"%.4f",std(h)))")
        }
#endif

        // 4. Transformer blocks — each block adds its own scale_shift_table to the shared mods
        for block in blocks {
            h = block(h, cross: cross, mods: mods,
                      freqsT: freqsT, freqsH: freqsH, freqsW: freqsW,
                      tPos: tPos, hPos: hPos, wPos: wPos)
        }

#if DEBUG
        // [DIT-DBG] Post-blocks diagnostics
        if dbgThisCall {
            func fmtStd(_ a: MLXArray) -> String {
                let f = a.asType(.float32); eval(f)
                let s = sqrt(pow(f - f.mean(), 2).mean()).item(Float.self)
                return String(format:"%.4f", s)
            }
            func fmtMn(_ a: MLXArray) -> String {
                String(format:"%.4f", a.asType(.float32).mean().item(Float.self))
            }
            print("[DIT-DBG] h after all blocks: std=\(fmtStd(h))")

            // Block 0 scaleMSA — should be near 0 at σ=1 if weights are correct
            let st0b = blocks[0].shiftTable.asType(.float32).reshaped([-1, c.dim])
            let scaleMSA = (st0b[1] + mods[0, 1]).asType(.float32); eval(scaleMSA)
            print("[DIT-DBG] block0 scaleMSA (full): mean=\(fmtMn(scaleMSA)) std=\(fmtStd(scaleMSA))")

            // outNorm then scale/shift (+ timeVec per P8.4)
            let hNorm = outNorm(h)
            print("[DIT-DBG] h after outNorm: std=\(fmtStd(hNorm))")
            let outMod = (outShiftTable.reshaped([-1, c.dim]).expandedDimensions(axis: 0)
                          + timeVec.expandedDimensions(axis: 1)).asType(.float32); eval(outMod)
            let hScaled = hNorm * (1 + outMod[0..., 1, 0...]) + outMod[0..., 0, 0...]
            print("[DIT-DBG] h after scale+shift (pre-projOut): std=\(fmtStd(hScaled))")
            print("[DIT-DBG] projOut(h) std=\(fmtStd(projOut(hScaled)))")
        }
#endif

        // 5. Final norm + output projection — scale/shift = outShiftTable + temb (P8.4)
        //    outShiftTable: [1, 2, dim] (or [2, dim]) + temb [B, 1, dim] → [B, 2, dim]; chunk(2) → [shift, scale]
        let outMod = outShiftTable.reshaped([-1, c.dim]).expandedDimensions(axis: 0)
                     + timeVec.expandedDimensions(axis: 1)      // [B, 2, dim]
        let shift = outMod[0..., 0, 0...]
        let scale = outMod[0..., 1, 0...]
        h = outNorm(h) * (1 + scale) + shift
        h = projOut(h)   // [B, S, pT*pH*pW*outDim]

        // 6. Unpatch pixel-shuffle: [B, S, pT·pH·pW·outDim] → [B, nT·pT, nH·pH, nW·pW, outDim]
        //    Reshape splits last dim into (pT,pH,pW,outDim), then interleave with spatial grid.
        let pT = c.patchT, pH = c.patchH, pW = c.patchW
        h = h.reshaped([B, nT, nH, nW, pT, pH, pW, c.outDim])
        h = h.transposed(0, 1, 4, 2, 5, 3, 6, 7)  // [B, nT, pT, nH, pH, nW, pW, outDim]
        return h.reshaped([B, nT * pT, nH * pH, nW * pW, c.outDim])
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
    @ModuleInfo(key: "fc1")  var fc1:  Linear   // freqDim → dim      (time_embedder.linear_1)
    @ModuleInfo(key: "fc2")  var fc2:  Linear   // dim → dim          (time_embedder.linear_2)
    let freqDim: Int

    init(config: WanDiTConfig) {
        freqDim = config.freqDim
        self._fc1.wrappedValue  = Linear(config.freqDim, config.dim)
        self._fc2.wrappedValue  = Linear(config.dim, config.dim)
        super.init()
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let sinEmb = sinusoidalEmbed(t, dim: freqDim)
        // One activation between fc1 and fc2 only — mirrors TimestepEmbedding.forward()
        return fc2(silu(fc1(sinEmb)))   // [B, dim=1536] — time_proj expands this to 6·dim
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
        // PixArtAlphaTextProjection uses gelu_tanh (approximate GELU)
        fc2(geluApproximate(fc1(x)))
    }
}

nonisolated final class WanDiTBlock: Module {
    @ModuleInfo(key: "norm1")        var norm1:        LayerNorm    // self-attn pre-norm — affine=False, no weights in checkpoint
    @ModuleInfo(key: "norm2")        var norm2:        LayerNorm    // cross-attn pre-norm — affine, loaded from blocks.{i}.norm2.*
    @ModuleInfo(key: "norm3")        var norm3:        LayerNorm    // FFN pre-norm — affine=False, no weights in checkpoint
    @ModuleInfo(key: "shiftTable")   var shiftTable:   MLXArray    // [6, dim] or [1, 6, dim]
    @ModuleInfo(key: "selfAttn")     var selfAttn:     WanDiTSelfAttn
    @ModuleInfo(key: "crossAttn")    var crossAttn:    WanDiTCrossAttn
    @ModuleInfo(key: "ffn")          var ffn:          WanDiTFFN

    init(config: WanDiTConfig) {
        self._norm1.wrappedValue        = LayerNorm(dimensions: config.dim, eps: 1e-6, affine: false)
        self._norm2.wrappedValue        = LayerNorm(dimensions: config.dim, eps: 1e-6)
        self._norm3.wrappedValue        = LayerNorm(dimensions: config.dim, eps: 1e-6, affine: false)
        self._shiftTable.wrappedValue   = MLXArray.zeros([6, config.dim])
        self._selfAttn.wrappedValue     = WanDiTSelfAttn(config: config)
        self._crossAttn.wrappedValue    = WanDiTCrossAttn(config: config)
        self._ffn.wrappedValue          = WanDiTFFN(config: config)
        super.init()
    }

    func callAsFunction(
        _ x:      MLXArray,    // [B, S, dim]
        cross:    MLXArray,    // [B, Seq, 1536] — textEmbed output for cross-attn
        mods:     MLXArray,    // [B, 6, dim] — shared timeProj(silu(temb)) adaLN modulation
        freqsT:   MLXArray, freqsH: MLXArray, freqsW: MLXArray,
        tPos:     MLXArray, hPos:   MLXArray, wPos:   MLXArray
    ) -> MLXArray {
        // adaLN-zero: modulation = scale_shift_table + shared time_proj (P8.1)
        let mod = shiftTable + mods
        let shiftMSA = mod[0..., 0, 0...]
        let scaleMSA = mod[0..., 1, 0...]
        let gateMSA  = mod[0..., 2, 0...]
        let shiftMLP = mod[0..., 3, 0...]
        let scaleMLP = mod[0..., 4, 0...]
        let gateMLP  = mod[0..., 5, 0...]

        // Self-attention with adaLN-zero — pre-norm is norm1 (affine=False, P8.2)
        var xN = norm1(x)
        xN = xN * (1 + scaleMSA.expandedDimensions(axis: 1)) + shiftMSA.expandedDimensions(axis: 1)
        var h  = x + gateMSA.expandedDimensions(axis: 1) * selfAttn(xN, freqsT: freqsT, freqsH: freqsH, freqsW: freqsW, tPos: tPos, hPos: hPos, wPos: wPos)

        // Cross-attention — pre-norm is norm2, the only affine LayerNorm in the checkpoint (P8.2)
        h = h + crossAttn(norm2(h), cross: cross)

        // FFN with adaLN-zero — pre-norm is norm3 (affine=False, P8.2)
        xN = norm3(h)
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
        self._normQ.wrappedValue = RMSNorm(dimensions: D)   // checkpoint norm_q: [1536]
        self._normK.wrappedValue = RMSNorm(dimensions: D)
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
        // QK-norm on full [B,S,D] before head split (checkpoint norm weight = [D=1536])
        var q = splitH(normQ(toQ(x)))   // [B, S, H, hD]
        var k = splitH(normK(toK(x)))
        let v = splitH(toV(x))

        // RoPE
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
        self._normQ.wrappedValue = RMSNorm(dimensions: D)   // checkpoint norm_q: [1536]
        self._normK.wrappedValue = RMSNorm(dimensions: D)
        self._toQ.wrappedValue   = Linear(D, D)
        self._toK.wrappedValue   = Linear(D, D)
        self._toV.wrappedValue   = Linear(D, D)
        self._toOut.wrappedValue = Linear(D, D)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cross: MLXArray) -> MLXArray {
        let B = x.dim(0)
        // QK-norm on full [B,S,D] before head split (checkpoint norm weight = [D=1536])
        let q = normQ(toQ(x)).reshaped([B, -1, nH, hD]).transposed(0, 2, 1, 3)
        let k = normK(toK(cross)).reshaped([B, -1, nH, hD]).transposed(0, 2, 1, 3)
        let v = toV(cross).reshaped([B, -1, nH, hD]).transposed(0, 2, 1, 3)
        let scale = Float(1.0 / sqrt(Double(hD)))
        let w = softmax((matmul(q, k.transposed(0, 1, 3, 2)) * scale).asType(.float32), axis: -1).asType(x.dtype)
        return toOut(matmul(w, v).transposed(0, 2, 1, 3).reshaped([B, -1, D]))
    }
}

nonisolated final class WanDiTFFN: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear   // dim → ffnDim
    @ModuleInfo(key: "fc2") var fc2: Linear   // ffnDim → dim
    init(config: WanDiTConfig) {
        self._fc1.wrappedValue = Linear(config.dim, config.ffnDim)
        self._fc2.wrappedValue = Linear(config.ffnDim, config.dim)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return fc2(gelu(fc1(x)))
    }
}
