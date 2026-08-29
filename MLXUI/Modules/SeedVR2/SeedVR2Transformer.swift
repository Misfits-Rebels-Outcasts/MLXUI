import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Helpers

/// SwiGLU hidden dim: round_up(2/3 * dim * expandRatio, 256).
private func svHidden(_ dim: Int, expand: Int) -> Int {
    let raw = 2 * dim * expand / 3
    return (raw + 255) / 256 * 256
}

/// Affine-free LayerNorm over the last axis, computed in float32 for stability.
/// Equivalent to `nn.LayerNorm(affine=False)`. No checkpoint weights.
/// Required before every ada scale/shift to prevent exponential token growth across blocks.
private func preNorm(_ x: MLXArray) -> MLXArray {
    let xF   = x.asType(.float32)
    let mean = xF.mean(axes: [-1], keepDims: true)
    let diff = xF - mean
    let std  = MLX.sqrt((diff * diff).mean(axes: [-1], keepDims: true) + 1e-5)
    return (diff / std).asType(x.dtype)
}

// MARK: - RoPE

/// Per-block learned RoPE. The checkpoint stores `freqs` as [headDim/6] base theta values
/// for the H and W spatial components of a 3-D T/H/W RoPE.
nonisolated final class SeedVR2RoPE: Module {
    @ParameterInfo(key: "freqs") var freqs: MLXArray  // [headDim/6]

    init(ropeDim: Int) {
        self._freqs.wrappedValue = MLXArray.zeros([ropeDim / 6])
        super.init()
    }

    func apply(_ x: MLXArray, nT: Int, nH: Int, nW: Int) -> MLXArray {
        let d     = x.dim(-1)
        let dHalf = freqs.dim(0)  // 21 = d/6
        let dHW   = 2 * dHalf    // 42
        let dT    = d - 2 * dHW  // 44
        let L     = nT * nH * nW

        let (_, hPos, wPos) = wanMakePositions(nT: nT, nH: nH, nW: nW)
        let fF = freqs.asType(.float32)
        let aH = hPos.asType(.float32).expandedDimensions(axis: 1) * fF.expandedDimensions(axis: 0)
        let aW = wPos.asType(.float32).expandedDimensions(axis: 1) * fF.expandedDimensions(axis: 0)
        let aT = MLXArray.zeros([L, dT / 2])

        let dtype = x.dtype
        let fT = aT.expandedDimensions(axis: 0).expandedDimensions(axis: 2).asType(dtype)
        let fH = aH.expandedDimensions(axis: 0).expandedDimensions(axis: 2).asType(dtype)
        let fW = aW.expandedDimensions(axis: 0).expandedDimensions(axis: 2).asType(dtype)

        return MLX.concatenated([
            wanApplyRoPE1D(x[.ellipsis, 0 ..< dT], freqs: fT),
            wanApplyRoPE1D(x[.ellipsis, dT ..< dT + dHW], freqs: fH),
            wanApplyRoPE1D(x[.ellipsis, dT + dHW ..< d], freqs: fW),
        ], axis: -1)
    }
}

// MARK: - Ada Modulation

/// Six learned per-block adaptive modulation vectors.
/// These are ADDED to the global time-conditioned emb_in output (which provides the
/// timestep-varying base modulation). Each vector is [dim].
nonisolated final class SeedVR2AdaParams: Module {
    @ParameterInfo(key: "attn_shift") var attnShift: MLXArray
    @ParameterInfo(key: "attn_scale") var attnScale: MLXArray
    @ParameterInfo(key: "attn_gate")  var attnGate:  MLXArray
    @ParameterInfo(key: "mlp_shift")  var mlpShift:  MLXArray
    @ParameterInfo(key: "mlp_scale")  var mlpScale:  MLXArray
    @ParameterInfo(key: "mlp_gate")   var mlpGate:   MLXArray

    init(dim: Int) {
        self._attnShift.wrappedValue = MLXArray.zeros([dim])
        self._attnScale.wrappedValue = MLXArray.zeros([dim])
        self._attnGate.wrappedValue  = MLXArray.zeros([dim])
        self._mlpShift.wrappedValue  = MLXArray.zeros([dim])
        self._mlpScale.wrappedValue  = MLXArray.zeros([dim])
        self._mlpGate.wrappedValue   = MLXArray.zeros([dim])
        super.init()
    }
}

nonisolated final class SeedVR2Ada: Module {
    @ModuleInfo(key: "params_vid") var paramsVid: SeedVR2AdaParams
    @ModuleInfo(key: "params_txt") var paramsTxt: SeedVR2AdaParams?

    init(dim: Int, isMM: Bool) {
        self._paramsVid.wrappedValue = SeedVR2AdaParams(dim: dim)
        if isMM { self._paramsTxt.wrappedValue = SeedVR2AdaParams(dim: dim) }
        super.init()
    }
}

// MARK: - SwiGLU MLP (split gate/up projections)

/// One SwiGLU branch: silu(proj_in_gate(x)) × proj_in(x), then proj_out.
/// Gate and up are stored as separate Linear layers in the checkpoint.
nonisolated final class SeedVR2SwiGLUBranch: Module {
    @ModuleInfo(key: "proj_in")      var projIn:     Linear
    @ModuleInfo(key: "proj_in_gate") var projInGate: Linear
    @ModuleInfo(key: "proj_out")     var projOut:    Linear

    init(dim: Int, hidDim: Int) {
        self._projIn.wrappedValue     = Linear(dim, hidDim, bias: false)
        self._projInGate.wrappedValue = Linear(dim, hidDim, bias: false)
        self._projOut.wrappedValue    = Linear(hidDim, dim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = MLXNN.silu(projInGate(x))
        return projOut(gate * projIn(x))
    }
}

nonisolated final class SeedVR2MMSwiGLU: Module {
    @ModuleInfo(key: "vid") var vid: SeedVR2SwiGLUBranch
    @ModuleInfo(key: "txt") var txt: SeedVR2SwiGLUBranch?

    init(dim: Int, expandRatio: Int, isMM: Bool) {
        let hid = svHidden(dim, expand: expandRatio)
        self._vid.wrappedValue = SeedVR2SwiGLUBranch(dim: dim, hidDim: hid)
        if isMM { self._txt.wrappedValue = SeedVR2SwiGLUBranch(dim: dim, hidDim: hid) }
        super.init()
    }
}

// MARK: - Multi-Modal Attention

/// Joint vid+txt global attention with:
/// - Per-head QK RMSNorm (norm_q_vid, norm_k_vid) applied before RoPE
/// - 3-D RoPE on video tokens
// Checkpoint confirms ALL 32 blocks have txt attention components — they are never optional.
nonisolated final class SeedVR2MMAttention: Module {
    @ModuleInfo(key: "proj_qkv_vid") var projQkvVid: Linear
    @ModuleInfo(key: "proj_qkv_txt") var projQkvTxt: Linear
    @ModuleInfo(key: "proj_out_vid") var projOutVid: Linear
    @ModuleInfo(key: "proj_out_txt") var projOutTxt: Linear
    @ModuleInfo(key: "norm_q_vid")   var normQVid:   RMSNorm
    @ModuleInfo(key: "norm_k_vid")   var normKVid:   RMSNorm
    @ModuleInfo(key: "norm_q_txt")   var normQTxt:   RMSNorm
    @ModuleInfo(key: "norm_k_txt")   var normKTxt:   RMSNorm
    @ModuleInfo(key: "rope")         var rope:       SeedVR2RoPE

    let heads: Int; let headDim: Int

    init(dim: Int, heads: Int, headDim: Int, ropeDim: Int) {
        self.heads = heads; self.headDim = headDim
        let qkvDim = heads * headDim
        self._projQkvVid.wrappedValue = Linear(dim, 3 * qkvDim, bias: false)
        self._projQkvTxt.wrappedValue = Linear(dim, 3 * qkvDim, bias: false)
        self._projOutVid.wrappedValue = Linear(qkvDim, dim, bias: true)
        self._projOutTxt.wrappedValue = Linear(qkvDim, dim, bias: true)
        self._normQVid.wrappedValue   = RMSNorm(dimensions: headDim)
        self._normKVid.wrappedValue   = RMSNorm(dimensions: headDim)
        self._normQTxt.wrappedValue   = RMSNorm(dimensions: headDim)
        self._normKTxt.wrappedValue   = RMSNorm(dimensions: headDim)
        self._rope.wrappedValue       = SeedVR2RoPE(ropeDim: ropeDim)
        super.init()
    }

    func callAsFunction(
        vid: MLXArray, txt: MLXArray, nT: Int, nH: Int, nW: Int
    ) -> (vid: MLXArray, txt: MLXArray) {
        let B = vid.dim(0), H = heads, D = headDim
        let Lv = vid.dim(1), Lt = txt.dim(1)

        func splitQKV(_ x: MLXArray, L: Int) -> (MLXArray, MLXArray, MLXArray) {
            let r = x.reshaped([B, L, 3, H, D]).transposed(2, 0, 3, 1, 4)
            return (r[0], r[1], r[2])
        }

        var (vq, vk, vv) = splitQKV(projQkvVid(vid), L: Lv)
        vq = normQVid(vq)
        vk = normKVid(vk)
        vq = rope.apply(vq.transposed(0, 2, 1, 3), nT: nT, nH: nH, nW: nW).transposed(0, 2, 1, 3)
        vk = rope.apply(vk.transposed(0, 2, 1, 3), nT: nT, nH: nH, nW: nW).transposed(0, 2, 1, 3)

        var (tq, tk, tv) = splitQKV(projQkvTxt(txt), L: Lt)
        tq = normQTxt(tq); tk = normKTxt(tk)
        let q = MLX.concatenated([vq, tq], axis: 2)
        let k = MLX.concatenated([vk, tk], axis: 2)
        let v = MLX.concatenated([vv, tv], axis: 2)

        let scale = 1.0 / sqrt(Float(D))
        let out  = scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        let flat = out.transposed(0, 2, 1, 3).reshaped([B, -1, H * D])
        return (projOutVid(flat[0..., ..<Lv, 0...]),
                projOutTxt(flat[0..., Lv ..< Lv + Lt, 0...]))
    }
}

// MARK: - Transformer Block

nonisolated final class SeedVR2TransformerBlock: Module {
    @ModuleInfo(key: "attn") var attn: SeedVR2MMAttention
    @ModuleInfo(key: "mlp")  var mlp:  SeedVR2MMSwiGLU
    @ModuleInfo(key: "ada")  var ada:  SeedVR2Ada

    init(c: SeedVR2Config, isMM: Bool) {
        self._attn.wrappedValue = SeedVR2MMAttention(
            dim: c.vidDim, heads: c.heads, headDim: c.headDim, ropeDim: c.ropeDim)
        self._mlp.wrappedValue  = SeedVR2MMSwiGLU(dim: c.vidDim, expandRatio: c.expandRatio, isMM: isMM)
        self._ada.wrappedValue  = SeedVR2Ada(dim: c.vidDim, isMM: isMM)
        super.init()
    }

    /// emb: [B, 6*dim] global time-conditioned modulation from emb_in.
    /// Ada for this block = global emb reshaped to [B, 6, dim] + per-block learned offset.
    func callAsFunction(
        vid: MLXArray, txt: MLXArray, emb: MLXArray, nT: Int, nH: Int, nW: Int
    ) -> (vid: MLXArray, txt: MLXArray) {
        let B = vid.dim(0)
        let dim = vid.dim(-1)
        let e = emb.reshaped([B, 6, dim]).asType(vid.dtype)  // [B, 6, dim]

        // Global slot i + per-block learned offset → [B, 1, dim] for broadcasting.
        // preNorm() normalizes tokens to unit std before each scale/shift, preventing
        // exponential growth. Same pattern as WanVideo's norm1/norm3 (affine=false LayerNorm).
        func vSlot(_ i: Int, _ p: MLXArray) -> MLXArray {
            (e[0..., i, 0...] + p.expandedDimensions(axis: 0)).expandedDimensions(axis: 1)
        }
        let v = ada.paramsVid
        let vShiftA = vSlot(0, v.attnShift); let vScaleA = vSlot(1, v.attnScale)
        let vGateA  = vSlot(2, v.attnGate)
        let vShiftM = vSlot(3, v.mlpShift);  let vScaleM = vSlot(4, v.mlpScale)
        let vGateM  = vSlot(5, v.mlpGate)

        var vidX = vid, txtX = txt

        if let pt = ada.paramsTxt {
            // MM block: modulate both vid and txt
            func tSlot(_ i: Int, _ p: MLXArray) -> MLXArray {
                (e[0..., i, 0...] + p.expandedDimensions(axis: 0)).expandedDimensions(axis: 1)
            }
            let tShiftA = tSlot(0, pt.attnShift); let tScaleA = tSlot(1, pt.attnScale)
            let tGateA  = tSlot(2, pt.attnGate)
            let tShiftM = tSlot(3, pt.mlpShift);  let tScaleM = tSlot(4, pt.mlpScale)
            let tGateM  = tSlot(5, pt.mlpGate)

            let vAttnIn = preNorm(vidX) * (1 + vScaleA) + vShiftA
            let tAttnIn = preNorm(txtX) * (1 + tScaleA) + tShiftA
            let (vAttnOut, tAttnOut) = attn(vid: vAttnIn, txt: tAttnIn, nT: nT, nH: nH, nW: nW)
            vidX = vidX + vGateA * vAttnOut
            txtX = txtX + tGateA * tAttnOut

            let vMlpOut = mlp.vid(preNorm(vidX) * (1 + vScaleM) + vShiftM)
            let tMlpOut = mlp.txt!(preNorm(txtX) * (1 + tScaleM) + tShiftM)
            vidX = vidX + vGateM * vMlpOut
            txtX = txtX + tGateM * tMlpOut
        } else {
            // Non-MM block: vid gets full ada modulation; txt enters attention pre-normalized
            // (no ada params_txt) to keep V values bounded.
            let vAttnIn = preNorm(vidX) * (1 + vScaleA) + vShiftA
            let (vAttnOut, tAttnOut) = attn(vid: vAttnIn, txt: preNorm(txtX), nT: nT, nH: nH, nW: nW)
            vidX = vidX + vGateA * vAttnOut
            txtX = txtX + tAttnOut
            vidX = vidX + vGateM * mlp.vid(preNorm(vidX) * (1 + vScaleM) + vShiftM)
        }

        return (vidX, txtX)
    }
}

// MARK: - Patch Embed / Unpack

/// [B, C, T, H, W] → [B, nT×nH×nW, dim] via volume patching.
nonisolated final class SeedVR2PatchIn: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    let pT: Int; let pH: Int; let pW: Int

    init(inCh: Int, dim: Int, patchSize: [Int]) {
        precondition(patchSize.count == 3)
        self.pT = patchSize[0]; self.pH = patchSize[1]; self.pW = patchSize[2]
        self._proj.wrappedValue = Linear(inCh * pT * pH * pW, dim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (tokens: MLXArray, nT: Int, nH: Int, nW: Int) {
        let (B, C, T, H, W) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        let nT = T / pT; let nH = H / pH; let nW = W / pW
        let x8 = x.reshaped([B, C, nT, pT, nH, pH, nW, pW])
                   .transposed(0, 2, 4, 6, 1, 3, 5, 7)
                   .reshaped([B, nT * nH * nW, C * pT * pH * pW])
        return (proj(x8.asType(.bfloat16)), nT, nH, nW)
    }
}

/// [B, L, dim] → [B, outCh, T, H, W].
nonisolated final class SeedVR2PatchOut: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    let pT: Int; let pH: Int; let pW: Int; let outCh: Int

    init(dim: Int, outCh: Int, patchSize: [Int]) {
        precondition(patchSize.count == 3)
        self.pT = patchSize[0]; self.pH = patchSize[1]; self.pW = patchSize[2]
        self.outCh = outCh
        self._proj.wrappedValue = Linear(dim, outCh * pT * pH * pW, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, nT: Int, nH: Int, nW: Int) -> MLXArray {
        let B = x.dim(0)
        return proj(x)
            .reshaped([B, nT, nH, nW, outCh, pT, pH, pW])
            .transposed(0, 4, 1, 5, 2, 6, 3, 7)
            .reshaped([B, outCh, nT * pT, nH * pH, nW * pW])
    }
}

// MARK: - Timestep Embedding

/// Sinusoidal timestep → projIn → silu → projHid → silu → projOut.
/// projOut outputs 6×vidDim — these are the global time-conditioned ada params.
nonisolated final class SeedVR2TimeEmbedding: Module {
    @ModuleInfo(key: "proj_in")  var projIn:  Linear
    @ModuleInfo(key: "proj_hid") var projHid: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    let freqDim: Int

    init(freqDim: Int = 256, vidDim: Int) {
        self.freqDim = freqDim
        self._projIn.wrappedValue  = Linear(freqDim, vidDim, bias: true)
        self._projHid.wrappedValue = Linear(vidDim, vidDim, bias: true)
        self._projOut.wrappedValue = Linear(vidDim, 6 * vidDim, bias: true)
        super.init()
    }

    /// Returns [B, 6×vidDim] — the global ada modulation vector for all blocks.
    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let tF  = t.asType(.float32).reshaped([-1])
        let emb = sinusoidalEmbed(tF, dim: freqDim)
        var h   = MLXNN.silu(projIn(emb.asType(.bfloat16)))
        h = MLXNN.silu(projHid(h))
        return projOut(h)
    }

    private func sinusoidalEmbed(_ t: MLXArray, dim: Int) -> MLXArray {
        let half  = dim / 2
        let freqs = exp(
            -log(10000.0) * MLXArray(Int32(0) ..< Int32(half)).asType(.float32) / Float(half)
        )
        let args = t.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        return MLX.concatenated([cos(args), sin(args)], axis: -1)
    }
}

// MARK: - SeedVR2Transformer

nonisolated final class SeedVR2Transformer: Module {
    @ModuleInfo(key: "vid_in")      var vidIn:      SeedVR2PatchIn
    @ModuleInfo(key: "txt_in")      var txtIn:      Linear
    @ModuleInfo(key: "emb_in")      var embIn:      SeedVR2TimeEmbedding
    @ModuleInfo(key: "blocks")      var blocks:     [SeedVR2TransformerBlock]
    @ModuleInfo(key: "vid_out_norm") var vidOutNorm: RMSNorm
    @ModuleInfo(key: "vid_out")     var vidOut:     SeedVR2PatchOut
    @ParameterInfo(key: "out_scale") var outScale:  MLXArray  // [vidDim]
    @ParameterInfo(key: "out_shift") var outShift:  MLXArray  // [vidDim]

    init(c: SeedVR2Config = SeedVR2Config()) {
        self._vidIn.wrappedValue      = SeedVR2PatchIn(
            inCh: c.vidInChannels, dim: c.vidDim, patchSize: c.patchSize)
        self._txtIn.wrappedValue      = Linear(c.txtInDim, c.vidDim, bias: true)
        self._embIn.wrappedValue      = SeedVR2TimeEmbedding(freqDim: 256, vidDim: c.vidDim)
        self._blocks.wrappedValue     = (0 ..< c.numLayers).map { i in
            SeedVR2TransformerBlock(c: c, isMM: i < c.mmLayers)
        }
        self._vidOutNorm.wrappedValue = RMSNorm(dimensions: c.vidDim, eps: c.normEps)
        self._vidOut.wrappedValue     = SeedVR2PatchOut(
            dim: c.vidDim, outCh: c.vidOutChannels, patchSize: c.patchSize)
        self._outScale.wrappedValue   = MLXArray.zeros([c.vidDim])
        self._outShift.wrappedValue   = MLXArray.zeros([c.vidDim])
        super.init()
    }

    /// - x: [B, vidInChannels, T, H, W]
    /// - textEmb: [L_txt, txtInDim]
    /// - timestep: scalar or [B] float in [0, 1000]
    func callAsFunction(_ x: MLXArray, textEmb: MLXArray, timestep: MLXArray) -> MLXArray {
        let (vidTokens, nT, nH, nW) = vidIn(x)  // [B, L_vid, vidDim]
        let B = vidTokens.dim(0)

        // Project text embedding and tile to batch
        let txtProj   = txtIn(textEmb.asType(.bfloat16)).expandedDimensions(axis: 0)
        let txtTokens = MLX.tiled(txtProj, repetitions: [B, 1, 1])

        // Global time-conditioned ada: [B, 6*vidDim]
        let emb = embIn(timestep)

        func hasNaN(_ a: MLXArray) -> Bool {
            let f = a.asType(.float32); eval(f)
            return MLX.any(MLX.isNaN(f)).item(Bool.self)
        }
        func rng2(_ a: MLXArray) -> String {
            let f = a.asType(.float32); eval(f)
            let mn = f.min(keepDims: false).item(Float.self)
            let mx = f.max(keepDims: false).item(Float.self)
            return "[\(String(format: "%.3f", mn)), \(String(format: "%.3f", mx))]"
        }
        print("[SeedVR2-D] emb: \(rng2(emb))")

        var vid = vidTokens
        var txt = txtTokens
        for (i, block) in blocks.enumerated() {
            (vid, txt) = block(vid: vid, txt: txt, emb: emb, nT: nT, nH: nH, nW: nW)
            eval(vid); eval(txt)
            if hasNaN(vid) || hasNaN(txt) {
                print("[SeedVR2-D] NaN first at block \(i): vid=\(rng2(vid)) txt=\(rng2(txt))")
                break
            }
            if i == 0 || i == 9 || i == 10 {
                print("[SeedVR2-D] block \(i): vid=\(rng2(vid)) txt=\(rng2(txt))")
            }
        }

        // Output: norm → final scale/shift → unpatch
        let normed = vidOutNorm(vid)  // [B, L, vidDim]
        let scale  = outScale.expandedDimensions(axis: 0).expandedDimensions(axis: 0).asType(normed.dtype)
        let shift  = outShift.expandedDimensions(axis: 0).expandedDimensions(axis: 0).asType(normed.dtype)
        let modulated = normed * (1 + scale) + shift
        return vidOut(modulated, nT: nT, nH: nH, nW: nW)
    }
}
