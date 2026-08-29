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

// MARK: - RoPE

/// Per-block learned rotary position embedding.
/// The checkpoint stores `freqs` as a 1-D tensor of shape [headDim/6] — the base theta values
/// for the H and W spatial components of a 3-D T/H/W RoPE.
/// Per-token angles are computed via outer(position, freqs) at inference time.
nonisolated final class SeedVR2RoPE: Module {
    @ParameterInfo(key: "freqs") var freqs: MLXArray  // [headDim/6] learned base theta values

    init(ropeDim: Int) {
        self._freqs.wrappedValue = MLXArray.zeros([ropeDim / 6])
        super.init()
    }

    /// Apply 3-D RoPE to x: [B, L, H, headDim] where L = nT×nH×nW.
    /// headDim is split as: dT = headDim − 2·dHW, dHW = 2·(headDim/6).
    /// freqs ([headDim/6]) are per-block learned theta values used for H and W components.
    /// T uses zero angles (identity rotation) — valid for single-frame (nT = 1).
    func apply(_ x: MLXArray, nT: Int, nH: Int, nW: Int) -> MLXArray {
        let d     = x.dim(-1)     // headDim = 128
        let dHalf = freqs.dim(0)  // 21 = d / 6
        let dHW   = 2 * dHalf    // 42
        let dT    = d - 2 * dHW  // 44
        let L     = nT * nH * nW

        let (_, hPos, wPos) = wanMakePositions(nT: nT, nH: nH, nW: nW)

        // Outer product: position [L,1] × theta [1,dHalf] → per-token angles [L, dHalf]
        let fF = freqs.asType(.float32)
        let aH = hPos.asType(.float32).expandedDimensions(axis: 1) * fF.expandedDimensions(axis: 0)
        let aW = wPos.asType(.float32).expandedDimensions(axis: 1) * fF.expandedDimensions(axis: 0)
        // T component: zero angles → identity rotation (all T positions = 0 when nT = 1)
        let aT = MLXArray.zeros([L, dT / 2])

        // Expand to [1, L, 1, *] for broadcasting with x [B, L, H, d]
        let dtype = x.dtype
        let fT = aT.expandedDimensions(axis: 0).expandedDimensions(axis: 2).asType(dtype)
        let fH = aH.expandedDimensions(axis: 0).expandedDimensions(axis: 2).asType(dtype)
        let fW = aW.expandedDimensions(axis: 0).expandedDimensions(axis: 2).asType(dtype)

        let xT = x[.ellipsis, 0 ..< dT]
        let xH = x[.ellipsis, dT ..< dT + dHW]
        let xW = x[.ellipsis, dT + dHW ..< d]

        return MLX.concatenated([
            wanApplyRoPE1D(xT, freqs: fT),
            wanApplyRoPE1D(xH, freqs: fH),
            wanApplyRoPE1D(xW, freqs: fW),
        ], axis: -1)
    }
}

// MARK: - AdaModulation

/// Adaptive modulation (AdaLN-Zero style). One linear per block maps the time embedding
/// to 6×vidDim parameters (shift/scale/gate for attn and mlp). MM layers emit 12×vidDim
/// (6 for vid, 6 for txt).
nonisolated final class SeedVR2AdaModulation: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    let isMM: Bool

    init(vidDim: Int, isMM: Bool) {
        self.isMM = isMM
        self._proj.wrappedValue = Linear(vidDim, isMM ? 12 * vidDim : 6 * vidDim, bias: true)
        super.init()
    }

    /// Returns modulation params: vid [B,6,dim], txt [B,6,dim]?.
    /// Non-MM proj output: [B, 6*dim] → vp=[B,6,dim], tp=nil.
    /// MM proj output:     [B,12*dim] → first half → vp=[B,6,dim], second half → tp=[B,6,dim].
    func params(from emb: MLXArray) -> (vid: MLXArray, txt: MLXArray?) {
        let h = proj(MLXNN.silu(emb))       // [B, 6*dim or 12*dim]
        let B = h.dim(0)
        let outDim = h.dim(-1)
        let vidOut = isMM ? h[0..., ..<(outDim/2)] : h
        let dim    = vidOut.dim(-1) / 6
        let vp = vidOut.reshaped([B, 6, dim])
        if isMM {
            let tp = h[0..., (outDim/2)...].reshaped([B, 6, dim])
            return (vp, tp)
        }
        return (vp, nil)
    }
}

/// Apply AdaLN: normed * (1 + scale) + shift. p: [B, 6, dim], normed: [B, L, dim].
private func modulate(_ normed: MLXArray, scale: MLXArray, shift: MLXArray) -> MLXArray {
    let sc = scale.expandedDimensions(axis: 1)  // [B, 1, dim]
    let sh = shift.expandedDimensions(axis: 1)
    return normed * (1 + sc) + sh
}

// MARK: - Multi-Modal SwiGLU MLP

nonisolated final class SeedVR2MMSwiGLU: Module {
    @ModuleInfo(key: "vidGateUp") var vidGateUp: Linear
    @ModuleInfo(key: "vidDown")   var vidDown:   Linear
    @ModuleInfo(key: "txtGateUp") var txtGateUp: Linear?
    @ModuleInfo(key: "txtDown")   var txtDown:   Linear?

    init(dim: Int, expandRatio: Int, isMM: Bool) {
        let hid = svHidden(dim, expand: expandRatio)
        self._vidGateUp.wrappedValue = Linear(dim, 2 * hid, bias: true)
        self._vidDown.wrappedValue   = Linear(hid, dim, bias: true)
        if isMM {
            self._txtGateUp.wrappedValue = Linear(dim, 2 * hid, bias: true)
            self._txtDown.wrappedValue   = Linear(hid, dim, bias: true)
        }
        super.init()
    }

    func callAsFunction(vid: MLXArray, txt: MLXArray) -> (vid: MLXArray, txt: MLXArray) {
        let vGU = vidGateUp(vid)
        let vH  = vGU.dim(-1) / 2
        let vOut = vidDown(MLXNN.silu(vGU[.ellipsis, 0 ..< vH]) * vGU[.ellipsis, vH...])
        guard let tGU = txtGateUp, let tD = txtDown else { return (vOut, txt) }
        let tGUr = tGU(txt)
        let tH   = tGUr.dim(-1) / 2
        let tOut = tD(MLXNN.silu(tGUr[.ellipsis, 0 ..< tH]) * tGUr[.ellipsis, tH...])
        return (vOut, tOut)
    }
}

// MARK: - Multi-Modal Attention (global, with RoPE on vid)

/// Joint vid+txt global attention. Video tokens receive 3-D RoPE; text tokens attend but are
/// not rotated. A full windowed-attention implementation matching the training config ([4,3,3]
/// windows would improve large-image quality; for single-frame inference the difference is minimal.
nonisolated final class SeedVR2MMAttention: Module {
    @ModuleInfo(key: "toQkvVid") var toQkvVid: Linear
    @ModuleInfo(key: "toQkvTxt") var toQkvTxt: Linear?  // mm layers only
    @ModuleInfo(key: "toOutVid") var toOutVid: Linear
    @ModuleInfo(key: "toOutTxt") var toOutTxt: Linear?
    @ModuleInfo(key: "rope")     var rope:     SeedVR2RoPE

    let heads:   Int
    let headDim: Int

    init(dim: Int, heads: Int, headDim: Int, ropeDim: Int, isMM: Bool) {
        self.heads   = heads
        self.headDim = headDim
        let qkvDim = heads * headDim
        self._toQkvVid.wrappedValue = Linear(dim, 3 * qkvDim, bias: true)
        self._toOutVid.wrappedValue = Linear(qkvDim, dim, bias: true)
        self._rope.wrappedValue     = SeedVR2RoPE(ropeDim: ropeDim)
        if isMM {
            self._toQkvTxt.wrappedValue = Linear(dim, 3 * qkvDim, bias: true)
            self._toOutTxt.wrappedValue = Linear(qkvDim, dim, bias: true)
        }
        super.init()
    }

    func callAsFunction(
        vid: MLXArray,   // [B, L_vid, dim]
        txt: MLXArray,   // [B, L_txt, dim]
        nT: Int, nH: Int, nW: Int
    ) -> (vid: MLXArray, txt: MLXArray) {
        let B = vid.dim(0)
        let H = heads, D = headDim
        let Lv = vid.dim(1), Lt = txt.dim(1)

        func splitQKV(_ x: MLXArray, L: Int) -> (MLXArray, MLXArray, MLXArray) {
            let r = x.reshaped([B, L, 3, H, D]).transposed(2, 0, 3, 1, 4)
            return (r[0], r[1], r[2])
        }

        let vqkv = toQkvVid(vid)
        var (vq, vk, vv) = splitQKV(vqkv, L: Lv)

        // Apply 3-D RoPE to vid q/k: transpose to [B, L, H, D] for rope, then back
        vq = rope.apply(vq.transposed(0, 2, 1, 3), nT: nT, nH: nH, nW: nW).transposed(0, 2, 1, 3)
        vk = rope.apply(vk.transposed(0, 2, 1, 3), nT: nT, nH: nH, nW: nW).transposed(0, 2, 1, 3)

        // Optionally include txt tokens
        let (q, k, v, txtStart): (MLXArray, MLXArray, MLXArray, Int)
        if let tqkv = toQkvTxt {
            let tqkvOut = tqkv(txt)
            let (tq, tk, tv) = splitQKV(tqkvOut, L: Lt)
            q = MLX.concatenated([vq, tq], axis: 2)
            k = MLX.concatenated([vk, tk], axis: 2)
            v = MLX.concatenated([vv, tv], axis: 2)
            txtStart = Lv
        } else {
            (q, k, v, txtStart) = (vq, vk, vv, Lv)
        }

        // SDPA: q/k/v are [B, H, L, D]
        let scale = 1.0 / sqrt(Float(D))
        let out = scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil)  // [B, H, L, D]
        let flat = out.transposed(0, 2, 1, 3).reshaped([B, -1, H * D])

        let vidFlat = flat[0..., ..<Lv, 0...]
        let vidOut  = toOutVid(vidFlat)

        if let tOut = toOutTxt {
            let txtFlat = flat[0..., txtStart ..< txtStart+Lt, 0...]
            return (vidOut, tOut(txtFlat))
        }
        return (vidOut, txt)
    }
}

// MARK: - Transformer Block

nonisolated final class SeedVR2TransformerBlock: Module {
    @ModuleInfo(key: "vidNorm1") var vidNorm1: RMSNorm
    @ModuleInfo(key: "vidNorm2") var vidNorm2: RMSNorm
    @ModuleInfo(key: "txtNorm1") var txtNorm1: RMSNorm?
    @ModuleInfo(key: "txtNorm2") var txtNorm2: RMSNorm?
    @ModuleInfo(key: "attn")     var attn:     SeedVR2MMAttention
    @ModuleInfo(key: "mlp")      var mlp:      SeedVR2MMSwiGLU
    @ModuleInfo(key: "ada")      var ada:      SeedVR2AdaModulation

    init(c: SeedVR2Config, isMM: Bool) {
        let d = c.vidDim
        self._vidNorm1.wrappedValue = RMSNorm(dimensions: d, eps: c.normEps)
        self._vidNorm2.wrappedValue = RMSNorm(dimensions: d, eps: c.normEps)
        if isMM {
            self._txtNorm1.wrappedValue = RMSNorm(dimensions: d, eps: c.normEps)
            self._txtNorm2.wrappedValue = RMSNorm(dimensions: d, eps: c.normEps)
        }
        self._attn.wrappedValue = SeedVR2MMAttention(
            dim: d, heads: c.heads, headDim: c.headDim, ropeDim: c.ropeDim, isMM: isMM)
        self._mlp.wrappedValue  = SeedVR2MMSwiGLU(dim: d, expandRatio: c.expandRatio, isMM: isMM)
        self._ada.wrappedValue  = SeedVR2AdaModulation(vidDim: d, isMM: isMM)
        super.init()
    }

    func callAsFunction(
        vid: MLXArray,   // [B, L_vid, dim]
        txt: MLXArray,   // [B, L_txt, dim]
        emb: MLXArray,   // [B, dim] time embedding
        nT: Int, nH: Int, nW: Int
    ) -> (vid: MLXArray, txt: MLXArray) {
        let (vp, tp) = ada.params(from: emb)
        // Indices: 0=attn_shift, 1=attn_scale, 2=attn_gate, 3=mlp_shift, 4=mlp_scale, 5=mlp_gate
        let vAS = vp[0..., 0, 0...]; let vASc = vp[0..., 1, 0...]; let vAG = vp[0..., 2, 0...]
        let vMS = vp[0..., 3, 0...]; let vMSc = vp[0..., 4, 0...]; let vMG = vp[0..., 5, 0...]

        var vidX = vid
        var txtX = txt

        // Attention with AdaLN
        let vAttnIn = modulate(vidNorm1(vid), scale: vASc, shift: vAS)
        var tAttnIn: MLXArray = txt
        if let tn1 = txtNorm1, let tp2 = tp {
            let tAS = tp2[0..., 0, 0...]; let tASc = tp2[0..., 1, 0...]; let tAG = tp2[0..., 2, 0...]
            tAttnIn = modulate(tn1(txt), scale: tASc, shift: tAS)
            let (vAttnOut, tAttnOut) = attn(vid: vAttnIn, txt: tAttnIn, nT: nT, nH: nH, nW: nW)
            vidX = vidX + vAG.expandedDimensions(axis: 1) * vAttnOut
            txtX = txtX + tAG.expandedDimensions(axis: 1) * tAttnOut
        } else {
            let (vAttnOut, _) = attn(vid: vAttnIn, txt: tAttnIn, nT: nT, nH: nH, nW: nW)
            vidX = vidX + vAG.expandedDimensions(axis: 1) * vAttnOut
        }

        // MLP with AdaLN
        let vMlpIn = modulate(vidNorm2(vidX), scale: vMSc, shift: vMS)
        var tMlpIn: MLXArray = txtX
        if let tn2 = txtNorm2, let tp2 = tp {
            let tMS = tp2[0..., 3, 0...]; let tMSc = tp2[0..., 4, 0...]; let tMG = tp2[0..., 5, 0...]
            tMlpIn = modulate(tn2(txtX), scale: tMSc, shift: tMS)
            let (vMlpOut, tMlpOut) = mlp(vid: vMlpIn, txt: tMlpIn)
            vidX = vidX + vMG.expandedDimensions(axis: 1) * vMlpOut
            txtX = txtX + tMG.expandedDimensions(axis: 1) * tMlpOut
        } else {
            let (vMlpOut, _) = mlp(vid: vMlpIn, txt: tMlpIn)
            vidX = vidX + vMG.expandedDimensions(axis: 1) * vMlpOut
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
        // [B, C, nT, pT, nH, pH, nW, pW] → [B, nT, nH, nW, C*pT*pH*pW]
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
        let patches = proj(x)  // [B, L, outCh*pT*pH*pW]
        return patches
            .reshaped([B, nT, nH, nW, outCh, pT, pH, pW])
            .transposed(0, 4, 1, 5, 2, 6, 3, 7)
            .reshaped([B, outCh, nT * pT, nH * pH, nW * pW])
    }
}

// MARK: - Timestep Embedding

/// Sinusoidal timestep → MLP (projIn → silu → projHid → silu → projOut).
nonisolated final class SeedVR2TimeEmbedding: Module {
    @ModuleInfo(key: "projIn")  var projIn:  Linear
    @ModuleInfo(key: "projHid") var projHid: Linear
    @ModuleInfo(key: "projOut") var projOut: Linear

    let freqDim: Int

    init(freqDim: Int = 256, vidDim: Int) {
        self.freqDim = freqDim
        self._projIn.wrappedValue  = Linear(freqDim, vidDim, bias: true)
        self._projHid.wrappedValue = Linear(vidDim, vidDim, bias: true)
        self._projOut.wrappedValue = Linear(vidDim, vidDim, bias: true)
        super.init()
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        // t: scalar or [B] timestep in [0, 1000]
        let tF = t.asType(.float32).reshaped([-1])   // [B]
        let emb = sinusoidalEmbed(tF, dim: freqDim)  // [B, freqDim]
        var h = MLXNN.silu(projIn(emb.asType(.bfloat16)))
        h = MLXNN.silu(projHid(h))
        return projOut(h)
    }

    private func sinusoidalEmbed(_ t: MLXArray, dim: Int) -> MLXArray {
        let half = dim / 2
        let freqs = exp(
            -log(10000.0) * MLXArray(Int32(0) ..< Int32(half)).asType(.float32) / Float(half)
        )
        let args = t.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        return MLX.concatenated([cos(args), sin(args)], axis: -1)
    }
}

// MARK: - SeedVR2Transformer

nonisolated final class SeedVR2Transformer: Module {
    @ModuleInfo(key: "patchIn")    var patchIn:    SeedVR2PatchIn
    @ModuleInfo(key: "txtIn")      var txtIn:      Linear
    @ModuleInfo(key: "embIn")      var embIn:      SeedVR2TimeEmbedding
    @ModuleInfo(key: "blocks")     var blocks:     [SeedVR2TransformerBlock]
    @ModuleInfo(key: "vidOutNorm") var vidOutNorm: RMSNorm
    @ModuleInfo(key: "vidOut")     var vidOut:     SeedVR2PatchOut

    init(c: SeedVR2Config = SeedVR2Config()) {
        self._patchIn.wrappedValue    = SeedVR2PatchIn(
            inCh: c.vidInChannels, dim: c.vidDim, patchSize: c.patchSize)
        self._txtIn.wrappedValue      = Linear(c.txtInDim, c.vidDim, bias: true)
        self._embIn.wrappedValue      = SeedVR2TimeEmbedding(freqDim: 256, vidDim: c.vidDim)
        self._blocks.wrappedValue     = (0 ..< c.numLayers).map { i in
            SeedVR2TransformerBlock(c: c, isMM: i < c.mmLayers)
        }
        self._vidOutNorm.wrappedValue = RMSNorm(dimensions: c.vidDim, eps: c.normEps)
        self._vidOut.wrappedValue     = SeedVR2PatchOut(
            dim: c.vidDim, outCh: c.vidOutChannels, patchSize: c.patchSize)
        super.init()
    }

    /// Forward pass.
    /// - Parameters:
    ///   - x: input latent [B, vidInChannels, T, H, W]
    ///   - textEmb: precomputed text embedding [L_txt, txtInDim]
    ///   - timestep: scalar or [B] float in [0, 1000]
    func callAsFunction(_ x: MLXArray, textEmb: MLXArray, timestep: MLXArray) -> MLXArray {
        let (vidTokens, nT, nH, nW) = patchIn(x)  // [B, L_vid, vidDim]
        let B = vidTokens.dim(0)

        // Project fixed text embedding and tile to batch size
        let txtProj   = txtIn(textEmb.asType(.bfloat16)).expandedDimensions(axis: 0) // [1, L_txt, dim]
        let txtTokens = MLX.tiled(txtProj, repetitions: [B, 1, 1])                  // [B, L_txt, dim]

        // Timestep embedding
        let emb = embIn(timestep)  // [B, vidDim]

        // Run blocks
        var vid = vidTokens
        var txt = txtTokens
        for block in blocks {
            (vid, txt) = block(vid: vid, txt: txt, emb: emb, nT: nT, nH: nH, nW: nW)
        }

        // Output: norm → unpatch
        let normed = vidOutNorm(vid)
        return vidOut(normed, nT: nT, nH: nH, nW: nW)  // [B, vidOutChannels, T, H, W]
    }
}
