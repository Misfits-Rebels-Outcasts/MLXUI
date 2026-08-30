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

/// Affine-free RMSNorm (ones weight) over the last axis.
private func rmsNormOnes(_ x: MLXArray, eps: Float) -> MLXArray {
    let d = x.shape[x.ndim - 1]
    return MLXFast.rmsNorm(x, weight: MLXArray.ones([d]).asType(x.dtype), eps: eps)
}

private func ceilDiv(_ a: Int, _ b: Int) -> Int { (a + b - 1) / b }

// MARK: - Window partitioner

/// Gather/scatter indices that reorder video tokens into variable-size windows.
struct SeedVR2WindowPartitioner {
    let forwardIdx: MLXArray
    let reverseIdx: MLXArray
    let windowShapes: [[Int]]
    let windowCounts: [Int]

    init(vidShape: [[Int]], window: [Int], shift: Bool = false) {
        var forward: [Int32] = []
        var shapes: [[Int]] = []
        var counts: [Int] = []
        var base = 0
        for s in vidShape {
            let (t, h, w) = (s[0], s[1], s[2])
            let wins = Self.makeWindows(t: t, h: h, w: w, num: window, shift: shift)
            counts.append(wins.count)
            for win in wins {
                let (t0, t1, h0, h1, w0, w1) = win
                shapes.append([t1 - t0, h1 - h0, w1 - w0])
                for tt in t0 ..< t1 {
                    for hh in h0 ..< h1 {
                        for ww in w0 ..< w1 {
                            forward.append(Int32(base + tt * (h * w) + hh * w + ww))
                        }
                    }
                }
            }
            base += t * h * w
        }
        self.forwardIdx = MLXArray(forward)
        self.windowShapes = shapes
        self.windowCounts = counts
        self.reverseIdx = argSort(MLXArray(forward), axis: 0)
    }

    func partition(_ x: MLXArray) -> MLXArray { x[forwardIdx] }
    func reverse(_ x: MLXArray) -> MLXArray { x[reverseIdx] }

    /// Window bounds (t0,t1,h0,h1,w0,w1), iterating iw (outer) → ih → it (inner).
    static func makeWindows(t: Int, h: Int, w: Int, num: [Int], shift: Bool = false) -> [(Int, Int, Int, Int, Int, Int)] {
        let (rnt, rnh, rnw) = (num[0], num[1], num[2])
        let scale = (Double(45 * 80) / Double(h * w)).squareRoot()
        let resizedH = Int((Double(h) * scale).rounded(.toNearestOrEven))
        let resizedW = Int((Double(w) * scale).rounded(.toNearestOrEven))
        let wh = ceilDiv(resizedH, rnh)
        let ww = ceilDiv(resizedW, rnw)
        let wt = ceilDiv(min(t, 30), rnt)

        let st: Double, sh: Double, sw: Double
        let nt: Int, nh: Int, nw: Int
        if shift {
            st = wt < t ? 0.5 : 0
            sh = wh < h ? 0.5 : 0
            sw = ww < w ? 0.5 : 0
            nt = st > 0 ? ceilDiv(Int((Double(t) - st).rounded(.up)), wt) + 1 : 1
            nh = sh > 0 ? ceilDiv(Int((Double(h) - sh).rounded(.up)), wh) + 1 : 1
            nw = sw > 0 ? ceilDiv(Int((Double(w) - sw).rounded(.up)), ww) + 1 : 1
        } else {
            st = 0; sh = 0; sw = 0
            nt = ceilDiv(t, wt); nh = ceilDiv(h, wh); nw = ceilDiv(w, ww)
        }

        var out: [(Int, Int, Int, Int, Int, Int)] = []
        for iw in 0 ..< nw {
            let w0 = max(Int((Double(iw) - sw) * Double(ww)), 0)
            let w1 = min(Int((Double(iw) - sw + 1) * Double(ww)), w)
            if w1 <= w0 { continue }
            for ih in 0 ..< nh {
                let h0 = max(Int((Double(ih) - sh) * Double(wh)), 0)
                let h1 = min(Int((Double(ih) - sh + 1) * Double(wh)), h)
                if h1 <= h0 { continue }
                for it in 0 ..< nt {
                    let t0 = max(Int((Double(it) - st) * Double(wt)), 0)
                    let t1 = min(Int((Double(it) - st + 1) * Double(wt)), t)
                    if t1 <= t0 { continue }
                    out.append((t0, t1, h0, h1, w0, w1))
                }
            }
        }
        return out
    }
}

// MARK: - RoPE

/// Axial 3-D rotary embedding. `freqs` (21 values) is loaded from the checkpoint.
nonisolated final class SeedVR2RoPE: Module {
    @ParameterInfo(key: "freqs") var freqs: MLXArray  // [headDim/6]

    init(dim: Int = 128) {
        self._freqs.wrappedValue = MLXArray.zeros([dim / 3 / 2])
        super.init()
    }

    private var freqDimPerAxis: Int { freqs.shape[0] * 2 }   // 42
    private var ropeAxes: Int { 3 }

    /// Axial freqs over a grid `dims` (with an optional offset for the first axis).
    /// Returns [dims..., freqDimPerAxis * dims.count].
    private func axialFreqs(_ dims: [Int], temporalOffset: Int = 0) -> MLXArray {
        let fdpa = freqDimPerAxis
        let f32 = freqs.asType(.float32)
        var parts: [MLXArray] = []
        for (ind, d) in dims.enumerated() {
            let start = ind == 0 ? temporalOffset : 0
            let pos = MLXArray((start ..< (start + d)).map { Float($0) })
            var af = outer(pos, f32)                    // [d, 21]
            af = repeated(af, count: 2, axis: -1)       // [d, 42]
            var shape = Array(repeating: 1, count: dims.count) + [fdpa]
            shape[ind] = d
            af = af.reshaped(shape)
            af = broadcast(af, to: dims + [fdpa])
            parts.append(af)
        }
        return concatenated(parts, axis: -1)            // [dims..., 42*3]
    }

    private static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        var r = x.reshaped(Array(s.dropLast()) + [-1, 2])
        let x1 = r[.ellipsis, 0]
        let x2 = r[.ellipsis, 1]
        r = stacked([-x2, x1], axis: -1)
        return r.reshaped(s)
    }

    /// freqs: [..., rotDim]; t: [N, heads, headDim]. Rotate the first rotDim dims.
    private static func applyRotary(_ freqs: MLXArray, _ t: MLXArray) -> MLXArray {
        let rotDim = freqs.shape[freqs.ndim - 1]
        let dim = t.shape[t.ndim - 1]
        let tMid = t[.ellipsis, 0 ..< rotDim].asType(.float32)
        let f = freqs.asType(.float32)
        var out = tMid * cos(f) + rotateHalf(tMid) * sin(f)
        out = out.asType(t.dtype)
        if dim > rotDim {
            return concatenated([out, t[.ellipsis, rotDim ..< dim]], axis: -1)
        }
        return out
    }

    /// Video-only RoPE. windowShapes: [[t,h,w]] per window.
    func applyVid(_ vidQ: MLXArray, _ vidK: MLXArray, windowShapes: [[Int]]) -> (MLXArray, MLXArray) {
        var parts: [MLXArray] = []
        for s in windowShapes {
            let vf = axialFreqs(s).reshaped([-1, freqDimPerAxis * ropeAxes])
            parts.append(vf)
        }
        let vidFreqs = concatenated(parts, axis: 0).expandedDimensions(axis: 1)
        return (Self.applyRotary(vidFreqs, vidQ), Self.applyRotary(vidFreqs, vidK))
    }

    /// Multi-modal RoPE: vid temporal positions are offset by txtLen.
    func applyMM(_ vidQ: MLXArray, _ vidK: MLXArray, windowShapes: [[Int]],
                 _ txtQ: MLXArray, _ txtK: MLXArray, txtLens: [Int]) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        var vidParts: [MLXArray] = []
        var txtParts: [MLXArray] = []
        for (i, s) in windowShapes.enumerated() {
            let (f, h, w) = (s[0], s[1], s[2])
            let tl = txtLens[i]
            let full = axialFreqs([tl + f, h, w])
            let vidSlice = full[tl ..< (tl + f)].reshaped([-1, freqDimPerAxis * ropeAxes])
            vidParts.append(vidSlice)
            let txt1d = axialFreqs([tl])
            txtParts.append(tiled(txt1d, repetitions: [1, ropeAxes]))
        }
        let vidFreqs = concatenated(vidParts, axis: 0).expandedDimensions(axis: 1)
        let txtFreqs = concatenated(txtParts, axis: 0).expandedDimensions(axis: 1)
        return (Self.applyRotary(vidFreqs, vidQ), Self.applyRotary(vidFreqs, vidK),
                Self.applyRotary(txtFreqs, txtQ), Self.applyRotary(txtFreqs, txtK))
    }
}

// MARK: - Ada Modulation

nonisolated final class SeedVR2AdaParams: Module {
    @ParameterInfo(key: "attn_shift") var attnShift: MLXArray
    @ParameterInfo(key: "attn_scale") var attnScale: MLXArray
    @ParameterInfo(key: "attn_gate")  var attnGate:  MLXArray
    @ParameterInfo(key: "mlp_shift")  var mlpShift:  MLXArray
    @ParameterInfo(key: "mlp_scale")  var mlpScale:  MLXArray
    @ParameterInfo(key: "mlp_gate")   var mlpGate:   MLXArray

    init(_ dim: Int) {
        self._attnShift.wrappedValue = MLXArray.zeros([dim])
        self._attnScale.wrappedValue = MLXArray.ones([dim])
        self._attnGate.wrappedValue  = MLXArray.zeros([dim])
        self._mlpShift.wrappedValue  = MLXArray.zeros([dim])
        self._mlpScale.wrappedValue  = MLXArray.ones([dim])
        self._mlpGate.wrappedValue   = MLXArray.zeros([dim])
        super.init()
    }

    func shift(_ layer: Int) -> MLXArray { layer == 0 ? attnShift : mlpShift }
    func scale(_ layer: Int) -> MLXArray { layer == 0 ? attnScale : mlpScale }
    func gate(_ layer: Int) -> MLXArray { layer == 0 ? attnGate : mlpGate }
}

nonisolated final class SeedVR2Ada: Module {
    @ModuleInfo(key: "params_all") var paramsAll: SeedVR2AdaParams?
    @ModuleInfo(key: "params_vid") var paramsVid: SeedVR2AdaParams?
    @ModuleInfo(key: "params_txt") var paramsTxt: SeedVR2AdaParams?
    let shared: Bool
    let isLastLayer: Bool

    init(dim: Int, shared: Bool, isLastLayer: Bool) {
        self.shared = shared
        self.isLastLayer = isLastLayer
        if shared {
            self._paramsAll.wrappedValue = SeedVR2AdaParams(dim)
        } else {
            self._paramsVid.wrappedValue = SeedVR2AdaParams(dim)
            self._paramsTxt.wrappedValue = isLastLayer ? nil : SeedVR2AdaParams(dim)
        }
        super.init()
    }

    private func apply(_ hidden: MLXArray, _ emb: MLXArray, _ p: SeedVR2AdaParams, _ layer: Int, _ modeOut: Bool) -> MLXArray {
        let mod = emb[0..., 0..., layer]  // [B, dim, 3]
        if modeOut {
            let gate = (mod[.ellipsis, 2] + p.gate(layer)).expandedDimensions(axis: 1)
            return hidden * gate
        } else {
            let shift = (mod[.ellipsis, 0] + p.shift(layer)).expandedDimensions(axis: 1)
            let scale = (mod[.ellipsis, 1] + p.scale(layer)).expandedDimensions(axis: 1)
            return hidden * scale + shift
        }
    }

    func modulateVid(_ hidden: MLXArray, _ emb: MLXArray, _ layer: Int, _ modeOut: Bool) -> MLXArray {
        apply(hidden, emb, shared ? paramsAll! : paramsVid!, layer, modeOut)
    }

    func modulateTxt(_ hidden: MLXArray, _ emb: MLXArray, _ layer: Int, _ modeOut: Bool) -> MLXArray {
        if isLastLayer { return hidden }
        return apply(hidden, emb, shared ? paramsAll! : paramsTxt!, layer, modeOut)
    }
}

// MARK: - SwiGLU MLP

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
        projOut(MLXNN.silu(projInGate(x)) * projIn(x))
    }
}

nonisolated final class SeedVR2MMSwiGLU: Module {
    @ModuleInfo(key: "all") var all: SeedVR2SwiGLUBranch?
    @ModuleInfo(key: "vid") var vid: SeedVR2SwiGLUBranch?
    @ModuleInfo(key: "txt") var txt: SeedVR2SwiGLUBranch?
    let shared: Bool
    let isLastLayer: Bool

    init(dim: Int, expandRatio: Int, shared: Bool, isLastLayer: Bool) {
        let hid = svHidden(dim, expand: expandRatio)
        self.shared = shared
        self.isLastLayer = isLastLayer
        if shared {
            self._all.wrappedValue = SeedVR2SwiGLUBranch(dim: dim, hidDim: hid)
        } else {
            self._vid.wrappedValue = SeedVR2SwiGLUBranch(dim: dim, hidDim: hid)
            self._txt.wrappedValue = isLastLayer ? nil : SeedVR2SwiGLUBranch(dim: dim, hidDim: hid)
        }
        super.init()
    }

    func callAsFunction(_ vidIn: MLXArray, _ txtIn: MLXArray) -> (MLXArray, MLXArray) {
        let v = (shared ? all! : vid!)(vidIn)
        if isLastLayer { return (v, txtIn) }
        let t = (shared ? all! : txt!)(txtIn)
        return (v, t)
    }
}

// MARK: - Multi-Modal Windowed Attention

nonisolated final class SeedVR2MMAttention: Module {
    @ModuleInfo(key: "proj_qkv_vid") var projQkvVid: Linear
    @ModuleInfo(key: "proj_out_vid") var projOutVid: Linear
    @ModuleInfo(key: "norm_q_vid")   var normQVid:   RMSNorm
    @ModuleInfo(key: "norm_k_vid")   var normKVid:   RMSNorm
    @ModuleInfo(key: "proj_qkv_txt") var projQkvTxt: Linear
    @ModuleInfo(key: "proj_out_txt") var projOutTxt: Linear
    @ModuleInfo(key: "norm_q_txt")   var normQTxt:   RMSNorm
    @ModuleInfo(key: "norm_k_txt")   var normKTxt:   RMSNorm
    @ModuleInfo(key: "rope")         var rope:       SeedVR2RoPE

    let heads: Int, headDim: Int, scale: Float, window: [Int], ropeOnText: Bool, shift: Bool

    init(dim: Int, heads: Int, headDim: Int, ropeDim: Int, ropeOnText: Bool, window: [Int], shift: Bool) {
        self.heads = heads
        self.headDim = headDim
        self.scale = powf(Float(headDim), -0.5)
        self.window = window
        self.ropeOnText = ropeOnText
        self.shift = shift
        let inner = heads * headDim
        self._projQkvVid.wrappedValue = Linear(dim, 3 * inner, bias: false)
        self._projOutVid.wrappedValue = Linear(inner, dim, bias: true)
        self._normQVid.wrappedValue   = RMSNorm(dimensions: headDim)
        self._normKVid.wrappedValue   = RMSNorm(dimensions: headDim)
        self._projQkvTxt.wrappedValue = Linear(dim, 3 * inner, bias: false)
        self._projOutTxt.wrappedValue = Linear(inner, dim, bias: true)
        self._normQTxt.wrappedValue   = RMSNorm(dimensions: headDim)
        self._normKTxt.wrappedValue   = RMSNorm(dimensions: headDim)
        self._rope.wrappedValue       = SeedVR2RoPE(dim: ropeDim)
        super.init()
    }

    /// vid [1,L,dim], txt [1,Lt,dim]. vidShape [[t,h,w]]. txtLen scalar.
    func callAsFunction(_ vid: MLXArray, _ txt: MLXArray, vidShape: [[Int]], txtLen: Int) -> (MLXArray, MLXArray) {
        let (B, L) = (vid.shape[0], vid.shape[1])
        let Lt = txt.shape[1]
        let inner = heads * headDim

        // 1. project to qkv: [N, 3, heads, headDim]
        var qkvVid = projQkvVid(vid.reshaped([-1, vid.shape[2]])).reshaped([-1, 3, heads, headDim])
        let qkvTxt = projQkvTxt(txt.reshaped([-1, txt.shape[2]])).reshaped([-1, 3, heads, headDim])

        let part = SeedVR2WindowPartitioner(vidShape: vidShape, window: window, shift: shift)
        qkvVid = part.partition(qkvVid)

        // 2. normalize q,k; replicate text into every window
        let qVid = normQVid(qkvVid[0..., 0])
        let kVid = normKVid(qkvVid[0..., 1])
        let vVid = qkvVid[0..., 2]
        let qTxt = normQTxt(qkvTxt[0..., 0])
        let kTxt = normKTxt(qkvTxt[0..., 1])
        let vTxt = qkvTxt[0..., 2]
        let nWin = part.windowShapes.count
        let qTxtTiled = tiled(qTxt, repetitions: [nWin, 1, 1])
        let kTxtTiled = tiled(kTxt, repetitions: [nWin, 1, 1])
        let vTxtTiled = tiled(vTxt, repetitions: [nWin, 1, 1])

        // 3. RoPE
        let qV: MLXArray, kV: MLXArray, qT: MLXArray, kT: MLXArray
        if ropeOnText {
            let txtLens = Array(repeating: txtLen, count: nWin)
            (qV, kV, qT, kT) = rope.applyMM(qVid, kVid, windowShapes: part.windowShapes,
                                            qTxtTiled, kTxtTiled, txtLens: txtLens)
        } else {
            (qV, kV) = rope.applyVid(qVid, kVid, windowShapes: part.windowShapes)
            qT = qTxtTiled; kT = kTxtTiled
        }

        // 4. per-window SDPA over [vid ++ text]
        let vidLens = part.windowShapes.map { $0[0] * $0[1] * $0[2] }
        var vidOutBlocks: [MLXArray] = []
        var txtOutBlocks: [MLXArray] = []
        var vOff = 0
        for i in 0 ..< nWin {
            let vl = vidLens[i]
            let tOff = i * Lt
            let q = concatenated([qV[vOff ..< (vOff + vl)], qT[tOff ..< (tOff + Lt)]], axis: 0)
                .expandedDimensions(axis: 0).transposed(0, 2, 1, 3)
            let k = concatenated([kV[vOff ..< (vOff + vl)], kT[tOff ..< (tOff + Lt)]], axis: 0)
                .expandedDimensions(axis: 0).transposed(0, 2, 1, 3)
            let v = concatenated([vVid[vOff ..< (vOff + vl)], vTxtTiled[tOff ..< (tOff + Lt)]], axis: 0)
                .expandedDimensions(axis: 0).transposed(0, 2, 1, 3)
            var o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
            o = o.transposed(0, 2, 1, 3).squeezed(axis: 0).reshaped([-1, inner])
            vidOutBlocks.append(o[0 ..< vl])
            txtOutBlocks.append(o[vl ..< (vl + Lt)])
            vOff += vl
        }

        // 5. coalesce: vid scattered back; text averaged across windows
        let vidOut = part.reverse(concatenated(vidOutBlocks, axis: 0))
        let txtOut = mean(stacked(txtOutBlocks, axis: 0), axis: 0)
        return (projOutVid(vidOut).reshaped([B, L, -1]),
                projOutTxt(txtOut).reshaped([B, Lt, -1]))
    }
}

// MARK: - Transformer Block

nonisolated final class SeedVR2TransformerBlock: Module {
    @ModuleInfo(key: "attn") var attn: SeedVR2MMAttention
    @ModuleInfo(key: "mlp")  var mlp:  SeedVR2MMSwiGLU
    @ModuleInfo(key: "ada")  var ada:  SeedVR2Ada
    let isLastLayer: Bool
    let eps: Float

    init(c: SeedVR2Config, shared: Bool, isLastLayer: Bool, shift: Bool) {
        self.isLastLayer = isLastLayer
        self.eps = c.normEps
        self._attn.wrappedValue = SeedVR2MMAttention(
            dim: c.vidDim, heads: c.heads, headDim: c.headDim, ropeDim: c.ropeDim,
            ropeOnText: c.ropeOnText, window: c.window, shift: shift)
        self._mlp.wrappedValue = SeedVR2MMSwiGLU(
            dim: c.vidDim, expandRatio: c.expandRatio, shared: shared, isLastLayer: isLastLayer)
        self._ada.wrappedValue = SeedVR2Ada(dim: c.vidDim, shared: shared, isLastLayer: isLastLayer)
        super.init()
    }

    func callAsFunction(
        vid: MLXArray, txt: MLXArray, emb: MLXArray, vidShape: [[Int]], txtLen: Int
    ) -> (vid: MLXArray, txt: MLXArray) {
        var vidX = vid, txtX = txt

        var vidAttn = ada.modulateVid(rmsNormOnes(vidX, eps: eps), emb, 0, false)
        var txtAttn = ada.modulateTxt(rmsNormOnes(txtX, eps: eps), emb, 0, false)
        (vidAttn, txtAttn) = attn(vidAttn, txtAttn, vidShape: vidShape, txtLen: txtLen)
        vidAttn = ada.modulateVid(vidAttn, emb, 0, true)
        txtAttn = ada.modulateTxt(txtAttn, emb, 0, true)
        vidX = vidX + vidAttn
        if !isLastLayer { txtX = txtX + txtAttn }

        var vidMlp = ada.modulateVid(rmsNormOnes(vidX, eps: eps), emb, 1, false)
        let txtNorm = isLastLayer ? txtX : rmsNormOnes(txtX, eps: eps)
        var txtMlp = ada.modulateTxt(txtNorm, emb, 1, false)
        (vidMlp, txtMlp) = mlp(vidMlp, txtMlp)
        vidMlp = ada.modulateVid(vidMlp, emb, 1, true)
        txtMlp = ada.modulateTxt(txtMlp, emb, 1, true)
        vidX = vidX + vidMlp
        if !isLastLayer { txtX = txtX + txtMlp }

        return (vidX, txtX)
    }
}

// MARK: - Patch Embed / Unpack

nonisolated final class SeedVR2PatchIn: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    let pT: Int; let pH: Int; let pW: Int

    init(inCh: Int, dim: Int, patchSize: [Int]) {
        self.pT = patchSize[0]; self.pH = patchSize[1]; self.pW = patchSize[2]
        self._proj.wrappedValue = Linear(inCh * pT * pH * pW, dim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (tokens: MLXArray, nT: Int, nH: Int, nW: Int) {
        let (B, C, T, H, W) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        let nT = T / pT; let nH = H / pH; let nW = W / pW
        // [B, C, nT, pT, nH, pH, nW, pW] → [B, nT, nH, nW, pT, pH, pW, C] → token dim = pixel-major (t,h,w,C)
        var vid = x.reshaped([B, C, nT, pT, nH, pH, nW, pW])
        vid = vid.transposed(0, 2, 4, 6, 3, 5, 7, 1)
        vid = vid.reshaped([B, nT, nH, nW, C * pT * pH * pW])
        vid = proj(vid.asType(.bfloat16))
        return (vid.reshaped([B, nT * nH * nW, vid.dim(-1)]), nT, nH, nW)
    }
}

nonisolated final class SeedVR2PatchOut: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    let pT: Int; let pH: Int; let pW: Int; let outCh: Int

    init(dim: Int, outCh: Int, patchSize: [Int]) {
        self.pT = patchSize[0]; self.pH = patchSize[1]; self.pW = patchSize[2]
        self.outCh = outCh
        self._proj.wrappedValue = Linear(dim, outCh * pT * pH * pW, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, nT: Int, nH: Int, nW: Int) -> MLXArray {
        let B = x.dim(0)
        var vid = proj(x)   // [B, L, outCh*pT*pH*pW]
        // [B, nT, nH, nW, pT, pH, pW, outCh] → transpose → [B, outCh, nT, pT, nH, pH, nW, pW]
        vid = vid.reshaped([B, nT, nH, nW, pT, pH, pW, outCh])
        vid = vid.transposed(0, 7, 1, 4, 2, 5, 3, 6)
        return vid.reshaped([B, outCh, nT * pT, nH * pH, nW * pW])
    }
}

// MARK: - Timestep Embedding

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

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let tF  = t.asType(.float32).reshaped([-1])
        let emb = sinusoidalEmbed(tF, dim: freqDim)
        var h   = MLXNN.silu(projIn(emb.asType(.bfloat16)))
        h = MLXNN.silu(projHid(h))
        return projOut(h)
    }

    private func sinusoidalEmbed(_ t: MLXArray, dim: Int) -> MLXArray {
        let half  = dim / 2
        let freqs = exp(-log(10000.0) * MLXArray((0 ..< half).map { Float($0) }) / Float(half))
        let args = t.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        return MLX.concatenated([sin(args), cos(args)], axis: -1)
    }
}

// MARK: - SeedVR2Transformer

nonisolated final class SeedVR2Transformer: Module {
    @ModuleInfo(key: "vid_in")       var vidIn:      SeedVR2PatchIn
    @ModuleInfo(key: "txt_in")       var txtIn:      Linear
    @ModuleInfo(key: "emb_in")       var embIn:      SeedVR2TimeEmbedding
    @ModuleInfo(key: "blocks")       var blocks:     [SeedVR2TransformerBlock]
    @ModuleInfo(key: "vid_out_norm") var vidOutNorm: RMSNorm
    @ModuleInfo(key: "vid_out")      var vidOut:     SeedVR2PatchOut
    @ParameterInfo(key: "out_shift") var outShift:   MLXArray
    @ParameterInfo(key: "out_scale") var outScale:   MLXArray

    let vidDim: Int

    init(c: SeedVR2Config = SeedVR2Config()) {
        self.vidDim = c.vidDim
        self._vidIn.wrappedValue = SeedVR2PatchIn(
            inCh: c.vidInChannels, dim: c.vidDim, patchSize: c.patchSize)
        self._txtIn.wrappedValue = Linear(c.txtInDim, c.vidDim, bias: true)
        self._embIn.wrappedValue = SeedVR2TimeEmbedding(freqDim: 256, vidDim: c.vidDim)
        self._blocks.wrappedValue = (0 ..< c.numLayers).map { i in
            SeedVR2TransformerBlock(c: c, shared: i >= c.mmLayers, isLastLayer: i == c.numLayers - 1, shift: i % 2 == 1)
        }
        self._vidOutNorm.wrappedValue = RMSNorm(dimensions: c.vidDim, eps: c.normEps)
        self._vidOut.wrappedValue = SeedVR2PatchOut(
            dim: c.vidDim, outCh: c.vidOutChannels, patchSize: c.patchSize)
        self._outShift.wrappedValue = MLXArray.zeros([c.vidDim])
        self._outScale.wrappedValue = MLXArray.ones([c.vidDim])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, textEmb: MLXArray, timestep: MLXArray) -> MLXArray {
        let (vidTokens, nT, nH, nW) = vidIn(x)
        let B = vidTokens.dim(0)

        let txtProj   = txtIn(textEmb.asType(.bfloat16)).expandedDimensions(axis: 0)
        let txtTokens = MLX.tiled(txtProj, repetitions: [B, 1, 1])
        let txtLen = txtTokens.dim(1)

        var emb = embIn(timestep)
        emb = emb.reshaped([B, vidDim, 2, 3])

        var vid = vidTokens
        var txt = txtTokens
        let vidShape = [[nT, nH, nW]]
        for block in blocks {
            (vid, txt) = block(vid: vid, txt: txt, emb: emb, vidShape: vidShape, txtLen: txtLen)
        }

        vid = vidOutNorm(vid)
        let mod = emb[0..., 0..., 0]
        let shiftA = mod[.ellipsis, 0].expandedDimensions(axis: 1)
        let scaleA = mod[.ellipsis, 1].expandedDimensions(axis: 1)
        vid = vid * (scaleA + outScale) + (shiftA + outShift)

        return vidOut(vid, nT: nT, nH: nH, nW: nW)
    }
}
