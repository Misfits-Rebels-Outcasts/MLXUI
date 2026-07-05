import Foundation
import MLX
import MLXNN
import MLXFast

/// SAM ViT-B encoder for DeepSeek-OCR-2 — a from-scratch Swift/MLX port of
/// `Blaizzy/mlx-vlm`'s `mlx_vlm/models/deepseekocr/sam.py`. Windowed/global attention blocks with
/// **decomposed relative-position embeddings**, a Conv neck, and two stride-2 downsample convs
/// (`net_2`/`net_3`) reducing the patch grid by 4× and lifting channels to `final_out_chans` (896 for
/// OCR-2). Input NHWC `(B, S, S, 3)` → `(B, S/64, S/64, 896)` (1024→16×16, 768→12×12). No Swift SAM
/// exists upstream, so everything here is hand-written. All classes `nonisolated` (MainActor default).

// MARK: - Utility functions

// `internal` (not `private`) so the rel-pos dtype regression can be unit-tested directly.
enum SAMMath {
    /// Partition NHWC into non-overlapping `window`² windows with bottom/right padding.
    nonisolated static func windowPartition(_ x: MLXArray, _ window: Int) -> (MLXArray, (Int, Int)) {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let padH = (window - h % window) % window
        let padW = (window - w % window) % window
        var xp = x
        if padH > 0 || padW > 0 {
            xp = padded(x, widths: [[0, 0], [0, padH], [0, padW], [0, 0]])
        }
        let hp = h + padH, wp = w + padW
        xp = xp.reshaped(b, hp / window, window, wp / window, window, c)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(-1, window, window, c)
        return (xp, (hp, wp))
    }

    nonisolated static func windowUnpartition(
        _ windows: MLXArray, _ window: Int, _ padHW: (Int, Int), _ hw: (Int, Int)
    ) -> MLXArray {
        let (hp, wp) = padHW
        let (h, w) = hw
        let b = windows.dim(0) / (hp * wp / window / window)
        var x = windows.reshaped(b, hp / window, wp / window, window, window, -1)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(b, hp, wp, -1)
        if hp > h || wp > w {
            x = x[0..., 0 ..< h, 0 ..< w, 0...]
        }
        return x
    }

    /// Relative-position table for (q_size, k_size), linearly interpolated to `2·max-1` if needed.
    nonisolated static func getRelPos(_ qSize: Int, _ kSize: Int, _ relPos: MLXArray) -> MLXArray {
        let maxRelDist = 2 * max(qSize, kSize) - 1
        var resized: MLXArray
        if relPos.dim(0) != maxRelDist {
            let f = relPos.asType(.float32)
            let r = f.reshaped(1, relPos.dim(0), -1).transposed(0, 2, 1)   // (1, C, L)
            let l = r.dim(2)
            let scale = Float(l) / Float(maxRelDist)
            let indices = MLXArray(0 ..< maxRelDist).asType(.float32) * scale
            let idxFloor = floor(indices).asType(.int32)
            let idxCeil = minimum(idxFloor + Int32(1), MLXArray(Int32(l - 1)))
            let weight = indices - idxFloor.asType(.float32)
            let lo = take(r, idxFloor, axis: 2)
            let hi = take(r, idxCeil, axis: 2)
            let interp = lo * (1 - weight) + hi * weight
            resized = interp.reshaped(-1, maxRelDist).transposed(1, 0)     // (maxRelDist, C)
        } else {
            resized = relPos
        }
        let qk = Float(max(Double(kSize) / Double(qSize), 1.0))
        let kq = Float(max(Double(qSize) / Double(kSize), 1.0))
        let qCoords = MLXArray(0 ..< qSize).asType(.float32).reshaped(qSize, 1) * qk
        let kCoords = MLXArray(0 ..< kSize).asType(.float32).reshaped(1, kSize) * kq
        let relCoords = (qCoords - kCoords) + Float(kSize - 1) * kq
        // Cast back to `relPos`'s dtype: the interpolation branch upcasts to float32, and a float32
        // rel-pos bias would make SDPA reject the mask ("must promote to output type bfloat16") when
        // the model runs in bf16 — hit only by non-1024² tiles (e.g. 768² local patches, grid 48 ≠ 64).
        // Mirrors `getAbsPos`, which likewise restores the input dtype after interpolating.
        return resized[relCoords.asType(.int32)].asType(relPos.dtype)     // (qSize, kSize, C)
    }

    /// Decomposed relative-position biases `(rel_h, rel_w)` for the attention map.
    nonisolated static func addDecomposedRelPos(
        _ q: MLXArray, _ relPosH: MLXArray, _ relPosW: MLXArray,
        _ qSize: (Int, Int), _ kSize: (Int, Int)
    ) -> (MLXArray, MLXArray) {
        let (qH, qW) = qSize
        let (kH, kW) = kSize
        let rh = getRelPos(qH, kH, relPosH)   // (qH, kH, dim)
        let rw = getRelPos(qW, kW, relPosW)   // (qW, kW, dim)
        let b = q.dim(0)
        let dim = q.dim(2)
        let rQ = q.reshaped(b, qH, qW, dim)
        let relH = einsum("bhwc,hkc->bhwk", rQ, rh).reshaped(b, qH * qW, kH, 1)
        let relW = einsum("bhwc,wkc->bhwk", rQ, rw).reshaped(b, qH * qW, 1, kW)
        return (relH, relW)
    }

    /// Bicubic-interpolate absolute pos-embed `(1, src, src, C)` to `(1, tgt, tgt, C)` when sizes differ.
    nonisolated static func getAbsPos(_ absPos: MLXArray, _ tgtSize: Int) -> MLXArray {
        let srcSize = absPos.dim(1)
        guard srcSize != tgtSize else { return absPos }
        let scale = Float(tgtSize) / Float(srcSize)
        let up = Upsample(scaleFactor: [scale, scale], mode: .cubic(alignCorners: false))
        return up(absPos.asType(.float32)).asType(absPos.dtype)
    }
}

// MARK: - Blocks

nonisolated final class SAMMLPBlock: Module {
    @ModuleInfo(key: "lin1") var lin1: Linear
    @ModuleInfo(key: "lin2") var lin2: Linear

    init(embeddingDim: Int, mlpDim: Int) {
        self._lin1.wrappedValue = Linear(embeddingDim, mlpDim)
        self._lin2.wrappedValue = Linear(mlpDim, embeddingDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { lin2(gelu(lin1(x))) }
}

nonisolated final class SAMAttention: Module {
    let numHeads: Int
    let scale: Float
    let useRelPos: Bool

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "rel_pos_h") var relPosH: MLXArray
    @ParameterInfo(key: "rel_pos_w") var relPosW: MLXArray

    init(dim: Int, numHeads: Int, useRelPos: Bool, inputSize: (Int, Int)) {
        self.numHeads = numHeads
        let headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)
        self.useRelPos = useRelPos
        self._qkv.wrappedValue = Linear(dim, dim * 3, bias: true)
        self._proj.wrappedValue = Linear(dim, dim)
        // Sized for the block's attention window (windowed) or full grid (global).
        self._relPosH.wrappedValue = MLXArray.zeros([2 * inputSize.0 - 1, headDim])
        self._relPosW.wrappedValue = MLXArray.zeros([2 * inputSize.1 - 1, headDim])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w) = (x.dim(0), x.dim(1), x.dim(2))
        let qkvFlat = qkv(x).reshaped(b, h * w, 3, numHeads, -1).transposed(2, 0, 3, 1, 4)
        let qkvR = qkvFlat.reshaped(3, b * numHeads, h * w, -1)
        let q = qkvR[0], k = qkvR[1], v = qkvR[2]   // (B*heads, HW, hd)

        var maskMode: MLXFast.ScaledDotProductAttentionMaskMode = .none
        if useRelPos {
            let (relH, relW) = SAMMath.addDecomposedRelPos(q, relPosH, relPosW, (h, w), (h, w))
            let rhR = relH.reshaped(b, numHeads, relH.dim(1), relH.dim(2), relH.dim(3))
            let rwR = relW.reshaped(b, numHeads, relW.dim(1), relW.dim(2), relW.dim(3))
            let bias = (rhR + rwR).reshaped(b, numHeads, relH.dim(1), relH.dim(2) * relW.dim(3))
            maskMode = .array(bias)
        }

        let qh = q.reshaped(b, numHeads, h * w, -1)
        let kh = k.reshaped(b, numHeads, h * w, -1)
        let vh = v.reshaped(b, numHeads, h * w, -1)
        let out = MLXFast.scaledDotProductAttention(
            queries: qh, keys: kh, values: vh, scale: scale, mask: maskMode)
            .reshaped(b, numHeads, h, w, -1)
            .transposed(0, 2, 3, 1, 4)
            .reshaped(b, h, w, -1)
        return proj(out)
    }
}

nonisolated final class SAMBlock: Module {
    let windowSize: Int

    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: SAMAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SAMMLPBlock

    init(dim: Int, numHeads: Int, mlpRatio: Float, useRelPos: Bool, windowSize: Int, inputSize: (Int, Int)) {
        self.windowSize = windowSize
        self._norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._attn.wrappedValue = SAMAttention(
            dim: dim, numHeads: numHeads, useRelPos: useRelPos,
            inputSize: windowSize == 0 ? inputSize : (windowSize, windowSize))
        self._norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._mlp.wrappedValue = SAMMLPBlock(embeddingDim: dim, mlpDim: Int(Float(dim) * mlpRatio))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shortcut = x
        var h = norm1(x)
        var padHW: (Int, Int) = (0, 0)
        var origHW: (Int, Int) = (0, 0)
        if windowSize > 0 {
            origHW = (h.dim(1), h.dim(2))
            let (windows, p) = SAMMath.windowPartition(h, windowSize)
            h = windows
            padHW = p
        }
        h = attn(h)
        if windowSize > 0 {
            h = SAMMath.windowUnpartition(h, windowSize, padHW, origHW)
        }
        let out = shortcut + h
        return out + mlp(norm2(out))
    }
}

nonisolated final class SAMPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d

    init(patchSize: Int, inChannels: Int, embedDim: Int) {
        self._proj.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: embedDim,
            kernelSize: IntOrPair(patchSize), stride: IntOrPair(patchSize))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

// MARK: - Encoder

nonisolated final class SAMEncoder: Module {
    let useAbsPos = true

    @ModuleInfo(key: "patch_embed") var patchEmbed: SAMPatchEmbed
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [SAMBlock]
    @ModuleInfo(key: "neck") var neck: (Conv2d, LayerNorm, Conv2d, LayerNorm)
    @ModuleInfo(key: "net_2") var net2: Conv2d
    @ModuleInfo(key: "net_3") var net3: Conv2d

    init(_ v: DeepSeekOCRConfig.Vision) {
        let embedDim = v.samWidth
        let outChans = 256
        let grid = v.samImageSize / v.samPatchSize   // 64
        self._patchEmbed.wrappedValue = SAMPatchEmbed(
            patchSize: v.samPatchSize, inChannels: 3, embedDim: embedDim)
        self._posEmbed.wrappedValue = MLXArray.zeros([1, grid, grid, embedDim])

        let globals = Set(v.samGlobalAttnIndexes)
        self._blocks.wrappedValue = (0 ..< v.samLayers).map { i in
            SAMBlock(
                dim: embedDim, numHeads: v.samHeads, mlpRatio: 4.0, useRelPos: true,
                windowSize: globals.contains(i) ? 0 : v.samWindowSize,
                inputSize: (grid, grid))
        }
        self._neck.wrappedValue = (
            Conv2d(inputChannels: embedDim, outputChannels: outChans, kernelSize: IntOrPair(1), bias: false),
            LayerNorm(dimensions: outChans, eps: 1e-6),
            Conv2d(inputChannels: outChans, outputChannels: outChans, kernelSize: IntOrPair(3), padding: IntOrPair(1), bias: false),
            LayerNorm(dimensions: outChans, eps: 1e-6)
        )
        self._net2.wrappedValue = Conv2d(
            inputChannels: outChans, outputChannels: 512, kernelSize: IntOrPair(3),
            stride: IntOrPair(2), padding: IntOrPair(1), bias: false)
        self._net3.wrappedValue = Conv2d(
            inputChannels: 512, outputChannels: v.samFinalOutChannels, kernelSize: IntOrPair(3),
            stride: IntOrPair(2), padding: IntOrPair(1), bias: false)
        super.init()
    }

    /// `x`: NHWC `(B, S, S, 3)` → `(B, S/64, S/64, finalOutChannels)`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = patchEmbed(x)
        if useAbsPos {
            h = h + SAMMath.getAbsPos(posEmbed, h.dim(1))
        }
        for block in blocks {
            h = block(h)
        }
        h = neck.1(neck.0(h))
        h = neck.3(neck.2(h))
        h = net2(h)
        h = net3(h)
        return h
    }

    /// Conv-weight transpose (PyTorch `(out,in,kH,kW)` → MLX `(out,kH,kW,in)`) for the SAM conv keys,
    /// mirroring `sam.py`'s weight handling. Operates on the (already key-remapped) `sam_model.*` keys.
    nonisolated static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        for (key, value) in weights {
            if isConvWeight(key), value.ndim == 4, !isMLXConvShape(value) {
                out[key] = value.transposed(0, 2, 3, 1)
            } else {
                out[key] = value
            }
        }
        return out
    }

    private nonisolated static func isConvWeight(_ key: String) -> Bool {
        guard key.contains("sam_model") else { return false }
        return key.hasSuffix("patch_embed.proj.weight")
            || key.hasSuffix("neck.0.weight") || key.hasSuffix("neck.2.weight")
            || key.hasSuffix("net_2.weight") || key.hasSuffix("net_3.weight")
    }

    nonisolated static func isMLXConvShape(_ a: MLXArray) -> Bool {
        guard a.ndim == 4 else { return false }
        let outC = a.dim(0), kH = a.dim(1), kW = a.dim(2)
        return outC >= kH && outC >= kW && kH == kW
    }
}
