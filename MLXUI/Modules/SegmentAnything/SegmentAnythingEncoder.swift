import Foundation
import MLX
import MLXNN

// MARK: - Patch Embedding

/// `patch_embeddings.projection`: Conv2d(3 → hiddenSize, patchSize², stride patchSize, no bias).
nonisolated final class SAM3PatchEmbeddings: Module {
    @ModuleInfo var projection: Conv2d

    init(hiddenSize: Int, patchSize: Int) {
        _projection.wrappedValue = Conv2d(inputChannels: 3, outputChannels: hiddenSize,
                                          kernelSize: .init(patchSize), stride: .init(patchSize),
                                          padding: 0, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { projection(x) }
}

// MARK: - Embeddings

/// Adds the (loaded) learned absolute positional embeddings to the patch tokens.
/// The checkpoint stores them flat as `[1, 24×24, hiddenSize]`; SAM3 tiles the 24×24
/// grid (not interpolates) up to the 72×72 patch grid (`tile_abs_pos=True`).
nonisolated final class SAM3Embeddings: Module {
    @ModuleInfo var patchEmbeddings: SAM3PatchEmbeddings
    var positionEmbeddings: MLXArray   // [1, 576, hiddenSize], loaded manually (a buffer, not a param)

    init(hiddenSize: Int, patchSize: Int, pretrainGrid: Int) {
        _patchEmbeddings.wrappedValue = SAM3PatchEmbeddings(hiddenSize: hiddenSize, patchSize: patchSize)
        positionEmbeddings = MLXArray.zeros([1, pretrainGrid * pretrainGrid, hiddenSize])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let patches = patchEmbeddings(x)   // (B, H, W, D)
        let (b, h, w, d) = (patches.dim(0), patches.dim(1), patches.dim(2), patches.dim(3))
        let g = Int(sqrt(Float(positionEmbeddings.dim(1))))
        var pe = positionEmbeddings.reshaped([1, g, g, d])
        let repsH = h / g, repsW = w / g
        if repsH > 1 || repsW > 1 {
            pe = MLX.tiled(pe, repetitions: [1, repsH, repsW, 1])
        }
        return patches + pe
    }
}

// MARK: - RoPE (2-D axial)

/// Mirrors `compute_axial_cis` + `apply_rotary_enc` from `sam3/model/vitdet.py`.
/// head_dim = 64 → 32 complex frequencies (16 for x/column, 16 for y/row),
/// `freqs[k] = 10000^(-k/16)`.
private enum SAM3RoPE {
    nonisolated static let freqs: MLXArray = {
        var f = [Float]()
        for k in 0 ..< 16 { f.append(Float(pow(10000.0, -Double(k) / 16.0))) }
        return MLXArray(f)
    }()

    /// x: [B, H, L, D]; positions: [L, 2] of (row, col) already scaled (1/3 for the global grid).
    nonisolated static func apply(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let (b, h, l, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let half = d / 2
        let row = positions[0..., 0].asType(.float32)   // [L]
        let col = positions[0..., 1].asType(.float32)   // [L]
        let xf = col.expandedDimensions(axis: -1) * freqs   // [L, 16]  (x = column)
        let yf = row.expandedDimensions(axis: -1) * freqs   // [L, 16]  (y = row)
        let angles = concatenated([xf, yf], axis: -1)       // [L, 32]
        let c = cos(angles).expandedDimensions(axes: [0, 1]) // [1, 1, L, 32]
        let s = sin(angles).expandedDimensions(axes: [0, 1])
        let xr = x.reshaped([b, h, l, half, 2])
        let a = xr[.ellipsis, 0]
        let bb = xr[.ellipsis, 1]
        let ar = a * c - bb * s
        let br = a * s + bb * c
        return stacked([ar, br], axis: -1).reshaped([b, h, l, d])
    }
}

// MARK: - Grid position helper

/// [H*W, 2] Float grid of (row, col) scaled by `scale`.
private nonisolated func makeGridPositions(h: Int, w: Int, scale: Float) -> MLXArray {
    var rows = [Float](), cols = [Float]()
    for r in 0 ..< h { for _ in 0 ..< w { rows.append(Float(r) * scale) } }
    for _ in 0 ..< h { for c in 0 ..< w { cols.append(Float(c) * scale) } }
    return stacked([MLXArray(rows), MLXArray(cols)], axis: -1)
}

// MARK: - Attention

/// Weight keys: `attention.q_proj/k_proj/v_proj/o_proj` (the checkpoint splits the fused qkv).
nonisolated final class SAM3Attention: Module {
    @ModuleInfo var qProj: Linear
    @ModuleInfo var kProj: Linear
    @ModuleInfo var vProj: Linear
    @ModuleInfo var oProj: Linear
    let numHeads: Int

    init(hiddenSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        _qProj.wrappedValue   = Linear(hiddenSize, hiddenSize)
        _kProj.wrappedValue   = Linear(hiddenSize, hiddenSize)
        _vProj.wrappedValue   = Linear(hiddenSize, hiddenSize)
        _oProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let (b, l, d) = (x.dim(0), x.dim(1), x.dim(2))
        let h = numHeads, hd = d / numHeads

        func proj(_ m: Linear, _ t: MLXArray) -> MLXArray {
            m(t).reshaped([b, l, h, hd]).transposed(0, 2, 1, 3)
        }
        let q = SAM3RoPE.apply(proj(qProj, x), positions: positions)
        let k = SAM3RoPE.apply(proj(kProj, x), positions: positions)
        let v = proj(vProj, x)
        let attn = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: pow(Float(hd), -0.5), mask: nil)
        return oProj(attn.transposed(0, 2, 1, 3).reshaped([b, l, d]))
    }
}

// MARK: - MLP

/// Weight keys: `mlp.fc1.*`, `mlp.fc2.*`
nonisolated final class SAM3MLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _fc1.wrappedValue = Linear(hiddenSize, intermediateSize)
        _fc2.wrappedValue = Linear(intermediateSize, hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(MLXNN.gelu(fc1(x))) }
}

// MARK: - ViT Layer

/// Weight keys relative to `backbone.layers.N`: `layer_norm1/layer_norm2`, `attention.*`, `mlp.*`.
/// Window attention for all but the global layers {7,15,23,31} (window 24 on the 72 grid → 3×3 windows).
nonisolated final class SAM3ViTLayer: Module {
    @ModuleInfo var layerNorm1: LayerNorm
    @ModuleInfo var layerNorm2: LayerNorm
    @ModuleInfo var attention: SAM3Attention
    @ModuleInfo var mlp: SAM3MLP
    let windowSize: Int   // 0 = global

    init(hiddenSize: Int, numHeads: Int, intermediateSize: Int, windowSize: Int) {
        _layerNorm1.wrappedValue = LayerNorm(dimensions: hiddenSize)
        _layerNorm2.wrappedValue = LayerNorm(dimensions: hiddenSize)
        _attention.wrappedValue  = SAM3Attention(hiddenSize: hiddenSize, numHeads: numHeads)
        _mlp.wrappedValue        = SAM3MLP(hiddenSize: hiddenSize, intermediateSize: intermediateSize)
        self.windowSize = windowSize
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let flat = x.reshaped([b, h * w, d])

        let attnOut: MLXArray
        if windowSize > 0 {
            attnOut = windowedAttn(x, b: b, h: h, w: w, d: d)
        } else {
            let positions = makeGridPositions(h: h, w: w, scale: 1.0 / 3.0)
            attnOut = attention(layerNorm1(flat), positions: positions)
        }

        let r1 = flat + attnOut
        let r2 = r1 + mlp(layerNorm2(r1))
        return r2.reshaped([b, h, w, d])
    }

    private func windowedAttn(_ x: MLXArray, b: Int, h: Int, w: Int, d: Int) -> MLXArray {
        let ws = windowSize
        let nH = h / ws, nW = w / ws
        let windows = x.reshaped([b, nH, ws, nW, ws, d])
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped([b * nH * nW, ws * ws, d])
        let winPos = makeGridPositions(h: ws, w: ws, scale: 1.0)
        let attn = attention(layerNorm1(windows), positions: winPos)
        return attn.reshaped([b, nH, nW, ws, ws, d])
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped([b, h * w, d])
    }
}

// MARK: - ViT Backbone

/// `detector_model.vision_encoder.backbone.*`. 1008px input → 72×72 tokens (stride 14).
/// `ln_pre` before the blocks; no post-norm and no neck conv (the neck is a separate module).
nonisolated final class SAM3ViTBackbone: Module {
    @ModuleInfo var embeddings: SAM3Embeddings
    @ModuleInfo var layers: [SAM3ViTLayer]
    @ModuleInfo var layerNorm: LayerNorm   // ln_pre (checkpoint key: backbone.layer_norm.*)

    init(hiddenSize: Int = 1024,
         numLayers: Int = 32,
         numHeads: Int = 16,
         intermediateSize: Int = 4736,
         imageSize: Int = 1008,
         patchSize: Int = 14,
         windowSize: Int = 24,
         globalAttnLayers: Set<Int> = [7, 15, 23, 31]) {
        _embeddings.wrappedValue = SAM3Embeddings(hiddenSize: hiddenSize, patchSize: patchSize, pretrainGrid: 24)
        _layers.wrappedValue = (0 ..< numLayers).map { i in
            SAM3ViTLayer(hiddenSize: hiddenSize, numHeads: numHeads,
                         intermediateSize: intermediateSize,
                         windowSize: globalAttnLayers.contains(i) ? 0 : windowSize)
        }
        _layerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
        super.init()
    }

    /// x: [B, imageSize, imageSize, 3] → [B, 72, 72, 1024]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = embeddings(x)
        h = layerNorm(h)          // ln_pre, normalized over the last dim
        for layer in layers { h = layer(h) }
        return h
    }
}
