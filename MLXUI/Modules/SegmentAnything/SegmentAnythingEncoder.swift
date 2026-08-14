import Foundation
import MLX
import MLXNN

// MARK: - Patch Embedding

nonisolated final class SAM3PatchEmbeddings: Module {
    @ModuleInfo var projection: Conv2d

    init(hiddenSize: Int, patchSize: Int) {
        _projection.wrappedValue = Conv2d(inputChannels: 3, outputChannels: hiddenSize,
                                          kernelSize: .init(patchSize), stride: .init(patchSize),
                                          bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { projection(x) }
}

nonisolated final class SAM3Embeddings: Module {
    @ModuleInfo var patchEmbeddings: SAM3PatchEmbeddings
    var positionEmbeddings: MLXArray

    init(hiddenSize: Int, patchSize: Int, gridH: Int, gridW: Int) {
        _patchEmbeddings.wrappedValue = SAM3PatchEmbeddings(hiddenSize: hiddenSize, patchSize: patchSize)
        positionEmbeddings = MLXArray.zeros([1, gridH, gridW, hiddenSize])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let patches = patchEmbeddings(x)   // (B, H, W, D)
        let (_, h, w, d) = (patches.dim(0), patches.dim(1), patches.dim(2), patches.dim(3))
        // Checkpoint stores position embeddings flat (1, H*W, D); reshape to match spatial patches.
        let pe = positionEmbeddings.ndim == 3
            ? positionEmbeddings.reshaped([1, h, w, d])
            : positionEmbeddings
        return patches + pe
    }
}

// MARK: - RoPE (2-D)

private enum SAM3RoPE {
    nonisolated static func apply(_ x: MLXArray, positions: MLXArray, theta: Float = 10000.0) -> MLXArray {
        let headDim = x.dim(3)
        let halfDim = headDim / 2
        let idx = MLXArray.arange(0, halfDim).asType(.float32)
        let f = pow(Float(theta), -(idx / Float(halfDim)))

        let rowPos = positions[.ellipsis, 0].expandedDimensions(axis: -1).asType(.float32)
        let colPos = positions[.ellipsis, 1].expandedDimensions(axis: -1).asType(.float32)
        let rowAngle = rowPos * f[0 ..< halfDim / 2]
        let colAngle = colPos * f[halfDim / 2 ..< halfDim]
        let angles = concatenated([rowAngle, colAngle], axis: -1)

        let x1 = x[.ellipsis, 0 ..< halfDim]
        let x2 = x[.ellipsis, halfDim ..< headDim]
        let ca = cos(angles).expandedDimensions(axes: [0, 1])
        let sa = sin(angles).expandedDimensions(axes: [0, 1])
        return concatenated([x1 * ca - x2 * sa, x1 * sa + x2 * ca], axis: -1)
    }
}

// MARK: - Grid position helper

/// Build [H*W, 2] Int32 grid of (row, col) indices.
private nonisolated func makeGridPositions(h: Int, w: Int) -> MLXArray {
    let rows = (0 ..< h).flatMap { r in [Int32](repeating: Int32(r), count: w) }
    let cols = (0 ..< h).flatMap { _ in (0 ..< w).map { Int32($0) } }
    return stacked([MLXArray(rows), MLXArray(cols)], axis: -1)   // [H*W, 2]
}

// MARK: - Attention

/// Weight keys (relative to layer root):
///   `attention.q_proj.*`, `attention.k_proj.*`, `attention.v_proj.*`, `attention.out_proj.*`
nonisolated final class SAM3Attention: Module {
    @ModuleInfo var qProj: Linear
    @ModuleInfo var kProj: Linear
    @ModuleInfo var vProj: Linear
    @ModuleInfo var outProj: Linear
    let numHeads: Int

    init(hiddenSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        _qProj.wrappedValue   = Linear(hiddenSize, hiddenSize)
        _kProj.wrappedValue   = Linear(hiddenSize, hiddenSize)
        _vProj.wrappedValue   = Linear(hiddenSize, hiddenSize)
        _outProj.wrappedValue = Linear(hiddenSize, hiddenSize)
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
        return outProj(attn.transposed(0, 2, 1, 3).reshaped([b, l, d]))
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

/// Weight keys relative to `backbone.layers.N`:
///   `layer_norm1.*`, `layer_norm2.*`, `attention.*`, `mlp.*`
///
/// Windowed attention is used when `windowSize > 0` AND the spatial grid is
/// evenly divisible by `windowSize`; otherwise falls back to global attention.
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

        // Use windowed attention only when the grid divides evenly — avoids padding/cropping.
        let useWindow = windowSize > 0 && h % windowSize == 0 && w % windowSize == 0
        let attnOut: MLXArray
        if useWindow {
            attnOut = windowedAttn(x, b: b, h: h, w: w, d: d)
        } else {
            let positions = makeGridPositions(h: h, w: w)
            attnOut = attention(layerNorm1(flat), positions: positions)
        }

        let r1 = flat + attnOut
        let r2 = r1 + mlp(layerNorm2(r1))
        return r2.reshaped([b, h, w, d])
    }

    // Pre-condition: h % windowSize == 0 && w % windowSize == 0
    private func windowedAttn(_ x: MLXArray, b: Int, h: Int, w: Int, d: Int) -> MLXArray {
        let ws = windowSize
        let nH = h / ws, nW = w / ws
        let windows = x.reshaped([b, nH, ws, nW, ws, d])
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped([b * nH * nW, ws * ws, d])
        let winPos = makeGridPositions(h: ws, w: ws)
        let attn = attention(layerNorm1(windows), positions: winPos)
        return attn.reshaped([b, nH, nW, ws, ws, d])
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped([b, h * w, d])
    }
}

// MARK: - ViT Backbone

/// Matches `detector_model.vision_encoder.backbone.*` in mlx-community/sam3-4bit.
/// Global attention at `globalAttnLayers`; all others use windowed attention
/// (falling back to global when the grid is not evenly divisible by windowSize).
/// Final 1×1 neck conv: hiddenSize → neckChannels (256).
nonisolated final class SAM3ViTBackbone: Module {
    @ModuleInfo var embeddings: SAM3Embeddings
    @ModuleInfo var layers: [SAM3ViTLayer]
    @ModuleInfo var layerNorm: LayerNorm
    @ModuleInfo var neckConv: Conv2d
    let gridH: Int
    let gridW: Int

    init(hiddenSize: Int = 1024,
         numLayers: Int = 32,
         numHeads: Int = 16,
         intermediateSize: Int = 4736,
         imageSize: Int = 336,
         patchSize: Int = 14,
         windowSize: Int = 14,
         globalAttnLayers: Set<Int> = [7, 15, 23, 31],
         neckChannels: Int = 256) {
        gridH = imageSize / patchSize
        gridW = imageSize / patchSize
        _embeddings.wrappedValue = SAM3Embeddings(hiddenSize: hiddenSize, patchSize: patchSize,
                                                   gridH: gridH, gridW: gridW)
        _layers.wrappedValue = (0 ..< numLayers).map { i in
            SAM3ViTLayer(hiddenSize: hiddenSize, numHeads: numHeads,
                         intermediateSize: intermediateSize,
                         windowSize: globalAttnLayers.contains(i) ? 0 : windowSize)
        }
        _layerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
        _neckConv.wrappedValue  = Conv2d(inputChannels: hiddenSize, outputChannels: neckChannels,
                                          kernelSize: .init(1), bias: false)
        super.init()
    }

    /// x: [B, imageSize, imageSize, 3] → [B, gridH, gridW, neckChannels]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = embeddings(x)
        for layer in layers { h = layer(h) }
        let (b, gh, gw, d) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
        h = layerNorm(h.reshaped([b, gh * gw, d])).reshaped([b, gh, gw, d])
        return neckConv(h)
    }
}
