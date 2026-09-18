import Foundation
import MLX
import MLXNN

// MARK: - Prompt Encoder (SAM2-style)

/// Encodes point prompts into sparse token embeddings, matching SAM3's tracker prompt encoder.
/// Checkpoint keys (`tracker_model.prompt_encoder.*`):
///   `point_embed.weight`              → `pointEmbed`  [4, 256] (0=neg, 1=pos, 2=boxTL, 3=boxBR)
///   `not_a_point_embed.weight`        → `notAPointEmbed` [1, 256]
///   `no_mask_embed.weight`            → `noMaskEmbed` [1, 256]
///   `shared_embedding.positional_embedding` → `gaussianMatrix` [2, 128] (loaded manually)
nonisolated final class SAM3PromptEncoder: Module {
    @ModuleInfo var pointEmbed: Embedding
    @ModuleInfo var notAPointEmbed: Embedding
    @ModuleInfo var noMaskEmbed: Embedding
    var gaussianMatrix: MLXArray   // [2, 128]

    init(promptDim: Int = 256) {
        _pointEmbed.wrappedValue = Embedding(embeddingCount: 4, dimensions: promptDim)
        _notAPointEmbed.wrappedValue = Embedding(embeddingCount: 1, dimensions: promptDim)
        _noMaskEmbed.wrappedValue = Embedding(embeddingCount: 1, dimensions: promptDim)
        gaussianMatrix = MLXArray.zeros([2, 128])
        super.init()
    }

    /// points: [N, 2] in 1008-px space; labels: [N] (0=background, 1=foreground).
    /// Returns sparse [1, N+1, promptDim] (appends a `not_a_point` padding token).
    func callAsFunction(points: MLXArray, labels: MLXArray) -> MLXArray {
        let pts = (points + 0.5).asType(.float32)          // shift to pixel center
        let padded = concatenated([pts, MLXArray.zeros([1, 2])], axis: 0)          // [N+1, 2]
        let paddedLabels = concatenated([labels, MLXArray([Int32(-1)])], axis: 0)  // [N+1]

        let pe = encodePoints(padded)                      // [N+1, 256]

        let notAPoint = notAPointEmbed(MLXArray([Int32(0)]))  // [1, 256]
        let isPad = (paddedLabels .== Int32(-1)).asType(.float32).expandedDimensions(axis: -1)  // [N+1, 1]
        let peReplaced = pe * (1 - isPad) + notAPoint * isPad

        let idx = clip(paddedLabels, min: Int32(0), max: Int32(3))
        let typeEmb = pointEmbed(idx) * (1 - isPad)        // [N+1, 256]

        return (peReplaced + typeEmb).expandedDimensions(axis: 0)   // [1, N+1, 256]
    }

    /// `PositionEmbeddingRandom`: coords → 2*coords-1 → @gaussian → 2π → [sin, cos].
    private func encodePoints(_ points: MLXArray) -> MLXArray {
        let coords = points / 1008.0                       // [0, 1]
        let c = 2 * coords - 1                             // [-1, 1]
        let proj = matmul(c, gaussianMatrix)               // [N, 128]
        let angles = 2 * Float.pi * proj
        return concatenated([sin(angles), cos(angles)], axis: -1)  // [N, 256]
    }

    /// Dense positional encoding for the 72×72 image embedding grid → [1, 72, 72, 256] (NHWC).
    func densePositionalEncoding(grid: Int = 72) -> MLXArray {
        var coords = [Float]()
        for r in 0 ..< grid {
            for c in 0 ..< grid {
                coords.append((Float(c) + 0.5) / Float(grid))
                coords.append((Float(r) + 0.5) / Float(grid))
            }
        }
        let xy = MLXArray(coords, [grid * grid, 2])        // (x, y)
        let c = 2 * xy - 1
        let proj = matmul(c, gaussianMatrix)               // [G*G, 128]
        let angles = 2 * Float.pi * proj
        let pe = concatenated([sin(angles), cos(angles)], axis: -1)  // [G*G, 256]
        return pe.reshaped([1, grid, grid, 256])
    }
}

// MARK: - Mask decoder attention (no RoPE, optional downsample)

nonisolated final class SAM3MaskAttention: Module {
    @ModuleInfo var qProj: Linear
    @ModuleInfo var kProj: Linear
    @ModuleInfo var vProj: Linear
    @ModuleInfo var oProj: Linear
    let numHeads: Int
    let internalDim: Int

    init(embeddingDim: Int, numHeads: Int, downsampleRate: Int = 1, kvInDim: Int? = nil) {
        let internalChannels = embeddingDim / downsampleRate
        let kvDim = kvInDim ?? embeddingDim
        self.numHeads = numHeads
        self.internalDim = internalChannels
        _qProj.wrappedValue = Linear(embeddingDim, internalChannels)
        _kProj.wrappedValue = Linear(kvDim, internalChannels)
        _vProj.wrappedValue = Linear(kvDim, internalChannels)
        _oProj.wrappedValue = Linear(internalChannels, embeddingDim)
        super.init()
    }

    func callAsFunction(_ query: MLXArray, _ key: MLXArray, _ value: MLXArray) -> MLXArray {
        let b = query.dim(0), ql = query.dim(1)
        let h = numHeads, hd = internalDim / numHeads
        func heads(_ t: MLXArray) -> MLXArray {
            t.reshaped([b, -1, h, hd]).transposed(0, 2, 1, 3)
        }
        let q = heads(qProj(query))
        let k = heads(kProj(key))
        let v = heads(vProj(value))
        let attn = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: pow(Float(hd), -0.5), mask: nil)
        let out = attn.transposed(0, 2, 1, 3).reshaped([b, ql, internalDim])
        return oProj(out)
    }
}

// MARK: - Mask decoder MLP

/// `proj_in` → relu → [layers.N] → relu → `proj_out` (optional sigmoid).
nonisolated final class SAM3MaskMLP: Module {
    @ModuleInfo var projIn: Linear
    @ModuleInfo var layers: [Linear]
    @ModuleInfo var projOut: Linear
    let sigmoidOutput: Bool

    init(inputDim: Int, hiddenDim: Int, outputDim: Int, numLayers: Int, sigmoidOutput: Bool = false) {
        _projIn.wrappedValue = Linear(inputDim, hiddenDim)
        _layers.wrappedValue = (0 ..< max(0, numLayers - 2)).map { _ in Linear(hiddenDim, hiddenDim) }
        _projOut.wrappedValue = Linear(hiddenDim, outputDim)
        self.sigmoidOutput = sigmoidOutput
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = MLXNN.relu(projIn(x))
        for layer in layers { h = MLXNN.relu(layer(h)) }
        h = projOut(h)
        return sigmoidOutput ? sigmoid(h) : h
    }
}

/// Transformer MLP block: `proj_in` (256→mlpDim) → relu → `proj_out` (mlpDim→256).
nonisolated final class SAM3MLPBlock: Module {
    @ModuleInfo var projIn: Linear
    @ModuleInfo var projOut: Linear
    init(embeddingDim: Int, mlpDim: Int) {
        _projIn.wrappedValue = Linear(embeddingDim, mlpDim)
        _projOut.wrappedValue = Linear(mlpDim, embeddingDim)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { projOut(MLXNN.relu(projIn(x))) }
}

// MARK: - Two-way transformer

nonisolated final class SAM3TwoWayAttentionBlock: Module {
    @ModuleInfo var selfAttn: SAM3MaskAttention
    @ModuleInfo var layerNorm1: LayerNorm
    @ModuleInfo var crossAttnTokenToImage: SAM3MaskAttention
    @ModuleInfo var layerNorm2: LayerNorm
    @ModuleInfo var mlp: SAM3MLPBlock
    @ModuleInfo var layerNorm3: LayerNorm
    @ModuleInfo var layerNorm4: LayerNorm
    @ModuleInfo var crossAttnImageToToken: SAM3MaskAttention
    let skipFirstLayerPe: Bool

    init(embeddingDim: Int, numHeads: Int, mlpDim: Int, downsampleRate: Int, skipFirstLayerPe: Bool) {
        _selfAttn.wrappedValue = SAM3MaskAttention(embeddingDim: embeddingDim, numHeads: numHeads)
        _layerNorm1.wrappedValue = LayerNorm(dimensions: embeddingDim)
        _crossAttnTokenToImage.wrappedValue = SAM3MaskAttention(embeddingDim: embeddingDim, numHeads: numHeads, downsampleRate: downsampleRate)
        _layerNorm2.wrappedValue = LayerNorm(dimensions: embeddingDim)
        _mlp.wrappedValue = SAM3MLPBlock(embeddingDim: embeddingDim, mlpDim: mlpDim)
        _layerNorm3.wrappedValue = LayerNorm(dimensions: embeddingDim)
        _layerNorm4.wrappedValue = LayerNorm(dimensions: embeddingDim)
        _crossAttnImageToToken.wrappedValue = SAM3MaskAttention(embeddingDim: embeddingDim, numHeads: numHeads, downsampleRate: downsampleRate)
        self.skipFirstLayerPe = skipFirstLayerPe
        super.init()
    }

    func callAsFunction(queries: MLXArray, keys: MLXArray, queryPe: MLXArray, keyPe: MLXArray) -> (MLXArray, MLXArray) {
        var queries = queries
        var keys = keys

        if skipFirstLayerPe {
            queries = selfAttn(queries, queries, queries)
        } else {
            let q = queries + queryPe
            queries = queries + selfAttn(q, q, queries)
        }
        queries = layerNorm1(queries)

        var q = queries + queryPe
        var k = keys + keyPe
        queries = queries + crossAttnTokenToImage(q, k, keys)
        queries = layerNorm2(queries)

        queries = queries + mlp(queries)
        queries = layerNorm3(queries)

        q = queries + queryPe
        k = keys + keyPe
        keys = keys + crossAttnImageToToken(k, q, queries)
        keys = layerNorm4(keys)

        return (queries, keys)
    }
}

nonisolated final class SAM3TwoWayTransformer: Module {
    @ModuleInfo var layers: [SAM3TwoWayAttentionBlock]
    @ModuleInfo var finalAttnTokenToImage: SAM3MaskAttention
    @ModuleInfo var layerNormFinalAttn: LayerNorm

    init(depth: Int, embeddingDim: Int, numHeads: Int, mlpDim: Int, downsampleRate: Int = 2) {
        _layers.wrappedValue = (0 ..< depth).map { i in
            SAM3TwoWayAttentionBlock(embeddingDim: embeddingDim, numHeads: numHeads, mlpDim: mlpDim,
                                     downsampleRate: downsampleRate, skipFirstLayerPe: i == 0)
        }
        _finalAttnTokenToImage.wrappedValue = SAM3MaskAttention(embeddingDim: embeddingDim, numHeads: numHeads, downsampleRate: downsampleRate)
        _layerNormFinalAttn.wrappedValue = LayerNorm(dimensions: embeddingDim)
        super.init()
    }

    /// imageEmbedding [B, H, W, C] (NHWC), imagePe [B, H, W, C], pointEmbedding [B, N, C].
    /// Returns (queries [B, N, C], keys [B, H*W, C]).
    func callAsFunction(imageEmbedding: MLXArray, imagePe: MLXArray, pointEmbedding: MLXArray) -> (MLXArray, MLXArray) {
        let c = imageEmbedding.dim(3)
        let imageFlat = imageEmbedding.reshaped([imageEmbedding.dim(0), -1, c])  // [B, H*W, C]
        let peFlat = imagePe.reshaped([imagePe.dim(0), -1, c])                  // [B, H*W, C]

        var queries = pointEmbedding
        var keys = imageFlat
        for layer in layers {
            (queries, keys) = layer(queries: queries, keys: keys, queryPe: pointEmbedding, keyPe: peFlat)
        }
        let q = queries + pointEmbedding
        let k = keys + peFlat
        queries = queries + finalAttnTokenToImage(q, k, keys)
        queries = layerNormFinalAttn(queries)
        return (queries, keys)
    }
}

// MARK: - Mask decoder

/// `tracker_model.mask_decoder.*` — the SAM2-style transformer mask decoder.
nonisolated final class SAM3MaskDecoder: Module {
    @ModuleInfo var transformer: SAM3TwoWayTransformer
    @ModuleInfo var iouToken: Embedding
    @ModuleInfo var maskTokens: Embedding
    @ModuleInfo var objScoreToken: Embedding
    @ModuleInfo var outputHypernetworksMlps: [SAM3MaskMLP]
    @ModuleInfo var iouPredictionHead: SAM3MaskMLP
    @ModuleInfo var predObjScoreHead: SAM3MaskMLP
    @ModuleInfo var convS0: Conv2d
    @ModuleInfo var convS1: Conv2d
    @ModuleInfo var upscaleConv1: ConvTransposed2d
    @ModuleInfo var upscaleLayerNorm: GroupNorm
    @ModuleInfo var upscaleConv2: ConvTransposed2d
    let numMaskTokens: Int

    init(transformerDim: Int = 256,
         numHeads: Int = 8,
         numMultimaskOutputs: Int = 3,
         iouHeadDepth: Int = 3,
         iouHeadHiddenDim: Int = 256) {
        _transformer.wrappedValue = SAM3TwoWayTransformer(depth: 2, embeddingDim: transformerDim,
                                                          numHeads: numHeads, mlpDim: 2048)
        _iouToken.wrappedValue = Embedding(embeddingCount: 1, dimensions: transformerDim)
        _maskTokens.wrappedValue = Embedding(embeddingCount: numMultimaskOutputs + 1, dimensions: transformerDim)
        _objScoreToken.wrappedValue = Embedding(embeddingCount: 1, dimensions: transformerDim)
        _outputHypernetworksMlps.wrappedValue = (0 ..< (numMultimaskOutputs + 1)).map { _ in
            SAM3MaskMLP(inputDim: transformerDim, hiddenDim: transformerDim, outputDim: transformerDim / 8, numLayers: 3)
        }
        _iouPredictionHead.wrappedValue = SAM3MaskMLP(inputDim: transformerDim, hiddenDim: iouHeadHiddenDim,
                                                      outputDim: numMultimaskOutputs + 1, numLayers: iouHeadDepth, sigmoidOutput: true)
        _predObjScoreHead.wrappedValue = SAM3MaskMLP(inputDim: transformerDim, hiddenDim: transformerDim, outputDim: 1, numLayers: 3)
        _convS0.wrappedValue = Conv2d(inputChannels: transformerDim, outputChannels: transformerDim / 8, kernelSize: .init(1))
        _convS1.wrappedValue = Conv2d(inputChannels: transformerDim, outputChannels: transformerDim / 4, kernelSize: .init(1))
        _upscaleConv1.wrappedValue = ConvTransposed2d(inputChannels: transformerDim, outputChannels: transformerDim / 4,
                                                     kernelSize: .init(2), stride: .init(2))
        _upscaleLayerNorm.wrappedValue = GroupNorm(groupCount: 1, dimensions: transformerDim / 4,
                                                   eps: 1e-6, affine: true, pytorchCompatible: true)
        _upscaleConv2.wrappedValue = ConvTransposed2d(inputChannels: transformerDim / 4, outputChannels: transformerDim / 8,
                                                     kernelSize: .init(2), stride: .init(2))
        self.numMaskTokens = numMultimaskOutputs + 1
        super.init()
    }

    /// imageEmbedding: [B, 72, 72, 256] (NHWC); imagePe: [B, 72, 72, 256];
    /// sparsePrompts: [B, N, 256]; densePrompts: [B, 72, 72, 256];
    /// highResFeatures: [featS0 [B,288,288,32], featS1 [B,144,144,64]] (already conv_s0/s1 projected).
    /// Returns masks [B, numMaskTokens, 288, 288] (logits) and iou [B, numMaskTokens].
    func callAsFunction(imageEmbedding: MLXArray, imagePe: MLXArray,
                        sparsePrompts: MLXArray, densePrompts: MLXArray,
                        highResFeatures: [MLXArray]) -> (masks: MLXArray, iou: MLXArray) {
        let b = imageEmbedding.dim(0)

        let outputTokens = concatenated([objScoreToken(MLXArray([Int32(0)])),
                                         iouToken(MLXArray([Int32(0)])),
                                         maskTokens(MLXArray((0 ..< numMaskTokens).map { Int32($0) }))], axis: 0)  // [6, 256]
        let tokens = concatenated([outputTokens.expandedDimensions(axis: 0), sparsePrompts], axis: 1)  // [B, 6+N, 256]

        let src = imageEmbedding + densePrompts  // [B, 72, 72, 256]
        let (hs, keys) = transformer(imageEmbedding: src, imagePe: imagePe, pointEmbedding: tokens)

        let iouTokenOut = hs[0..., 1, 0...]     // [B, 256]
        let maskTokensOut = stacked((0 ..< numMaskTokens).map { hs[0..., 2 + $0, 0...] }, axis: 1)  // [B, 4, 256]

        // output upscaling with high-res features, on the transformer-updated image features
        let srcOut = keys.reshaped([b, 72, 72, 256])
        var upscaled = upscaleConv1(srcOut)                 // [B, 144, 144, 64]
        upscaled = upscaleLayerNorm(upscaled) + highResFeatures[1]  // add feat_s1
        upscaled = MLXNN.gelu(upscaled)
        upscaled = upscaleConv2(upscaled) + highResFeatures[0]      // add feat_s0  [B, 288, 288, 32]
        upscaled = MLXNN.gelu(upscaled)

        // hypernetworks: per-token 1×1 dot product
        let hyperIn = stacked((0 ..< numMaskTokens).map { i in
            outputHypernetworksMlps[i](maskTokensOut[0..., i, 0...])  // [B, 32]
        }, axis: 1)  // [B, 4, 32]

        let upFlat = upscaled.reshaped([b, -1, upscaled.dim(3)]).transposed(0, 2, 1)  // [B, 32, 288*288]
        let masks = matmul(hyperIn, upFlat).reshaped([b, numMaskTokens, upscaled.dim(1), upscaled.dim(2)])  // [B,4,288,288]

        let iou = iouPredictionHead(iouTokenOut)  // [B, 4]

        return (masks, iou)
    }
}
