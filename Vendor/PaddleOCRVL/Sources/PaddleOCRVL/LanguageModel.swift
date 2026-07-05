import Foundation
import MLX
import MLXNN
import MLXFast

public class RoPE: Module {
    let dimensions: Int
    let traditional: Bool
    let base: Float
    let scale: Float

    public init(dimensions: Int, traditional: Bool = false, base: Float = 10_000, scale: Float = 1.0) {
        self.dimensions = dimensions
        self.traditional = traditional
        self.base = base
        self.scale = scale
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: dimensions,
            traditional: traditional,
            base: base,
            scale: scale,
            offset: offset
        )
    }
}

public class KVCache {
    var keys: MLXArray?
    var values: MLXArray?
    public var offset: Int = 0

    public init() {}

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if let existingKeys = self.keys, let existingValues = self.values {
            self.keys = concatenated([existingKeys, keys], axis: 2)
            self.values = concatenated([existingValues, values], axis: 2)
        } else {
            self.keys = keys
            self.values = values
        }
        self.offset += keys.dim(2)
        return (self.keys!, self.values!)
    }

    public func reset() {
        keys = nil
        values = nil
        offset = 0
    }
}

/// Rotate the last dim's halves: `[-x2, x1]`.
private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let d = x.dim(-1)
    let x1 = x[.ellipsis, 0 ..< (d / 2)]
    let x2 = x[.ellipsis, (d / 2) ..< d]
    return concatenated([-x2, x1], axis: -1)
}

/// Qwen2-VL-style 3D mRoPE cos/sin tables for `positionIds` `[3, L]` (temporal, height, width).
/// Returns `[L, headDim]` tables selecting each frequency band from the axis given by
/// `section` (doubled and cycled t,h,w,t,h,w). For text tokens all 3 rows are equal, so this
/// reduces to ordinary 1D RoPE.
private func mropeCosSin(
    positionIds: MLXArray, headDim: Int, theta: Float, section: [Int]
) -> (MLXArray, MLXArray) {
    let half = headDim / 2
    var invFreqVals = [Float]()
    var i = 0
    while i < half {
        invFreqVals.append(1.0 / pow(theta, Float(2 * i) / Float(headDim)))
        i += 1
    }
    let L = positionIds.dim(1)
    let invFreq = MLXArray(invFreqVals).reshaped(1, 1, half)          // [1,1,half]
    let pos = positionIds.asType(.float32).reshaped(3, L, 1)          // [3,L,1]
    let freqs = pos * invFreq                                         // [3,L,half]
    let emb = concatenated([freqs, freqs], axis: -1)                  // [3,L,headDim]
    let cosF = cos(emb)
    let sinF = sin(emb)

    // Select bands: sections doubled and cycled across axes t,h,w,t,h,w.
    let doubled = section + section
    var cosParts = [MLXArray]()
    var sinParts = [MLXArray]()
    var start = 0
    for (k, size) in doubled.enumerated() {
        let axis = k % 3
        cosParts.append(cosF[axis, 0..., start ..< (start + size)])
        sinParts.append(sinF[axis, 0..., start ..< (start + size)])
        start += size
    }
    return (concatenated(cosParts, axis: -1), concatenated(sinParts, axis: -1))  // each [L,headDim]
}

public class ERNIEAttention: Module {
    let config: PaddleOCRVLTextConfig
    let scale: Float
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let mropeSection: [Int]

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    public init(config: PaddleOCRVLTextConfig) {
        self.config = config
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        // Use the config's explicit head_dim (128 on 1.5) — it is NOT hiddenSize/numHeads.
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)
        self.mropeSection = config.mropeSection

        let dim = config.hiddenSize

        self._qProj.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, dim, bias: false)

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray?,
        cache: KVCache?,
        positionIds: MLXArray
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        queries = queries.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)

        let (cosT, sinT) = mropeCosSin(
            positionIds: positionIds, headDim: headDim, theta: config.ropeTheta, section: mropeSection)
        let c = cosT.reshaped(1, 1, L, headDim)
        let s = sinT.reshaped(1, 1, L, headDim)
        queries = queries * c + rotateHalf(queries) * s
        keys = keys * c + rotateHalf(keys) * s

        if let cache = cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        if numKVHeads < numHeads {
            let repeats = numHeads / numKVHeads
            keys = expandedKVHeads(keys, repeats: repeats)
            values = expandedKVHeads(values, repeats: repeats)
        }

        var scores = matmul(queries, keys.transposed(0, 1, 3, 2)) * scale

        if let mask = mask {
            scores = scores + mask
        }

        let weights = softmax(scores, axis: -1)
        var output = matmul(weights, values)

        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return oProj(output)
    }

    private func expandedKVHeads(_ x: MLXArray, repeats: Int) -> MLXArray {
        let (B, nKVHeads, L, D) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let expanded = x.expandedDimensions(axis: 2)
        let repeated = MLX.repeated(expanded, count: repeats, axis: 2)
        return repeated.reshaped(B, nKVHeads * repeats, L, D)
    }
}

public class ERNIEMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public init(config: PaddleOCRVLTextConfig) {
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize

        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

public class RMSNorm: Module {
    let eps: Float
    var weight: MLXArray

    public init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self.weight = MLXArray.ones([dimensions])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

public class ERNIEDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: ERNIEAttention
    @ModuleInfo(key: "mlp") var mlp: ERNIEMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(config: PaddleOCRVLTextConfig) {
        self._selfAttn.wrappedValue = ERNIEAttention(config: config)
        self._mlp.wrappedValue = ERNIEMLP(config: config)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray?,
        cache: KVCache?,
        positionIds: MLXArray
    ) -> MLXArray {
        var h = selfAttn(inputLayerNorm(x), mask: mask, cache: cache, positionIds: positionIds)
        h = x + h
        let mlpOutput = mlp(postAttentionLayerNorm(h))
        return h + mlpOutput
    }
}

public class ERNIEModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [ERNIEDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let config: PaddleOCRVLTextConfig

    public init(config: PaddleOCRVLTextConfig) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

        var decoderLayers: [ERNIEDecoderLayer] = []
        for _ in 0..<config.numHiddenLayers {
            decoderLayers.append(ERNIEDecoderLayer(config: config))
        }
        self._layers.wrappedValue = decoderLayers

        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    public func callAsFunction(
        _ inputIds: MLXArray,
        cache: [KVCache]?,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        let h = embedTokens(inputIds)
        return forward(h, cache: cache, positionIds: positionIds)
    }

    public func forward(
        _ inputsEmbeds: MLXArray,
        cache: [KVCache]?,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        var h = inputsEmbeds
        let mask = createAttentionMask(h: h, cache: cache)

        let L = inputsEmbeds.dim(1)
        let offset = cache?.first?.offset ?? 0
        // Fallback to sequential positions (equivalent to 1D RoPE) when the caller doesn't
        // supply mRoPE 3D position ids.
        let pids = positionIds ?? Self.sequentialPositionIds(offset: offset, length: L)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i], positionIds: pids)
        }

        return norm(h)
    }

    /// `[3, length]` where every row is `offset ..< offset+length` (all axes identical).
    static func sequentialPositionIds(offset: Int, length: Int) -> MLXArray {
        let row = MLXArray(Int32(offset) ..< Int32(offset + length)).reshaped(1, length)
        return concatenated([row, row, row], axis: 0)
    }

    private func createAttentionMask(h: MLXArray, cache: [KVCache]?) -> MLXArray? {
        let n = h.dim(1)
        let offset = cache?.first?.offset ?? 0

        if n == 1 {
            return nil
        }

        var rinds = MLXArray(Int32(0) ..< Int32(offset + n))
        var linds = offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rinds
        linds = linds[0..., .newAxis]
        rinds = rinds[.newAxis]

        let mask = linds .>= rinds
        let additiveMask = MLX.where(mask, MLXArray(Float(0)), MLXArray(Float(-1e9)))

        return additiveMask.reshaped(1, 1, n, offset + n)
    }

    public func getEmbedding(_ inputIds: MLXArray) -> MLXArray {
        embedTokens(inputIds)
    }
}

public class ERNIELanguageModel: Module {
    @ModuleInfo(key: "model") var model: ERNIEModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let config: PaddleOCRVLTextConfig
    public let vocabSize: Int

    public init(config: PaddleOCRVLTextConfig) {
        self.config = config
        self.vocabSize = config.vocabSize

        self._model.wrappedValue = ERNIEModelInner(config: config)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }

        super.init()
    }

    public func callAsFunction(
        _ inputIds: MLXArray,
        cache: [KVCache]?
    ) -> MLXArray {
        var out = model(inputIds, cache: cache)
        out = computeLogits(out)
        return out
    }

    public func forward(
        inputsEmbeds: MLXArray,
        cache: [KVCache]?
    ) -> MLXArray {
        var out = model.forward(inputsEmbeds, cache: cache)
        out = computeLogits(out)
        return out
    }

    private func computeLogits(_ hiddenStates: MLXArray) -> MLXArray {
        if let lmHead = lmHead {
            return lmHead(hiddenStates)
        } else {
            return model.embedTokens.asLinear(hiddenStates)
        }
    }

    public func getInputEmbeddings(_ inputIds: MLXArray) -> MLXArray {
        model.getEmbedding(inputIds)
    }

    public func newCache() -> [KVCache] {
        (0..<config.numHiddenLayers).map { _ in KVCache() }
    }
}
