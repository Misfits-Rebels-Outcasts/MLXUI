import Foundation
import MLX
import MLXFast
import MLXNN

/// The shared MLX ModernBERT encoder stack (token embeddings → transformer layers → final
/// norm), used by both the standalone embedder (`ModernBERTEmbedder`, SUP-4) and Laya's
/// decision encoder (LY-4, `RSI/DelegateLayaBacklog.md` §LY-0 gate **C**). Ported from
/// `laya_mlx/model.py`: `EncoderConfig.from_dict`, `Embeddings`, `EncoderAttention`,
/// `EncoderMLP`, `EncoderLayer`, `attention_masks`, `ModernBert`.
///
/// Key layout matches the raw HF ModernBERT checkpoint one-for-one: `embeddings.tok_embeddings`,
/// `embeddings.norm`, `layers.<n>.{attn_norm,attn.{Wqkv,Wo},mlp_norm,mlp.{Wi,Wo}}`,
/// `final_norm`. Laya's raw checkpoint already prefixes these with `encoder.` (falls out of
/// `LayaDecisionModel.encoder` holding this type — no rename needed); the embedder's raw
/// checkpoint needs only its own `model.` prefix stripped (see `ModernBERTEmbedder.sanitize`).
///
/// Sliding-window attention (`slidingWindowEnabled`) is Laya-only: the embedder has never run
/// it (SUP-4's header: "v1 uses full attention … identical … for inputs ≤ the local window"),
/// so it keeps that behavior — byte-identical — by constructing this encoder with the flag off.
nonisolated struct ModernBERTEncoderConfig: Sendable {
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var normEps: Float = 1e-5
    var normBias: Bool = false
    var attentionBias: Bool = false
    var mlpBias: Bool = false
    var localAttention: Int = 128
    var globalAttnEveryNLayers: Int = 3
    var globalRopeTheta: Float = 160_000
    var localRopeTheta: Float = 10_000
    /// `layer_types` from the checkpoint config, when present (Laya's `encoder/config.json`
    /// ships this per `model.py::EncoderConfig.from_dict`); `nil` falls back to the
    /// `global_attn_every_n_layers` pattern (the embedder's checkpoint has no `layer_types`).
    var layerTypes: [String]?
    /// `rope_parameters.<kind>.rope_theta`, when present (Laya's transformers-5 config
    /// shape); falls back to `globalRopeTheta`/`localRopeTheta` (the embedder's checkpoint
    /// has no `rope_parameters`).
    var ropeThetaByType: [String: Float]?

    var headDim: Int { hiddenSize / numAttentionHeads }

    /// `layer_types[i]` if present, else `i % global_attn_every_n_layers == 0 ? full : sliding`.
    func layerType(_ index: Int) -> String {
        if let layerTypes, index < layerTypes.count { return layerTypes[index] }
        return index % globalAttnEveryNLayers == 0 ? "full_attention" : "sliding_attention"
    }

    /// `rope_parameters[kind].rope_theta` else the global/local fallback theta.
    func ropeBase(_ kind: String) -> Float {
        if let v = ropeThetaByType?[kind] { return v }
        return kind == "full_attention" ? globalRopeTheta : localRopeTheta
    }
}

/// LayerNorm with a weight but **no bias** (ModernBERT's `norm_bias: false` for every
/// shipping checkpoint this encoder loads). MLXNN's `LayerNorm` always carries a bias when
/// affine, which wouldn't match either checkpoint's params.
nonisolated final class ModernBERTLayerNormNoBias: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(_ dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = (x - mean).square().mean(axis: -1, keepDims: true)
        return weight * (x - mean) * rsqrt(variance + eps)
    }
}

/// Boolean key masks, ported from `laya_mlx/model.py::attention_masks`: inclusive local
/// distance `<= local_attention / 2`. Padded queries may see valid keys (never used as keys
/// or pooled outputs, so valid-token results are unchanged — the Python docstring's own
/// justification, carried over verbatim).
nonisolated func modernBERTAttentionMasks(attentionMask: MLXArray, window: Int) -> (full: MLXArray, sliding: MLXArray) {
    let valid = attentionMask.asType(.bool)                          // [B, L]
    let full = valid.expandedDimensions(axes: [1, 2])                  // [B, 1, 1, L]
    let length = valid.dim(1)
    let positions = MLXArray(Int32(0) ..< Int32(length))               // [L]
    let diff = abs(positions.expandedDimensions(axis: 1) - positions.expandedDimensions(axis: 0))  // [L, L]
    let local0 = diff .<= MLXArray(Int32(window / 2))                  // [L, L]
    let notValid = logicalNot(valid).expandedDimensions(axes: [1, 3])  // [B, 1, L, 1]
    let local = logicalAnd(
        logicalOr(local0.expandedDimensions(axes: [0, 1]), notValid),
        full)
    return (full, local)
}

nonisolated private final class ModernBERTAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float
    @ModuleInfo(key: "Wqkv") var wqkv: Linear
    @ModuleInfo(key: "Wo") var wo: Linear
    let rope: RoPE

    init(_ config: ModernBERTEncoderConfig, layerType: String) {
        numHeads = config.numAttentionHeads
        headDim = config.headDim
        scale = pow(Float(headDim), -0.5)
        _wqkv.wrappedValue = Linear(config.hiddenSize, 3 * config.hiddenSize, bias: config.attentionBias)
        _wo.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.attentionBias)
        rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeBase(layerType))
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (b, length) = (x.dim(0), x.dim(1))
        let qkv = wqkv(x).reshaped(b, length, 3, numHeads, headDim).transposed(2, 0, 3, 1, 4)
        let q = rope(qkv[0])
        let k = rope(qkv[1])
        let v = qkv[2]
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        return wo(out.transposed(0, 2, 1, 3).reshaped(b, length, numHeads * headDim))
    }
}

nonisolated private final class ModernBERTMLP: Module {
    @ModuleInfo(key: "Wi") var wi: Linear
    @ModuleInfo(key: "Wo") var wo: Linear

    init(_ config: ModernBERTEncoderConfig) {
        _wi.wrappedValue = Linear(config.hiddenSize, 2 * config.intermediateSize, bias: config.mlpBias)
        _wo.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = wi(x).split(parts: 2, axis: -1)
        return wo(gelu(parts[0]) * parts[1])
    }
}

nonisolated private final class ModernBERTEncoderLayer: Module {
    @ModuleInfo(key: "attn_norm") var attnNorm: ModernBERTLayerNormNoBias?
    @ModuleInfo(key: "attn") var attn: ModernBERTAttention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: ModernBERTLayerNormNoBias
    @ModuleInfo(key: "mlp") var mlp: ModernBERTMLP
    let layerType: String

    init(_ config: ModernBERTEncoderConfig, index: Int) {
        layerType = config.layerType(index)
        if index != 0 {
            _attnNorm.wrappedValue = ModernBERTLayerNormNoBias(config.hiddenSize, eps: config.normEps)
        }
        _attn.wrappedValue = ModernBERTAttention(config, layerType: layerType)
        _mlpNorm.wrappedValue = ModernBERTLayerNormNoBias(config.hiddenSize, eps: config.normEps)
        _mlp.wrappedValue = ModernBERTMLP(config)
    }

    func callAsFunction(_ x: MLXArray, masks: (full: MLXArray, sliding: MLXArray)) -> MLXArray {
        let mask = layerType == "full_attention" ? masks.full : masks.sliding
        var h = x + attn(attnNorm?(x) ?? x, mask: mask)
        h = h + mlp(mlpNorm(h))
        return h
    }
}

nonisolated private final class ModernBERTEmbeddings: Module {
    @ModuleInfo(key: "tok_embeddings") var tokEmbeddings: Embedding
    @ModuleInfo(key: "norm") var norm: ModernBERTLayerNormNoBias

    init(_ config: ModernBERTEncoderConfig) {
        _tokEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _norm.wrappedValue = ModernBERTLayerNormNoBias(config.hiddenSize, eps: config.normEps)
    }

    func callAsFunction(_ ids: MLXArray) -> MLXArray { norm(tokEmbeddings(ids)) }
}

/// The shared encoder: embeddings → layers → final norm. `slidingWindowEnabled` gates whether
/// `sliding_attention` layers actually get the local window mask (Laya: on) or the same
/// full-sequence mask every layer uses (the embedder: off, unchanged from before Gate C).
nonisolated final class ModernBERTEncoder: Module {
    @ModuleInfo(key: "embeddings") fileprivate var embeddings: ModernBERTEmbeddings
    fileprivate let layers: [ModernBERTEncoderLayer]
    @ModuleInfo(key: "final_norm") fileprivate var finalNorm: ModernBERTLayerNormNoBias
    let config: ModernBERTEncoderConfig
    let slidingWindowEnabled: Bool

    init(_ config: ModernBERTEncoderConfig, slidingWindowEnabled: Bool) {
        self.config = config
        self.slidingWindowEnabled = slidingWindowEnabled
        _embeddings.wrappedValue = ModernBERTEmbeddings(config)
        layers = (0 ..< config.numHiddenLayers).map { ModernBERTEncoderLayer(config, index: $0) }
        _finalNorm.wrappedValue = ModernBERTLayerNormNoBias(config.hiddenSize, eps: config.normEps)
    }

    /// Last-layer hidden states for `inputIds` [B, L]. `attentionMask` [B, L] (1 keep / 0 pad).
    func callAsFunction(inputIds: MLXArray, attentionMask: MLXArray) -> MLXArray {
        var h = embeddings(inputIds)
        let masks = modernBERTAttentionMasks(attentionMask: attentionMask, window: config.localAttention)
        // Sliding-window off: every layer gets the same full-sequence mask (byte-identical to
        // the pre-Gate-C embedder, which never built a local window).
        let effectiveMasks = slidingWindowEnabled ? masks : (full: masks.full, sliding: masks.full)
        for layer in layers { h = layer(h, masks: effectiveMasks) }
        return finalNorm(h)
    }
}
