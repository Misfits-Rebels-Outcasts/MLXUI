import Foundation
import MLX
import MLXNN
import MLXFast
import MLXLMCommon

/// The DeepSeek-V2 MoE text decoder for DeepSeek-OCR-2 — a from-scratch port of
/// `Blaizzy/mlx-vlm`'s `mlx_vlm/models/deepseekocr/language.py` for **this** checkpoint's config:
/// `use_mla:false` ⇒ plain **Llama/MHA** attention (not MLA), softmax + greedy MoE gate (not V3's
/// sigmoid/noaux), `first_k_dense_replace:1` (layer 0 dense, rest MoE), 64 routed + 2 shared experts.
/// Reuses the public `SwitchGLU` for the expert stack and the public attention helpers
/// (`createAttentionMask`/`applyRotaryPosition`/`attentionWithCacheUpdate`). Adds an `inputEmbedding`
/// decode path for the vision merge. Module tree matches the (remapped) checkpoint:
/// `language_model.model.{embed_tokens,layers,norm}` + `language_model.lm_head`.
/// All classes `nonisolated` (MainActor default).

nonisolated final class DSV2Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    let rope: RoPE

    init(_ c: DeepSeekOCRConfig) {
        self.numHeads = c.numAttentionHeads
        self.numKVHeads = c.numKeyValueHeads
        let headDim = c.hiddenSize / c.numAttentionHeads
        self.scale = pow(Float(headDim), -0.5)
        let bias = c.attentionBias
        self._wq.wrappedValue = Linear(c.hiddenSize, numHeads * headDim, bias: bias)
        self._wk.wrappedValue = Linear(c.hiddenSize, numKVHeads * headDim, bias: bias)
        self._wv.wrappedValue = Linear(c.hiddenSize, numKVHeads * headDim, bias: bias)
        self._wo.wrappedValue = Linear(numHeads * headDim, c.hiddenSize, bias: bias)
        self.rope = RoPE(dimensions: headDim, traditional: false, base: c.ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = wq(x).reshaped(b, l, numHeads, -1).transposed(0, 2, 1, 3)
        var k = wk(x).reshaped(b, l, numKVHeads, -1).transposed(0, 2, 1, 3)
        let v = wv(x).reshaped(b, l, numKVHeads, -1).transposed(0, 2, 1, 3)
        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)
        let out = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(b, l, -1)
        return wo(out)
    }
}

/// Base for the two feed-forward variants so a decoder layer can hold either under the `mlp` key.
nonisolated class DSV2FFN: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray { fatalError("abstract") }
}

nonisolated final class DSV2MLP: DSV2FFN {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

/// Router gate: softmax scores + greedy top-k selection, scaled by `routed_scaling_factor`.
nonisolated final class DSV2MoEGate: Module {
    let topK: Int
    let nRouted: Int
    let scaling: Float
    @ParameterInfo(key: "weight") var weight: MLXArray   // (n_routed, hidden)

    init(_ c: DeepSeekOCRConfig) {
        self.topK = c.numExpertsPerTok
        self.nRouted = c.nRoutedExperts
        self.scaling = c.routedScalingFactor
        self._weight.wrappedValue = MLXArray.zeros([c.nRoutedExperts, c.hiddenSize])
        super.init()
    }

    /// Returns `(indices, scores)` each `(B, L, topK)`.
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let gates = matmul(x, weight.transposed())          // (B, L, n_routed)
        let scores = softmax(gates, axis: -1, precise: true)
        let part = argPartition(scores, kth: nRouted - topK, axis: -1)
        let inds = part[.ellipsis, (nRouted - topK)...]     // top-k (unordered) → (B, L, topK)
        let selected = takeAlong(scores, inds, axis: -1) * scaling
        return (inds, selected)
    }
}

nonisolated final class DSV2MoE: DSV2FFN {
    let numExpertsPerTok: Int
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") var gate: DSV2MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DSV2MLP

    init(_ c: DeepSeekOCRConfig) {
        self.numExpertsPerTok = c.numExpertsPerTok
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: c.hiddenSize, hiddenDims: c.moeIntermediateSize, numExperts: c.nRoutedExperts)
        self._gate.wrappedValue = DSV2MoEGate(c)
        self._sharedExperts.wrappedValue = DSV2MLP(
            hiddenSize: c.hiddenSize, intermediateSize: c.moeIntermediateSize * c.nSharedExperts)
        super.init()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = switchMLP(x, inds)                       // (B, L, topK, hidden)
        y = (y * scores.expandedDimensions(axis: -1)).sum(axis: -2)   // (B, L, hidden)
        return y + sharedExperts(x)
    }
}

nonisolated final class DSV2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DSV2Attention
    @ModuleInfo(key: "mlp") var mlp: DSV2FFN
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ c: DeepSeekOCRConfig, layerIndex: Int) {
        self._selfAttn.wrappedValue = DSV2Attention(c)
        // Layer < first_k_dense_replace is dense; the rest are MoE.
        self._mlp.wrappedValue = layerIndex >= c.firstKDenseReplace
            ? DSV2MoE(c)
            : DSV2MLP(hiddenSize: c.hiddenSize, intermediateSize: c.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

nonisolated final class DSV2Model: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [DSV2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ c: DeepSeekOCRConfig) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.hiddenSize)
        self._layers.wrappedValue = (0 ..< c.numHiddenLayers).map { DSV2DecoderLayer(c, layerIndex: $0) }
        self._norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray?, cache: [KVCache]?, inputEmbedding: MLXArray?) -> MLXArray {
        var h = inputEmbedding ?? embedTokens(inputs!)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

/// `language_model`: the DeepSeek-V2 model + its `lm_head` (a sibling, per the checkpoint's remap
/// `lm_head.` → `language_model.lm_head.`).
nonisolated final class DeepSeekOCRLanguage: Module, KVCacheDimensionProvider {
    @ModuleInfo(key: "model") var model: DSV2Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    let kvHeads: [Int]

    init(_ c: DeepSeekOCRConfig) {
        self._model.wrappedValue = DSV2Model(c)
        self._lmHead.wrappedValue = Linear(c.hiddenSize, c.vocabSize, bias: false)
        self.kvHeads = (0 ..< c.numHiddenLayers).map { _ in c.numKeyValueHeads }
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil) -> MLXArray {
        lmHead(model(inputs, cache: cache, inputEmbedding: inputEmbedding))
    }

    /// Join per-expert weights `mlp.experts.{e}.{proj}` → the stacked `mlp.switch_mlp.{proj}` that
    /// `SwitchGLU` expects (mirrors `language.py`'s `LanguageModel.sanitize`). Operates on the full
    /// (already key-remapped) weight dict.
    nonisolated static func sanitizeExperts(
        _ weights: [String: MLXArray], numLayers: Int, numExperts: Int
    ) -> [String: MLXArray] {
        var w = weights
        for l in 0 ..< numLayers {
            let prefix = "language_model.model.layers.\(l)"
            for proj in ["gate_proj", "down_proj", "up_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    guard w["\(prefix).mlp.experts.0.\(proj).\(suffix)"] != nil else { continue }
                    var toJoin: [MLXArray] = []
                    for e in 0 ..< numExperts {
                        if let arr = w.removeValue(forKey: "\(prefix).mlp.experts.\(e).\(proj).\(suffix)") {
                            toJoin.append(arr)
                        }
                    }
                    w["\(prefix).mlp.switch_mlp.\(proj).\(suffix)"] = stacked(toJoin)
                }
            }
        }
        return w
    }
}
