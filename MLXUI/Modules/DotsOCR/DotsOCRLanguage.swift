import Foundation
import MLX
import MLXNN
import MLXFast
import MLXLMCommon

/// The Qwen2 text decoder for dots.ocr — a hand-rolled port matching MLXLLM's `Qwen2.swift`
/// (which is package-internal and can't be reused) plus dots' `TextConfig`. Adds an
/// `inputEmbedding` entry point so the VLM can inject merged image+text embeddings, mirroring
/// mlx-swift-lm's own VLM language stacks. Module tree matches the (key-remapped) checkpoint:
/// `language_model.model.{embed_tokens,layers,norm,lm_head}` — see `DotsOCRWeights.remapKey`.
///
/// `nonisolated` throughout (app target defaults to MainActor; MLX `Module` is nonisolated).

nonisolated final class DotsQwen2Attention: Module {
    let heads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    init(_ c: DotsOCRConfig) {
        self.heads = c.numAttentionHeads
        self.kvHeads = c.numKeyValueHeads
        let headDim = c.headDim
        self.scale = pow(Float(headDim), -0.5)
        // Qwen2: q/k/v carry bias, o_proj does not.
        self._wq.wrappedValue = Linear(c.hiddenSize, heads * headDim, bias: c.attentionBias)
        self._wk.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: c.attentionBias)
        self._wv.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: c.attentionBias)
        self._wo.wrappedValue = Linear(heads * headDim, c.hiddenSize, bias: false)
        self.rope = RoPE(dimensions: headDim, traditional: false, base: c.ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = wq(x).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var k = wk(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        let v = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)

        let out = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return wo(out)
    }
}

nonisolated final class DotsQwen2MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ c: DotsOCRConfig) {
        self._gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        self._up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        self._down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

nonisolated final class DotsQwen2Block: Module {
    @ModuleInfo(key: "self_attn") var attention: DotsQwen2Attention
    @ModuleInfo(key: "mlp") var mlp: DotsQwen2MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ c: DotsOCRConfig) {
        self._attention.wrappedValue = DotsQwen2Attention(c)
        self._mlp.wrappedValue = DotsQwen2MLP(c)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// Inner Qwen2 (`language_model.model.*`): embeddings, decoder layers, final norm, and — matching
/// dots' checkpoint layout — the `lm_head` (dots maps `lm_head.` → `language_model.model.lm_head.`).
nonisolated final class DotsQwen2Inner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [DotsQwen2Block]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ c: DotsOCRConfig) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.hiddenSize)
        self._layers.wrappedValue = (0 ..< c.numHiddenLayers).map { _ in DotsQwen2Block(c) }
        self._norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        self._lmHead.wrappedValue = Linear(c.hiddenSize, c.vocabSize, bias: false)
        super.init()
    }

    /// Run the decoder stack. Uses `inputEmbedding` when provided (VLM merge), else embeds `inputs`.
    func callAsFunction(_ inputs: MLXArray?, cache: [KVCache]?, inputEmbedding: MLXArray?) -> MLXArray {
        var h = inputEmbedding ?? embedTokens(inputs!)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return lmHead(norm(h))
    }
}

/// `language_model` wrapper: owns the inner Qwen2 model and exposes the `LMOutput` interface the
/// VLM step loop expects.
nonisolated final class DotsLanguageModel: Module, KVCacheDimensionProvider {
    @ModuleInfo(key: "model") var model: DotsQwen2Inner
    let kvHeads: [Int]

    init(_ c: DotsOCRConfig) {
        self._model.wrappedValue = DotsQwen2Inner(c)
        self.kvHeads = (0 ..< c.numHiddenLayers).map { _ in c.numKeyValueHeads }
        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
    ) -> LMOutput {
        LMOutput(logits: model(inputs, cache: cache, inputEmbedding: inputEmbedding))
    }
}
