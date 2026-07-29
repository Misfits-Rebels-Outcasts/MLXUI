//
//  Qwen35Model.swift
//  MLXUI — Qwen3.5 (qwen3_5) hybrid GatedDeltaNet port, Slice QB3.
//
//  Assembles the text tower: the SwiGLU MLP, the hybrid decoder layer (linear-attention
//  GatedDeltaNet on `(idx+1) % full_attention_interval != 0`, gated full attention
//  otherwise), the layer stack, and the `LLMModel` head. Mirrors `Qwen3_5DecoderLayer`
//  / `Qwen3_5Model` in Blaizzy/mlx-vlm `models/qwen3_5/language.py`.
//
//  CACHE SCOPE (QB3): the graph is wired and conforms to `LLMModel` so `load()` +
//  weight-shaping work, but only the full-attention layers use their KV-cache slot.
//  The linear layers run their **prefill** forward (zero state) and ignore their slot —
//  the real conv+SSM decode cache and the hybrid generate loop are Slice QB4. So this
//  builds and can be instantiated, but is not yet a correct incremental generator.
//

import Foundation
import MLX
import MLXFast
import MLXLLM
import MLXLMCommon
import MLXNN

/// SwiGLU MLP. Mirrors `Qwen3_5MLP`.
nonisolated final class Qwen35MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(dim: Int, hiddenDim: Int) {
        self._gate.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._up.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._down.wrappedValue = Linear(hiddenDim, dim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(Qwen35Math.silu(gate(x)) * up(x))
    }
}

/// One decoder layer — either linear-attention (`linear_attn`) or gated full-attention
/// (`self_attn`), selected by the layer schedule. Mirrors `Qwen3_5DecoderLayer`.
nonisolated final class Qwen35DecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35GatedDeltaNet?
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention?
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Qwen35MLP

    init(_ config: Qwen35TextConfig, layerIndex: Int) {
        self.isLinear = config.isLinearLayer(layerIndex)
        if isLinear {
            self._linearAttn.wrappedValue = Qwen35GatedDeltaNet(config)
        } else {
            self._selfAttn.wrappedValue = Qwen35Attention(config)
        }
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._mlp.wrappedValue = Qwen35MLP(dim: config.hiddenSize, hiddenDim: config.intermediateSize)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let normed = inputLayerNorm(x)
        let r: MLXArray
        if let selfAttn {
            r = selfAttn(normed, cos: cos, sin: sin, mask: mask, cache: cache)
        } else if let linearAttn {
            // QB3: prefill-only (zero state). Incremental conv+SSM decode is QB4.
            r = linearAttn(normed)
        } else {
            r = normed
        }
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// The layer stack: `embed_tokens` → 64 hybrid layers → `norm`. Mirrors `Qwen3_5Model`.
nonisolated final class Qwen35ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen35DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let rope: Qwen35RoPE

    init(_ config: Qwen35TextConfig) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
            Qwen35DecoderLayer(config, layerIndex: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        let rotaryDim = Int((Double(config.headDim) * config.ropeParameters.partialRotaryFactor).rounded(.down))
        self.rope = Qwen35RoPE(
            rotaryDim: rotaryDim,
            base: config.ropeParameters.ropeTheta,
            mropeSection: config.ropeParameters.mropeSection
        )
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)

        // Text position ids: arange(offset, offset+L). Per-axis (3-D) ids arrive with QB7.
        let offset = cache?.first?.offset ?? 0
        let positionIds = Qwen35RoPE.textPositionIds(offset: offset, length: h.dim(1))
        let (cos, sin) = rope.cosSin(positionIds: positionIds, dtype: h.dtype)

        for (i, layer) in layers.enumerated() {
            h = layer(h, cos: cos, sin: sin, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

/// The Qwen3.5 text model + `lm_head`. Conforms to `LLMModel` so mlx-swift-lm's loader
/// and generate loop can drive it. Mirrors `Qwen3_5ForCausalLM` (text path).
nonisolated final class Qwen35Model: Module, LLMModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let kvHeads: [Int]
    let config: Qwen35TextConfig

    fileprivate let model: Qwen35ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: Qwen35TextConfig) {
        self.config = config
        self.vocabularySize = config.vocabSize
        // One cache slot per layer (full-attn layers use it; linear layers ignore theirs
        // until QB4 replaces this with a proper hybrid cache).
        self.kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
        self.model = Qwen35ModelInner(config)
        super.init()
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// Transformer layers LoRA could adapt (satisfies `LoRAModel`); unused for inference.
    var loraLayers: [Module] { model.layers }

    /// Reshape the depthwise conv weight `[convDim, 1, K]` → `[convDim, K]` for every
    /// linear layer, and drop `lm_head` when embeddings are tied. Further key remapping /
    /// quantization handling lands in QB5.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = weights
        for (key, value) in w where key.hasSuffix("conv1d.weight") && value.ndim == 3 {
            w[key] = value.reshaped(value.dim(0), value.dim(2))
        }
        if config.tieWordEmbeddings {
            w["lm_head.weight"] = nil
        }
        return w
    }
}
