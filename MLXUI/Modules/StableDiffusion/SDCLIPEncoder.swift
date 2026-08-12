import Foundation
import MLX
import MLXNN
import MLXFast

/// The SDXL CLIP text-encoder architecture, ported from `ml-explore/mlx-examples/stable_diffusion`
/// (`clip.py` → `CLIPTextModel`). SDXL's **two** text encoders (CLIP-L in `text_encoder/`,
/// OpenCLIP ViT-G/14 in `text_encoder_2/`) share this exact structure — only the config and the
/// MLP activation differ — so the layers live here once and each encoder is a thin wrapper.
///
/// Checkpoint facts (from the `stabilityai/sdxl-turbo` safetensors headers, SD-ENG1):
/// - plain **fp16** weights → `Linear`/`Embedding`, not the 4-bit quantized variants FLUX uses;
/// - key prefix `text_model.` with single-segment `@ModuleInfo` keys so `update(parameters:)`
///   descends correctly (`text_model.embeddings.{token,position}_embedding`,
///   `text_model.encoder.layers.{i}.{layer_norm1,self_attn,mlp,layer_norm2}`, `text_model.final_layer_norm`);
/// - both encoders carry `LayerNorm` with bias (affine true).
///
/// **SDXL difference vs FLUX:** the UNet cross-attends to the **full hidden sequence**
/// (`last_hidden_state`, the final `final_layer_norm` output), NOT the pooled EOS vector. So
/// `callAsFunction` returns `[1, 77, dim]`, not `[dim]`. The `text_encoder_2` checkpoint's
/// `text_projection` key is left undeclared (never applied) — `update(parameters:verify:.none)`
/// ignores it.

/// A CLIP text-encoder config for SDXL. The two encoders differ only in dims/layers/activation.
nonisolated struct SDCLIPConfig: Sendable {
    let dModel: Int
    let dFF: Int
    let heads: Int
    let headDim: Int
    let vocab: Int
    let maxLength: Int
    let layers: Int
    /// MLP activation: CLIP-L uses quick-GELU, OpenCLIP-G uses plain GELU.
    enum Activation {
        case quickGELU, gelu
        func apply(_ x: MLXArray) -> MLXArray {
            switch self {
            case .quickGELU:
                return x * sigmoid(1.702 * x)
            case .gelu:
                return MLXNN.gelu(x)
            }
        }
    }
    let activation: Activation

    /// CLIP-L — `text_encoder/` (768-dim, 12 layers, 12 heads, 3072 FFN, quick-GELU).
    static let clipL = SDCLIPConfig(
        dModel: 768, dFF: 3072, heads: 12, headDim: 64,
        vocab: 49408, maxLength: 77, layers: 12, activation: .quickGELU)
}

/// CLIP-L text encoder → `[1, 77, 768]` (full hidden sequence, no pooling).
nonisolated final class SDCLIPEncoder: Module {
    @ModuleInfo(key: "text_model") var textModel: SDCLIPCoreTextModel

    override init() {
        self._textModel.wrappedValue = SDCLIPCoreTextModel(config: SDCLIPConfig.clipL)
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        textModel(tokens)
    }
}

/// The causal attention mask: upper-triangular `-3.4e38`, shape `(B, 1, L, L)` (cast to fp16
/// this saturates to -inf, which the attention treats as "ignore").
nonisolated extension SDCLIPCoreTextModel {
    static func causalMask(batch: Int, seqLen: Int) -> MLXArray {
        let tri = tril(MLXArray.ones([seqLen, seqLen]), k: 0)
        let mask = (1 - tri) * -3.4e38
        return broadcast(mask.reshaped([1, 1, seqLen, seqLen]), to: [batch, 1, seqLen, seqLen])
    }
}

/// `CLIPTextModel`: embeddings → encoder layers → final LayerNorm. Returns the **full
/// sequence** `[1, seqLen, dModel]` (SDXL cross-attention conditioning), not a pooled vector.
nonisolated final class SDCLIPCoreTextModel: Module {
    let config: SDCLIPConfig

    @ModuleInfo(key: "embeddings") var embeddings: SDCLIPEmbeddings
    @ModuleInfo(key: "encoder") var encoder: SDCLIPEncoderCore
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(config: SDCLIPConfig) {
        self.config = config
        self._embeddings.wrappedValue = SDCLIPEmbeddings(config: config)
        self._encoder.wrappedValue = SDCLIPEncoderCore(config: config)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5, affine: true, bias: true)
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let seqLen = tokens.dim(1)
        let positionIds = MLXArray.arange(seqLen).reshaped([1, seqLen])
        var hidden = embeddings(tokens, positionIds: positionIds)
        let mask = Self.causalMask(batch: hidden.dim(0), seqLen: seqLen).asType(hidden.dtype)
        hidden = encoder(hidden, mask: mask)
        hidden = finalLayerNorm(hidden)
        return hidden
    }
}

nonisolated final class SDCLIPEmbeddings: Module {
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    init(config: SDCLIPConfig) {
        self._tokenEmbedding.wrappedValue = Embedding(embeddingCount: config.vocab, dimensions: config.dModel)
        self._positionEmbedding.wrappedValue = Embedding(embeddingCount: config.maxLength, dimensions: config.dModel)
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray, positionIds: MLXArray) -> MLXArray {
        tokenEmbedding(tokens) + positionEmbedding(positionIds)
    }
}

nonisolated final class SDCLIPEncoderCore: Module {
    let config: SDCLIPConfig
    @ModuleInfo(key: "layers") var layers: [SDCLIPEncoderLayer]

    init(config: SDCLIPConfig) {
        self.config = config
        self._layers.wrappedValue = (0 ..< config.layers).map { _ in SDCLIPEncoderLayer(config: config) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var hidden = x
        for layer in layers { hidden = layer(hidden, mask: mask) }
        return hidden
    }
}

nonisolated final class SDCLIPEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SDCLIPSelfAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SDCLIPMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    init(config: SDCLIPConfig) {
        self._selfAttn.wrappedValue = SDCLIPSelfAttention(config: config)
        self._layerNorm1.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5, affine: true, bias: true)
        self._mlp.wrappedValue = SDCLIPMLP(config: config)
        self._layerNorm2.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5, affine: true, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var hidden = x + selfAttn(layerNorm1(x), mask: mask)
        hidden = hidden + mlp(layerNorm2(hidden))
        return hidden
    }
}

nonisolated final class SDCLIPSelfAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let config: SDCLIPConfig

    init(config: SDCLIPConfig) {
        self.config = config
        func lin() -> Linear { Linear(config.dModel, config.dModel, bias: true) }
        self._qProj.wrappedValue = lin()
        self._kProj.wrappedValue = lin()
        self._vProj.wrappedValue = lin()
        self._outProj.wrappedValue = lin()
        super.init()
    }

    private func reshapeAndTranspose(_ x: MLXArray) -> MLXArray {
        x.reshaped([1, -1, config.heads, config.headDim]).transposed(0, 2, 1, 3)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let query = reshapeAndTranspose(qProj(x))
        let key = reshapeAndTranspose(kProj(x))
        let value = reshapeAndTranspose(vProj(x))
        let scale = 1.0 / sqrt(Float(query.dim(3)))
        let attn = scaledDotProductAttention(queries: query, keys: key, values: value, scale: scale, mask: mask)
        let out = attn.transposed(0, 2, 1, 3).reshaped([1, -1, config.heads * config.headDim])
        return outProj(out)
    }
}

nonisolated final class SDCLIPMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    let config: SDCLIPConfig

    init(config: SDCLIPConfig) {
        self.config = config
        self._fc1.wrappedValue = Linear(config.dModel, config.dFF, bias: true)
        self._fc2.wrappedValue = Linear(config.dFF, config.dModel, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(config.activation.apply(fc1(x)))
    }
}
