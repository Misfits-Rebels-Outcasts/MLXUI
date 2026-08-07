import Foundation
import MLX
import MLXNN
import MLXFast

/// The FLUX CLIP text encoder, ported from mflux (`flux_text_encoder/clip_encoder/`) in the
/// checkpoint's key/quant format (AM4f). All Linear/Embedding 4-bit affine-quantized (group 64);
/// the LayerNorms are affine BF16 **with bias**.
///
/// Config: 768-dim, 12 layers, 12 heads, head dim 64, vocab 49408, 77-token window,
/// quick-GELU MLP (3072 intermediate).
nonisolated final class FluxCLIPEncoder: Module {

    static let dModel = 768
    static let dFF = 3072
    static let heads = 12
    static let headDim = 64
    static let vocab = 49408
    static let maxLength = 77
    static let layers = 12

    @ModuleInfo(key: "text_model") var textModel: FluxCLIPTextModel

    override init() {
        self._textModel.wrappedValue = FluxCLIPTextModel()
        super.init()
    }

    /// The causal attention mask mflux builds: upper-triangular -3.4e38, shape `(B, 1, L, L)`.
    static func causalMask(batch: Int, seqLen: Int) -> MLXArray {
        let tri = tril(MLXArray.ones([seqLen, seqLen]), k: 0)
        let mask = (1 - tri) * -3.4e38
        return broadcast(mask.reshaped([1, 1, seqLen, seqLen]), to: [batch, 1, seqLen, seqLen])
    }

    /// Returns the **pooled** output (last hidden state at the EOS/argmax token).
    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        textModel(tokens)
    }
}

/// `CLIPTextModel`: embeddings → encoder layers → final LayerNorm → pooled at EOS.
/// `@ModuleInfo` keys are single segments to ensure `update(parameters:)` descends correctly.
nonisolated final class FluxCLIPTextModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: FluxCLIPEmbeddings
    @ModuleInfo(key: "encoder") var encoder: FluxCLIPEncoderCore
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    override init() {
        let c = FluxCLIPEncoder.self
        self._embeddings.wrappedValue = FluxCLIPEmbeddings()
        self._encoder.wrappedValue = FluxCLIPEncoderCore()
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: c.dModel, eps: 1e-5, affine: true, bias: true)
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let seqLen = tokens.dim(1)
        let positionIds = MLXArray.arange(seqLen).reshaped([1, seqLen])
        var hidden = embeddings(tokens, positionIds: positionIds)
        let mask = FluxCLIPEncoder.causalMask(batch: hidden.dim(0), seqLen: seqLen).asType(hidden.dtype)
        hidden = encoder(hidden, mask: mask)
        hidden = finalLayerNorm(hidden)
        let eosIndex = argMax(tokens, axis: -1)[0]
        return hidden[0, eosIndex]
    }
}

nonisolated final class FluxCLIPEmbeddings: Module {
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: QuantizedEmbedding
    @ModuleInfo(key: "position_embedding") var positionEmbedding: QuantizedEmbedding

    override init() {
        let c = FluxCLIPEncoder.self
        self._tokenEmbedding.wrappedValue = QuantizedEmbedding(
            embeddingCount: c.vocab, dimensions: c.dModel, groupSize: 64, bits: 4, mode: .affine)
        self._positionEmbedding.wrappedValue = QuantizedEmbedding(
            embeddingCount: c.maxLength, dimensions: c.dModel, groupSize: 64, bits: 4, mode: .affine)
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray, positionIds: MLXArray) -> MLXArray {
        tokenEmbedding(tokens) + positionEmbedding(positionIds)
    }
}

nonisolated final class FluxCLIPEncoderCore: Module {
    @ModuleInfo(key: "layers") var layers: [FluxCLIPEncoderLayer]

    override init() {
        self._layers.wrappedValue = (0 ..< FluxCLIPEncoder.layers).map { _ in FluxCLIPEncoderLayer() }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var hidden = x
        for layer in layers { hidden = layer(hidden, mask: mask) }
        return hidden
    }
}

nonisolated final class FluxCLIPEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: FluxCLIPSelfAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: FluxCLIPMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    override init() {
        let c = FluxCLIPEncoder.self
        self._selfAttn.wrappedValue = FluxCLIPSelfAttention()
        self._layerNorm1.wrappedValue = LayerNorm(dimensions: c.dModel, eps: 1e-5, affine: true, bias: true)
        self._mlp.wrappedValue = FluxCLIPMLP()
        self._layerNorm2.wrappedValue = LayerNorm(dimensions: c.dModel, eps: 1e-5, affine: true, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var hidden = x + selfAttn(layerNorm1(x), mask: mask)
        hidden = hidden + mlp(layerNorm2(hidden))
        return hidden
    }
}

nonisolated final class FluxCLIPSelfAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: QuantizedLinear
    @ModuleInfo(key: "k_proj") var kProj: QuantizedLinear
    @ModuleInfo(key: "v_proj") var vProj: QuantizedLinear
    @ModuleInfo(key: "out_proj") var outProj: QuantizedLinear

    override init() {
        let c = FluxCLIPEncoder.self
        func ql() -> QuantizedLinear {
            QuantizedLinear(c.dModel, c.dModel, bias: true, groupSize: 64, bits: 4, mode: .affine)
        }
        self._qProj.wrappedValue = ql()
        self._kProj.wrappedValue = ql()
        self._vProj.wrappedValue = ql()
        self._outProj.wrappedValue = ql()
        super.init()
    }

    private static func reshapeAndTranspose(_ x: MLXArray) -> MLXArray {
        x.reshaped([1, -1, FluxCLIPEncoder.heads, FluxCLIPEncoder.headDim]).transposed(0, 2, 1, 3)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let query = Self.reshapeAndTranspose(qProj(x))
        let key = Self.reshapeAndTranspose(kProj(x))
        let value = Self.reshapeAndTranspose(vProj(x))
        let scale = 1.0 / sqrt(Float(query.dim(3)))
        let attn = scaledDotProductAttention(queries: query, keys: key, values: value, scale: scale, mask: mask)
        let out = attn.transposed(0, 2, 1, 3).reshaped([1, -1, FluxCLIPEncoder.heads * FluxCLIPEncoder.headDim])
        return outProj(out)
    }
}

nonisolated final class FluxCLIPMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: QuantizedLinear
    @ModuleInfo(key: "fc2") var fc2: QuantizedLinear

    override init() {
        let c = FluxCLIPEncoder.self
        self._fc1.wrappedValue = QuantizedLinear(c.dModel, c.dFF, bias: true, groupSize: 64, bits: 4, mode: .affine)
        self._fc2.wrappedValue = QuantizedLinear(c.dFF, c.dModel, bias: true, groupSize: 64, bits: 4, mode: .affine)
        super.init()
    }

    /// `quick_gelu`: x · σ(1.702·x)
    static func quickGELU(_ x: MLXArray) -> MLXArray {
        x * sigmoid(1.702 * x)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(Self.quickGELU(fc1(x)))
    }
}
