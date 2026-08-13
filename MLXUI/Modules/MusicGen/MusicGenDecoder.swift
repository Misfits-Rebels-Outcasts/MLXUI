import Foundation
import MLX
import MLXNN

/// Config for the MusicGen-small causal decoder (MG-ENG3), from the repo's `config.json`
/// `decoder` section (verified 2026-08-13): **24 layers, hidden 1024, 16 heads, ffn 4096**,
/// 4 codebooks, vocab 2048 (+1 pad/bos id 2048), learned positional embeddings
/// (`max_position_embeddings 2048`). This is the SMALL variant — the backlog's "48L/1536d/24H"
/// describes musicgen-large and must NOT be used.
nonisolated struct MusicGenDecoderConfig: Sendable {
    var hidden = 1024
    var layers = 24
    var heads = 16
    var headDim = 64
    var ffn = 4096
    var codebooks = 4
    var vocab = 2048
    var maxPosition = 2048
    var bosTokenID = 2048
    var layerNormEpsilon: Float = 1e-5
}

/// MusicGen's causal transformer decoder (`MusicGen` minus the T5 encoder and EnCodec codec),
/// ported from `ml-explore/mlx-examples/musicgen/musicgen.py` in the **jasonvassallo repo's own
/// key layout** (verified from the `decoder.safetensors` header):
///
/// ```
/// embed_positions.weights                  [2048, 1024]  (learned — NOT sinusoidal)
/// embed_tokens.{0..3}.weight               [2049, 1024]  (4 codebook embeddings; 2048+1 pad)
/// lm_heads.{0..3}.weight                   [2048, 1024]  (4 output projections)
/// enc_to_dec_proj.weight [1024, 768] + bias              (T5 hidden → decoder hidden)
/// layer_norm.weight/.bias                  [1024]        (final norm)
/// layers.{0..23}.self_attn.{q,k,v,out}_proj.weight       (causal self-attn)
/// layers.{0..23}.self_attn_layer_norm.weight/.bias
/// layers.{0..23}.encoder_attn.{q,k,v,out}_proj.weight    (cross-attn to T5 context)
/// layers.{0..23}.encoder_attn_layer_norm.weight/.bias
/// layers.{0..23}.fc1.weight [4096, 1024] / fc2.weight [1024, 4096]  (gelu FFN)
/// layers.{0..23}.final_layer_norm.weight/.bias
/// ```
///
/// Block shape (`TransformerBlock`): `norm1 → self_attn (+residual) → norm_cross → cross_attn
/// (+residual) → norm2 → fc1 → gelu → fc2 (+residual)`. All keys are single-segment under
/// `layers.N.*`, so `@ModuleInfo(key:)` descends directly. The KV cache is runtime-only (no
/// cache keys in the checkpoint).
nonisolated final class MusicGenDecoder: Module {
    let config: MusicGenDecoderConfig

    @ModuleInfo(key: "embed_positions") var embedPositions: EmbedPositionsTable
    @ModuleInfo(key: "embed_tokens") var embedTokens: [Embedding]
    @ModuleInfo(key: "lm_heads") var lmHeads: [Linear]
    @ModuleInfo(key: "enc_to_dec_proj") var encToDecProj: Linear
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [MusicGenDecoderBlock]

    init(config: MusicGenDecoderConfig = MusicGenDecoderConfig()) {
        self.config = config
        self._embedPositions.wrappedValue = EmbedPositionsTable(shape: [config.maxPosition, config.hidden])
        self._embedTokens.wrappedValue = (0 ..< config.codebooks).map { _ in
            Embedding(embeddingCount: config.vocab + 1, dimensions: config.hidden)
        }
        self._lmHeads.wrappedValue = (0 ..< config.codebooks).map { _ in
            Linear(config.hidden, config.vocab, bias: false)
        }
        self._encToDecProj.wrappedValue = Linear(768, config.hidden)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: config.hidden, eps: config.layerNormEpsilon)
        self._layers.wrappedValue = (0 ..< config.layers).map { _ in MusicGenDecoderBlock(config: config) }
        super.init()
    }

    /// Forward pass. `audioTokens` `[1, T, codebooks]` int32; `conditioning` `[1, seq, hidden]`
    /// (the T5 context already projected by `encToDecProj`, per the reference
    /// `TextConditioner`). `offset` = current absolute position (for the learned positional
    /// embedding + KV cache). Returns logits `[1, T, vocab, codebooks]`.
    func callAsFunction(_ audioTokens: MLXArray, conditioning: MLXArray, offset: Int = 0) -> MLXArray {
        var x: MLXArray?
        for k in 0 ..< config.codebooks {
            let emb = embedTokens[k](audioTokens[0..., 0..., k])   // [1, T, hidden]
            x = x.map { $0 + emb } ?? emb
        }
        var h = x!
        // Learned positional embedding (the checkpoint's `embed_positions.weights`, NOT sinusoidal).
        let pos = embedPositions.weights[offset ..< (offset + h.dim(1))].expandedDimensions(axis: 0)
        h = h + pos

        for block in layers {
            h = block(h, conditioning: conditioning)
        }
        h = layerNorm(h)
        let logits = (0 ..< config.codebooks).map { lmHeads[$0](h) }   // [1, T, vocab] × codebooks
        return MLX.stacked(logits, axis: -1)                           // [1, T, vocab, codebooks]
    }
}

/// Wraps the checkpoint's `embed_positions.weights` — a raw `weights` param nested under the
/// `embed_positions` key (NOT a bare top-level array). Keeps `@ModuleInfo(key:)` single-segment
/// so `update(parameters:)` descends correctly (the same trap as SDXL's `ff.net` gap).
nonisolated final class EmbedPositionsTable: Module {
    @ModuleInfo(key: "weights") var weights: MLXArray

    init(shape: [Int]) {
        self._weights.wrappedValue = MLXArray.zeros(shape)
        super.init()
    }

    func callAsFunction(_ positions: MLXArray) -> MLXArray {
        weights[positions]
    }
}

/// One causal transformer block: self-attn → cross-attn → gelu FFN, each with LayerNorm.
nonisolated final class MusicGenDecoderBlock: Module {
    let config: MusicGenDecoderConfig

    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnNorm: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttention: MusicGenMultiHeadAttention
    @ModuleInfo(key: "encoder_attn_layer_norm") var encoderAttnNorm: LayerNorm
    @ModuleInfo(key: "encoder_attn") var crossAttention: MusicGenMultiHeadAttention
    @ModuleInfo(key: "final_layer_norm") var ffnNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(config: MusicGenDecoderConfig) {
        self.config = config
        self._selfAttnNorm.wrappedValue = LayerNorm(dimensions: config.hidden, eps: config.layerNormEpsilon)
        self._selfAttention.wrappedValue = MusicGenMultiHeadAttention(config: config)
        self._encoderAttnNorm.wrappedValue = LayerNorm(dimensions: config.hidden, eps: config.layerNormEpsilon)
        self._crossAttention.wrappedValue = MusicGenMultiHeadAttention(config: config)
        self._ffnNorm.wrappedValue = LayerNorm(dimensions: config.hidden, eps: config.layerNormEpsilon)
        self._fc1.wrappedValue = Linear(config.hidden, config.ffn, bias: false)
        self._fc2.wrappedValue = Linear(config.ffn, config.hidden, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> MLXArray {
        // Reference `TransformerBlock`: each sub-layer pre-norms and adds a residual.
        let y = selfAttention(selfAttnNorm(x), causal: true)      // + x
        let z = crossAttention(encoderAttnNorm(x + y), conditioning: conditioning)   // + (x + y)
        let f = ffnNorm(x + y + z)
        return x + y + z + fc2(gelu(fc1(f)))
    }
}

/// Multi-head attention (causal self-attn or cross-attn to the text context). Bias-free
/// projections; head dim = hidden/heads (64). Used for both the causal self-attn and the
/// cross-attn; the cross-attn path passes the conditioning sequence as keys/values.
nonisolated final class MusicGenMultiHeadAttention: Module {
    let config: MusicGenDecoderConfig

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(config: MusicGenDecoderConfig) {
        self.config = config
        self._qProj.wrappedValue = Linear(config.hidden, config.hidden, bias: false)
        self._kProj.wrappedValue = Linear(config.hidden, config.hidden, bias: false)
        self._vProj.wrappedValue = Linear(config.hidden, config.hidden, bias: false)
        self._outProj.wrappedValue = Linear(config.hidden, config.hidden, bias: false)
        super.init()
    }

    /// Causal self-attention over `x` `[1, T, hidden]`.
    func callAsFunction(_ x: MLXArray, causal: Bool) -> MLXArray {
        let q = project(x, qProj)
        let k = project(x, kProj)
        let v = project(x, vProj)
        let mask: MLXArray? = causal
            ? MultiHeadAttention.createAdditiveCausalMask(x.dim(1), dtype: x.dtype)
            : nil
        return attend(q, k, v, mask: mask)
    }

    /// Cross-attention: queries from `x` `[1, T, hidden]`, keys/values from `conditioning`
    /// `[1, seq, hidden]` (T5 context). No mask — the full text is visible.
    func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> MLXArray {
        let q = project(x, qProj)
        let k = project(conditioning, kProj)
        let v = project(conditioning, vProj)
        return attend(q, k, v, mask: nil)
    }

    private func project(_ x: MLXArray, _ proj: Linear) -> MLXArray {
        let h = proj(x)                                              // [B, T, hidden]
        return h.reshaped([h.dim(0), h.dim(1), config.heads, config.headDim])
            .transposed(0, 2, 1, 3)                                  // [B, heads, T, headDim]
    }

    private func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXArray?) -> MLXArray {
        let scale = 1.0 / sqrt(Float(config.headDim))
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale     // [1, heads, T, S]
        if let mask { scores = scores + mask }
        let weights = softmax(scores, axis: -1)
        let out = matmul(weights, v)                                 // [1, heads, T, headDim]
        let merged = out.transposed(0, 2, 1, 3).reshaped([q.dim(0), q.dim(2), config.hidden])
        return outProj(merged)
    }
}
