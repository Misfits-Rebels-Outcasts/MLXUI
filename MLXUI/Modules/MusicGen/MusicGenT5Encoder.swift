import Foundation
import MLX
import MLXNN
import Tokenizers

/// Config for the MusicGen T5-base text encoder (MG-ENG1). T5-base: d_model 768, 12 layers,
/// 12 heads (d_kv 64), d_ff 3072, vocab 32128, relu FFN, RMSNorm. Matches `t5_config.json`
/// shipped by `jasonvassallo/mlx-musicgen-small`.
nonisolated struct MusicGenT5Config: Sendable {
    var dModel = 768
    var layers = 12
    var heads = 12
    var dKV = 64
    var dFF = 3072
    var vocab = 32128
    var numBuckets = 32
    var maxDistance = 128
    var layerNormEpsilon: Float = 1e-6
}

/// Encoder-only T5-base, in the **jasonvassallo repo's own key layout** (verified from the
/// `t5.safetensors` header). The repo remapped the upstream HF T5 keys to `block`-style names:
///
/// ```
/// shared.weight                                  [32128, 768]  (tied embedding)
/// encoder.embed_tokens.weight                   [32128, 768]  (duplicate of shared)
/// encoder.block.{0..11}.self_attn.{q,k,v,o}.weight  [768, 768]  (bias-free)
/// encoder.block.{0..11}.self_attn.relative_attention_bias.weight  [32, 12]
/// encoder.block.{0..11}.self_attn_norm.weight   [768]  (RMSNorm — weight only)
/// encoder.block.{0..11}.ff.wi.weight            [3072, 768]
/// encoder.block.{0..11}.ff.wo.weight            [768, 3072]
/// encoder.block.{0..11}.ff_norm.weight          [768]  (RMSNorm)
/// encoder.final_layer_norm.weight               [768]  (RMSNorm)
/// ```
///
/// This is NOT the Flux T5 layout (`FluxT5Encoder` uses gated-gelu `wi_0/wi_1` and the
/// `SelfAttention` sub-module) — MusicGen's T5 is a plain **relu** FFN with bias-free
/// `self_attn.{q,k,v,o}` naming. Output: `[1, seq, 768]` hidden states after the final
/// RMSNorm (full sequence, not pooled).
nonisolated final class MusicGenT5Encoder: Module {
    @ModuleInfo(key: "shared") var shared: Embedding
    @ModuleInfo(key: "encoder") var encoder: MusicGenT5Stack

    init(config: MusicGenT5Config = MusicGenT5Config()) {
        self._shared.wrappedValue = Embedding(embeddingCount: config.vocab, dimensions: config.dModel)
        self._encoder.wrappedValue = MusicGenT5Stack(config: config)
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        encoder(shared(tokens))
    }
}

/// The encoder stack: embedding → 12 blocks → final RMSNorm.
nonisolated final class MusicGenT5Stack: Module {
    @ModuleInfo(key: "block") var blocks: [MusicGenT5Block]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: RMSNorm

    init(config: MusicGenT5Config) {
        self._blocks.wrappedValue = (0 ..< config.layers).map { _ in MusicGenT5Block(config: config) }
        self._finalLayerNorm.wrappedValue = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for block in blocks { h = block(h) }
        return finalLayerNorm(h)
    }
}

/// One T5 encoder block: RMSNorm → self-attention (+residual) → RMSNorm → relu FFN (+residual).
nonisolated final class MusicGenT5Block: Module {
    @ModuleInfo(key: "self_attn_norm") var selfAttnNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttention: MusicGenT5SelfAttention
    @ModuleInfo(key: "ff_norm") var ffNorm: RMSNorm
    @ModuleInfo(key: "ff") var ff: MusicGenT5FFN

    init(config: MusicGenT5Config) {
        self._selfAttnNorm.wrappedValue = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        self._selfAttention.wrappedValue = MusicGenT5SelfAttention(config: config)
        self._ffNorm.wrappedValue = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        self._ff.wrappedValue = MusicGenT5FFN(config: config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Pre-norm + residual on both sublayers (reference `TransformerEncoderLayer`:
        // `x = x + attention(ln1(x))`, then `return x + dense(ln2(x))`). Dropping the residuals
        // discards `x` entirely and feeds the raw attention output into the FFN norm.
        let y = x + selfAttention(selfAttnNorm(x))
        return y + ff(ffNorm(y))
    }
}

/// T5 self-attention with a learned relative-position bias (bidirectional, 32 buckets).
/// All projections bias-free, matching the checkpoint.
nonisolated final class MusicGenT5SelfAttention: Module {
    let config: MusicGenT5Config

    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "relative_attention_bias") var relativeAttentionBias: Embedding

    init(config: MusicGenT5Config) {
        self.config = config
        self._q.wrappedValue = Linear(config.dModel, config.dKV * config.heads, bias: false)
        self._k.wrappedValue = Linear(config.dModel, config.dKV * config.heads, bias: false)
        self._v.wrappedValue = Linear(config.dModel, config.dKV * config.heads, bias: false)
        self._o.wrappedValue = Linear(config.dKV * config.heads, config.dModel, bias: false)
        self._relativeAttentionBias.wrappedValue = Embedding(
            embeddingCount: config.numBuckets, dimensions: config.heads)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let q = Self.reshapeHeads(self.q(x), config: config)
        let k = Self.reshapeHeads(self.k(x), config: config)
        let v = Self.reshapeHeads(self.v(x), config: config)

        var scores = matmul(q, k.transposed(0, 1, 3, 2))
        scores = scores + bias(seqLength: x.dim(1))
        let weights = softmax(scores, axis: -1)
        let out = matmul(weights, v)
        let merged = out.transposed(0, 2, 1, 3).reshaped([x.dim(0), x.dim(1), config.dModel])
        return o(merged)
    }

    private static func reshapeHeads(_ x: MLXArray, config: MusicGenT5Config) -> MLXArray {
        x.reshaped([x.dim(0), x.dim(1), config.heads, config.dKV]).transposed(0, 2, 1, 3)
    }

    /// mflux/mlx-examples `RelativePositionBias` (bidirectional). Buckets relative positions
    /// `memory - query` into 32 buckets (16 for negatives' half, log-spaced beyond `max_exact`),
    /// embeds them, returns `[heads, seq, seq]` additive bias.
    private func bias(seqLength: Int) -> MLXArray {
        let contextPosition = MLXArray.arange(seqLength).expandedDimensions(axis: -1)
        let memoryPosition = MLXArray.arange(seqLength).expandedDimensions(axis: 0)
        let relativePosition = memoryPosition - contextPosition
        let buckets = Self.relativePositionBuckets(
            relativePosition, numBuckets: config.numBuckets, maxDistance: config.maxDistance)
        let values = relativeAttentionBias(buckets).transposed(2, 0, 1)
        return values.expandedDimensions(axis: 0)
    }

    /// T5 `_relative_position_bucket` for the bidirectional case (mlx-examples `t5.py`).
    /// Note the reference halves `num_buckets` (32 → 16) up front and uses THAT for both the
    /// log-spaced `scale` and the upper clamp — positives live in 16..31, negatives in 0..15.
    static func relativePositionBuckets(_ relativePosition: MLXArray, numBuckets: Int, maxDistance: Int) -> MLXArray {
        var buckets = MLXArray.zeros(like: relativePosition)
        let half = numBuckets / 2
        buckets = buckets + MLX.where(relativePosition .> 0, MLXArray(Int32(half)), MLXArray(Int32(0))).asType(.int32)
        let absPos = abs(relativePosition)
        let maxExact = half / 2
        let isSmall = absPos .< MLXArray(Int32(maxExact))
        let scale = Double(half - maxExact) / log(Double(maxDistance) / Double(maxExact))
        let large = MLXArray(Int32(maxExact)) + (log(absPos.asType(.float32) / Float(maxExact)) * Float(scale)).asType(.int32)
        let largeClamped = minimum(large, MLXArray(Int32(half - 1)))
        buckets = buckets + MLX.where(isSmall, absPos, largeClamped)
        return buckets
    }
}

/// T5 relu FFN (`dense_act_fn: "relu"`): `wi → relu → wo`, bias-free.
nonisolated final class MusicGenT5FFN: Module {
    @ModuleInfo(key: "wi") var wi: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(config: MusicGenT5Config) {
        self._wi.wrappedValue = Linear(config.dModel, config.dFF, bias: false)
        self._wo.wrappedValue = Linear(config.dFF, config.dModel, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        wo(relu(wi(x)))
    }
}

/// Wraps the fast T5 SentencePiece tokenizer shipped as `tokenizer.json` (the repo has no
/// `spiece.model`). Loaded from the installed model dir via swift-transformers (precedent:
/// `ModernBERTEmbedder`). Truncates to `maxTokens` (T5 default 512) — the encoder is
/// encoder-only, so no BOS/EOS decoration is applied.
nonisolated final class MusicGenTokenizer: @unchecked Sendable {
    private let tokenizer: Tokenizer

    init(url: URL) async throws {
        self.tokenizer = try await AutoTokenizer.from(modelFolder: url)
    }

    /// Tokenize `text` → token id array (Int32), truncated to `maxTokens` (T5 default 512).
    /// An empty prompt yields a single pad token.
    func encode(_ text: String, maxTokens: Int = 512) -> [Int32] {
        let ids = tokenizer.encode(text: text)
        if ids.isEmpty {
            let pad = tokenizer.convertTokenToId("<pad>") ?? 0
            return [Int32(pad)]
        }
        return ids.prefix(maxTokens).map(Int32.init)
    }
}
