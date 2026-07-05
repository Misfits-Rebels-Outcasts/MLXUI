import Foundation
import MLX
import MLXFast
import MLXNN
import Tokenizers

// SUP-4 — a self-contained MLX ModernBERT encoder + embedding runner for
// `nomicai-modernbert-embed-base` (model_type `modernbert`), which `MLXEmbedders` doesn't
// support. We deliberately DON'T go through `MLXEmbedders` (its `EmbeddingModel` must return
// `EmbeddingModelOutput`, whose init is internal → not constructible from this module). Instead
// we load + run the model standalone, exactly like the Kokoro/Whisper/Voxtral engines, and pool
// ourselves. See journal/2026-41.
//
// ModernBERT: RoPE (per-layer global/local theta) instead of learned positions; pre-norm blocks
// with a GeGLU MLP; no biases on norms/linears; layer 0 skips its attention norm.
//
// ⚠️ Numeric correctness is unverified in CI (needs an on-device parity smoke — checklist row 11).
// v1 uses full attention (no sliding-window mask); identical to ModernBERT for inputs ≤ the local
// window (128 tokens), which covers the short texts the embedding UI sends.

nonisolated struct ModernBERTConfiguration: Decodable, Sendable {
    var hiddenSize = 768
    var numHiddenLayers = 22
    var numAttentionHeads = 12
    var intermediateSize = 1152
    var vocabularySize = 50368
    var normEps: Float = 1e-5
    var globalRopeTheta: Float = 160_000
    var localRopeTheta: Float = 10_000
    var globalAttnEveryNLayers = 3

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case vocabularySize = "vocab_size"
        case normEps = "norm_eps"
        case globalRopeTheta = "global_rope_theta"
        case localRopeTheta = "local_rope_theta"
        case globalAttnEveryNLayers = "global_attn_every_n_layers"
    }

    init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = (try? c.decodeIfPresent(Int.self, forKey: .hiddenSize)) ?? hiddenSize
        numHiddenLayers = (try? c.decodeIfPresent(Int.self, forKey: .numHiddenLayers)) ?? numHiddenLayers
        numAttentionHeads = (try? c.decodeIfPresent(Int.self, forKey: .numAttentionHeads)) ?? numAttentionHeads
        intermediateSize = (try? c.decodeIfPresent(Int.self, forKey: .intermediateSize)) ?? intermediateSize
        vocabularySize = (try? c.decodeIfPresent(Int.self, forKey: .vocabularySize)) ?? vocabularySize
        normEps = (try? c.decodeIfPresent(Float.self, forKey: .normEps)) ?? normEps
        globalRopeTheta = (try? c.decodeIfPresent(Float.self, forKey: .globalRopeTheta)) ?? globalRopeTheta
        localRopeTheta = (try? c.decodeIfPresent(Float.self, forKey: .localRopeTheta)) ?? localRopeTheta
        globalAttnEveryNLayers = (try? c.decodeIfPresent(Int.self, forKey: .globalAttnEveryNLayers)) ?? globalAttnEveryNLayers
    }
}

/// LayerNorm with a weight but **no bias** (ModernBERT's `norm_bias: false`). MLXNN's `LayerNorm`
/// always carries a bias when affine, which wouldn't match the checkpoint's params.
private nonisolated final class LayerNormNoBias: Module {
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

private nonisolated final class Attention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float
    @ModuleInfo(key: "Wqkv") var wqkv: Linear
    @ModuleInfo(key: "Wo") var wo: Linear
    let rope: RoPE

    init(_ config: ModernBERTConfiguration, ropeTheta: Float) {
        numHeads = config.numAttentionHeads
        headDim = config.hiddenSize / config.numAttentionHeads
        scale = pow(Float(headDim), -0.5)
        _wqkv.wrappedValue = Linear(config.hiddenSize, 3 * config.hiddenSize, bias: false)
        _wo.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        rope = RoPE(dimensions: headDim, traditional: false, base: ropeTheta)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let qkv = wqkv(x).reshaped(B, L, 3, numHeads, headDim).transposed(2, 0, 3, 1, 4)
        let q = rope(qkv[0])   // [B, H, L, D]
        let k = rope(qkv[1])
        let v = qkv[2]
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        return wo(out.transposed(0, 2, 1, 3).reshaped(B, L, numHeads * headDim))
    }
}

private nonisolated final class MLPBlock: Module {
    @ModuleInfo(key: "Wi") var wi: Linear
    @ModuleInfo(key: "Wo") var wo: Linear

    init(_ config: ModernBERTConfiguration) {
        _wi.wrappedValue = Linear(config.hiddenSize, 2 * config.intermediateSize, bias: false)
        _wo.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = wi(x).split(parts: 2, axis: -1)
        return wo(gelu(parts[0]) * parts[1])
    }
}

private nonisolated final class EncoderLayer: Module {
    @ModuleInfo(key: "attn_norm") var attnNorm: LayerNormNoBias?
    @ModuleInfo(key: "attn") var attn: Attention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: LayerNormNoBias
    @ModuleInfo(key: "mlp") var mlp: MLPBlock

    init(_ config: ModernBERTConfiguration, layerIndex: Int) {
        let isGlobal = layerIndex % config.globalAttnEveryNLayers == 0
        let theta = isGlobal ? config.globalRopeTheta : config.localRopeTheta
        if layerIndex != 0 {
            _attnNorm.wrappedValue = LayerNormNoBias(config.hiddenSize, eps: config.normEps)
        }
        _attn.wrappedValue = Attention(config, ropeTheta: theta)
        _mlpNorm.wrappedValue = LayerNormNoBias(config.hiddenSize, eps: config.normEps)
        _mlp.wrappedValue = MLPBlock(config)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x + attn(attnNorm?(x) ?? x, mask: mask)
        h = h + mlp(mlpNorm(h))
        return h
    }
}

/// The ModernBERT encoder (embeddings + layers + final norm) and a standalone embed entry point.
nonisolated final class ModernBERTEmbedder: Module {
    @ModuleInfo(key: "tok_embeddings") var tokEmbeddings: Embedding
    @ModuleInfo(key: "embeddings_norm") fileprivate var embeddingsNorm: LayerNormNoBias
    fileprivate let layers: [EncoderLayer]
    @ModuleInfo(key: "final_norm") fileprivate var finalNorm: LayerNormNoBias

    init(_ config: ModernBERTConfiguration) {
        _tokEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _embeddingsNorm.wrappedValue = LayerNormNoBias(config.hiddenSize, eps: config.normEps)
        layers = (0 ..< config.numHiddenLayers).map { EncoderLayer(config, layerIndex: $0) }
        _finalNorm.wrappedValue = LayerNormNoBias(config.hiddenSize, eps: config.normEps)
    }

    /// Last-layer hidden states for `inputIds` [B, L]. `attentionMask` [B, L] (1 keep / 0 pad).
    func hiddenStates(inputIds: MLXArray, attentionMask: MLXArray) -> MLXArray {
        var h = embeddingsNorm(tokEmbeddings(inputIds))
        // Additive mask [B,1,1,L]: 0 for real tokens, -inf for padding.
        let mask = attentionMask.asType(h.dtype).expandedDimensions(axes: [1, 2]).log()
        for layer in layers { h = layer(h, mask: mask) }
        return finalNorm(h)
    }

    /// Map HF ModernBERT weight names onto this module's keys, keeping only what we model.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (rawKey, value) in weights {
            var key = rawKey
            if key.hasPrefix("model.") { key = String(key.dropFirst("model.".count)) }
            key = key
                .replacingOccurrences(of: "embeddings.tok_embeddings.", with: "tok_embeddings.")
                .replacingOccurrences(of: "embeddings.norm.", with: "embeddings_norm.")
            // Keep only encoder params; drop MLM head / position_ids / anything else.
            if key.hasPrefix("tok_embeddings.") || key.hasPrefix("embeddings_norm.")
                || key.hasPrefix("layers.") || key.hasPrefix("final_norm.") {
                out[key] = value
            }
        }
        return out
    }

    /// Load config + safetensors from an installed model directory (the A2 pattern).
    static func fromDirectory(_ directory: URL) throws -> ModernBERTEmbedder {
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(ModernBERTConfiguration.self, from: configData)
        let model = ModernBERTEmbedder(config)

        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        var weights: [String: MLXArray] = [:]
        for file in files where file.pathExtension == "safetensors" {
            try MLX.loadArrays(url: file).forEach { weights[$0.key] = $0.value }
        }
        let sanitized = model.sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        eval(model)
        return model
    }

    /// Embed one or more texts → L2-normalized vectors (mean pooling over tokens).
    static func embed(_ texts: [String], modelDirectory: URL) async throws -> [[Float]] {
        do {
            let model = try fromDirectory(modelDirectory)
            let tokenizer = try await AutoTokenizer.from(modelFolder: modelDirectory)

            let encoded = texts.map { tokenizer.encode(text: $0, addSpecialTokens: true) }
            let maxLen = max(1, encoded.map(\.count).max() ?? 1)
            let ids = stacked(encoded.map { row in
                MLXArray((row + Array(repeating: 0, count: maxLen - row.count)).map { Int32($0) })
            })
            let maskRows = encoded.map { row in
                (0 ..< maxLen).map { Float($0 < row.count ? 1 : 0) }
            }
            let mask = stacked(maskRows.map { MLXArray($0) })   // [B, L]

            let hidden = model.hiddenStates(inputIds: ids, attentionMask: mask)  // [B, L, H]
            let m = mask.expandedDimensions(axis: -1)                            // [B, L, 1]
            let summed = (hidden * m).sum(axis: 1)                               // [B, H]
            let counts = MLX.maximum(m.sum(axis: 1), MLXArray(Float(1e-9)))      // [B, 1]
            let meanPooled = summed / counts
            let l2 = MLX.sqrt((meanPooled * meanPooled).sum(axis: -1, keepDims: true))
            let normalized = meanPooled / MLX.maximum(l2, MLXArray(Float(1e-9)))
            normalized.eval()
            return (0 ..< normalized.dim(0)).map { normalized[$0].asArray(Float.self) }
        } catch {
            throw StageError.engineFailure(stage: "ModernBERT Embedding", underlying: error)
        }
    }
}
