import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import Tokenizers

// A self-contained MLX EmbeddingGemma runner for `mlx-community/embeddinggemma-300m-4bit`
// (model_type `gemma3_text`). We deliberately DON'T go through `MLXEmbedders`: its
// `EmbeddingGemma` reassigns its `@ModuleInfo dense` in `sanitize(weights:)` after `init` already
// set it to `[]`, which trips MLX's "please use Model.update(modules:)…" fatalError — the crash the
// user hit. We also fix two correctness issues in that port for the standalone path:
//
//  1. **Bidirectional attention.** embeddinggemma's config sets `use_bidirectional_attention: true`
//     (it's an encoder). The `MLXEmbedders` port applies *causal* masks via `createAttentionMask`,
//     which would corrupt embeddings. Here every layer sees a full bidirectional mask (only padding
//     is masked out). Inputs ≤ `sliding_window` (512) make sliding-window == full attention, which
//     covers the short texts the embedding UI sends — same simplification as ModernBERTEmbedder.
//  2. **Dense head built in `init`** (never reassigned), with dimensions inferred from the actual
//     checkpoint weights, so 4-bit quantization + `update(verify: .all)` line up exactly.
//
// Runs standalone like the ModernBERT/Kokoro/Whisper engines: load → mean-pool → dense → L2-norm.
// Reuses `Gemma.RMSNorm` / `Gemma.clipResidual` from MLXLMCommon (both public).
//
// ⚠️ Numeric correctness is unverified in CI (the checkpoint is Gemma-license gated). Needs an
// on-device parity smoke test. See journal/2026-33-embeddings.

// MARK: - Configuration

nonisolated struct EmbeddingGemmaConfiguration: Decodable, Sendable {
    var hiddenSize = 768
    var hiddenLayers = 24
    var intermediateSize = 1152
    var attentionHeads = 3
    var headDim = 256
    var kvHeads = 1
    var rmsNormEps: Float = 1e-6
    var vocabularySize = 262144
    var ropeTheta: Float = 1_000_000
    var ropeLocalBaseFreq: Float = 10_000
    var queryPreAttnScalar: Float = 256
    var slidingWindow = 512
    var slidingWindowPattern = 6
    var quantGroupSize: Int?
    var quantBits: Int?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case quantization
    }

    enum QuantizationKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }

    init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = (try? c.decodeIfPresent(Int.self, forKey: .hiddenSize)) ?? hiddenSize
        hiddenLayers = (try? c.decodeIfPresent(Int.self, forKey: .hiddenLayers)) ?? hiddenLayers
        intermediateSize = (try? c.decodeIfPresent(Int.self, forKey: .intermediateSize)) ?? intermediateSize
        attentionHeads = (try? c.decodeIfPresent(Int.self, forKey: .attentionHeads)) ?? attentionHeads
        headDim = (try? c.decodeIfPresent(Int.self, forKey: .headDim)) ?? headDim
        kvHeads = (try? c.decodeIfPresent(Int.self, forKey: .kvHeads)) ?? kvHeads
        rmsNormEps = (try? c.decodeIfPresent(Float.self, forKey: .rmsNormEps)) ?? rmsNormEps
        vocabularySize = (try? c.decodeIfPresent(Int.self, forKey: .vocabularySize)) ?? vocabularySize
        ropeTheta = (try? c.decodeIfPresent(Float.self, forKey: .ropeTheta)) ?? ropeTheta
        ropeLocalBaseFreq = (try? c.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq)) ?? ropeLocalBaseFreq
        queryPreAttnScalar = (try? c.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar)) ?? queryPreAttnScalar
        slidingWindow = (try? c.decodeIfPresent(Int.self, forKey: .slidingWindow)) ?? slidingWindow
        slidingWindowPattern = (try? c.decodeIfPresent(Int.self, forKey: .slidingWindowPattern)) ?? slidingWindowPattern
        if let q = try? c.nestedContainer(keyedBy: QuantizationKeys.self, forKey: .quantization) {
            quantGroupSize = try? q.decodeIfPresent(Int.self, forKey: .groupSize)
            quantBits = try? q.decodeIfPresent(Int.self, forKey: .bits)
        }
    }
}

// MARK: - Attention

private nonisolated final class EmbeddingGemmaAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: Gemma.RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma.RMSNorm
    let rope: RoPE

    init(_ config: EmbeddingGemmaConfiguration, layerIdx: Int) {
        let dim = config.hiddenSize
        nHeads = config.attentionHeads
        nKVHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(config.queryPreAttnScalar, -0.5)

        _queryProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        _keyProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        _valueProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        _outputProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)
        _queryNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _keyNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        // Sliding-window layers use the local RoPE base; global layers use theta. Bidirectional
        // masking doesn't change which base a layer uses, so keep the per-layer selection.
        let isSliding = (layerIdx + 1) % config.slidingWindowPattern != 0
        rope = RoPE(
            dimensions: headDim, traditional: false,
            base: isSliding ? config.ropeLocalBaseFreq : config.ropeTheta)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = queryProj(x).reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        var keys = keyProj(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        let values = valueProj(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        queries = rope(queryNorm(queries))
        keys = rope(keyNorm(keys))

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return outputProj(output)
    }
}

// MARK: - MLP

private nonisolated final class EmbeddingGemmaMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Transformer block

private nonisolated final class EmbeddingGemmaBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: EmbeddingGemmaAttention
    @ModuleInfo var mlp: EmbeddingGemmaMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma.RMSNorm

    init(_ config: EmbeddingGemmaConfiguration, layerIdx: Int) {
        _selfAttention.wrappedValue = EmbeddingGemmaAttention(config, layerIdx: layerIdx)
        mlp = EmbeddingGemmaMLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        _inputLayerNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let r = selfAttention(inputLayerNorm(x), mask: mask)
        let h = Gemma.clipResidual(x, postAttentionLayerNorm(r))
        let r2 = mlp(preFeedforwardLayerNorm(h))
        return Gemma.clipResidual(h, postFeedforwardLayerNorm(r2))
    }
}

// MARK: - Backbone

private nonisolated final class EmbeddingGemmaBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo fileprivate var layers: [EmbeddingGemmaBlock]
    @ModuleInfo var norm: Gemma.RMSNorm

    let config: EmbeddingGemmaConfiguration

    init(_ config: EmbeddingGemmaConfiguration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { EmbeddingGemmaBlock(config, layerIdx: $0) }
        _norm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// `inputs` [B, L] token ids, `mask` an additive attention mask [B, 1, 1, L] (0 keep / -inf pad).
    func callAsFunction(_ inputs: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = embedTokens(inputs)
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
        h = h * scale.asType(h.dtype)
        // SDPA requires the mask dtype to promote to the compute dtype; the quantized model runs in
        // float16, and a float32 mask won't promote to it. Match the mask to the hidden dtype.
        let m = mask?.asType(h.dtype)
        for layer in layers { h = layer(h, mask: m) }
        return norm(h)
    }
}

// MARK: - EmbeddingGemma

nonisolated final class EmbeddingGemmaEmbedder: Module {
    @ModuleInfo(key: "model") fileprivate var backbone: EmbeddingGemmaBackbone
    @ModuleInfo fileprivate var dense: [Module]

    /// `denseSpecs` are the (in, out) dimensions of each projection-head Linear, inferred from the
    /// checkpoint in `fromDirectory`. Building them here in `init` (rather than reassigning later)
    /// is what avoids the `@ModuleInfo` reassignment fatalError.
    init(_ config: EmbeddingGemmaConfiguration, denseSpecs: [(inDim: Int, outDim: Int)]) {
        _backbone.wrappedValue = EmbeddingGemmaBackbone(config)
        _dense.wrappedValue = denseSpecs.map { Linear($0.inDim, $0.outDim, bias: false) }
        super.init()
    }

    /// Pooled, projected, L2-normalized sentence embeddings [B, D].
    func pooledEmbeddings(inputs: MLXArray, attentionMask: MLXArray) -> MLXArray {
        // Bidirectional additive mask [B, 1, 1, L]: 0 for real tokens, -inf for padding.
        let additive = attentionMask.asType(.float32).expandedDimensions(axes: [1, 2]).log()
        let hidden = backbone(inputs, mask: additive)   // [B, L, H]

        // Mean pooling over non-padding tokens.
        let keep = attentionMask.expandedDimensions(axis: -1).asType(hidden.dtype)  // [B, L, 1]
        let summed = (hidden * keep).sum(axis: 1)                                   // [B, H]
        let counts = MLX.maximum(keep.sum(axis: 1), MLXArray(Float(1e-9)))          // [B, 1]
        var out = summed / counts

        // Projection head (dense.0 → dense.1), no activation between (sentence-transformers Dense).
        for layer in dense {
            if let linear = layer as? Linear {
                out = linear(out)
            } else if let quantized = layer as? QuantizedLinear {
                out = quantized(out)
            }
        }

        // L2 normalize so cosine similarity == dot product.
        let l2 = MLX.sqrt((out * out).sum(axis: -1, keepDims: true))
        return out.asType(.float32) / MLX.maximum(l2.asType(.float32), MLXArray(Float(1e-9)))
    }

    /// Rename/filter checkpoint keys onto this module's structure.
    static func sanitize(
        weights: [String: MLXArray], vocabularySize: Int
    ) -> [String: MLXArray] {
        var processed = weights

        // Support a `language_model.` prefix (weights lifted from a VLM checkpoint).
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processed = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        // Truncate any extra padding rows the embedding table may carry (keep weight/scales/biases
        // aligned, since row-0 truncation is safe for 4-bit packing).
        for suffix in ["weight", "scales", "biases"] {
            let key = "model.embed_tokens.\(suffix)"
            if let w = processed[key], w.dim(0) > vocabularySize {
                processed[key] = w[0 ..< vocabularySize]
            }
        }

        // Drop keys we don't model.
        return processed.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq") && !key.contains("lm_head")
        }
    }

    /// Load config + safetensors from an installed model directory (the A2 pattern), applying
    /// 4-bit quantization before `update` (custom loaders must quantize first — otherwise
    /// `update` rejects the `scales`/`biases` keys).
    static func fromDirectory(_ directory: URL) throws -> EmbeddingGemmaEmbedder {
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(EmbeddingGemmaConfiguration.self, from: configData)

        var raw: [String: MLXArray] = [:]
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        for file in files where file.pathExtension == "safetensors" {
            try MLX.loadArrays(url: file).forEach { raw[$0.key] = $0.value }
        }
        let weights = sanitize(weights: raw, vocabularySize: config.vocabularySize)

        // Infer the projection-head Linear dims from the actual checkpoint so shapes line up under
        // quantization. For a quantized layer, `scales` is [out, in / groupSize]; otherwise `weight`
        // is [out, in].
        let groupSize = config.quantGroupSize ?? 64
        var denseSpecs: [(inDim: Int, outDim: Int)] = []
        var i = 0
        while true {
            if let scales = weights["dense.\(i).scales"] {
                denseSpecs.append((inDim: scales.dim(1) * groupSize, outDim: scales.dim(0)))
            } else if let w = weights["dense.\(i).weight"] {
                denseSpecs.append((inDim: w.dim(1), outDim: w.dim(0)))
            } else {
                break
            }
            i += 1
        }

        let model = EmbeddingGemmaEmbedder(config, denseSpecs: denseSpecs)

        if config.quantBits != nil || config.quantGroupSize != nil {
            let bits = config.quantBits ?? 4
            quantize(model: model) { path, _ in
                weights["\(path).scales"] != nil ? (groupSize, bits, QuantizationMode.affine) : nil
            }
        }

        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)
        return model
    }

    /// Embed one or more texts → L2-normalized vectors. Matches ModernBERTEmbedder.embed's contract.
    static func embed(_ texts: [String], modelDirectory: URL) async throws -> [[Float]] {
        do {
            let model = try fromDirectory(modelDirectory)
            let tokenizer = try await AutoTokenizer.from(modelFolder: modelDirectory)

            let encoded = texts.map { tokenizer.encode(text: $0, addSpecialTokens: true) }
            let maxLen = max(1, encoded.map(\.count).max() ?? 1)
            let ids = stacked(encoded.map { row in
                MLXArray((row + Array(repeating: 0, count: maxLen - row.count)).map { Int32($0) })
            })
            let mask = stacked(encoded.map { row in
                MLXArray((0 ..< maxLen).map { Float($0 < row.count ? 1 : 0) })
            })   // [B, L]

            let pooled = model.pooledEmbeddings(inputs: ids, attentionMask: mask)
            pooled.eval()
            return (0 ..< pooled.dim(0)).map { pooled[$0].asArray(Float.self) }
        } catch {
            throw StageError.engineFailure(stage: "EmbeddingGemma", underlying: error)
        }
    }
}
