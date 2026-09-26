import Foundation
import MLX
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

/// The ModernBERT encoder (embeddings + layers + final norm) and a standalone embed entry point.
/// Gate C (`RSI/DelegateLayaBacklog.md`, LY-4): built on the shared `Core/ModernBERTEncoder.swift`
/// (extracted so Laya's decision encoder doesn't duplicate this file's ~180 lines), constructed
/// with `slidingWindowEnabled: false` — this checkpoint has never run a local-window mask (see
/// the header comment above), so behavior is unchanged: every layer still gets the same
/// full-sequence mask it always did. Only the *representation* of that mask moved, from an
/// additive float log-mask to the shared encoder's native boolean mask — mathematically the
/// same "attend fully to valid positions, `-inf` to padding" (both feed the same
/// `MLXFast.scaledDotProductAttention` primitive, which accepts either), not a semantic change.
nonisolated final class ModernBERTEmbedder: Module {
    @ModuleInfo(key: "encoder") var encoder: ModernBERTEncoder

    init(_ config: ModernBERTEncoderConfig) {
        _encoder.wrappedValue = ModernBERTEncoder(config, slidingWindowEnabled: false)
    }

    /// Last-layer hidden states for `inputIds` [B, L]. `attentionMask` [B, L] (1 keep / 0 pad).
    func hiddenStates(inputIds: MLXArray, attentionMask: MLXArray) -> MLXArray {
        encoder(inputIds: inputIds, attentionMask: attentionMask)
    }

    /// Map HF ModernBERT weight names onto this module's keys, keeping only what we model.
    /// The shared encoder's own layout (`embeddings.tok_embeddings`, `embeddings.norm`,
    /// `layers.<n>.*`, `final_norm`) already matches this checkpoint's raw keys one-for-one
    /// once `model.` is stripped — the only rename left is routing them under this class's
    /// `encoder` submodule.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (rawKey, value) in weights {
            var key = rawKey
            if key.hasPrefix("model.") { key = String(key.dropFirst("model.".count)) }
            // Keep only encoder params; drop MLM head / position_ids / anything else.
            if key.hasPrefix("embeddings.") || key.hasPrefix("layers.") || key.hasPrefix("final_norm.") {
                out["encoder." + key] = value
            }
        }
        return out
    }

    /// Load config + safetensors from an installed model directory (the A2 pattern).
    static func fromDirectory(_ directory: URL) throws -> ModernBERTEmbedder {
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let hfConfig = try JSONDecoder().decode(ModernBERTConfiguration.self, from: configData)
        let config = ModernBERTEncoderConfig(
            vocabularySize: hfConfig.vocabularySize, hiddenSize: hfConfig.hiddenSize,
            intermediateSize: hfConfig.intermediateSize, numHiddenLayers: hfConfig.numHiddenLayers,
            numAttentionHeads: hfConfig.numAttentionHeads, normEps: hfConfig.normEps,
            localAttention: 128, globalAttnEveryNLayers: hfConfig.globalAttnEveryNLayers,
            globalRopeTheta: hfConfig.globalRopeTheta, localRopeTheta: hfConfig.localRopeTheta)
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
