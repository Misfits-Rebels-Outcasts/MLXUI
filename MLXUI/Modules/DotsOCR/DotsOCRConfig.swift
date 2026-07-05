import Foundation

/// Decoded `config.json` for a dots.ocr / dots.mocr checkpoint (`model_type: dots_ocr`).
///
/// The **text** hyper-parameters (a Qwen2 decoder) live at the JSON **top level** — the upstream
/// `configuration_dots` keeps no nested `text_config` for these checkpoints — while the `dots_vit`
/// vision tower is nested under `vision_config`. Field set verified against
/// `mlx-community/dots.mocr-4bit`'s `config.json`. Pure Foundation (no MLX): decoded in the engine,
/// then handed to the model (later slice).
struct DotsOCRConfig: Codable, Sendable {

    /// The `dots_vit` NaViT vision tower config (`vision_config`).
    struct Vision: Codable, Sendable {
        var embedDim: Int
        var hiddenSize: Int
        var intermediateSize: Int
        var numHiddenLayers: Int
        var numAttentionHeads: Int
        var numChannels: Int
        var patchSize: Int
        var postNorm: Bool
        var rmsNormEps: Float
        var spatialMergeSize: Int
        var temporalPatchSize: Int
        var useBias: Bool

        enum CodingKeys: String, CodingKey {
            case embedDim = "embed_dim"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numChannels = "num_channels"
            case patchSize = "patch_size"
            case postNorm = "post_norm"
            case rmsNormEps = "rms_norm_eps"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
            case useBias = "use_bias"
        }
    }

    /// Affine-quantization block (present on the 4-bit MLX checkpoints).
    struct Quantization: Codable, Sendable {
        var groupSize: Int
        var bits: Int

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
        }
    }

    // Text (Qwen2) — flat at the top level.
    var modelType: String
    var hiddenSize: Int
    var intermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var vocabSize: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var attentionBias: Bool
    var tieWordEmbeddings: Bool
    var maxPositionEmbeddings: Int
    var imageTokenId: Int
    var videoTokenId: Int

    var visionConfig: Vision
    var quantization: Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case attentionBias = "attention_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case visionConfig = "vision_config"
        case quantization
    }

    /// Head dimension shared by the text decoder (Qwen2 GQA).
    var headDim: Int { hiddenSize / numAttentionHeads }
}

/// Weight-key remapping from the HF checkpoint layout to our module's property layout — a 1:1
/// mirror of `dots_ocr.py`'s `Model.sanitize` (prefix rewrites only). Kept pure/string-only so it's
/// unit-testable; the array-level bits (vision conv-weight transpose) live in the vision tower's own
/// `sanitize` (later slice). **Order matters:** `model.vision_tower.` must be checked before the
/// broader `model.` prefix.
enum DotsOCRWeights {
    static func remapKey(_ key: String) -> String {
        if key.hasPrefix("model.vision_tower.") {
            return "vision_tower." + key.dropFirst("model.vision_tower.".count)
        } else if key.hasPrefix("model.") {
            return "language_model.model." + key.dropFirst("model.".count)
        } else if key.hasPrefix("lm_head.") {
            return "language_model.model.lm_head." + key.dropFirst("lm_head.".count)
        }
        return key
    }
}
