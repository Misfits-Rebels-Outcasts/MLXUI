//
//  Qwen35Config.swift
//  MLXUI — Qwen3.5 (qwen3_5) hybrid GatedDeltaNet port, Slice 1.
//
//  Decodes the checkpoint `config.json` for prism-ml/Ternary-Bonsai-*-mlx-2bit
//  (model_type "qwen3_5", text model_type "qwen3_5_text"). Mirrors the reference
//  layout in Blaizzy/mlx-vlm `models/qwen3_5/config.py` (TextConfig / VisionConfig /
//  ModelConfig). Pure Codable — no MLX dependency — so it is unit-testable and
//  build-verifiable on its own.
//
//  NOTE: this is a from-reference port in progress. Only the text tower is modelled
//  here; the vision tower (Qwen3-VL) config is decoded but not yet consumed by a runner.
//

import Foundation

/// RoPE parameters for the full-attention layers. Qwen3.5 uses interleaved multimodal
/// RoPE (`mrope`) with a partial rotary factor (only a fraction of `head_dim` is rotated).
struct Qwen35RopeParameters: Codable, Sendable {
    var mropeInterleaved: Bool
    var mropeSection: [Int]
    var partialRotaryFactor: Double
    var ropeTheta: Double
    var ropeType: String

    enum CodingKeys: String, CodingKey {
        case mropeInterleaved = "mrope_interleaved"
        case mropeSection = "mrope_section"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeTheta = "rope_theta"
        case ropeType = "rope_type"
    }

    init(
        mropeInterleaved: Bool = true,
        mropeSection: [Int] = [11, 11, 10],
        partialRotaryFactor: Double = 0.25,
        ropeTheta: Double = 10_000_000,
        ropeType: String = "default"
    ) {
        self.mropeInterleaved = mropeInterleaved
        self.mropeSection = mropeSection
        self.partialRotaryFactor = partialRotaryFactor
        self.ropeTheta = ropeTheta
        self.ropeType = ropeType
    }

    init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.mropeInterleaved = try c.decodeIfPresent(Bool.self, forKey: .mropeInterleaved) ?? true
        self.mropeSection = try c.decodeIfPresent([Int].self, forKey: .mropeSection) ?? [11, 11, 10]
        self.partialRotaryFactor = try c.decodeIfPresent(Double.self, forKey: .partialRotaryFactor) ?? 0.25
        self.ropeTheta = try c.decodeIfPresent(Double.self, forKey: .ropeTheta) ?? 10_000_000
        self.ropeType = try c.decodeIfPresent(String.self, forKey: .ropeType) ?? "default"
    }
}

/// The Qwen3.5 text tower configuration (`text_config`). Combines standard gated
/// full-attention layers with GatedDeltaNet linear-attention layers on a fixed schedule:
/// every `fullAttentionInterval`-th layer is full attention, the rest are linear.
struct Qwen35TextConfig: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var intermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var vocabSize: Int
    var maxPositionEmbeddings: Int
    var fullAttentionInterval: Int
    var attentionBias: Bool
    var attnOutputGate: Bool
    var tieWordEmbeddings: Bool
    var partialRotaryFactor: Double
    var ropeParameters: Qwen35RopeParameters

    // GatedDeltaNet (linear-attention) dimensions.
    var linearNumValueHeads: Int
    var linearNumKeyHeads: Int
    var linearKeyHeadDim: Int
    var linearValueHeadDim: Int
    var linearConvKernelDim: Int

    /// The full per-layer schedule from the checkpoint ("linear_attention" / "full_attention").
    /// Derivable from `fullAttentionInterval`, but decoded verbatim so the port matches the
    /// checkpoint exactly even if the pattern ever changes.
    var layerTypes: [String]

    /// True when layer `index` is a GatedDeltaNet (linear-attention) layer.
    /// Matches the reference: `is_linear = (layer_idx + 1) % full_attention_interval != 0`.
    func isLinearLayer(_ index: Int) -> Bool {
        if index >= 0, index < layerTypes.count {
            return layerTypes[index] == "linear_attention"
        }
        return (index + 1) % fullAttentionInterval != 0
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case fullAttentionInterval = "full_attention_interval"
        case attentionBias = "attention_bias"
        case attnOutputGate = "attn_output_gate"
        case tieWordEmbeddings = "tie_word_embeddings"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeParameters = "rope_parameters"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case layerTypes = "layer_types"
    }

    init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func i(_ k: CodingKeys, _ d: Int) throws -> Int { try c.decodeIfPresent(Int.self, forKey: k) ?? d }

        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_5_text"
        self.hiddenSize = try i(.hiddenSize, 5120)
        self.intermediateSize = try i(.intermediateSize, 17408)
        self.numHiddenLayers = try i(.numHiddenLayers, 64)
        self.numAttentionHeads = try i(.numAttentionHeads, 24)
        self.numKeyValueHeads = try i(.numKeyValueHeads, 4)
        self.headDim = try i(.headDim, 256)
        self.rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabSize = try i(.vocabSize, 248320)
        self.maxPositionEmbeddings = try i(.maxPositionEmbeddings, 262144)
        self.fullAttentionInterval = try i(.fullAttentionInterval, 4)
        self.attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.attnOutputGate = try c.decodeIfPresent(Bool.self, forKey: .attnOutputGate) ?? true
        self.tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.partialRotaryFactor = try c.decodeIfPresent(Double.self, forKey: .partialRotaryFactor) ?? 0.25
        self.ropeParameters = try c.decodeIfPresent(Qwen35RopeParameters.self, forKey: .ropeParameters)
            ?? Qwen35RopeParameters()
        self.linearNumValueHeads = try i(.linearNumValueHeads, 48)
        self.linearNumKeyHeads = try i(.linearNumKeyHeads, 16)
        self.linearKeyHeadDim = try i(.linearKeyHeadDim, 128)
        self.linearValueHeadDim = try i(.linearValueHeadDim, 128)
        self.linearConvKernelDim = try i(.linearConvKernelDim, 4)
        self.layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
    }
}

/// The Qwen3.5 vision tower configuration (`vision_config`), decoded for completeness.
/// Not yet consumed — the vision runner is a later port slice.
struct Qwen35VisionConfig: Codable, Sendable {
    var depth: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var numHeads: Int
    var inChannels: Int
    var patchSize: Int
    var spatialMergeSize: Int
    var temporalPatchSize: Int
    var outHiddenSize: Int
    var numPositionEmbeddings: Int

    enum CodingKeys: String, CodingKey {
        case depth
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHeads = "num_heads"
        case inChannels = "in_channels"
        case patchSize = "patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case temporalPatchSize = "temporal_patch_size"
        case outHiddenSize = "out_hidden_size"
        case numPositionEmbeddings = "num_position_embeddings"
    }

    init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func i(_ k: CodingKeys, _ d: Int) throws -> Int { try c.decodeIfPresent(Int.self, forKey: k) ?? d }
        self.depth = try i(.depth, 24)
        self.hiddenSize = try i(.hiddenSize, 1152)
        self.intermediateSize = try i(.intermediateSize, 4304)
        self.numHeads = try i(.numHeads, 16)
        self.inChannels = try i(.inChannels, 3)
        self.patchSize = try i(.patchSize, 16)
        self.spatialMergeSize = try i(.spatialMergeSize, 2)
        self.temporalPatchSize = try i(.temporalPatchSize, 2)
        self.outHiddenSize = try i(.outHiddenSize, 5120)
        self.numPositionEmbeddings = try i(.numPositionEmbeddings, 2304)
    }
}

/// Top-level `config.json` for the Qwen3.5 multimodal checkpoint.
struct Qwen35Config: Codable, Sendable {
    var modelType: String
    var textConfig: Qwen35TextConfig
    var visionConfig: Qwen35VisionConfig?
    var imageTokenId: Int
    var videoTokenId: Int
    var visionStartTokenId: Int
    var visionEndTokenId: Int
    var tieWordEmbeddings: Bool
    var languageModelOnly: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case visionStartTokenId = "vision_start_token_id"
        case visionEndTokenId = "vision_end_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
        case languageModelOnly = "language_model_only"
    }

    init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_5"
        self.textConfig = try c.decode(Qwen35TextConfig.self, forKey: .textConfig)
        self.visionConfig = try c.decodeIfPresent(Qwen35VisionConfig.self, forKey: .visionConfig)
        self.imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 248056
        self.videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 248057
        self.visionStartTokenId = try c.decodeIfPresent(Int.self, forKey: .visionStartTokenId) ?? 248045
        self.visionEndTokenId = try c.decodeIfPresent(Int.self, forKey: .visionEndTokenId) ?? 248046
        self.tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.languageModelOnly = try c.decodeIfPresent(Bool.self, forKey: .languageModelOnly) ?? false
    }
}
