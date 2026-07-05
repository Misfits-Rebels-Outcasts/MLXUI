import Foundation

/// Decoded `config.json` for a DeepSeek-OCR-2 checkpoint (`model_type: deepseekocr_2`). The
/// DeepSeek-V2 text hyper-parameters live at the JSON top level (also mirrored under
/// `language_config`); the vision tower is a nested `vision_config.width.{sam_vit_b, qwen2-0-5b}`
/// dict; the projector is `projector_config`. Many text/vision fields are absent from this checkpoint
/// and fall back to the upstream `config.py` defaults (folded in here). Verified against
/// `mlx-community/DeepSeek-OCR-2-bf16`. Pure Foundation (no MLX).
struct DeepSeekOCRConfig: Codable, Sendable {

    // MARK: Text (DeepSeek-V2, Llama/MHA — `use_mla:false`, `qk_nope_head_dim:0`)
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var vocabSize: Int
    var maxPositionEmbeddings: Int
    var firstKDenseReplace: Int
    var nRoutedExperts: Int
    var nSharedExperts: Int
    var numExpertsPerTok: Int
    var nGroup: Int
    var topkGroup: Int
    var topkMethod: String
    var scoringFunc: String
    var routedScalingFactor: Float
    var ropeTheta: Float
    var rmsNormEps: Float
    var attentionBias: Bool

    // MARK: Top-level multimodal
    var tileTag: String
    var globalViewPos: String
    var imageTokenIndex: Int

    // MARK: Sub-configs
    var vision: Vision
    var projector: Projector

    /// `head_dim` for the LM (Llama MHA): `qk_nope_head_dim == 0` ⇒ hidden/heads.
    var headDim: Int { hiddenSize / numAttentionHeads }

    /// The `dots_vit`-style vision config: a SAM ViT-B feeding a Qwen2-0.5B decoder-as-encoder.
    /// Only `sam_vit_b`'s {width, layers, heads, global_attn_indexes, downsample_channels} and
    /// `qwen2-0-5b.dim` come from JSON; the rest are `config.py` defaults.
    struct Vision: Codable, Sendable {
        // SAM ViT-B
        var samWidth: Int
        var samLayers: Int
        var samHeads: Int
        var samGlobalAttnIndexes: [Int]
        var samDownsampleChannels: [Int]
        var samPatchSize: Int = 16
        var samWindowSize: Int = 14
        var samImageSize: Int = 1024
        var samFinalOutChannels: Int = 896   // OCR-2 uses 896 (vs 1024 in OCR-1)
        // Qwen2-0.5B decoder-as-encoder
        var qwen2Dim: Int
        var qwen2Layers: Int = 24
        var qwen2Heads: Int = 14
        var qwen2KVHeads: Int = 2
        var qwen2IntermediateSize: Int = 4864
        var qwen2RMSNormEps: Float = 1e-6
        var qwen2RopeTheta: Float = 1_000_000
    }

    struct Projector: Codable, Sendable {
        var inputDim: Int
        var nEmbed: Int
        var projectorType: String
    }

    // MARK: Decoding

    private enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case firstKDenseReplace = "first_k_dense_replace"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case topkMethod = "topk_method"
        case scoringFunc = "scoring_func"
        case routedScalingFactor = "routed_scaling_factor"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case tileTag = "tile_tag"
        case globalViewPos = "global_view_pos"
        case imageTokenIndex = "image_token_index"
        case visionConfig = "vision_config"
        case projectorConfig = "projector_config"
    }

    private enum VisionKeys: String, CodingKey { case width; case imageSize = "image_size" }
    private enum WidthKeys: String, CodingKey {
        case qwen2 = "qwen2-0-5b"
        case sam = "sam_vit_b"
    }
    private enum Qwen2Keys: String, CodingKey {
        case dim, layers, heads
        case kvHeads = "kv_heads"
        case intermediateSize = "intermediate_size"
    }
    private enum SamKeys: String, CodingKey {
        case width, heads, layers
        case globalAttnIndexes = "global_attn_indexes"
        case downsampleChannels = "downsample_channels"
        case finalOutChannels = "final_out_chans"
    }
    private enum ProjectorKeys: String, CodingKey {
        case inputDim = "input_dim"
        case nEmbed = "n_embed"
        case projectorType = "projector_type"
    }

    init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func i(_ k: CodingKeys, _ d: Int) throws -> Int { try c.decodeIfPresent(Int.self, forKey: k) ?? d }
        func f(_ k: CodingKeys, _ d: Float) throws -> Float { try c.decodeIfPresent(Float.self, forKey: k) ?? d }
        func s(_ k: CodingKeys, _ d: String) throws -> String { try c.decodeIfPresent(String.self, forKey: k) ?? d }
        func b(_ k: CodingKeys, _ d: Bool) throws -> Bool { try c.decodeIfPresent(Bool.self, forKey: k) ?? d }

        hiddenSize = try i(.hiddenSize, 1280)
        intermediateSize = try i(.intermediateSize, 6848)
        moeIntermediateSize = try i(.moeIntermediateSize, 896)
        numHiddenLayers = try i(.numHiddenLayers, 12)
        numAttentionHeads = try i(.numAttentionHeads, 10)
        numKeyValueHeads = try i(.numKeyValueHeads, 10)
        vocabSize = try i(.vocabSize, 129280)
        maxPositionEmbeddings = try i(.maxPositionEmbeddings, 8192)
        firstKDenseReplace = try i(.firstKDenseReplace, 1)
        nRoutedExperts = try i(.nRoutedExperts, 64)
        nSharedExperts = try i(.nSharedExperts, 2)
        numExpertsPerTok = try i(.numExpertsPerTok, 6)
        nGroup = try i(.nGroup, 1)
        topkGroup = try i(.topkGroup, 1)
        topkMethod = try s(.topkMethod, "greedy")
        scoringFunc = try s(.scoringFunc, "softmax")           // absent in JSON → config.py default
        routedScalingFactor = try f(.routedScalingFactor, 1.0) // absent → default
        ropeTheta = try f(.ropeTheta, 10_000)                  // absent → default
        rmsNormEps = try f(.rmsNormEps, 1e-6)                  // absent → default
        attentionBias = try b(.attentionBias, false)          // absent → default
        tileTag = try s(.tileTag, "2D")
        globalViewPos = try s(.globalViewPos, "head")
        imageTokenIndex = try i(.imageTokenIndex, 128815)      // absent → config.py default

        // vision_config.width.{sam_vit_b, qwen2-0-5b}
        let vc = try c.nestedContainer(keyedBy: VisionKeys.self, forKey: .visionConfig)
        let width = try vc.nestedContainer(keyedBy: WidthKeys.self, forKey: .width)
        let qwen2 = try width.nestedContainer(keyedBy: Qwen2Keys.self, forKey: .qwen2)
        let sam = try width.nestedContainer(keyedBy: SamKeys.self, forKey: .sam)
        vision = Vision(
            samWidth: try sam.decodeIfPresent(Int.self, forKey: .width) ?? 768,
            samLayers: try sam.decodeIfPresent(Int.self, forKey: .layers) ?? 12,
            samHeads: try sam.decodeIfPresent(Int.self, forKey: .heads) ?? 12,
            samGlobalAttnIndexes: try sam.decodeIfPresent([Int].self, forKey: .globalAttnIndexes) ?? [2, 5, 8, 11],
            samDownsampleChannels: try sam.decodeIfPresent([Int].self, forKey: .downsampleChannels) ?? [512, 1024],
            samImageSize: try vc.decodeIfPresent(Int.self, forKey: .imageSize) ?? 1024,
            samFinalOutChannels: try sam.decodeIfPresent(Int.self, forKey: .finalOutChannels) ?? 896,
            qwen2Dim: try qwen2.decodeIfPresent(Int.self, forKey: .dim) ?? 896,
            qwen2Layers: try qwen2.decodeIfPresent(Int.self, forKey: .layers) ?? 24,
            qwen2Heads: try qwen2.decodeIfPresent(Int.self, forKey: .heads) ?? 14,
            qwen2KVHeads: try qwen2.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 2,
            qwen2IntermediateSize: try qwen2.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4864)

        // projector_config
        let pc = try c.nestedContainer(keyedBy: ProjectorKeys.self, forKey: .projectorConfig)
        projector = Projector(
            inputDim: try pc.decodeIfPresent(Int.self, forKey: .inputDim) ?? 896,
            nEmbed: try pc.decodeIfPresent(Int.self, forKey: .nEmbed) ?? 1280,
            projectorType: try pc.decodeIfPresent(String.self, forKey: .projectorType) ?? "linear")
    }

    func encode(to encoder: Swift.Encoder) throws {
        // Decode-only in practice; a minimal encoder keeps `Codable` conformance cheap.
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(vocabSize, forKey: .vocabSize)
    }
}

/// Weight-key remapping from the HF checkpoint layout to our module's property layout — a 1:1 mirror
/// of `deepseekocr_2.py`'s `transform_key`. Order-sensitive: qwen2-encoder rewrites first (so the
/// broad `model.layers`/`model.norm`/`model.embed_tokens` LM rewrites skip qwen2 keys via the guards),
/// then LM, then vision/sam/projector, the `view_seperator`→`view_separator` typo fix, and `lm_head`.
enum DeepSeekOCRWeights {
    static func remapKey(_ key0: String) -> String {
        var key = key0

        if key.contains("qwen2_model.model.model.layers") {
            key = key.replacingOccurrences(
                of: "model.qwen2_model.model.model.layers", with: "vision_model.qwen2_encoder.layers")
        }
        if key.contains("qwen2_model.model.model.norm") {
            key = key.replacingOccurrences(
                of: "model.qwen2_model.model.model.norm", with: "vision_model.qwen2_encoder.norm")
        }
        if key.contains("model.qwen2_model.query_1024") {
            key = key.replacingOccurrences(
                of: "model.qwen2_model.query_1024.weight", with: "vision_model.qwen2_encoder.query_1024")
            key = key.replacingOccurrences(
                of: "model.qwen2_model.query_1024", with: "vision_model.qwen2_encoder.query_1024")
        }
        if key.contains("model.qwen2_model.query_768") {
            key = key.replacingOccurrences(
                of: "model.qwen2_model.query_768.weight", with: "vision_model.qwen2_encoder.query_768")
            key = key.replacingOccurrences(
                of: "model.qwen2_model.query_768", with: "vision_model.qwen2_encoder.query_768")
        }

        let isQwen = key.contains("qwen2")
        let isLang = key.contains("language_model")
        if key.contains("model.layers"), !isLang, !isQwen {
            key = key.replacingOccurrences(of: "model.layers", with: "language_model.model.layers")
        }
        if key.contains("model.embed_tokens"), !isLang, !isQwen {
            key = key.replacingOccurrences(of: "model.embed_tokens", with: "language_model.model.embed_tokens")
        }
        if key.contains("model.norm"), !isLang, !isQwen {
            key = key.replacingOccurrences(of: "model.norm", with: "language_model.model.norm")
        }
        if key.contains("model.vision_model") {
            key = key.replacingOccurrences(of: "model.vision_model", with: "vision_model")
        }
        if key.contains("model.sam_model") {
            key = key.replacingOccurrences(of: "model.sam_model", with: "sam_model")
        }
        if key.contains("model.projector") {
            key = key.replacingOccurrences(of: "model.projector", with: "projector")
        }
        if key.contains("model.view_seperator") {   // HF typo (e instead of a)
            key = key.replacingOccurrences(of: "model.view_seperator", with: "view_separator")
        }
        if key.contains("lm_head.weight"), !key.contains("language_model") {
            key = key.replacingOccurrences(of: "lm_head.weight", with: "language_model.lm_head.weight")
        }
        return key
    }
}
