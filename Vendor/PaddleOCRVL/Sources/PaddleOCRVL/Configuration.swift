import Foundation

public struct PaddleOCRVLVisionConfig: Codable, Sendable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numChannels: Int
    public var imageSize: Int
    public var patchSize: Int
    public var hiddenAct: String
    public var layerNormEps: Float
    public var attentionDropout: Float
    public var intermediateSize: Int
    public var projectorHiddenAct: String

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numChannels = "num_channels"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case hiddenAct = "hidden_act"
        case layerNormEps = "layer_norm_eps"
        case attentionDropout = "attention_dropout"
        case intermediateSize = "intermediate_size"
        case projectorHiddenAct = "projector_hidden_act"
    }

    public init(
        hiddenSize: Int = 1024,
        numHiddenLayers: Int = 24,
        numAttentionHeads: Int = 16,
        numChannels: Int = 3,
        imageSize: Int = 448,
        patchSize: Int = 14,
        hiddenAct: String = "gelu",
        layerNormEps: Float = 1e-6,
        attentionDropout: Float = 0.0,
        intermediateSize: Int = 4096,
        projectorHiddenAct: String = "gelu"
    ) {
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numChannels = numChannels
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.hiddenAct = hiddenAct
        self.layerNormEps = layerNormEps
        self.attentionDropout = attentionDropout
        self.intermediateSize = intermediateSize
        self.projectorHiddenAct = projectorHiddenAct
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 24
        self.numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        self.numChannels = try container.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
        self.imageSize = try container.decodeIfPresent(Int.self, forKey: .imageSize) ?? 448
        self.patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu"
        self.layerNormEps = try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        self.attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4096
        self.projectorHiddenAct = try container.decodeIfPresent(String.self, forKey: .projectorHiddenAct) ?? "gelu"
    }
}

public struct PaddleOCRVLTextConfig: Codable, Sendable {
    public var vocabSize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    /// Per-head dimension. On PaddleOCR-VL 1.5 this is set explicitly (128) and is NOT
    /// `hiddenSize / numAttentionHeads` (1024/16 = 64), so q/k/v are wider than the hidden size.
    public var headDim: Int
    public var hiddenAct: String
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var tieWordEmbeddings: Bool
    /// 3D mRoPE section split across (temporal, height, width). Sums to headDim/2. Parsed from
    /// `rope_scaling.mrope_section` in `PaddleOCRVLConfig.load` (not via Codable).
    public var mropeSection: [Int] = [16, 24, 24]

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(
        vocabSize: Int = 48000,
        hiddenSize: Int = 896,
        intermediateSize: Int = 4864,
        numHiddenLayers: Int = 24,
        numAttentionHeads: Int = 14,
        numKeyValueHeads: Int = 2,
        headDim: Int? = nil,
        hiddenAct: String = "silu",
        maxPositionEmbeddings: Int = 32768,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 10_000,
        tieWordEmbeddings: Bool = false
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim ?? (hiddenSize / numAttentionHeads)
        self.hiddenAct = hiddenAct
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 48000
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 896
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4864
        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 24
        self.numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 14
        self.numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 2
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (self.hiddenSize / self.numAttentionHeads)
        self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        self.maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        self.tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

public struct PaddleOCRVLConfig: Codable, Sendable {
    /// Quantization parameters from the checkpoint's `config.json` `quantization` block.
    /// Present only for quantized checkpoints (e.g. mlx-community 4-bit); nil = full precision.
    public struct Quantization: Codable, Sendable {
        public var groupSize: Int
        public var bits: Int
        public init(groupSize: Int, bits: Int) {
            self.groupSize = groupSize
            self.bits = bits
        }
    }

    public var visionConfig: PaddleOCRVLVisionConfig
    public var textConfig: PaddleOCRVLTextConfig
    public var imageTokenIndex: Int
    public var visionStartTokenId: Int
    public var visionEndTokenId: Int
    public var visionTokenId: Int
    /// Global quantization params, or nil for full-precision checkpoints. Parsed in `load`,
    /// not via `Codable` (kept out of `CodingKeys` deliberately).
    public var quantization: Quantization? = nil
    /// End-of-sequence token id (from config.json). Parsed in `load`.
    public var eosTokenId: Int = 2

    enum CodingKeys: String, CodingKey {
        case visionConfig = "vision_config"
        case textConfig = "text_config"
        case imageTokenIndex = "image_token_index"
        case visionStartTokenId = "vision_start_token_id"
        case visionEndTokenId = "vision_end_token_id"
        case visionTokenId = "vision_token_id"
    }

    public init(
        visionConfig: PaddleOCRVLVisionConfig = PaddleOCRVLVisionConfig(),
        textConfig: PaddleOCRVLTextConfig = PaddleOCRVLTextConfig(),
        imageTokenIndex: Int = 151655,
        visionStartTokenId: Int = 151652,
        visionEndTokenId: Int = 151653,
        visionTokenId: Int = 151654,
        quantization: Quantization? = nil
    ) {
        self.visionConfig = visionConfig
        self.textConfig = textConfig
        self.imageTokenIndex = imageTokenIndex
        self.visionStartTokenId = visionStartTokenId
        self.visionEndTokenId = visionEndTokenId
        self.visionTokenId = visionTokenId
        self.quantization = quantization
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.visionConfig = try container.decodeIfPresent(PaddleOCRVLVisionConfig.self, forKey: .visionConfig) ?? PaddleOCRVLVisionConfig()
        self.textConfig = try container.decodeIfPresent(PaddleOCRVLTextConfig.self, forKey: .textConfig) ?? PaddleOCRVLTextConfig()
        self.imageTokenIndex = try container.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? 151655
        self.visionStartTokenId = try container.decodeIfPresent(Int.self, forKey: .visionStartTokenId) ?? 151652
        self.visionEndTokenId = try container.decodeIfPresent(Int.self, forKey: .visionEndTokenId) ?? 151653
        self.visionTokenId = try container.decodeIfPresent(Int.self, forKey: .visionTokenId) ?? 151654
    }

    public static func load(from directory: URL) throws -> PaddleOCRVLConfig {
        let configURL = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)

        let raw = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] ?? [:]
        let decoder = JSONDecoder()

        let visionConfig: PaddleOCRVLVisionConfig
        if let visionDict = raw["vision_config"] as? [String: Any] {
            let visionData = try JSONSerialization.data(withJSONObject: visionDict, options: [])
            visionConfig = try decoder.decode(PaddleOCRVLVisionConfig.self, from: visionData)
        } else {
            visionConfig = PaddleOCRVLVisionConfig()
        }

        // PaddleOCR-VL 1.5's config.json is FLAT (LM params at top level, no nested `text_config`).
        // Fall back to decoding the whole config for the text model when `text_config` is absent.
        var textConfig: PaddleOCRVLTextConfig
        if let textDict = raw["text_config"] as? [String: Any] {
            let textData = try JSONSerialization.data(withJSONObject: textDict, options: [])
            textConfig = try decoder.decode(PaddleOCRVLTextConfig.self, from: textData)
        } else {
            textConfig = try decoder.decode(PaddleOCRVLTextConfig.self, from: data)
        }
        // mrope_section lives under rope_scaling (a nested dict), not a flat key.
        if let ropeScaling = raw["rope_scaling"] as? [String: Any],
           let section = ropeScaling["mrope_section"] as? [Int], !section.isEmpty {
            textConfig.mropeSection = section
        }

        // Image placeholder token: 1.5 uses `image_token_id`; older layouts used `vision_token_id`.
        let imageTokenIndex = (raw["image_token_id"] as? Int) ?? (raw["image_token_index"] as? Int) ?? 151655
        let visionStartTokenId = (raw["vision_start_token_id"] as? Int) ?? 151652
        let visionEndTokenId = (raw["vision_end_token_id"] as? Int) ?? 151653
        let visionTokenId = (raw["image_token_id"] as? Int) ?? (raw["vision_token_id"] as? Int) ?? 151654

        // Quantized checkpoints carry a `quantization` block (group_size + bits). Accept the
        // HF/mlx spellings; default to the mlx-community convention (64-group, 4-bit) if the
        // block exists but omits a field.
        var quantization: Quantization? = nil
        if let q = raw["quantization"] as? [String: Any] {
            let groupSize = (q["group_size"] as? Int) ?? (q["groupSize"] as? Int) ?? 64
            let bits = (q["bits"] as? Int) ?? 4
            quantization = Quantization(groupSize: groupSize, bits: bits)
        }

        // eos_token_id may be a single Int or an array of ints; take the first.
        var eosTokenId = 2
        if let e = raw["eos_token_id"] as? Int {
            eosTokenId = e
        } else if let arr = raw["eos_token_id"] as? [Int], let first = arr.first {
            eosTokenId = first
        }

        var config = PaddleOCRVLConfig(
            visionConfig: visionConfig,
            textConfig: textConfig,
            imageTokenIndex: imageTokenIndex,
            visionStartTokenId: visionStartTokenId,
            visionEndTokenId: visionEndTokenId,
            visionTokenId: visionTokenId,
            quantization: quantization
        )
        config.eosTokenId = eosTokenId
        return config
    }
}

public struct PaddleOCRVLTokens {
    public static let imageToken = "<image>"
    public static let visionStartToken = "<|vision_start|>"
    public static let visionEndToken = "<|vision_end|>"
    public static let visionPadToken = "<|vision_pad|>"
}

public enum PaddleOCRTask: String, CaseIterable, Sendable {
    case ocr = "ocr"
    case table = "table"
    case formula = "formula"
    case chart = "chart"

    public var prompt: String {
        switch self {
        case .ocr: return "OCR:"
        case .table: return "Table Recognition:"
        case .formula: return "Formula Recognition:"
        case .chart: return "Chart Recognition:"
        }
    }
}
