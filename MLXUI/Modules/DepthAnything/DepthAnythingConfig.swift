import Foundation

/// `config.json` of `depth-anything/Depth-Anything-V2-Small-hf` (pinned revision
/// `5426e4f0f36572d16453bbda7a8389317b1bef99`). The architecture is fixed in the Swift port; the file is
/// decoded and checked against it so a different Depth Anything checkpoint in the same folder is refused
/// rather than loaded with the wrong shapes.
///
/// The published `config.json` is **partial**: it omits every field the `transformers` classes default
/// (`depth_estimation_type`, `max_depth`, `num_hidden_layers`, `mlp_ratio`, `hidden_act`, `layer_norm_eps`,
/// `num_channels`, `qkv_bias`, `layerscale_value`, `apply_layernorm`, `use_mask_token`,
/// `use_swiglu_ffn`). Those decode with the same defaults `transformers` uses, so the checkpoint is read as
/// the reference reads it.
nonisolated struct DepthAnythingConfig: Decodable, Sendable, Equatable {

    nonisolated struct Backbone: Decodable, Sendable, Equatable {
        let hiddenSize: Int
        let numHiddenLayers: Int
        let numAttentionHeads: Int
        let mlpRatio: Int
        let hiddenAct: String
        let layerNormEps: Float
        let imageSize: Int
        let patchSize: Int
        let numChannels: Int
        let qkvBias: Bool
        let layerscaleValue: Float
        let applyLayernorm: Bool
        let reshapeHiddenStates: Bool
        let useMaskToken: Bool
        let useSwigluFFN: Bool
        let outIndices: [Int]

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case mlpRatio = "mlp_ratio"
            case hiddenAct = "hidden_act"
            case layerNormEps = "layer_norm_eps"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case numChannels = "num_channels"
            case qkvBias = "qkv_bias"
            case layerscaleValue = "layerscale_value"
            case applyLayernorm = "apply_layernorm"
            case reshapeHiddenStates = "reshape_hidden_states"
            case useMaskToken = "use_mask_token"
            case useSwigluFFN = "use_swiglu_ffn"
            case outIndices = "out_indices"
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
            numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12
            numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
            mlpRatio = try c.decodeIfPresent(Int.self, forKey: .mlpRatio) ?? 4
            hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu"
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
            imageSize = try c.decode(Int.self, forKey: .imageSize)
            patchSize = try c.decode(Int.self, forKey: .patchSize)
            numChannels = try c.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
            qkvBias = try c.decodeIfPresent(Bool.self, forKey: .qkvBias) ?? true
            layerscaleValue = try c.decodeIfPresent(Float.self, forKey: .layerscaleValue) ?? 1
            applyLayernorm = try c.decodeIfPresent(Bool.self, forKey: .applyLayernorm) ?? true
            reshapeHiddenStates = try c.decodeIfPresent(Bool.self, forKey: .reshapeHiddenStates) ?? true
            useMaskToken = try c.decodeIfPresent(Bool.self, forKey: .useMaskToken) ?? true
            useSwigluFFN = try c.decodeIfPresent(Bool.self, forKey: .useSwigluFFN) ?? false
            outIndices = try c.decode([Int].self, forKey: .outIndices)
        }

        /// For tests and the synthetic config in `DepthAnythingBackboneTests`.
        init(hiddenSize: Int, numHiddenLayers: Int = 12, numAttentionHeads: Int, mlpRatio: Int = 4,
             hiddenAct: String = "gelu", layerNormEps: Float = 1e-6, imageSize: Int, patchSize: Int,
             numChannels: Int = 3, qkvBias: Bool = true, layerscaleValue: Float = 1,
             applyLayernorm: Bool = true, reshapeHiddenStates: Bool = true, useMaskToken: Bool = true,
             useSwigluFFN: Bool = false, outIndices: [Int]) {
            self.hiddenSize = hiddenSize
            self.numHiddenLayers = numHiddenLayers
            self.numAttentionHeads = numAttentionHeads
            self.mlpRatio = mlpRatio
            self.hiddenAct = hiddenAct
            self.layerNormEps = layerNormEps
            self.imageSize = imageSize
            self.patchSize = patchSize
            self.numChannels = numChannels
            self.qkvBias = qkvBias
            self.layerscaleValue = layerscaleValue
            self.applyLayernorm = applyLayernorm
            self.reshapeHiddenStates = reshapeHiddenStates
            self.useMaskToken = useMaskToken
            self.useSwigluFFN = useSwigluFFN
            self.outIndices = outIndices
        }
    }

    let patchSize: Int
    let reassembleHiddenSize: Int
    let reassembleFactors: [Float]
    let neckHiddenSizes: [Int]
    let fusionHiddenSize: Int
    let headInIndex: Int
    let headHiddenSize: Int
    let depthEstimationType: String
    let maxDepth: Float
    let backboneConfig: Backbone

    enum CodingKeys: String, CodingKey {
        case patchSize = "patch_size"
        case reassembleHiddenSize = "reassemble_hidden_size"
        case reassembleFactors = "reassemble_factors"
        case neckHiddenSizes = "neck_hidden_sizes"
        case fusionHiddenSize = "fusion_hidden_size"
        case headInIndex = "head_in_index"
        case headHiddenSize = "head_hidden_size"
        case depthEstimationType = "depth_estimation_type"
        case maxDepth = "max_depth"
        case backboneConfig = "backbone_config"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        patchSize = try c.decode(Int.self, forKey: .patchSize)
        reassembleHiddenSize = try c.decode(Int.self, forKey: .reassembleHiddenSize)
        reassembleFactors = try c.decode([Float].self, forKey: .reassembleFactors)
        neckHiddenSizes = try c.decode([Int].self, forKey: .neckHiddenSizes)
        fusionHiddenSize = try c.decode(Int.self, forKey: .fusionHiddenSize)
        headInIndex = try c.decodeIfPresent(Int.self, forKey: .headInIndex) ?? -1
        headHiddenSize = try c.decode(Int.self, forKey: .headHiddenSize)
        depthEstimationType = try c.decodeIfPresent(String.self, forKey: .depthEstimationType) ?? "relative"
        let depth = try c.decodeIfPresent(Float.self, forKey: .maxDepth)
        maxDepth = (depth ?? 0) == 0 ? 1 : (depth ?? 1)   // the reference: `self.max_depth = self.max_depth if self.max_depth else 1`
        backboneConfig = try c.decode(Backbone.self, forKey: .backboneConfig)
    }

    /// For tests.
    init(patchSize: Int, reassembleHiddenSize: Int, reassembleFactors: [Float], neckHiddenSizes: [Int],
         fusionHiddenSize: Int, headInIndex: Int = -1, headHiddenSize: Int,
         depthEstimationType: String = "relative", maxDepth: Float = 1, backboneConfig: Backbone) {
        self.patchSize = patchSize
        self.reassembleHiddenSize = reassembleHiddenSize
        self.reassembleFactors = reassembleFactors
        self.neckHiddenSizes = neckHiddenSizes
        self.fusionHiddenSize = fusionHiddenSize
        self.headInIndex = headInIndex
        self.headHiddenSize = headHiddenSize
        self.depthEstimationType = depthEstimationType
        self.maxDepth = maxDepth
        self.backboneConfig = backboneConfig
    }

    /// Refuses a checkpoint this port was not written for. Naming the mismatch is the point: a silently
    /// different backbone would load with wrong shapes or produce plausible-looking garbage.
    func validate() throws {
        let b = backboneConfig
        guard b.hiddenSize == 384, b.numHiddenLayers == 12, b.numAttentionHeads == 6,
              b.patchSize == 14, b.numChannels == 3, b.mlpRatio == 4,
              b.hiddenAct == "gelu", b.qkvBias, !b.useSwigluFFN else {
            throw DepthAnythingError.unsupportedConfig(
                "backbone \(b.numHiddenLayers)x \(b.hiddenSize)-wide, \(b.numAttentionHeads) heads, patch \(b.patchSize), act \(b.hiddenAct)")
        }
        guard patchSize == 14, reassembleHiddenSize == 384,
              neckHiddenSizes.count == 4, reassembleFactors.count == 4,
              fusionHiddenSize == 64, headHiddenSize == 32,
              (headInIndex == -1 || headInIndex == 3) else {
            throw DepthAnythingError.unsupportedConfig(
                "neck \(neckHiddenSizes) factors \(reassembleFactors) fusion \(fusionHiddenSize) head \(headHiddenSize)")
        }
        guard reassembleFactors == [4, 2, 1, 0.5] else {
            throw DepthAnythingError.unsupportedConfig("reassemble factors \(reassembleFactors)")
        }
        guard b.outIndices == [3, 6, 9, 12] else {
            throw DepthAnythingError.unsupportedConfig("backbone out indices \(b.outIndices)")
        }
        guard depthEstimationType == "relative", maxDepth == 1 else {
            throw DepthAnythingError.unsupportedConfig(
                "depth estimation type \(depthEstimationType) (max depth \(maxDepth)); only relative depth is ported")
        }
        guard b.applyLayernorm, !b.reshapeHiddenStates, b.useMaskToken else {
            throw DepthAnythingError.unsupportedConfig(
                "backbone options apply_layernorm=\(b.applyLayernorm) reshape=\(b.reshapeHiddenStates) mask_token=\(b.useMaskToken)")
        }
        guard b.layerNormEps == 1e-6, b.layerscaleValue == 1 else {
            throw DepthAnythingError.unsupportedConfig("layer norm eps \(b.layerNormEps), layer scale \(b.layerscaleValue)")
        }
        guard b.imageSize == 518 else {
            throw DepthAnythingError.unsupportedConfig("backbone image size \(b.imageSize)")
        }
    }

    /// The 37 x 37 pretrained position-embedding grid (518 / 14).
    var pretrainedPatchGrid: Int { backboneConfig.imageSize / backboneConfig.patchSize }
}
