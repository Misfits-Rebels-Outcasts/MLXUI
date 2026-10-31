import Foundation

// FLUX.2 Klein 4B configuration (MP7-3a). Decoded from the checkpoint's own config.json files
// (transformer/, text_encoder/, vae/, scheduler/ of mlx-community/FLUX.2-klein-4B-bf16, revision 2f7dd4ac) and then checked
// against the dimensions the port is written for (RSI/notes/claude-code/modelpack/MP7-1-spike.md section 2). A checkpoint whose
// config differs is refused at decode with the field named, never loaded with the wrong shapes.
//
// The field set and the expected values follow VincentGourbin/flux-2-swift-mlx @ 315c27e (MIT, © 2025 Vincent Gourbin)
// `Flux2TransformerConfig.klein4B` and `Qwen3TextConfig.qwen3_4B`; the VAE and scheduler fields come from diffusers 0.37's
// `AutoencoderKLFlux2` and `FlowMatchEulerDiscreteScheduler` configs.

nonisolated enum Flux2KleinConfigError: Error, Equatable, LocalizedError {
    /// A config field differs from the Klein 4B value this port is built for.
    case unexpected(file: String, field: String, expected: String, actual: String)

    var errorDescription: String? {
        switch self {
        case let .unexpected(file, field, expected, actual):
            return "\(file): \(field) is \(actual), but this FLUX.2 Klein 4B build is written for \(expected)."
        }
    }
}

/// The Klein 4B dimensions, in one place. The decoders below compare against these.
nonisolated enum Flux2KleinDims {
    static let attentionHeadDim = 128
    static let attentionHeads = 24
    static let doubleBlocks = 5
    static let singleBlocks = 20
    static let transformerInChannels = 128      // 32 VAE channels x 2 x 2 patch
    static let jointAttentionDim = 7680         // 3 tapped Qwen3 layers x 2560
    static let axesDimsRope = [32, 32, 32, 32]
    static let ropeTheta = 2000.0
    static let timestepGuidanceChannels = 256

    static let textHidden = 2560
    static let textLayers = 36
    static let textHeads = 32
    static let textKVHeads = 8
    static let textHeadDim = 128
    static let textIntermediate = 9728
    static let textVocab = 151_936
    /// `hidden_states[k]` of the Qwen3 pass (index 0 is the embedding output) that feed the DiT, in feature order.
    static let tapLayers = [9, 18, 27]

    static let latentChannels = 32
    static let vaePatch = 2
    static let vaeScale = 8
    static let vaeBatchNormEps = 1e-4
}

nonisolated struct Flux2KleinTransformerConfig: Decodable, Equatable, Sendable {
    let attentionHeadDim: Int
    let numAttentionHeads: Int
    let numLayers: Int
    let numSingleLayers: Int
    let inChannels: Int
    let patchSize: Int
    let jointAttentionDim: Int
    let mlpRatio: Double
    let axesDimsRope: [Int]
    let ropeTheta: Double
    let timestepGuidanceChannels: Int
    let eps: Double
    let guidanceEmbeds: Bool

    var innerDim: Int { attentionHeadDim * numAttentionHeads }

    enum CodingKeys: String, CodingKey {
        case attentionHeadDim = "attention_head_dim", numAttentionHeads = "num_attention_heads", numLayers = "num_layers"
        case numSingleLayers = "num_single_layers", inChannels = "in_channels", patchSize = "patch_size"
        case jointAttentionDim = "joint_attention_dim", mlpRatio = "mlp_ratio", axesDimsRope = "axes_dims_rope"
        case ropeTheta = "rope_theta", timestepGuidanceChannels = "timestep_guidance_channels", eps
        case guidanceEmbeds = "guidance_embeds"
    }

    static func decode(from data: Data) throws -> Flux2KleinTransformerConfig {
        let c = try JSONDecoder().decode(Flux2KleinTransformerConfig.self, from: data)
        let f = "transformer/config.json"
        try expect(f, "attention_head_dim", Flux2KleinDims.attentionHeadDim, c.attentionHeadDim)
        try expect(f, "num_attention_heads", Flux2KleinDims.attentionHeads, c.numAttentionHeads)
        try expect(f, "num_layers", Flux2KleinDims.doubleBlocks, c.numLayers)
        try expect(f, "num_single_layers", Flux2KleinDims.singleBlocks, c.numSingleLayers)
        try expect(f, "in_channels", Flux2KleinDims.transformerInChannels, c.inChannels)
        try expect(f, "patch_size", 1, c.patchSize)
        try expect(f, "joint_attention_dim", Flux2KleinDims.jointAttentionDim, c.jointAttentionDim)
        try expect(f, "mlp_ratio", 3.0, c.mlpRatio)
        try expect(f, "axes_dims_rope", Flux2KleinDims.axesDimsRope, c.axesDimsRope)
        try expect(f, "rope_theta", Flux2KleinDims.ropeTheta, c.ropeTheta)
        try expect(f, "timestep_guidance_channels", Flux2KleinDims.timestepGuidanceChannels, c.timestepGuidanceChannels)
        try expect(f, "guidance_embeds", false, c.guidanceEmbeds)
        // The four rope axes must exactly cover one attention head.
        try expect(f, "sum(axes_dims_rope)", c.attentionHeadDim, c.axesDimsRope.reduce(0, +))
        return c
    }
}

nonisolated struct Flux2KleinTextEncoderConfig: Decodable, Equatable, Sendable {
    let hiddenSize: Int
    let intermediateSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let vocabSize: Int
    let rmsNormEps: Double
    let ropeTheta: Double
    let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", intermediateSize = "intermediate_size", numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads", numKeyValueHeads = "num_key_value_heads", headDim = "head_dim"
        case vocabSize = "vocab_size", rmsNormEps = "rms_norm_eps", ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    static func decode(from data: Data) throws -> Flux2KleinTextEncoderConfig {
        let c = try JSONDecoder().decode(Flux2KleinTextEncoderConfig.self, from: data)
        let f = "text_encoder/config.json"
        try expect(f, "hidden_size", Flux2KleinDims.textHidden, c.hiddenSize)
        try expect(f, "intermediate_size", Flux2KleinDims.textIntermediate, c.intermediateSize)
        try expect(f, "num_hidden_layers", Flux2KleinDims.textLayers, c.numHiddenLayers)
        try expect(f, "num_attention_heads", Flux2KleinDims.textHeads, c.numAttentionHeads)
        try expect(f, "num_key_value_heads", Flux2KleinDims.textKVHeads, c.numKeyValueHeads)
        try expect(f, "head_dim", Flux2KleinDims.textHeadDim, c.headDim)
        try expect(f, "vocab_size", Flux2KleinDims.textVocab, c.vocabSize)
        try expect(f, "rms_norm_eps", 1e-6, c.rmsNormEps)
        try expect(f, "rope_theta", 1_000_000.0, c.ropeTheta)
        try expect(f, "tie_word_embeddings", true, c.tieWordEmbeddings)
        // The DiT's text input is the concatenation of the tapped layers.
        try expect(f, "hidden_size x taps (vs joint_attention_dim)", Flux2KleinDims.jointAttentionDim,
                   c.hiddenSize * Flux2KleinDims.tapLayers.count)
        return c
    }
}

nonisolated struct Flux2KleinVAEConfig: Decodable, Equatable, Sendable {
    let inChannels: Int
    let outChannels: Int
    let latentChannels: Int
    let blockOutChannels: [Int]
    let layersPerBlock: Int
    let normNumGroups: Int
    let patchSize: [Int]
    let batchNormEps: Double
    let useQuantConv: Bool
    let usePostQuantConv: Bool
    let midBlockAddAttention: Bool

    enum CodingKeys: String, CodingKey {
        case inChannels = "in_channels", outChannels = "out_channels", latentChannels = "latent_channels"
        case blockOutChannels = "block_out_channels", layersPerBlock = "layers_per_block", normNumGroups = "norm_num_groups"
        case patchSize = "patch_size", batchNormEps = "batch_norm_eps", useQuantConv = "use_quant_conv"
        case usePostQuantConv = "use_post_quant_conv", midBlockAddAttention = "mid_block_add_attention"
    }

    static func decode(from data: Data) throws -> Flux2KleinVAEConfig {
        let c = try JSONDecoder().decode(Flux2KleinVAEConfig.self, from: data)
        let f = "vae/config.json"
        try expect(f, "in_channels", 3, c.inChannels)
        try expect(f, "out_channels", 3, c.outChannels)
        try expect(f, "latent_channels", Flux2KleinDims.latentChannels, c.latentChannels)
        try expect(f, "block_out_channels", [128, 256, 512, 512], c.blockOutChannels)
        try expect(f, "layers_per_block", 2, c.layersPerBlock)
        try expect(f, "norm_num_groups", 32, c.normNumGroups)
        try expect(f, "patch_size", [Flux2KleinDims.vaePatch, Flux2KleinDims.vaePatch], c.patchSize)
        try expect(f, "batch_norm_eps", Flux2KleinDims.vaeBatchNormEps, c.batchNormEps)
        try expect(f, "use_quant_conv", true, c.useQuantConv)
        try expect(f, "use_post_quant_conv", true, c.usePostQuantConv)
        try expect(f, "mid_block_add_attention", true, c.midBlockAddAttention)
        // The transformer's input channels are the VAE latent x patch area.
        try expect(f, "latent_channels x patch area (vs transformer in_channels)", Flux2KleinDims.transformerInChannels,
                   c.latentChannels * c.patchSize.reduce(1, *))
        return c
    }
}

nonisolated struct Flux2KleinSchedulerConfig: Decodable, Equatable, Sendable {
    let numTrainTimesteps: Int
    let shift: Double
    let useDynamicShifting: Bool
    let timeShiftType: String
    let baseShift: Double
    let maxShift: Double
    let baseImageSeqLen: Int
    let maxImageSeqLen: Int
    let invertSigmas: Bool
    let stochasticSampling: Bool

    enum CodingKeys: String, CodingKey {
        case numTrainTimesteps = "num_train_timesteps", shift, useDynamicShifting = "use_dynamic_shifting"
        case timeShiftType = "time_shift_type", baseShift = "base_shift", maxShift = "max_shift"
        case baseImageSeqLen = "base_image_seq_len", maxImageSeqLen = "max_image_seq_len"
        case invertSigmas = "invert_sigmas", stochasticSampling = "stochastic_sampling"
    }

    static func decode(from data: Data) throws -> Flux2KleinSchedulerConfig {
        let c = try JSONDecoder().decode(Flux2KleinSchedulerConfig.self, from: data)
        let f = "scheduler/scheduler_config.json"
        try expect(f, "num_train_timesteps", 1000, c.numTrainTimesteps)
        try expect(f, "use_dynamic_shifting", true, c.useDynamicShifting)
        try expect(f, "time_shift_type", "exponential", c.timeShiftType)
        try expect(f, "invert_sigmas", false, c.invertSigmas)
        try expect(f, "stochastic_sampling", false, c.stochasticSampling)
        return c
    }
}

private nonisolated func expect<T: Equatable & CustomStringConvertible>(_ file: String, _ field: String, _ expected: T, _ actual: T) throws {
    guard expected == actual else {
        throw Flux2KleinConfigError.unexpected(file: file, field: field, expected: "\(expected)", actual: "\(actual)")
    }
}
