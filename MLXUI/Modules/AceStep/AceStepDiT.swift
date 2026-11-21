import Foundation
import MLX
import MLXFast
import MLXNN

// MC5-3f: the ACE-Step DiT decoder (`AceStepDiTModel`, `modeling_acestep_v15_turbo.py`, ACE-Step/Ace-Step1.5 @ 19671f4): 24 layers of
// AdaLN self-attention (sliding-window or full by layer type), cross-attention to the projected condition, gated MLP, patch-2 in/out.
// Weights are the 3c `AceStepWeights.loadDecoder` layout (`proj_in.weight` [out, K, in], `proj_out.weight` [out, K, in]).
// Like upstream, no padding masks are used: the DiT input is never padded (beyond the patch-multiple zero pad) and the encoder mask is
// all-ones on this path, so the self-attention masks are the plain bidirectional full / sliding-window masks and cross-attention is unmasked.

nonisolated struct AceStepDiTConfig: Sendable, Equatable {
    var hiddenSize = 2048
    var intermediateSize = 6144
    var attentionHeads = 16
    var kvHeads = 8
    var headDim = 128
    var layers = 24
    var inChannels = 192
    var outChannels = 64
    var patchSize = 2
    var rmsNormEps: Float = 1e-6
    var ropeTheta: Float = 1_000_000
    var slidingWindow = 128
    var useSlidingWindow = true
    /// `true` = sliding attention, per layer index.
    var slidingLayers: [Bool] = (0..<24).map { $0 % 2 == 0 }

    init() {}

    init(json: Data) throws {
        guard let o = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw AceStepWeightsError.badHeader("config", "not a JSON object")
        }
        func int(_ k: String) throws -> Int {
            guard let v = (o[k] as? NSNumber)?.intValue else { throw AceStepWeightsError.badHeader("config", "missing '\(k)'") }
            return v
        }
        hiddenSize = try int("hidden_size")
        intermediateSize = try int("intermediate_size")
        attentionHeads = try int("num_attention_heads")
        kvHeads = try int("num_key_value_heads")
        headDim = try int("head_dim")
        layers = try int("num_hidden_layers")
        inChannels = try int("in_channels")
        outChannels = try int("audio_acoustic_hidden_dim")
        patchSize = try int("patch_size")
        slidingWindow = try int("sliding_window")
        rmsNormEps = (o["rms_norm_eps"] as? NSNumber)?.floatValue ?? 1e-6
        ropeTheta = (o["rope_theta"] as? NSNumber)?.floatValue ?? 1_000_000
        useSlidingWindow = (o["use_sliding_window"] as? NSNumber)?.boolValue ?? true
        guard let types = o["layer_types"] as? [String], types.count >= layers else {
            throw AceStepWeightsError.badHeader("config", "missing or short 'layer_types'")
        }
        slidingLayers = types.map { $0 == "sliding_attention" }
    }
}

// MARK: - Layers

/// Sinusoidal timestep embedding (t x 1000, 256 wide: cos then sin) -> MLP -> `(temb [B, H], proj [B, 6, H])`.
nonisolated final class AceStepTimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    @ModuleInfo(key: "time_proj") var timeProj: Linear
    static let inChannels = 256, scale: Float = 1000, maxPeriod: Float = 10_000

    init(hidden: Int) {
        _linear1.wrappedValue = Linear(Self.inChannels, hidden, bias: true)
        _linear2.wrappedValue = Linear(hidden, hidden, bias: true)
        _timeProj.wrappedValue = Linear(hidden, hidden * 6, bias: true)
        super.init()
    }

    static func sinusoid(_ t: MLXArray) -> MLXArray {
        let half = inChannels / 2
        let freqs = exp(-log(maxPeriod) * MLXArray(Array(0 ..< half).map { Float($0) }) / Float(half))
        let args = (t.asType(.float32) * scale).expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        return concatenated([cos(args), sin(args)], axis: -1)
    }

    func callAsFunction(_ t: MLXArray) -> (MLXArray, MLXArray) {
        let temb = linear2(silu(linear1(Self.sinusoid(t))))
        let proj = timeProj(silu(temb))
        return (temb, proj.reshaped(proj.dim(0), 6, -1))
    }
}

nonisolated final class AceStepDiTAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let heads: Int, kvHeads: Int, headDim: Int, scale: Float, rope: RoPE?

    init(_ c: AceStepDiTConfig, cross: Bool) {
        heads = c.attentionHeads; kvHeads = c.kvHeads; headDim = c.headDim
        scale = Float(pow(Double(c.headDim), -0.5))
        _qProj.wrappedValue = Linear(c.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.rmsNormEps)
        rope = cross ? nil : RoPE(dimensions: headDim, traditional: false, base: c.ropeTheta)
        super.init()
    }

    /// Self-attention when `context` is nil (RoPE on q and k, positions 0 ..< L), cross-attention otherwise (no RoPE).
    /// `mask`: additive `[1, 1, L, L]` or nil for none.
    func callAsFunction(_ x: MLXArray, context: MLXArray? = nil, mask: MLXArray? = nil) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let source = context ?? x
        let S = source.dim(1)
        var q = qNorm(qProj(x).reshaped(B, L, heads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(source).reshaped(B, S, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(source).reshaped(B, S, kvHeads, headDim).transposed(0, 2, 1, 3)
        if let rope { q = rope(q); k = rope(k) }
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = mask.map { .array($0) } ?? .none
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: maskMode)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, heads * headDim))
    }
}

nonisolated final class AceStepDiTLayer: Module {
    @ModuleInfo(key: "self_attn_norm") var selfNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: AceStepDiTAttention
    @ModuleInfo(key: "cross_attn_norm") var crossNorm: RMSNorm
    @ModuleInfo(key: "cross_attn") var crossAttn: AceStepDiTAttention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: RMSNorm
    let mlp: AceStepEncoderMLP
    @ParameterInfo(key: "scale_shift_table") var table: MLXArray
    let sliding: Bool

    init(_ c: AceStepDiTConfig, sliding: Bool) {
        self.sliding = sliding
        _selfNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _selfAttn.wrappedValue = AceStepDiTAttention(c, cross: false)
        _crossNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _crossAttn.wrappedValue = AceStepDiTAttention(c, cross: true)
        _mlpNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        var m = AceStepConditionConfig()
        m.hiddenSize = c.hiddenSize; m.intermediateSize = c.intermediateSize
        mlp = AceStepEncoderMLP(m)
        _table.wrappedValue = MLXArray.zeros([1, 6, c.hiddenSize])
        super.init()
    }

    /// `proj` `[B, 6, H]` (timestep projection). Shift / scale / gate for the attention, then for the MLP.
    func callAsFunction(_ x: MLXArray, proj: MLXArray, context: MLXArray, selfMask: MLXArray?) -> MLXArray {
        let parts = split(table + proj, parts: 6, axis: 1)                                     // 6 x [B, 1, H]
        var h = x
        h = h + selfAttn(selfNorm(h) * (1 + parts[1]) + parts[0], mask: selfMask) * parts[2]
        h = h + crossAttn(crossNorm(h), context: context)
        return h + mlp(mlpNorm(h) * (1 + parts[4]) + parts[3]) * parts[5]
    }
}

/// `proj_in`: Conv1d(kernel = stride = patch) as a patch matmul. `weight` `[out, K, in]`: patch `[B, T', K x in]` @ `[K x in, out]`.
nonisolated final class AceStepPatchIn: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    init(inChannels: Int, outChannels: Int, patch: Int) {
        _weight.wrappedValue = MLXArray.zeros([outChannels, patch, inChannels])
        _bias.wrappedValue = MLXArray.zeros([outChannels])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, T, C) = (x.dim(0), x.dim(1), x.dim(2))
        let (O, K) = (weight.dim(0), weight.dim(1))
        let patches = x.reshaped(B, T / K, K * C)
        return matmul(patches, weight.reshaped(O, K * C).transposed()) + bias
    }
}

/// `proj_out`: ConvTranspose1d(kernel = stride = patch) as a matmul + interleave. `weight` `[out, K, in]`.
nonisolated final class AceStepPatchOut: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    init(inChannels: Int, outChannels: Int, patch: Int) {
        _weight.wrappedValue = MLXArray.zeros([outChannels, patch, inChannels])
        _bias.wrappedValue = MLXArray.zeros([outChannels])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, T) = (x.dim(0), x.dim(1))
        let (O, K, I) = (weight.dim(0), weight.dim(1), weight.dim(2))
        let y = matmul(x, weight.reshaped(O * K, I).transposed())                              // [B, T, O x K]
        return y.reshaped(B, T, O, K).transposed(0, 1, 3, 2).reshaped(B, T * K, O) + bias
    }
}

// MARK: - Model

nonisolated final class AceStepDiT: Module, @unchecked Sendable {
    let config: AceStepDiTConfig
    let layers: [AceStepDiTLayer]
    @ModuleInfo(key: "proj_in") var projIn: AceStepPatchIn
    @ModuleInfo(key: "proj_out") var projOut: AceStepPatchOut
    @ModuleInfo(key: "time_embed") var timeEmbed: AceStepTimestepEmbedding
    @ModuleInfo(key: "time_embed_r") var timeEmbedR: AceStepTimestepEmbedding
    @ModuleInfo(key: "condition_embedder") var conditionEmbedder: Linear
    @ModuleInfo(key: "norm_out") var normOut: RMSNorm
    @ParameterInfo(key: "scale_shift_table") var table: MLXArray

    init(config c: AceStepDiTConfig) {
        config = c
        layers = (0 ..< c.layers).map { AceStepDiTLayer(c, sliding: c.slidingLayers[$0]) }
        _projIn.wrappedValue = AceStepPatchIn(inChannels: c.inChannels, outChannels: c.hiddenSize, patch: c.patchSize)
        _projOut.wrappedValue = AceStepPatchOut(inChannels: c.hiddenSize, outChannels: c.outChannels, patch: c.patchSize)
        _timeEmbed.wrappedValue = AceStepTimestepEmbedding(hidden: c.hiddenSize)
        _timeEmbedR.wrappedValue = AceStepTimestepEmbedding(hidden: c.hiddenSize)
        _conditionEmbedder.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        _normOut.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _table.wrappedValue = MLXArray.zeros([1, 2, c.hiddenSize])
        super.init()
    }

    /// The names (as `AceStepWeights.loadDecoder` returns them) a checkpoint must carry, and nothing else.
    static func expectedParameterNames(_ c: AceStepDiTConfig) -> [String] {
        var names = ["condition_embedder.weight", "condition_embedder.bias", "norm_out.weight", "scale_shift_table",
                     "proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias"]
        for p in ["time_embed", "time_embed_r"] {
            for l in ["linear_1", "linear_2", "time_proj"] { names += ["\(p).\(l).weight", "\(p).\(l).bias"] }
        }
        for i in 0 ..< c.layers {
            let l = "layers.\(i)"
            names += ["\(l).scale_shift_table", "\(l).self_attn_norm.weight", "\(l).cross_attn_norm.weight", "\(l).mlp_norm.weight"]
            for a in ["self_attn", "cross_attn"] {
                names += ["q_proj", "k_proj", "v_proj", "o_proj"].map { "\(l).\(a).\($0).weight" }
                names += ["\(l).\(a).q_norm.weight", "\(l).\(a).k_norm.weight"]
            }
            names += ["gate_proj", "up_proj", "down_proj"].map { "\(l).mlp.\($0).weight" }
        }
        return names
    }

    /// Strict: every expected tensor present, none left over (the first offender is named).
    static func load(config: AceStepDiTConfig, weights: [String: MLXArray]) throws -> AceStepDiT {
        try AceStepWeights.checkStrict(actual: weights.keys, expected: expectedParameterNames(config))
        let net = AceStepDiT(config: config)
        try net.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(net)
        return net
    }

    /// One velocity prediction. `hidden` `[B, T, 64]`, `context` `[B, T, 128]` (context_latents), `encoder` `[B, L, H]`
    /// (encoder_hidden_states, all tokens valid), `timestep` / `timestepR` `[B]`. Returns `[B, T, 64]`.
    func callAsFunction(hidden: MLXArray, context: MLXArray, encoder: MLXArray, timestep: MLXArray, timestepR: MLXArray) -> MLXArray {
        let (tembT, projT) = timeEmbed(timestep)
        let (tembR, projR) = timeEmbedR(timestep - timestepR)
        let temb = tembT + tembR, proj = projT + projR

        var x = concatenated([context, hidden], axis: -1)                                     // [B, T, 192]
        let original = x.dim(1)
        let pad = (config.patchSize - original % config.patchSize) % config.patchSize
        if pad > 0 { x = padded(x, widths: [IntOrPair(0), IntOrPair((0, pad)), IntOrPair(0)]) }
        var h = projIn(x)
        let cond = conditionEmbedder(encoder)

        let L = h.dim(1)
        let sliding = config.useSlidingWindow
            ? AceStepMask.bidirectional(length: L, keyMask: nil, window: config.slidingWindow, dtype: h.dtype) : nil
        for layer in layers {
            h = layer(h, proj: proj, context: cond, selfMask: layer.sliding ? sliding : nil)
        }
        let parts = split(table + temb.expandedDimensions(axis: 1), parts: 2, axis: 1)         // shift, scale: [B, 1, H]
        h = normOut(h) * (1 + parts[1]) + parts[0]
        return projOut(h)[0..., 0 ..< original, 0...]
    }
}
