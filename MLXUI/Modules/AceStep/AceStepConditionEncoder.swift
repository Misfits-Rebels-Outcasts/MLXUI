import Foundation
import MLX
import MLXFast
import MLXNN

// MC5-3e: the ACE-Step condition encoder on the no-LM text2music path. Upstream is the pinned remote-code class
// `AceStepConditionEncoder` (`modeling_acestep_v15_turbo.py`, ACE-Step/Ace-Step1.5 @ 19671f4): `text_projector`, `AceStepLyricEncoder`
// (8 layers), `AceStepTimbreEncoder` (4 layers) and `pack_sequences` (twice). The FSQ tokenizer / detokenizer / attention pooler are NOT
// ported: on this path their output is discarded (readme T16). Parity: `AceStepConditionEncoderTests` against upstream's own tiny-config output.

// MARK: - Config

nonisolated struct AceStepConditionConfig: Sendable, Equatable {
    var hiddenSize = 2048
    var intermediateSize = 6144
    var attentionHeads = 16
    var kvHeads = 8
    var headDim = 128
    var textHiddenDim = 1024
    var timbreHiddenDim = 64
    var lyricLayers = 8
    var timbreLayers = 4
    var rmsNormEps: Float = 1e-6
    var ropeTheta: Float = 1_000_000
    var slidingWindow = 128
    var useSlidingWindow = true
    /// `true` = sliding attention, per layer index (shared by the lyric and timbre encoders: layer `i` of either uses `layer_types[i]`).
    var slidingLayers: [Bool] = (0..<24).map { $0 % 2 == 0 }

    init() {}

    /// From an `AceStepConfig` JSON (`config.json` of the turbo folder, or the tiny fixture's metadata).
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
        textHiddenDim = try int("text_hidden_dim")
        timbreHiddenDim = try int("timbre_hidden_dim")
        lyricLayers = try int("num_lyric_encoder_hidden_layers")
        timbreLayers = try int("num_timbre_encoder_hidden_layers")
        slidingWindow = try int("sliding_window")
        rmsNormEps = (o["rms_norm_eps"] as? NSNumber)?.floatValue ?? 1e-6
        ropeTheta = (o["rope_theta"] as? NSNumber)?.floatValue ?? 1_000_000
        useSlidingWindow = (o["use_sliding_window"] as? NSNumber)?.boolValue ?? true
        guard let types = o["layer_types"] as? [String] else { throw AceStepWeightsError.badHeader("config", "missing 'layer_types'") }
        slidingLayers = types.map { $0 == "sliding_attention" }
        guard slidingLayers.count >= max(lyricLayers, timbreLayers) else {
            throw AceStepWeightsError.badHeader("config", "layer_types shorter than the encoders")
        }
    }
}

// MARK: - Masks

nonisolated enum AceStepMask {
    /// The most negative finite value of `dtype` (`torch.finfo(dtype).min`); upstream adds this, never `-inf`, so a fully masked row softmaxes
    /// to uniform instead of NaN.
    static func minValue(_ dtype: DType) -> Float {
        switch dtype {
        case .float16: return -65504
        case .bfloat16: return -3.3895314e38
        default: return -Float.greatestFiniteMagnitude
        }
    }

    /// `create_4d_mask(..., is_causal=False)`: additive `[B, 1, L, L]` mask. Keys outside `window` (|i-j| > window, when given) and padded
    /// keys (`keyMask` 0) get the minimum value, everything else 0.
    static func bidirectional(length L: Int, keyMask: MLXArray?, window: Int?, dtype: DType) -> MLXArray {
        let idx = MLXArray(Array(0 ..< L).map { Int32($0) })
        let diff = abs(idx.expandedDimensions(axis: 1) - idx.expandedDimensions(axis: 0))      // [L, L]
        var valid = MLXArray.ones([L, L], dtype: .bool)
        if let window { valid = valid & (diff .<= Int32(window)) }
        var valid4 = valid.expandedDimensions(axes: [0, 1])                                  // [1, 1, L, L]
        if let keyMask {
            let keys = keyMask.asType(.bool).reshaped(keyMask.dim(0), 1, 1, L)              // [B, 1, 1, L]
            valid4 = valid4 & keys
        }
        let zero = MLXArray(Float(0)), low = MLXArray(minValue(dtype))
        return MLX.where(valid4, zero, low).asType(dtype)
    }
}

// MARK: - Layers

nonisolated final class AceStepEncoderAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let heads: Int, kvHeads: Int, headDim: Int
    let scale: Float
    let rope: RoPE

    init(_ c: AceStepConditionConfig) {
        heads = c.attentionHeads; kvHeads = c.kvHeads; headDim = c.headDim
        scale = Float(pow(Double(c.headDim), -0.5))
        _qProj.wrappedValue = Linear(c.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.rmsNormEps)
        rope = RoPE(dimensions: headDim, traditional: false, base: c.ropeTheta)
        super.init()
    }

    /// Bidirectional self-attention with q/k RMSNorm per head, RoPE (positions 0 ..< L), GQA and an additive `[B, 1, L, L]` mask.
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(B, L, heads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(B, L, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = rope(q)
        k = rope(k)
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .array(mask))
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, heads * headDim))
    }
}

nonisolated final class AceStepEncoderMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ c: AceStepConditionConfig) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

nonisolated final class AceStepEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: AceStepEncoderAttention
    let mlp: AceStepEncoderMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    let sliding: Bool

    init(_ c: AceStepConditionConfig, sliding: Bool) {
        self.sliding = sliding
        _attention.wrappedValue = AceStepEncoderAttention(c)
        mlp = AceStepEncoderMLP(c)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let h = x + attention(inputNorm(x), mask: mask)
        return h + mlp(postNorm(h))
    }
}

/// The part the lyric and timbre encoders share: project, run the layers with the full or sliding mask by layer type, final norm.
nonisolated func aceStepRunEncoder(layers: [AceStepEncoderLayer], config c: AceStepConditionConfig, embedded: MLXArray,
                                   keyMask: MLXArray?) -> MLXArray {
    let L = embedded.dim(1)
    let full = AceStepMask.bidirectional(length: L, keyMask: keyMask, window: nil, dtype: embedded.dtype)
    let sliding = c.useSlidingWindow
        ? AceStepMask.bidirectional(length: L, keyMask: keyMask, window: c.slidingWindow, dtype: embedded.dtype) : nil
    var h = embedded
    for layer in layers { h = layer(h, mask: layer.sliding ? (sliding ?? full) : full) }
    return h
}

// MARK: - Lyric and timbre encoders

nonisolated final class AceStepLyricEncoder: Module {
    /// A Linear with bias (text_hidden -> hidden), not a token table.
    @ModuleInfo(key: "embed_tokens") var embedTokens: Linear
    let layers: [AceStepEncoderLayer]
    let norm: RMSNorm
    let config: AceStepConditionConfig

    init(_ c: AceStepConditionConfig) {
        config = c
        _embedTokens.wrappedValue = Linear(c.textHiddenDim, c.hiddenSize, bias: true)
        layers = (0 ..< c.lyricLayers).map { AceStepEncoderLayer(c, sliding: c.slidingLayers[$0]) }
        norm = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }

    /// `inputs` `[B, L, text_hidden]`, `mask` `[B, L]` (1 = real token) -> `[B, L, hidden]`.
    func callAsFunction(_ inputs: MLXArray, mask: MLXArray) -> MLXArray {
        norm(aceStepRunEncoder(layers: layers, config: config, embedded: embedTokens(inputs), keyMask: mask))
    }
}

nonisolated final class AceStepTimbreEncoder: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Linear
    let layers: [AceStepEncoderLayer]
    let norm: RMSNorm
    /// In the checkpoint, but upstream's forward has the prepend commented out: loaded (strict names), never used.
    @ParameterInfo(key: "special_token") var specialToken: MLXArray
    let config: AceStepConditionConfig

    init(_ c: AceStepConditionConfig) {
        config = c
        _embedTokens.wrappedValue = Linear(c.timbreHiddenDim, c.hiddenSize, bias: true)
        layers = (0 ..< c.timbreLayers).map { AceStepEncoderLayer(c, sliding: c.slidingLayers[$0]) }
        norm = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _specialToken.wrappedValue = MLXArray.zeros([1, 1, c.hiddenSize])
        super.init()
    }

    /// `referPacked` `[N, T, timbre_hidden]` (N reference clips), `order` `[N]` (the batch item each clip belongs to).
    /// Returns the unpacked `([B, maxCount, hidden], [B, maxCount])`. No padding mask is passed (as upstream): bidirectional full or
    /// sliding attention only; the embedding is the first position's output.
    func callAsFunction(_ referPacked: MLXArray, order: [Int]) -> (MLXArray, MLXArray) {
        let h = norm(aceStepRunEncoder(layers: layers, config: config, embedded: embedTokens(referPacked), keyMask: nil))
        let first = h[0..., 0, 0...]                                                           // [N, hidden]
        return Self.unpack(first, order: order)
    }

    /// `unpack_timbre_embeddings`: place each clip's embedding at (its batch item, its position among that item's clips), by one-hot matmul.
    static func unpack(_ packed: MLXArray, order: [Int]) -> (MLXArray, MLXArray) {
        let N = order.count, d = packed.dim(1)
        let B = (order.max() ?? -1) + 1
        var counts = [Int](repeating: 0, count: B)
        var position = [Int](repeating: 0, count: N)
        for (i, b) in order.enumerated() { position[i] = counts[b]; counts[b] += 1 }
        let maxCount = counts.max() ?? 0
        var oneHot = [Float](repeating: 0, count: N * B * maxCount)
        for i in 0 ..< N { oneHot[i * B * maxCount + order[i] * maxCount + position[i]] = 1 }
        let hot = MLXArray(oneHot, [N, B * maxCount]).asType(packed.dtype)
        let flat = matmul(hot.transposed(), packed)                                            // [B*maxCount, d]
        let mask = (hot.sum(axis: 0) .> 0).asType(.int32).reshaped(B, maxCount)
        return (flat.reshaped(B, maxCount, d), mask)
    }
}

/// The three parameter groups under upstream's names (`text_projector.weight`, `lyric_encoder.*`, `timbre_encoder.*`), so the checkpoint's
/// `encoder.*` keys (prefix stripped) install directly.
nonisolated final class AceStepConditionNetwork: Module {
    @ModuleInfo(key: "text_projector") var textProjector: Linear
    @ModuleInfo(key: "lyric_encoder") var lyricEncoder: AceStepLyricEncoder
    @ModuleInfo(key: "timbre_encoder") var timbreEncoder: AceStepTimbreEncoder

    init(_ c: AceStepConditionConfig) {
        _textProjector.wrappedValue = Linear(c.textHiddenDim, c.hiddenSize, bias: false)
        _lyricEncoder.wrappedValue = AceStepLyricEncoder(c)
        _timbreEncoder.wrappedValue = AceStepTimbreEncoder(c)
        super.init()
    }
}

// MARK: - Silence latent

nonisolated enum AceStepSilenceLatent {
    static let frames = 15_000, channels = 64
    /// The timbre input with no reference audio: the first 750 frames (30 s x 25 Hz). The literal 750, NOT `timbre_fix_frame` (dead config), T17.
    static let timbreFrames = 750

    /// The bundled raw f32 `[15000, 64]` resource (`MLXUI/Resources/AceStep/silence_latent_f32.bin`, R6).
    static func load(bundle: Bundle = .main) throws -> MLXArray {
        guard let url = bundle.url(forResource: "silence_latent_f32", withExtension: "bin") else {
            throw AceStepWeightsError.missingFile("silence_latent_f32.bin")
        }
        return try load(url: url)
    }

    static func load(url: URL) throws -> MLXArray {
        let data = try Data(contentsOf: url)
        guard data.count == frames * channels * 4 else { throw AceStepWeightsError.badHeader(url.lastPathComponent, "wrong size \(data.count)") }
        return MLXArray(data, [frames, channels], dtype: .float32)
    }

    /// `_get_silence_latent_slice`: exactly `length` frames, tiled when longer than the stored 15000.
    static func slice(_ silence: MLXArray, length: Int) -> MLXArray {
        if length <= silence.dim(0) { return silence[0 ..< length] }
        let repeats = (length + silence.dim(0) - 1) / silence.dim(0)
        return concatenated(Array(repeating: silence, count: repeats), axis: 0)[0 ..< length]
    }
}

// MARK: - Condition encoder

nonisolated final class AceStepConditionEncoder: @unchecked Sendable {
    let config: AceStepConditionConfig
    let network: AceStepConditionNetwork
    private let silenceLatent: MLXArray?
    private var cachedTimbre: (embedding: MLXArray, mask: MLXArray)?
    /// How many times the timbre encoder has actually run (the no-reference timbre is computed once and cached, T17).
    private(set) var timbreEvaluations = 0

    /// Names (without the `encoder.` prefix) every checkpoint must carry and nothing else: `text_projector.weight`, the lyric and timbre
    /// encoders. Computed from the config, so a layer count change cannot drift from the real 140-key set.
    static func expectedParameterNames(_ c: AceStepConditionConfig) -> [String] {
        var names = ["text_projector.weight"]
        func part(_ p: String, layers: Int) {
            names += ["\(p).embed_tokens.weight", "\(p).embed_tokens.bias", "\(p).norm.weight"]
            for i in 0 ..< layers {
                let l = "\(p).layers.\(i)"
                names += ["\(l).input_layernorm.weight", "\(l).post_attention_layernorm.weight"]
                names += ["q_proj", "k_proj", "v_proj", "o_proj"].map { "\(l).self_attn.\($0).weight" }
                names += ["\(l).self_attn.q_norm.weight", "\(l).self_attn.k_norm.weight"]
                names += ["gate_proj", "up_proj", "down_proj"].map { "\(l).mlp.\($0).weight" }
            }
        }
        part("lyric_encoder", layers: c.lyricLayers)
        part("timbre_encoder", layers: c.timbreLayers)
        names.append("timbre_encoder.special_token")
        return names
    }

    /// `weights` are the checkpoint's `encoder.*` tensors as `AceStepWeights.loadConditionEncoder` returns them (prefix included) or already
    /// stripped. Strict: the first missing or unused tensor is named.
    init(config: AceStepConditionConfig, weights: [String: MLXArray], silenceLatent: MLXArray? = nil) throws {
        self.config = config
        self.silenceLatent = silenceLatent
        var stripped: [String: MLXArray] = [:]
        for (k, v) in weights { stripped[k.hasPrefix("encoder.") ? String(k.dropFirst("encoder.".count)) : k] = v }
        try AceStepWeights.checkStrict(actual: stripped.keys, expected: Self.expectedParameterNames(config))
        let net = AceStepConditionNetwork(config)
        try net.update(parameters: ModuleParameters.unflattened(stripped), verify: [.all])
        eval(net)
        network = net
    }

    // MARK: Pieces

    func projectText(_ hidden: MLXArray) -> MLXArray { network.textProjector(hidden) }

    func encodeLyrics(_ embeds: MLXArray, mask: MLXArray) -> MLXArray { network.lyricEncoder(embeds, mask: mask) }

    func encodeTimbre(referPacked: MLXArray, order: [Int]) -> (MLXArray, MLXArray) {
        network.timbreEncoder(referPacked, order: order)
    }

    /// With no reference audio the timbre input is `silence_latent[:750]` as one clip with order `[0]`, so the output is the same every
    /// run: computed once and cached.
    func silenceTimbre() throws -> (embedding: MLXArray, mask: MLXArray) {
        if let cachedTimbre { return cachedTimbre }
        guard let silenceLatent else { throw AceStepWeightsError.missingFile("silence_latent_f32.bin") }
        let refer = AceStepSilenceLatent.slice(silenceLatent, length: AceStepSilenceLatent.timbreFrames)
            .expandedDimensions(axis: 0).asType(network.textProjector.weight.dtype)           // [1, 750, 64]
        let (emb, mask) = encodeTimbre(referPacked: refer, order: [0])
        eval(emb, mask)
        timbreEvaluations += 1
        cachedTimbre = (emb, mask)
        return (emb, mask)
    }

    /// `pack_sequences`: concatenate, then stable-sort so tokens with the larger mask value come first (ties keep their original order);
    /// the new mask is `arange(L) < number of valid tokens`.
    static func packSequences(_ h1: MLXArray, _ h2: MLXArray, mask1: MLXArray, mask2: MLXArray) throws -> (MLXArray, MLXArray) {
        let hidden = concatenated([h1, h2], axis: 1)                                          // [B, L, D]
        let mask = concatenated([mask1.asType(.int32), mask2.asType(.int32)], axis: 1)        // [B, L]
        let (B, L, D) = (hidden.dim(0), hidden.dim(1), hidden.dim(2))
        guard B > 0, L > 0, D > 0 else { throw AceStepEngineError.emptyTokens("tokens to pack") }
        let flat = try MLXErrorGuard.read(mask, as: Int32.self)
        var order: [Int32] = [], lengths: [Int32] = []
        for b in 0 ..< B {
            let row = Array(flat[b * L ..< (b + 1) * L])
            order += row.indices.sorted { row[$0] != row[$1] ? row[$0] > row[$1] : $0 < $1 }.map { Int32($0) }
            lengths.append(row.reduce(0) { $0 + $1 })
        }
        let idx = MLXArray(order, [B, L]).expandedDimensions(axis: 2)
        let gathered = takeAlong(hidden, broadcast(idx, to: [B, L, D]), axis: 1)
        let positions = MLXArray(Array(0 ..< L).map { Int32($0) }).expandedDimensions(axis: 0)
        let newMask = (positions .< MLXArray(lengths, [B, 1])).asType(.int32)
        return (gathered, newMask)
    }

    // MARK: Full forward

    /// `AceStepConditionEncoder.forward` for batch 1 on the no-reference path: text projected, lyrics encoded, the cached silence timbre, then
    /// lyrics + timbre packed, then that + text packed. Returns `(encoder_hidden_states [1, L, hidden], encoder_attention_mask [1, L])`.
    func encode(_ inputs: AceStepTextInputs) throws -> (hidden: MLXArray, mask: MLXArray) {
        let text = projectText(inputs.textHidden)
        let lyric = encodeLyrics(inputs.lyricEmbeds, mask: inputs.lyricMask)
        let timbre = try silenceTimbre()
        return try Self.packBoth(lyric: lyric, lyricMask: inputs.lyricMask, timbre: timbre.embedding, timbreMask: timbre.mask,
                             text: text, textMask: inputs.textMask)
    }

    static func packBoth(lyric: MLXArray, lyricMask: MLXArray, timbre: MLXArray, timbreMask: MLXArray,
                         text: MLXArray, textMask: MLXArray) throws -> (hidden: MLXArray, mask: MLXArray) {
        let (h1, m1) = try packSequences(lyric, timbre, mask1: lyricMask, mask2: timbreMask)
        return try packSequences(h1, text, mask1: m1, mask2: textMask)
    }

    // MARK: context_latents

    /// The latent length upstream uses for a silent target of `duration` seconds at 48 kHz: `int(duration * 48000) // 1920` frames
    /// (25 per second), at least 128 (`max_latent_length = max(128, ...)`).
    static func latentFrames(duration: Double) -> Int { max(128, Int(duration * 48_000) / 1920) }

    /// `[1, T, 128]` = `concat([silence_latent[:T] (64 ch), ones (64 ch chunk mask)], axis: -1)` for text2music.
    func contextLatents(duration: Double) throws -> MLXArray { try contextLatents(frames: Self.latentFrames(duration: duration)) }

    /// The same for an explicit frame count `T` (the engine passes the noise's own length, so the two always agree).
    func contextLatents(frames T: Int) throws -> MLXArray {
        guard let silenceLatent else { throw AceStepWeightsError.missingFile("silence_latent_f32.bin") }
        let src = AceStepSilenceLatent.slice(silenceLatent, length: T)
        let chunk = MLXArray.ones([T, src.dim(1)], dtype: src.dtype)
        return concatenated([src, chunk], axis: -1).expandedDimensions(axis: 0)
    }
}
