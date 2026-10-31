import Foundation
import MLX
import MLXNN

// FLUX.2 Klein MMDiT front end (MP7-3d): input embedders, the timestep MLP, the three shared modulation modules, the output AdaLayerNormContinuous and
// the rotary position embedding. Everything here sits outside the 5 double + 20 single blocks (3e / 3f).
// Structure follows VincentGourbin/flux-2-swift-mlx @ 315c27e (MIT, © 2025 Vincent Gourbin) `Flux2Transformer`, `Flux2Embeddings`, `Flux2Modulation`
// and `Flux2RoPE`; the numbers are checked against diffusers 0.37.1 `Flux2Transformer2DModel` (a tiny-config fp32 golden,
// MLXUITests/Fixtures/Flux2Klein/g3d_embedders.safetensors). diffusers is the arbiter wherever the two disagree.
//
// Keys are the diffusers snake_case names, so the checkpoint loads without a remap: `x_embedder.weight`, `context_embedder.weight`,
// `time_guidance_embed.timestep_embedder.linear_{1,2}.weight`, `{double_stream_modulation_img,double_stream_modulation_txt,single_stream_modulation}.linear.weight`,
// `norm_out.linear.weight`, `proj_out.weight`. Every Linear is bias-free.
//
// Timestep. The caller passes sigma in [0, 1]; it is multiplied by 1000 INSIDE `Flux2KleinTimeEmbed` (as diffusers does in the transformer's forward), the sinusoid
// is built in float32, ordered [cos, sin] (flip_sin_to_cos), downscale_freq_shift 0, max_period 10000.
//
// RoPE. Computed from the ids actually passed, every call. There is no cache: the reference keys its cache on the sequence length alone, so two different
// id layouts of equal length would alias (MP7-1 spike section 2.2).

/// The dimensions the front end is built from. `init(_:)` takes the validated real config; the memberwise init exists for the tiny test config.
nonisolated struct Flux2KleinEmbedderShape: Equatable, Sendable {
    var inChannels: Int
    var jointAttentionDim: Int
    var inner: Int
    var timestepChannels: Int
    var eps: Float
    var axesDims: [Int]
    var ropeTheta: Float

    init(_ c: Flux2KleinTransformerConfig) {
        self.init(inChannels: c.inChannels, jointAttentionDim: c.jointAttentionDim, inner: c.innerDim, timestepChannels: c.timestepGuidanceChannels,
                  eps: Float(c.eps), axesDims: c.axesDimsRope, ropeTheta: Float(c.ropeTheta))
    }

    init(inChannels: Int, jointAttentionDim: Int, inner: Int, timestepChannels: Int, eps: Float, axesDims: [Int], ropeTheta: Float) {
        self.inChannels = inChannels
        self.jointAttentionDim = jointAttentionDim
        self.inner = inner
        self.timestepChannels = timestepChannels
        self.eps = eps
        self.axesDims = axesDims
        self.ropeTheta = ropeTheta
    }
}

/// `Timesteps(num_channels, flip_sin_to_cos: true, downscale_freq_shift: 0)` then `TimestepEmbedding` (linear_1, SiLU, linear_2), no bias, no guidance branch.
nonisolated final class Flux2KleinTimestepEmbedder: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(channels: Int, inner: Int) {
        _linear1.wrappedValue = Linear(channels, inner, bias: false)
        _linear2.wrappedValue = Linear(inner, inner, bias: false)
        super.init()
    }

    func callAsFunction(_ projection: MLXArray) -> MLXArray { linear2(silu(linear1(projection))) }
}

/// `Flux2TimestepGuidanceEmbeddings` for a model without `guidance_embeds` (Klein): the timestep branch only.
nonisolated final class Flux2KleinTimeEmbed: Module {
    let channels: Int
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: Flux2KleinTimestepEmbedder

    init(channels: Int, inner: Int) {
        self.channels = channels
        _timestepEmbedder.wrappedValue = Flux2KleinTimestepEmbedder(channels: channels, inner: inner)
        super.init()
    }

    /// `get_timestep_embedding(t * 1000, channels, flip_sin_to_cos: true, downscale_freq_shift: 0)`: `[B] -> [B, channels]`, float32, `[cos, sin]`.
    static func projection(sigma: MLXArray, channels: Int) -> MLXArray {
        let half = channels / 2
        let t = sigma.asType(.float32).reshaped(-1) * 1000
        let exponent = MLXArray.arange(0, half).asType(.float32) * (-Float(log(10_000.0)) / Float(half))
        let angles = t.expandedDimensions(axis: 1) * exp(exponent).expandedDimensions(axis: 0)
        return concatenated([cos(angles), sin(angles)], axis: -1)
    }

    /// `sigma` is `[B]` in [0, 1]. The sinusoid is float32; it is cast to `dtype` (the model's compute dtype) before the MLP, as diffusers casts it to the timestep's dtype.
    func callAsFunction(sigma: MLXArray, dtype: DType = .float32) -> MLXArray {
        timestepEmbedder(Self.projection(sigma: sigma, channels: channels).asType(dtype))
    }
}

/// One `[shift, scale, gate]` set, each `[B, 1, inner]`.
nonisolated struct Flux2KleinModulationSet: @unchecked Sendable {
    let shift: MLXArray
    let scale: MLXArray
    let gate: MLXArray
}

/// `Flux2Modulation`: `linear(silu(temb))`, no bias, `sets` x (shift, scale, gate). Shared by every block; computed once per forward.
nonisolated final class Flux2KleinModulation: Module {
    let sets: Int
    let inner: Int
    @ModuleInfo(key: "linear") var linear: Linear

    init(inner: Int, sets: Int) {
        self.sets = sets
        self.inner = inner
        _linear.wrappedValue = Linear(inner, inner * 3 * sets, bias: false)
        super.init()
    }

    /// `[B, inner] -> [B, 3 * sets * inner]`.
    func callAsFunction(_ temb: MLXArray) -> MLXArray { linear(silu(temb)) }

    /// `Flux2Modulation.split`: a `[B, 3 * sets * inner]` (or `[B, 1, ...]`) tensor into `sets` triples, each piece `[B, 1, inner]`.
    func split(_ mod: MLXArray) -> [Flux2KleinModulationSet] {
        let m = mod.ndim == 2 ? mod.expandedDimensions(axis: 1) : mod
        let parts = MLX.split(m, parts: 3 * sets, axis: -1)
        return (0 ..< sets).map { Flux2KleinModulationSet(shift: parts[3 * $0], scale: parts[3 * $0 + 1], gate: parts[3 * $0 + 2]) }
    }

    func chunks(_ temb: MLXArray) -> [Flux2KleinModulationSet] { split(callAsFunction(temb)) }
}

/// `AdaLayerNormContinuous(inner, inner, elementwise_affine: false, eps, bias: false)`: `linear(silu(temb))` split (scale, shift), then `norm(x) * (1 + scale) + shift`.
nonisolated final class Flux2KleinAdaLayerNormContinuous: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    let norm: LayerNorm

    init(inner: Int, eps: Float) {
        _linear.wrappedValue = Linear(inner, inner * 2, bias: false)
        norm = LayerNorm(dimensions: inner, eps: eps, affine: false)
        super.init()
    }

    /// `x` is `[B, S, inner]`, `temb` is `[B, inner]`.
    func callAsFunction(_ x: MLXArray, temb: MLXArray) -> MLXArray {
        let emb = linear(silu(temb).asType(x.dtype))
        let parts = MLX.split(emb, parts: 2, axis: -1)       // scale first, then shift
        let scale = parts[0].expandedDimensions(axis: 1)
        let shift = parts[1].expandedDimensions(axis: 1)
        return norm(x) * (1 + scale) + shift
    }
}

/// Everything in `Flux2Transformer2DModel` that is not a block, under the diffusers keys. 3f owns the blocks and will hold this (or its parts) as children.
nonisolated final class Flux2KleinEmbedders: Module {
    let shape: Flux2KleinEmbedderShape
    @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "time_guidance_embed") var timeGuidanceEmbed: Flux2KleinTimeEmbed
    @ModuleInfo(key: "double_stream_modulation_img") var doubleModulationImg: Flux2KleinModulation
    @ModuleInfo(key: "double_stream_modulation_txt") var doubleModulationTxt: Flux2KleinModulation
    @ModuleInfo(key: "single_stream_modulation") var singleModulation: Flux2KleinModulation
    @ModuleInfo(key: "norm_out") var normOut: Flux2KleinAdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(_ s: Flux2KleinEmbedderShape) {
        shape = s
        _xEmbedder.wrappedValue = Linear(s.inChannels, s.inner, bias: false)
        _contextEmbedder.wrappedValue = Linear(s.jointAttentionDim, s.inner, bias: false)
        _timeGuidanceEmbed.wrappedValue = Flux2KleinTimeEmbed(channels: s.timestepChannels, inner: s.inner)
        _doubleModulationImg.wrappedValue = Flux2KleinModulation(inner: s.inner, sets: 2)
        _doubleModulationTxt.wrappedValue = Flux2KleinModulation(inner: s.inner, sets: 2)
        _singleModulation.wrappedValue = Flux2KleinModulation(inner: s.inner, sets: 1)
        _normOut.wrappedValue = Flux2KleinAdaLayerNormContinuous(inner: s.inner, eps: s.eps)
        _projOut.wrappedValue = Linear(s.inner, s.inChannels, bias: false)
        super.init()
    }

    /// The per-forward products the blocks consume.
    struct Conditioning: @unchecked Sendable {
        let temb: MLXArray
        let doubleImg: [Flux2KleinModulationSet]    // 2 sets
        let doubleTxt: [Flux2KleinModulationSet]    // 2 sets
        let single: [Flux2KleinModulationSet]       // 1 set
    }

    /// `sigma` is `[B]` in [0, 1] (multiplied by 1000 inside).
    func conditioning(sigma: MLXArray, dtype: DType = .float32) -> Conditioning {
        let temb = timeGuidanceEmbed(sigma: sigma, dtype: dtype)
        return Conditioning(temb: temb, doubleImg: doubleModulationImg.chunks(temb), doubleTxt: doubleModulationTxt.chunks(temb),
                            single: singleModulation.chunks(temb))
    }

    /// Final layers: `proj_out(norm_out(x, temb))`.
    func finalLayer(_ x: MLXArray, temb: MLXArray) -> MLXArray { projOut(normOut(x, temb: temb)) }
}

/// Rotary position embedding for the four id axes `[T, H, W, L]`, computed from the ids passed (never cached by length).
///
/// Verdict (MP7-3d, measured, see MP7-3d-report.md): `FluxLayers.embedND` already produces the same per-axis angles, so cos/sin are derived from it rather than
/// re-implemented; the rotation itself is `applyRotary`, in the `[B, S, H, D]` layout the Klein attention uses.
///
/// Owner ruling (Boo, 2026-10-31: "RoPE fp32 gap accepted, no change"): the cos/sin table is built in float32 on the GPU, which differs from a float64 reference by
/// about 2.4e-5 at positions near 500. That gap is accepted as owner-ruled. It is roughly 300x below bf16 resolution (~7.8e-3), and diffusers also computes this
/// table in float32 on Apple GPUs. No logic change; MP7-3d-FIX-1 pins it at the real positions (Flux2KleinRoPERealPositionTests, floor 1e-4).
nonisolated enum Flux2KleinRoPE {
    /// `cos`, `sin` as `[S, sum(axesDims)]`, each axis repeat-interleaved by 2 and the axes concatenated (diffusers `Flux2PosEmbed`). `ids` is `[S, axes]` (or `[1, S, axes]`).
    static func cosSin(ids: MLXArray, axesDims: [Int], theta: Float) -> (cos: MLXArray, sin: MLXArray) {
        let flat = ids.ndim == 3 ? ids[0] : ids
        let pos = flat.asType(.float32).expandedDimensions(axis: 0)                          // [1, S, axes]
        let pe = FluxLayers.embedND(ids: pos, dim: axesDims.reduce(0, +), theta: theta, axesDim: axesDims)   // [1, 1, S, half, 2, 2] = [[cos, -sin], [sin, cos]]
        let c = pe[0, 0, 0..., 0..., 0, 0]                                                    // [S, half]
        let s = pe[0, 0, 0..., 0..., 1, 0]
        let n = c.dim(0), half = c.dim(1)
        return (stacked([c, c], axis: -1).reshaped(n, 2 * half), stacked([s, s], axis: -1).reshaped(n, 2 * half))
    }

    /// `cos`, `sin` for the joint sequence the blocks see: text rows first, then image rows (`torch.cat([text, image])`).
    static func jointCosSin(textIDs: MLXArray, imageIDs: MLXArray, axesDims: [Int], theta: Float) -> (cos: MLXArray, sin: MLXArray) {
        let ids = concatenated([textIDs.ndim == 3 ? textIDs[0] : textIDs, imageIDs.ndim == 3 ? imageIDs[0] : imageIDs], axis: 0)
        return cosSin(ids: ids, axesDims: axesDims, theta: theta)
    }

    /// diffusers `apply_rotary_emb(x, (cos, sin), sequence_dim=1)`: `x` is `[B, S, H, D]`, `cos`/`sin` are `[S, D]`.
    /// Consecutive pairs rotate `(x_e, x_o) -> (x_e c - x_o s, x_o c + x_e s)`. Computed in float32, returned in `x`'s dtype.
    static func applyRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let shape = x.shape
        let xf = x.asType(.float32)
        let pairs = xf.reshaped(Array(shape.dropLast()) + [shape[shape.count - 1] / 2, 2])
        let rotated = stacked([-pairs[.ellipsis, 1], pairs[.ellipsis, 0]], axis: -1).reshaped(shape)
        let c = cos.asType(.float32).expandedDimensions(axes: [0, 2])
        let s = sin.asType(.float32).expandedDimensions(axes: [0, 2])
        return (xf * c + rotated * s).asType(x.dtype)
    }
}
