import Foundation
import MLX
import MLXNN
import MLXFast

/// The FLUX MMDiT transformer for `mlx-community/Flux-1.lite-8B-MLX-Q4`, ported from
/// **mflux** (`src/mflux/models/flux/model/flux_transformer/`) — the format the installed
/// checkpoint actually ships (mflux 0.7.1, 4-bit affine-quantized, diffusers-style keys like
/// `transformer_blocks.{i}.attn.to_q`). AM4e.
///
/// Checkpoint facts (from the safetensors headers):
/// - every Linear is 4-bit affine-quantized (`U32` weight packed 4-bit, `BF16` scales+biases,
///   group 64) → `QuantizedLinear(groupSize: 64, bits: 4, mode: .affine)`;
/// - context (T5) hidden = 4096 → `context_embedder` maps to 3072; CLIP pooled = 768;
/// - 8 double blocks + 38 single blocks, hidden 3072, 24 heads, head dim 128.

// MARK: - Shared constants

nonisolated enum FluxTransformerConfig {
    static let hidden = 3072
    static let heads = 24
    static let headDim = 128
    static let contextDim = 4096        // T5 → context_embedder input
    static let pooledDim = 768          // CLIP → text_embedder input
    static let latentChannels = 64      // x_embedder input (patch 1, 64 latent ch)
    static let numDoubleBlocks = 8
    static let numSingleBlocks = 38
    static let theta: Float = 10_000
    static let axesDim: [Int] = [16, 56, 56]
    static let eps: Float = 1e-6
    static let groupSize = 64
    static let bits = 4
}

// MARK: - Feed forward

/// `FeedForward`: Linear(3072→12288) → act → Linear(12288→3072). Keys `ff.linear1`/`ff.linear2`.
nonisolated final class FluxFeedForward: Module {
    @ModuleInfo(key: "linear1") var linear1: QuantizedLinear
    @ModuleInfo(key: "linear2") var linear2: QuantizedLinear
    let usesTanhGELU: Bool

    init(usesTanhGELU: Bool) {
        self.usesTanhGELU = usesTanhGELU
        self._linear1.wrappedValue = QuantizedLinear(
            FluxTransformerConfig.hidden, 4 * FluxTransformerConfig.hidden,
            bias: true, groupSize: FluxTransformerConfig.groupSize,
            bits: FluxTransformerConfig.bits, mode: .affine)
        self._linear2.wrappedValue = QuantizedLinear(
            4 * FluxTransformerConfig.hidden, FluxTransformerConfig.hidden,
            bias: true, groupSize: FluxTransformerConfig.groupSize,
            bits: FluxTransformerConfig.bits, mode: .affine)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = linear1(x)
        return linear2(usesTanhGELU ? GELU(approximation: .tanh)(h) : gelu(h))
    }
}

// MARK: - AdaLayerNormZero (double block)

nonisolated final class FluxAdaLayerNormZero: Module {
    @ModuleInfo(key: "linear") var linear: QuantizedLinear
    let norm = LayerNorm(dimensions: FluxTransformerConfig.hidden, eps: FluxTransformerConfig.eps, affine: false)

    override init() {
        self._linear.wrappedValue = QuantizedLinear(
            FluxTransformerConfig.hidden, 6 * FluxTransformerConfig.hidden,
            bias: true, groupSize: FluxTransformerConfig.groupSize,
            bits: FluxTransformerConfig.bits, mode: .affine)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, textEmbeddings: MLXArray)
        -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        let te = linear(silu(textEmbeddings))
        let chunk = 6 * FluxTransformerConfig.hidden / 6
        let shiftMSA = te[.ellipsis, 0 ..< chunk]
        let scaleMSA = te[.ellipsis, chunk ..< 2 * chunk]
        let gateMSA = te[.ellipsis, 2 * chunk ..< 3 * chunk]
        let shiftMLP = te[.ellipsis, 3 * chunk ..< 4 * chunk]
        let scaleMLP = te[.ellipsis, 4 * chunk ..< 5 * chunk]
        let gateMLP = te[.ellipsis, 5 * chunk ..< 6 * chunk]
        let modulated = norm(hiddenStates) * (1 + scaleMSA.expandedDimensions(axis: 1))
            + shiftMSA.expandedDimensions(axis: 1)
        return (modulated, gateMSA, shiftMLP, scaleMLP, gateMLP)
    }
}

// MARK: - AdaLayerNormZeroSingle (single block)

nonisolated final class FluxAdaLayerNormZeroSingle: Module {
    @ModuleInfo(key: "linear") var linear: QuantizedLinear
    let norm = LayerNorm(dimensions: FluxTransformerConfig.hidden, eps: FluxTransformerConfig.eps, affine: false)

    override init() {
        self._linear.wrappedValue = QuantizedLinear(
            FluxTransformerConfig.hidden, 3 * FluxTransformerConfig.hidden,
            bias: true, groupSize: FluxTransformerConfig.groupSize,
            bits: FluxTransformerConfig.bits, mode: .affine)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, textEmbeddings: MLXArray) -> (MLXArray, MLXArray) {
        let te = linear(silu(textEmbeddings))
        let chunk = 3 * FluxTransformerConfig.hidden / 3
        let shift = te[.ellipsis, 0 ..< chunk]
        let scale = te[.ellipsis, chunk ..< 2 * chunk]
        let gate = te[.ellipsis, 2 * chunk ..< 3 * chunk]
        let modulated = norm(hiddenStates) * (1 + scale.expandedDimensions(axis: 1))
            + shift.expandedDimensions(axis: 1)
        return (modulated, gate)
    }
}

// MARK: - Attention

nonisolated final class FluxJointAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: QuantizedLinear
    @ModuleInfo(key: "to_k") var toK: QuantizedLinear
    @ModuleInfo(key: "to_v") var toV: QuantizedLinear
    @ModuleInfo(key: "to_out") var toOut: [QuantizedLinear]
    @ModuleInfo(key: "add_q_proj") var addQProj: QuantizedLinear
    @ModuleInfo(key: "add_k_proj") var addKProj: QuantizedLinear
    @ModuleInfo(key: "add_v_proj") var addVProj: QuantizedLinear
    @ModuleInfo(key: "to_add_out") var toAddOut: QuantizedLinear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm

    let numHeads = FluxTransformerConfig.heads
    let headDim = FluxTransformerConfig.headDim

    override init() {
        let c = FluxTransformerConfig.self
        func ql() -> QuantizedLinear { QuantizedLinear(c.hidden, c.hidden, bias: true, groupSize: c.groupSize, bits: c.bits, mode: .affine) }
        func rms() -> RMSNorm { RMSNorm(dimensions: c.headDim) }
        self._toQ.wrappedValue = ql()
        self._toK.wrappedValue = ql()
        self._toV.wrappedValue = ql()
        self._toOut.wrappedValue = [ql()]
        self._addQProj.wrappedValue = ql()
        self._addKProj.wrappedValue = ql()
        self._addVProj.wrappedValue = ql()
        self._toAddOut.wrappedValue = ql()
        self._normQ.wrappedValue = rms()
        self._normK.wrappedValue = rms()
        self._normAddedQ.wrappedValue = rms()
        self._normAddedK.wrappedValue = rms()
        super.init()
    }

    func processQKV(_ x: MLXArray, q: QuantizedLinear, k: QuantizedLinear, v: QuantizedLinear, normQ: RMSNorm, normK: RMSNorm) -> (MLXArray, MLXArray, MLXArray) {
        let (b, l) = (x.dim(0), x.dim(1))
        var query = q(x).reshaped([b, l, numHeads, headDim]).transposed(0, 2, 1, 3)
        var key = k(x).reshaped([b, l, numHeads, headDim]).transposed(0, 2, 1, 3)
        let value = v(x).reshaped([b, l, numHeads, headDim]).transposed(0, 2, 1, 3)
        let qDtype = query.dtype, kDtype = key.dtype
        query = normQ(query.asType(.float32)).asType(qDtype)
        key = normK(key.asType(.float32)).asType(kDtype)
        return (query, key, value)
    }

    func callAsFunction(_ hiddenStates: MLXArray, _ encoderHiddenStates: MLXArray, rotary: MLXArray)
        -> (MLXArray, MLXArray) {
        let (query, key, value) = processQKV(hiddenStates, q: toQ, k: toK, v: toV, normQ: normQ, normK: normK)
        let (eq, ek, ev) = processQKV(encoderHiddenStates, q: addQProj, k: addKProj, v: addVProj, normQ: normAddedQ, normK: normAddedK)

        var q = concatenated([eq, query], axis: 2)
        var k = concatenated([ek, key], axis: 2)
        let v = concatenated([ev, value], axis: 2)

        (q, k) = applyRope(q, k, freqsCis: rotary)

        let scale = 1.0 / sqrt(Float(q.dim(3)))
        let attn = scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        let b = attn.dim(0)
        let hidden = attn.transposed(0, 2, 1, 3).reshaped([b, -1, numHeads * headDim])

        let txtLen = encoderHiddenStates.dim(1)
        let imageOut = hidden[0 ..< 1, txtLen ..< hidden.dim(1), 0 ..< hidden.dim(2)]
        let encoderOut = hidden[0 ..< 1, 0 ..< txtLen, 0 ..< hidden.dim(2)]
        return (toOut[0](imageOut), toAddOut(encoderOut))
    }

    private func applyRope(_ q: MLXArray, _ k: MLXArray, freqsCis: MLXArray) -> (MLXArray, MLXArray) {
        func rot(_ x: MLXArray) -> MLXArray {
            let xf = x.asType(.float32)
            let s = xf.shape
            let reshaped = xf.reshaped(Array(s.dropLast()) + [-1, 1, 2])
            let c0 = freqsCis[.ellipsis, 0]
            let c1 = freqsCis[.ellipsis, 1]
            let x0 = reshaped[.ellipsis, 0]
            let x1 = reshaped[.ellipsis, 1]
            let out = c0 * x0 + c1 * x1
            return out.reshaped(s)
        }
        return (rot(q), rot(k))
    }
}

nonisolated final class FluxSingleBlockAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: QuantizedLinear
    @ModuleInfo(key: "to_k") var toK: QuantizedLinear
    @ModuleInfo(key: "to_v") var toV: QuantizedLinear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm

    let numHeads = FluxTransformerConfig.heads
    let headDim = FluxTransformerConfig.headDim

    override init() {
        let c = FluxTransformerConfig.self
        func ql() -> QuantizedLinear { QuantizedLinear(c.hidden, c.hidden, bias: true, groupSize: c.groupSize, bits: c.bits, mode: .affine) }
        self._toQ.wrappedValue = ql()
        self._toK.wrappedValue = ql()
        self._toV.wrappedValue = ql()
        self._normQ.wrappedValue = RMSNorm(dimensions: c.headDim)
        self._normK.wrappedValue = RMSNorm(dimensions: c.headDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, rotary: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var query = toQ(x).reshaped([b, l, numHeads, headDim]).transposed(0, 2, 1, 3)
        var key = toK(x).reshaped([b, l, numHeads, headDim]).transposed(0, 2, 1, 3)
        let value = toV(x).reshaped([b, l, numHeads, headDim]).transposed(0, 2, 1, 3)
        let qDtype = query.dtype, kDtype = key.dtype
        query = normQ(query.asType(.float32)).asType(qDtype)
        key = normK(key.asType(.float32)).asType(kDtype)

        func rot(_ x: MLXArray) -> MLXArray {
            let xf = x.asType(.float32)
            let s = xf.shape
            let reshaped = xf.reshaped(Array(s.dropLast()) + [-1, 1, 2])
            let out = rotary[.ellipsis, 0] * reshaped[.ellipsis, 0]
                + rotary[.ellipsis, 1] * reshaped[.ellipsis, 1]
            return out.reshaped(s)
        }
        query = rot(query)
        key = rot(key)

        let scale = 1.0 / sqrt(Float(query.dim(3)))
        let attn = scaledDotProductAttention(queries: query, keys: key, values: value, scale: scale, mask: nil)
        return attn.transposed(0, 2, 1, 3).reshaped([b, -1, numHeads * headDim])
    }
}

// MARK: - Blocks

nonisolated final class FluxDoubleBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: FluxAdaLayerNormZero
    @ModuleInfo(key: "norm1_context") var norm1Context: FluxAdaLayerNormZero
    @ModuleInfo(key: "attn") var attn: FluxJointAttention
    let norm2 = LayerNorm(dimensions: FluxTransformerConfig.hidden, eps: FluxTransformerConfig.eps, affine: false)
    let norm2Context = LayerNorm(dimensions: FluxTransformerConfig.hidden, eps: FluxTransformerConfig.eps, affine: false)
    @ModuleInfo(key: "ff") var ff: FluxFeedForward
    @ModuleInfo(key: "ff_context") var ffContext: FluxFeedForward

    override init() {
        self._norm1.wrappedValue = FluxAdaLayerNormZero()
        self._norm1Context.wrappedValue = FluxAdaLayerNormZero()
        self._attn.wrappedValue = FluxJointAttention()
        self._ff.wrappedValue = FluxFeedForward(usesTanhGELU: false)
        self._ffContext.wrappedValue = FluxFeedForward(usesTanhGELU: true)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, _ encoderHiddenStates: MLXArray,
                        textEmbeddings: MLXArray, rotary: MLXArray) -> (MLXArray, MLXArray) {
        let (normHidden, gateMSA, shiftMLP, scaleMLP, gateMLP) = norm1(hiddenStates, textEmbeddings: textEmbeddings)
        let (normEnc, cGateMSA, cShiftMLP, cScaleMLP, cGateMLP) = norm1Context(encoderHiddenStates, textEmbeddings: textEmbeddings)

        let (attnOut, contextAttnOut) = attn(normHidden, normEnc, rotary: rotary)

        let hs = applyNormAndFF(hiddenStates, attnOut: attnOut, gateMLP: gateMLP, gateMSA: gateMSA,
                                scaleMLP: scaleMLP, shiftMLP: shiftMLP, norm: norm2, ff: ff)
        let enc = applyNormAndFF(encoderHiddenStates, attnOut: contextAttnOut, gateMLP: cGateMLP, gateMSA: cGateMSA,
                                 scaleMLP: cScaleMLP, shiftMLP: cShiftMLP, norm: norm2Context, ff: ffContext)
        return (enc, hs)
    }

    private func applyNormAndFF(_ hiddenStates: MLXArray, attnOut: MLXArray, gateMLP: MLXArray,
                                gateMSA: MLXArray, scaleMLP: MLXArray, shiftMLP: MLXArray,
                                norm: LayerNorm, ff: FluxFeedForward) -> MLXArray {
        var h = hiddenStates + gateMSA.expandedDimensions(axis: 1) * attnOut
        let nh = norm(h) * (1 + scaleMLP.expandedDimensions(axis: 1)) + shiftMLP.expandedDimensions(axis: 1)
        h = h + gateMLP.expandedDimensions(axis: 1) * ff(nh)
        return h
    }
}

nonisolated final class FluxSingleBlock: Module {
    @ModuleInfo(key: "norm") var norm: FluxAdaLayerNormZeroSingle
    @ModuleInfo(key: "attn") var attn: FluxSingleBlockAttention
    @ModuleInfo(key: "proj_mlp") var projMLP: QuantizedLinear
    @ModuleInfo(key: "proj_out") var projOut: QuantizedLinear

    override init() {
        let c = FluxTransformerConfig.self
        self._norm.wrappedValue = FluxAdaLayerNormZeroSingle()
        self._attn.wrappedValue = FluxSingleBlockAttention()
        self._projMLP.wrappedValue = QuantizedLinear(c.hidden, 4 * c.hidden, bias: true, groupSize: c.groupSize, bits: c.bits, mode: .affine)
        self._projOut.wrappedValue = QuantizedLinear(5 * c.hidden, c.hidden, bias: true, groupSize: c.groupSize, bits: c.bits, mode: .affine)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, textEmbeddings: MLXArray, rotary: MLXArray) -> MLXArray {
        let residual = x
        let (normHidden, gate) = norm(x, textEmbeddings: textEmbeddings)
        let attnOut = attn(normHidden, rotary: rotary)
        let mlpHidden = geluApproximate(projMLP(normHidden))
        let out = gate.expandedDimensions(axis: 1) * projOut(concatenated([attnOut, mlpHidden], axis: 2))
        return residual + out
    }
}

// MARK: - Time/Text embed

/// `TimeTextEmbed`: timestep + guidance + CLIP pooled embedders. Each is a two-layer MLP
/// with single-segment `@ModuleInfo` keys to ensure `update(parameters:)` descends correctly.
nonisolated final class FluxTimeTextEmbed: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: FluxEmbedMLP
    @ModuleInfo(key: "guidance_embedder") var guidanceEmbedder: FluxEmbedMLP
    @ModuleInfo(key: "text_embedder") var textEmbedder: FluxEmbedMLP

    override init() {
        let c = FluxTransformerConfig.self
        func mlp(_ inDim: Int) -> FluxEmbedMLP { FluxEmbedMLP(inputDim: inDim) }
        self._timestepEmbedder.wrappedValue = mlp(256)
        self._guidanceEmbedder.wrappedValue = mlp(256)
        self._textEmbedder.wrappedValue = mlp(c.pooledDim)
        super.init()
    }

    /// mflux `TimeTextEmbed._time_proj`: half-dim 128 sine/cosine, concatenated [sin, cos] then
    /// swapped halves (matches the mflux reference implementation).
    static func timeProj(_ timeSteps: MLXArray) -> MLXArray {
        let halfDim = 128
        var exponent = MLXArray.arange(0, halfDim).asType(.float32)
        exponent = -log(10_000.0) * exponent / Float(halfDim)
        let emb = exp(exponent)
        let e = timeSteps.asType(.float32).expandedDimensions(axis: -1) * emb.expandedDimensions(axis: 0)
        var out = concatenated([sin(e), cos(e)], axis: -1)
        let first = out[.ellipsis, halfDim ..< 2 * halfDim]
        let second = out[.ellipsis, 0 ..< halfDim]
        out = concatenated([first, second], axis: -1)
        return out
    }

    func callAsFunction(timeStep: MLXArray, pooledProjection: MLXArray, guidance: MLXArray) -> MLXArray {
        let t = Self.timeProj(timeStep)
        var emb = timestepEmbedder(t)
        emb = emb + guidanceEmbedder(Self.timeProj(guidance))
        let pooled = textEmbedder(pooledProjection)
        return (emb + pooled).asType(timeStep.dtype)
    }
}

/// A two-layer `Linear → silu → Linear` embedder. `@ModuleInfo` keys are single segments.
nonisolated final class FluxEmbedMLP: Module {
    @ModuleInfo(key: "linear_1") var linear1: QuantizedLinear
    @ModuleInfo(key: "linear_2") var linear2: QuantizedLinear

    init(inputDim: Int) {
        let c = FluxTransformerConfig.self
        self._linear1.wrappedValue = QuantizedLinear(inputDim, c.hidden, bias: true, groupSize: c.groupSize, bits: c.bits, mode: .affine)
        self._linear2.wrappedValue = QuantizedLinear(c.hidden, c.hidden, bias: true, groupSize: c.groupSize, bits: c.bits, mode: .affine)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

// MARK: - Final norm + projection

nonisolated final class FluxAdaLayerNormContinuous: Module {
    @ModuleInfo(key: "linear") var linear: QuantizedLinear
    let norm = LayerNorm(dimensions: FluxTransformerConfig.hidden, eps: FluxTransformerConfig.eps, affine: false)

    override init() {
        let c = FluxTransformerConfig.self
        self._linear.wrappedValue = QuantizedLinear(c.hidden, 2 * c.hidden, bias: true, groupSize: c.groupSize, bits: c.bits, mode: .affine)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, textEmbeddings: MLXArray) -> MLXArray {
        let te = linear(silu(textEmbeddings))
        let chunk = FluxTransformerConfig.hidden
        let scale = te[.ellipsis, 0 ..< chunk]
        let shift = te[.ellipsis, chunk ..< 2 * chunk]
        return norm(x) * (1 + scale.expandedDimensions(axis: 1)) + shift.expandedDimensions(axis: 1)
    }
}

// MARK: - Transformer

/// The full FLUX MMDiT. Keys: `x_embedder`, `time_text_embed`, `context_embedder`,
/// `transformer_blocks.{i}`, `single_transformer_blocks.{i}`, `norm_out`, `proj_out`.
nonisolated final class FluxTransformer: Module {
    @ModuleInfo(key: "x_embedder") var xEmbedder: QuantizedLinear
    @ModuleInfo(key: "time_text_embed") var timeTextEmbed: FluxTimeTextEmbed
    @ModuleInfo(key: "context_embedder") var contextEmbedder: QuantizedLinear
    @ModuleInfo(key: "transformer_blocks") var doubleBlocks: [FluxDoubleBlock]
    @ModuleInfo(key: "single_transformer_blocks") var singleBlocks: [FluxSingleBlock]
    @ModuleInfo(key: "norm_out") var normOut: FluxAdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") var projOut: QuantizedLinear

    override init() {
        let c = FluxTransformerConfig.self
        self._xEmbedder.wrappedValue = QuantizedLinear(c.latentChannels, c.hidden, bias: true, groupSize: c.groupSize, bits: c.bits, mode: .affine)
        self._timeTextEmbed.wrappedValue = FluxTimeTextEmbed()
        self._contextEmbedder.wrappedValue = QuantizedLinear(c.contextDim, c.hidden, bias: true, groupSize: c.groupSize, bits: c.bits, mode: .affine)
        self._doubleBlocks.wrappedValue = (0 ..< c.numDoubleBlocks).map { _ in FluxDoubleBlock() }
        self._singleBlocks.wrappedValue = (0 ..< c.numSingleBlocks).map { _ in FluxSingleBlock() }
        self._normOut.wrappedValue = FluxAdaLayerNormContinuous()
        self._projOut.wrappedValue = QuantizedLinear(c.hidden, c.latentChannels, bias: true, groupSize: c.groupSize, bits: c.bits, mode: .affine)
        super.init()
    }

    static func latentImageIDs(h: Int, w: Int) -> MLXArray {
        let latentH = h / 2, latentW = w / 2
        let rows = MLXArray.arange(latentH).expandedDimensions(axis: -1)
        let cols = MLXArray.arange(latentW).expandedDimensions(axis: 0)
        let j = broadcast(rows, to: [latentH, latentW]).asType(.int32)
        let k = broadcast(cols, to: [latentH, latentW]).asType(.int32)
        let ids = stacked([
            MLXArray.zeros([latentH, latentW], dtype: .int32), j, k
        ], axis: -1)
        return ids.reshaped([1, latentH * latentW, 3])
    }

    static func textIDs(seqLen: Int) -> MLXArray {
        MLXArray.zeros([1, seqLen, 3], dtype: .int32)
    }

    static func rotaryEmbedding(img: MLXArray, txt: MLXArray, dim: Int, theta: Float, axesDim: [Int]) -> MLXArray {
        let ids = concatenated([txt, img], axis: 1)
        let pes = (0 ..< axesDim.count).map { i in
            FluxLayers.rope(pos: ids[.ellipsis, i], dim: axesDim[i], theta: theta)
        }
        let pe = concatenated(pes, axis: -3)
        return pe.expandedDimensions(axis: 1)
    }

    func callAsFunction(
        img: MLXArray, imgIDs: MLXArray, txt: MLXArray, txtIDs: MLXArray,
        vec: MLXArray, timestep: MLXArray, guidance: MLXArray
    ) -> MLXArray {
        let c = FluxTransformerConfig.self
        var hidden = xEmbedder(img)
        var encoderHidden = contextEmbedder(txt)

        let textEmb = timeTextEmbed(timeStep: timestep, pooledProjection: vec, guidance: guidance)
        let rotary = FluxTransformer.rotaryEmbedding(img: imgIDs, txt: txtIDs, dim: c.headDim, theta: c.theta, axesDim: c.axesDim)

        for block in doubleBlocks {
            (encoderHidden, hidden) = block(hidden, encoderHidden, textEmbeddings: textEmb, rotary: rotary)
        }
        hidden = concatenated([encoderHidden, hidden], axis: 1)
        for block in singleBlocks {
            hidden = block(hidden, textEmbeddings: textEmb, rotary: rotary)
        }
        let txtLen = encoderHidden.dim(1)
        hidden = hidden[0 ..< 1, txtLen ..< hidden.dim(1), 0 ..< hidden.dim(2)]
        hidden = normOut(hidden, textEmbeddings: textEmb)
        return projOut(hidden)
    }
}
