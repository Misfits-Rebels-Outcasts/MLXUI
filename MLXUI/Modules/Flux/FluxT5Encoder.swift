import Foundation
import MLX
import MLXNN

/// The FLUX T5 text encoder, ported from mflux (`flux_text_encoder/t5_encoder/`) in the
/// checkpoint's own key/quant format (AM4f). All Linear/Embedding are 4-bit affine-quantized
/// (group 64); the LayerNorms are affine BF16.
///
/// Config: d_model 4096, d_ff 10240, 24 layers, 64 heads, head dim 64, vocab 32128,
/// `feed_forward_proj "gated-gelu"` (wi_0 × gelu(wi_1)), rel-pos 32 buckets / max-dist 128.
nonisolated final class FluxT5Encoder: Module {

    static let dModel = 4096
    static let dFF = 10240
    static let heads = 64
    static let headDim = 64
    static let vocab = 32128
    static let layers = 24

    @ModuleInfo(key: "shared") var shared: QuantizedEmbedding
    @ModuleInfo(key: "t5_blocks") var t5Blocks: [FluxT5Block]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: FluxT5LayerNorm

    override init() {
        let c = FluxT5Encoder.self
        self._shared.wrappedValue = QuantizedEmbedding(
            embeddingCount: c.vocab, dimensions: c.dModel, groupSize: 64, bits: 4, mode: .affine)
        self._t5Blocks.wrappedValue = (0 ..< c.layers).map { _ in FluxT5Block() }
        self._finalLayerNorm.wrappedValue = FluxT5LayerNorm()
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        var hidden = shared(tokens)
        for block in t5Blocks { hidden = block(hidden) }
        return finalLayerNorm(hidden)
    }
}

/// T5 LayerNorm — RMS-norm-like with a learned BF16 `weight` (affine, no bias).
nonisolated final class FluxT5LayerNorm: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray

    override init() {
        self._weight.wrappedValue = MLXArray.ones([FluxT5Encoder.dModel])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let variance = pow(x.asType(.float32), 2).mean(axis: -1, keepDims: true)
        let normalized = x * rsqrt(variance + 1e-6)
        return weight * normalized
    }
}

nonisolated final class FluxT5Block: Module {
    @ModuleInfo(key: "attention") var attention: FluxT5Attention
    @ModuleInfo(key: "ff") var ff: FluxT5FeedForward

    override init() {
        self._attention.wrappedValue = FluxT5Attention()
        self._ff.wrappedValue = FluxT5FeedForward()
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = attention(x)
        return ff(out)
    }
}

nonisolated final class FluxT5Attention: Module {
    @ModuleInfo(key: "SelfAttention") var selfAttention: FluxT5SelfAttention
    @ModuleInfo(key: "layer_norm") var layerNorm: FluxT5LayerNorm

    override init() {
        self._selfAttention.wrappedValue = FluxT5SelfAttention()
        self._layerNorm.wrappedValue = FluxT5LayerNorm()
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + selfAttention(layerNorm(x))
    }
}

/// T5 self-attention with learned relative-position bias. All projections bias-less.
nonisolated final class FluxT5SelfAttention: Module {
    @ModuleInfo(key: "q") var q: QuantizedLinear
    @ModuleInfo(key: "k") var k: QuantizedLinear
    @ModuleInfo(key: "v") var v: QuantizedLinear
    @ModuleInfo(key: "o") var o: QuantizedLinear
    @ModuleInfo(key: "relative_attention_bias") var relativeAttentionBias: QuantizedEmbedding

    override init() {
        let c = FluxT5Encoder.self
        func ql() -> QuantizedLinear {
            QuantizedLinear(c.dModel, c.dModel, bias: false, groupSize: 64, bits: 4, mode: .affine)
        }
        self._q.wrappedValue = ql()
        self._k.wrappedValue = ql()
        self._v.wrappedValue = ql()
        self._o.wrappedValue = ql()
        self._relativeAttentionBias.wrappedValue = QuantizedEmbedding(
            embeddingCount: 32, dimensions: c.heads, groupSize: 64, bits: 4, mode: .affine)
        super.init()
    }

    private static func shape(_ states: MLXArray) -> MLXArray {
        states.reshaped([1, -1, FluxT5Encoder.heads, FluxT5Encoder.headDim]).transposed(0, 2, 1, 3)
    }

    private static func unshape(_ states: MLXArray) -> MLXArray {
        states.transposed(0, 2, 1, 3).reshaped([1, -1, FluxT5Encoder.dModel])
    }

    /// mflux `_relative_position_bucket` (bidirectional, 32 buckets, max-dist 128).
    static func relativePositionBucket(_ rpos: MLXArray) -> MLXArray {
        let numBuckets = 32 / 2
        var relativeBuckets = MLXArray.zeros(like: rpos)
        relativeBuckets = relativeBuckets
            + MLX.where(rpos .> 0, MLXArray(Int32(numBuckets)), MLXArray(Int32(0))).asType(.int32)
        let absPos = abs(rpos)
        let maxExact = numBuckets / 2
        let isSmall = absPos .< maxExact
        let large = MLXArray(maxExact).asType(.int32) + floor(
            log(absPos.asType(.float32) / Float(maxExact)) / log(128.0 / Float(maxExact))
                * Float(numBuckets - maxExact)
        ).asType(.int32)
        let largeClamped = minimum(large, MLXArray(Int32(numBuckets - 1)))
        relativeBuckets = relativeBuckets + MLX.where(isSmall, absPos, largeClamped)
        return relativeBuckets
    }

    private func computeBias(seqLength: Int) -> MLXArray {
        let contextPosition = MLXArray.arange(seqLength).expandedDimensions(axis: -1)
        let memoryPosition = MLXArray.arange(seqLength).expandedDimensions(axis: 0)
        let relativePosition = memoryPosition - contextPosition
        let bucket = Self.relativePositionBucket(relativePosition)
        let values = relativeAttentionBias(bucket).transposed(2, 0, 1)
        return values.expandedDimensions(axis: 0)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let queryStates = Self.shape(q(x))
        let keyStates = Self.shape(k(x))
        let valueStates = Self.shape(v(x))
        var scores = matmul(queryStates, keyStates.transposed(0, 1, 3, 2))
        scores = scores + computeBias(seqLength: x.dim(1))
        let attnWeights = softmax(scores, axis: -1)
        let attnOutput = Self.unshape(matmul(attnWeights, valueStates))
        return o(attnOutput)
    }
}

/// T5 gated-GELU FFN (`feed_forward_proj "gated-gelu"`): wi_0 × gelu(wi_1) → wo.
nonisolated final class FluxT5FeedForward: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: FluxT5LayerNorm
    @ModuleInfo(key: "DenseReluDense") var denseReluDense: FluxT5DenseReluDense

    override init() {
        self._layerNorm.wrappedValue = FluxT5LayerNorm()
        self._denseReluDense.wrappedValue = FluxT5DenseReluDense()
        super.init()
    }

    /// `new_gelu`: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
    static func newGELU(_ x: MLXArray) -> MLXArray {
        let inner = sqrt(2.0 / .pi) * (x + 0.044715 * pow(x, 3.0))
        return 0.5 * x * (1 + tanh(inner))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + denseReluDense(layerNorm(x))
    }
}

/// The `DenseReluDense` container. Its `@ModuleInfo` keys are single segments (`wi_0`,
/// `wi_1`, `wo`) — a dotted key like `DenseReluDense.wo` in one `@ModuleInfo` would silently
/// fail to load under MLX's `update(parameters:)`.
nonisolated final class FluxT5DenseReluDense: Module {
    @ModuleInfo(key: "wi_0") var wi0: QuantizedLinear
    @ModuleInfo(key: "wi_1") var wi1: QuantizedLinear
    @ModuleInfo(key: "wo") var wo: QuantizedLinear

    override init() {
        let c = FluxT5Encoder.self
        func qlIn(_ inDim: Int, _ out: Int) -> QuantizedLinear {
            QuantizedLinear(inDim, out, bias: false, groupSize: 64, bits: 4, mode: .affine)
        }
        self._wi0.wrappedValue = qlIn(c.dModel, c.dFF)
        self._wi1.wrappedValue = qlIn(c.dModel, c.dFF)
        // wo maps the 10240-dim gated output back to 4096.
        self._wo.wrappedValue = qlIn(c.dFF, c.dModel)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gated = FluxT5FeedForward.newGELU(wi0(x)) * wi1(x)
        return wo(gated)
    }
}
