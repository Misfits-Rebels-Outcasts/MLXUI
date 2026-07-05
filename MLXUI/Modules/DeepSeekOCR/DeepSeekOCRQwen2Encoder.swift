import Foundation
import MLX
import MLXNN
import MLXFast

/// The Qwen2-0.5B "decoder-as-encoder" for DeepSeek-OCR-2 — a from-scratch port of
/// `Blaizzy/mlx-vlm`'s `mlx_vlm/models/deepseekocr_2/vision.py`. It takes the SAM tower's spatial
/// features, appends learnable query tokens (256 for 1024² / 144 for 768²), runs 24 Qwen2 decoder
/// layers under a **mixed attention mask** (image↔image bidirectional, image↛query, query→image,
/// query→query causal), and returns the query tokens. A linear projector then lifts them to the LM
/// embedding width. All classes `nonisolated` (MainActor default).

private enum DSQwen2Math {
    /// Rotary cos/sin for `positionIds` `(B, S)` → `(B, S, headDim)` (Qwen2 RoPE, on-the-fly inv_freq).
    nonisolated static func rotaryCosSin(
        positionIds: MLXArray, dim: Int, base: Float, dtype: DType
    ) -> (MLXArray, MLXArray) {
        let exponent = MLXArray(Array(stride(from: 0, to: dim, by: 2))).asType(.float32) / Float(dim)
        let invFreq = 1.0 / pow(MLXArray(base), exponent)               // (dim/2)
        let pos = positionIds.asType(.float32).expandedDimensions(axis: -1)  // (B, S, 1)
        let freqs = pos * invFreq.reshaped(1, 1, -1)                    // (B, S, dim/2)
        let emb = concatenated([freqs, freqs], axis: -1)               // (B, S, dim)
        return (cos(emb).asType(dtype), sin(emb).asType(dtype))
    }

    nonisolated static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let parts = split(x, parts: 2, axis: -1)
        return concatenated([-parts[1], parts[0]], axis: -1)
    }

    /// Apply RoPE to `q`/`k` `(B, heads, S, hd)` with cos/sin `(B, S, hd)`.
    nonisolated static func applyRope(
        _ q: MLXArray, _ k: MLXArray, _ cosT: MLXArray, _ sinT: MLXArray
    ) -> (MLXArray, MLXArray) {
        let c = cosT.expandedDimensions(axis: 1)   // (B, 1, S, hd)
        let s = sinT.expandedDimensions(axis: 1)
        return ((q * c) + (rotateHalf(q) * s), (k * c) + (rotateHalf(k) * s))
    }
}

nonisolated final class DSQwen2MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(dim: Int, intermediate: Int) {
        self._gate.wrappedValue = Linear(dim, intermediate, bias: false)
        self._up.wrappedValue = Linear(dim, intermediate, bias: false)
        self._down.wrappedValue = Linear(intermediate, dim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

nonisolated final class DSQwen2Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let kvGroups: Int
    let scale: Float
    let ropeBase: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    init(_ v: DeepSeekOCRConfig.Vision) {
        self.numHeads = v.qwen2Heads
        self.numKVHeads = v.qwen2KVHeads
        self.headDim = v.qwen2Dim / v.qwen2Heads
        self.kvGroups = v.qwen2Heads / v.qwen2KVHeads
        self.scale = pow(Float(headDim), -0.5)
        self.ropeBase = v.qwen2RopeTheta
        self._wq.wrappedValue = Linear(v.qwen2Dim, numHeads * headDim, bias: true)
        self._wk.wrappedValue = Linear(v.qwen2Dim, numKVHeads * headDim, bias: true)
        self._wv.wrappedValue = Linear(v.qwen2Dim, numKVHeads * headDim, bias: true)
        self._wo.wrappedValue = Linear(numHeads * headDim, v.qwen2Dim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, positionIds: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = wq(x).reshaped(b, l, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = wk(x).reshaped(b, l, numKVHeads, headDim).transposed(0, 2, 1, 3)
        var v = wv(x).reshaped(b, l, numKVHeads, headDim).transposed(0, 2, 1, 3)

        let (cosT, sinT) = DSQwen2Math.rotaryCosSin(
            positionIds: positionIds, dim: headDim, base: ropeBase, dtype: x.dtype)
        (q, k) = DSQwen2Math.applyRope(q, k, cosT, sinT)

        if kvGroups > 1 {
            k = repeated(k, count: kvGroups, axis: 1)
            v = repeated(v, count: kvGroups, axis: 1)
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .array(mask))
            .transposed(0, 2, 1, 3)
            .reshaped(b, l, -1)
        return wo(out)
    }
}

nonisolated final class DSQwen2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DSQwen2Attention
    @ModuleInfo(key: "mlp") var mlp: DSQwen2MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ v: DeepSeekOCRConfig.Vision) {
        self._selfAttn.wrappedValue = DSQwen2Attention(v)
        self._mlp.wrappedValue = DSQwen2MLP(dim: v.qwen2Dim, intermediate: v.qwen2IntermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: v.qwen2Dim, eps: v.qwen2RMSNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: v.qwen2Dim, eps: v.qwen2RMSNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, positionIds: MLXArray) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, positionIds: positionIds)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

nonisolated final class Qwen2Decoder2Encoder: Module {
    let dim: Int

    @ParameterInfo(key: "query_1024") var query1024: MLXArray   // (256, dim)
    @ParameterInfo(key: "query_768") var query768: MLXArray     // (144, dim)
    @ModuleInfo(key: "layers") var layers: [DSQwen2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ v: DeepSeekOCRConfig.Vision) {
        self.dim = v.qwen2Dim
        self._query1024.wrappedValue = MLXArray.zeros([256, v.qwen2Dim])
        self._query768.wrappedValue = MLXArray.zeros([144, v.qwen2Dim])
        self._layers.wrappedValue = (0 ..< v.qwen2Layers).map { _ in DSQwen2DecoderLayer(v) }
        self._norm.wrappedValue = RMSNorm(dimensions: v.qwen2Dim, eps: v.qwen2RMSNormEps)
        super.init()
    }

    /// SAM features `(B, H, W, dim)` → query tokens `(B, numQueries, dim)`.
    func callAsFunction(_ samFeatures: MLXArray) -> MLXArray {
        let b = samFeatures.dim(0)
        let flat = samFeatures.reshaped(b, -1, dim)   // (B, numImage, dim)
        let ni = flat.dim(1)
        let (query, nq) = ni == 144 ? (query768, 144) : (query1024, 256)

        let queries = broadcast(query.reshaped(1, nq, dim), to: [b, nq, dim])
        var h = concatenated([flat, queries], axis: 1)  // (B, ni+nq, dim)
        let seq = ni + nq

        let mask = Self.mixedMask(numImage: ni, numQueries: nq, dtype: h.dtype)
        let positionIds = broadcast(MLXArray(0 ..< seq).reshaped(1, seq), to: [b, seq])

        for layer in layers {
            h = layer(h, mask: mask, positionIds: positionIds)
        }
        h = norm(h)
        return h[0..., (seq - nq)..., 0...]   // last numQueries tokens
    }

    /// (1,1,seq,seq) additive mask: image↔image bidirectional, image↛query blocked, query→image
    /// allowed, query→query causal. Mirrors `vision.py`'s block construction (built directly here).
    nonisolated static func mixedMask(numImage ni: Int, numQueries nq: Int, dtype: DType) -> MLXArray {
        let neg: Float = -1e9
        // image rows: [attend all images (0), block queries (neg)]
        let imageRow = concatenated([MLXArray.zeros([ni, ni]), MLXArray.zeros([ni, nq]) + neg], axis: 1)
        // query→query causal: neg where col > row (future), else 0
        let rIdx = MLXArray(0 ..< nq).reshaped(nq, 1)
        let cIdx = MLXArray(0 ..< nq).reshaped(1, nq)
        let causal = MLX.where(cIdx .> rIdx, MLXArray(neg), MLXArray(Float(0)))
        // query rows: [attend all images (0), causal over queries]
        let queryRow = concatenated([MLXArray.zeros([nq, ni]), causal], axis: 1)
        let mask = concatenated([imageRow, queryRow], axis: 0).asType(dtype)  // (seq, seq)
        return mask.reshaped(1, 1, ni + nq, ni + nq)
    }
}

/// Vision model wrapper (`vision_model`): owns the Qwen2 encoder. The raw image `x` isn't used
/// (kept for API symmetry with the reference); the SAM features are the real input.
nonisolated final class DeepSeekOCRVisionModel: Module {
    @ModuleInfo(key: "qwen2_encoder") var qwen2Encoder: Qwen2Decoder2Encoder

    init(_ v: DeepSeekOCRConfig.Vision) {
        self._qwen2Encoder.wrappedValue = Qwen2Decoder2Encoder(v)
        super.init()
    }

    func callAsFunction(_ samFeatures: MLXArray) -> MLXArray { qwen2Encoder(samFeatures) }
}

/// Linear projector (`projector`): SAM/Qwen2 feature width → LM embedding width. `projector.layers`
/// is the single `Linear` (matches the reference `MlpProjector` with `projector_type == "linear"`).
nonisolated final class DeepSeekOCRProjector: Module {
    @ModuleInfo(key: "layers") var layers: Linear

    init(_ p: DeepSeekOCRConfig.Projector) {
        self._layers.wrappedValue = Linear(p.inputDim, p.nEmbed)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { layers(x) }
}
