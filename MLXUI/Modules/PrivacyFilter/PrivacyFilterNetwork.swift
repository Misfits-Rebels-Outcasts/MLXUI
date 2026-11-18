import Foundation
import MLX

/// PF-5, Traps 2/4/7 — the Privacy Filter transformer's arithmetic, ported line by line from
/// `transformers/models/openai_privacy_filter/modeling_openai_privacy_filter.py` (5.12.0), cross-
/// checked against `opf/_model/model.py` where the two differ.
///
/// **Differences from MLXLLM's gpt-oss that make this a port, not a reuse** (Trap 2):
/// - attention is **bidirectional** with a symmetric band `abs(q - kv) <= slidingWindow` (128), not
///   causal (and the band is *over the slice*, so a windowed call is exact — Trap 3);
/// - RoPE is **interleaved** (even/odd pairs), not split-half;
/// - `q` **and** `k` are each scaled by `head_dim^-0.25` (the module's `scaling`), not one
///   `head_dim^-0.5` on the scores;
/// - per-head attention **sinks** are concatenated as one extra logit column (raw values from the
///   checkpoint; the HF conversion already carries `opf`'s `log(2)` factor), softmaxed in fp32 and
///   dropped;
/// - the MoE router is `topk` **then** softmax over the top-k **then** `/k` (fp32), the expert
///   gate/up halves are a **concatenated** `[gate|up]` layout (not interleaved), and the experts
///   run in **fp32** and are multiplied by `k` after the accumulation (the reference's
///   "stay in fp32 for all linear projections / muls" comment).
///
/// Weight-layout note: HF's `gate_up_proj`/`down_proj`/`score.weight` are plain `nn.Parameter`s, so
/// they are stored as `[experts, in, out]` — already the `matmul` layout. The `nn.Linear`s store
/// `[out, in]`, so their forward is `x @ weight.T`.
///
/// Trap 7: this enum is synchronous and stateless; every array is built and consumed on the calling
/// thread, and the model shell evaluates before any array is read back to Swift.
nonisolated enum PrivacyFilterNetwork {

    // MARK: - Constants (hard-coded in the reference, not in the checkpoint config)

    /// `OpenAIPrivacyFilterExperts.alpha`.
    static let swigluAlpha: Float = 1.702
    /// `OpenAIPrivacyFilterExperts.limit` (the SwiGLU clamp).
    static let swigluLimit: Float = 7.0
    /// How many query rows one attention chunk covers. The band makes the working set
    /// `O(chunk * (chunk + 2 * band) * heads)`; the result is identical for any chunk size because
    /// every query/key dot product is over `head_dim` only.
    private static let attentionChunk = 1024

    // MARK: - YaRN RoPE (`modeling_rope_utils.py::_compute_yarn_parameters`)

    struct Rotary: Sendable {
        /// `inv_freq`, `headDim / 2` values, fp32.
        let invFreq: [Float]
        /// `attention_factor = 0.1 * ln(factor) + 1` (`get_mscale(factor)`).
        let attentionScale: Float
    }

    /// The YaRN inverse frequencies for this config, exactly as `_compute_yarn_parameters` builds
    /// them: `low` from `beta_fast` (32, the extrapolation-only boundary), `high` from `beta_slow`
    /// (1), a clamped linear ramp between them, `truncate == false` (no floor/ceil), then
    /// `interpolation * ramp + extrapolation * (1 - ramp)`.
    static func yarnRotary(_ config: PrivacyFilterConfig) -> Rotary {
        let dim = config.headDim
        let half = dim / 2
        let base = Double(config.ropeTheta)
        let factor = Double(config.ropeFactor)
        let originalMax = Double(config.originalMaxPositionEmbeddings)

        func correctionDimension(_ rotations: Double) -> Double {
            Double(dim) * log(originalMax / (rotations * 2 * Double.pi)) / (2 * log(base))
        }

        var low = correctionDimension(Double(config.ropeBetaFast))
        var high = correctionDimension(Double(config.ropeBetaSlow))
        if config.ropeTruncate {
            low = floor(low)
            high = ceil(high)
        }
        low = max(low, 0)
        high = min(high, Double(dim - 1))
        let lowF = Float(low)
        var highF = Float(high)
        if lowF == highF { highF += 0.001 }

        var invFreq = [Float](repeating: 0, count: half)
        let scale = Float(0.1 * log(factor) + 1)
        for index in 0 ..< half {
            let positionFrequency = powf(Float(config.ropeTheta), Float(2 * index) / Float(dim))
            let interpolation = 1 / (Float(config.ropeFactor) * positionFrequency)
            let extrapolation = 1 / positionFrequency
            let ramp = min(max((Float(index) - lowF) / (highF - lowF), 0), 1)
            // `inv_freq_extrapolation_factor = 1 - ramp`, then
            // `interpolation * (1 - factor) + extrapolation * factor` == `interpolation * ramp +
            // extrapolation * (1 - ramp)`, kept in the reference's operation order.
            let extrapolationFactor = 1 - ramp
            invFreq[index] = interpolation * (1 - extrapolationFactor) + extrapolation * extrapolationFactor
        }
        return Rotary(invFreq: invFreq, attentionScale: scale)
    }

    /// `cos`/`sin` for absolute positions `positions`, computed in fp32 and cast to the model's
    /// compute dtype (bf16) — `OpenAIPrivacyFilterRotaryEmbedding.forward` returns
    /// `cos.to(x.dtype), sin.to(x.dtype)`.
    static func rotaryCosSin(
        _ rotary: Rotary, headDim: Int, positions: [Int], dtype: DType = .bfloat16
    ) -> (cos: MLXArray, sin: MLXArray) {
        let angle = MLXArray(positions.map { Float($0) })                        // [T] fp32
        let inverse = MLXArray(rotary.invFreq)                                  // [half] fp32
        let frequencies = angle.expandedDimensions(axis: -1)
            * inverse.expandedDimensions(axis: 0)                               // [T, half] fp32
        let scaling = MLXArray(rotary.attentionScale)
        return (
            (MLX.cos(frequencies) * scaling).asType(dtype),
            (MLX.sin(frequencies) * scaling).asType(dtype))
    }

    /// Interleaved RoPE: `x[..., ::2]` and `x[..., 1::2]` rotated as pairs, then flattened back —
    /// `_apply_rotary_emb` in the reference. `x` is `[T, heads, headDim]`, `cos`/`sin` `[T, headDim/2]`.
    static func applyRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let shape = x.shape
        let half = shape[shape.count - 1] / 2
        let pairs = x.reshaped([shape[0], shape[1], half, 2])
        let even = pairs[.ellipsis, 0 ..< 1].squeezed(axis: -1)
        let odd = pairs[.ellipsis, 1 ..< 2].squeezed(axis: -1)
        let cosB = cos.expandedDimensions(axis: 1)   // [T, 1, half]
        let sinB = sin.expandedDimensions(axis: 1)
        let first = even * cosB - odd * sinB
        let second = odd * cosB + even * sinB
        return MLX.stacked([first, second], axis: -1).reshaped(shape)
    }

    // MARK: - Primitives

    /// `OpenAIPrivacyFilterRMSNorm`: variance and the weight multiply in fp32, result back in the
    /// input dtype. Weight is bf16 in the checkpoint.
    static func rmsNorm(_ x: MLXArray, weight: MLXArray, eps: Float) -> MLXArray {
        let dtype = x.dtype
        let wide = x.asType(.float32)
        let variance = (wide * wide).mean(axis: -1, keepDims: true)
        let normalized = wide * rsqrt(variance + MLXArray(eps))
        return (weight.asType(.float32) * normalized).asType(dtype)
    }

    /// `nn.Linear` / `F.linear`, i.e. torch's `addmm`: the bf16 GEMM accumulates in fp32, the bias
    /// is added **in fp32**, and the result is rounded to bf16 once.
    ///
    /// Doing it as `matmul` (which rounds the bf16 result) followed by `+ bias` (a second rounding)
    /// is not equivalent: measured here, torch's `F.linear(x, w, b)` differs from `x @ w.T + b` by
    /// 1 ulp on ~half of the elements. At this network's residual scale that double rounding is the
    /// dominant deviation from the reference, so keep the fused form.
    static func linear(_ x: MLXArray, weight: MLXArray, bias: MLXArray?) -> MLXArray {
        let dtype = x.dtype
        let wide = MLX.matmul(x.asType(.float32), weight.asType(.float32).swappedAxes(-1, -2))
        guard let bias else { return wide.asType(dtype) }
        return (wide + bias.asType(.float32)).asType(dtype)
    }

    // MARK: - Attention

    /// One block's self-attention: q/k/v projections, interleaved YaRN RoPE, `q` and `k` each scaled
    /// by `head_dim^-0.25`, the banded bidirectional mask `abs(i - j) <= slidingWindow`, the sink
    /// column, fp32 softmax, then `o_proj`.
    static func selfAttention(
        _ hidden: MLXArray,                     // [T, hidden] bf16
        layer: PrivacyFilterWeights.Layer,
        rotary: Rotary,
        config: PrivacyFilterConfig,
        positionOffset: Int
    ) -> MLXArray {
        let count = hidden.dim(0)
        let headDim = config.headDim
        let heads = config.numHeads
        let kvHeads = config.numKVHeads
        let group = heads / kvHeads

        let queries = linear(hidden, weight: layer.qProjWeight, bias: layer.qProjBias)
        let keys = linear(hidden, weight: layer.kProjWeight, bias: layer.kProjBias)
        let values = linear(hidden, weight: layer.vProjWeight, bias: layer.vProjBias)

        let (cos, sin) = rotaryCosSin(
            rotary, headDim: headDim, positions: Array(positionOffset ..< (positionOffset + count)))
        var q = applyRotary(queries.reshaped([count, heads, headDim]), cos: cos, sin: sin)
        var k = applyRotary(keys.reshaped([count, kvHeads, headDim]), cos: cos, sin: sin)
        // v is not rotary-encoded, but it must still be `[T, kvHeads, headDim]`: `bandedAttention`
        // transposes it to `[kvHeads, T, headDim]`. Passing the raw `[T, kvHeads*headDim]` here made
        // that 3-axis transpose fail, and the following subscript then trapped on the failed array
        // (a Trap 7 crash on the first real forward pass).
        let valueHeads = values.reshaped([count, kvHeads, headDim])

        // `self.scaling = head_dim**-0.25`, applied to both q and k *after* RoPE. A bf16 scalar
        // multiply would round the scale to bf16 first; the reference rounds `q * scaling` to bf16,
        // so multiply in fp32 and round once.
        let scale = Float(pow(Double(headDim), -0.25))
        q = (q.asType(.float32) * MLXArray(scale)).asType(.bfloat16)
        k = (k.asType(.float32) * MLXArray(scale)).asType(.bfloat16)

        let attended = bandedAttention(
            query: q, key: k, value: valueHeads,
            sinks: layer.sinks, kvHeads: kvHeads, group: group,
            headDim: headDim, band: config.slidingWindow)
        return linear(attended, weight: layer.oProjWeight, bias: layer.oProjBias)
    }

    /// The bidirectional banded attention. Queries are processed in chunks so memory is
    /// `O(T * (chunk + 2 * band) * heads)` instead of `O(T^2 * heads)` — linear in the sequence
    /// length, which is what makes the windowed long-input path (Trap 3) possible. Chunking is
    /// numerically transparent: every score is a `head_dim`-long dot product either way.
    ///
    /// `query` `[T, heads, headDim]`, `key`/`value` `[T, kvHeads, headDim]` (bf16, already scaled),
    /// `sinks` `[heads]` fp32. Returns `[T, heads * headDim]` bf16.
    static func bandedAttention(
        query: MLXArray, key: MLXArray, value: MLXArray,
        sinks: MLXArray, kvHeads: Int, group: Int,
        headDim: Int, band: Int
    ) -> MLXArray {
        let count = query.dim(0)
        // Heads are grouped as `num_key_value_groups` consecutive query heads per KV head
        // (`repeat_kv`'s expand/reshape), hence [kvHeads, group, T, headDim]. `reshaped` here is a
        // view when the caller already passed the 3-D layout, and normalises it when it did not.
        let q = query.reshaped([count, kvHeads, group, headDim]).transposed(1, 2, 0, 3)
        let k = key.reshaped([count, kvHeads, headDim]).transposed(1, 0, 2)
        let v = value.reshaped([count, kvHeads, headDim]).transposed(1, 0, 2)
        let sinkValues = sinks.reshaped([kvHeads, group]).asType(.float32)

        var chunks: [MLXArray] = []
        chunks.reserveCapacity((count + attentionChunk - 1) / attentionChunk)
        var start = 0
        while start < count {
            let end = min(start + attentionChunk, count)
            let queryCount = end - start
            let keyStart = max(0, start - band)
            let keyEnd = min(count, end + band)
            let keyCount = keyEnd - keyStart

            let qChunk = q[0..., 0..., start ..< end, 0...]                 // [kv, group, qc, d]
            let kBand = k[0..., keyStart ..< keyEnd, 0...]                  // [kv, kc, d]
            let vBand = v[0..., keyStart ..< keyEnd, 0...]

            // `qChunk` is batched over [kv, group] while the key band is only [kv]. A unit axis at
            // the group position lets matmul broadcast the band across the group (`kv` != `group`,
            // so without it the batch dims do not broadcast and the op fails → Trap 7 crash).
            let kBandT = kBand.swappedAxes(-1, -2).expandedDimensions(axis: 1)  // [kv, 1, d, kc]
            let scores = MLX.matmul(qChunk, kBandT)                         // bf16 [kv, group, qc, kc]

            // Band over *slice* indices (`sliding_window_bidirectional_overlay`): a query attends to
            // a key when their absolute distance is within the (inclusive) window.
            let queryIndex = MLXArray((start ..< end).map { Int32($0) }).reshaped([queryCount, 1])
            let keyIndex = MLXArray((keyStart ..< keyEnd).map { Int32($0) }).reshaped([1, keyCount])
            let inBand = MLX.abs(queryIndex - keyIndex) .<= MLXArray(Int32(band))
            // The reference's mask is **bf16** (`eager_mask` uses `inputs_embeds.dtype`), and the
            // additive mask is applied to the bf16 scores *before* the fp32 sinks promote the
            // concatenation (`Attn + mask` is a bf16 add in `eager_attention_forward`). Keep that
            // rounding: add in bf16, then promote. For a slice shorter than the band the mask is all
            // zeros, so this is a no-op there; it matters once the band actually clips.
            let bf16Mask = MLX.which(
                inBand, MLXArray(Float(0)), MLXArray(-Float.infinity)).asType(.bfloat16)
            let masked = (scores + bf16Mask).asType(.float32)               // [qc, kc] broadcast

            // The sinks are one extra fp32 logit column (`eager_attention_forward` concatenates
            // them before the fp32 softmax and drops them afterwards).
            let sinkColumn = MLX.broadcast(
                sinkValues.reshaped([kvHeads, group, 1, 1]),
                to: [kvHeads, group, queryCount, 1])
            let combined = MLX.concatenated([masked, sinkColumn], axis: -1)
            let probabilities = MLX.softmax(combined, axis: -1, precise: true)
            let weights = probabilities[0..., 0..., 0..., 0 ..< keyCount].asType(.bfloat16)
            // Same unit-axis trick for the value band: [kv, 1, kc, d] broadcasts over the group.
            chunks.append(
                MLX.matmul(weights, vBand.expandedDimensions(axis: 1)))       // [kv, group, qc, d]
            start = end
        }

        let merged = chunks.count == 1 ? chunks[0] : MLX.concatenated(chunks, axis: 2)
        return merged.transposed(2, 0, 1, 3).reshaped([count, kvHeads * group * headDim])
    }

    // MARK: - MoE feed-forward (Trap 4)

    /// One block's mixture-of-experts feed-forward, in the reference's fp32 order:
    /// norm → fp32 router → `topk` → softmax(fp32) → `/k` → gathered fp32 expert matmuls with the
    /// `[gate|up]` concatenated layout → SwiGLU clamps → down projection → fp32 weighted sum → `* k`
    /// → bf16.
    ///
    /// `gatherMM` does every expert projection in one call per stage; there is no 128-iteration
    /// Swift loop.
    static func mixtureOfExperts(
        _ normalized: MLXArray,                 // [T, hidden] bf16 (post `post_attention_layernorm`)
        layer: PrivacyFilterWeights.Layer,
        config: PrivacyFilterConfig
    ) -> MLXArray {                             // [T, hidden] bf16
        let count = normalized.dim(0)
        let hidden = config.hiddenSize
        let intermediate = config.intermediateSize
        let k = config.expertsPerToken
        let rows = count * k

        let wide = normalized.asType(.float32)                                  // [T, hidden]

        // Router: `F.linear(hidden.float(), weight.float(), bias.float())` then `torch.topk`.
        let routerLogits = MLX.matmul(wide, layer.routerWeight.asType(.float32).swappedAxes(-1, -2))
            + layer.routerBias.asType(.float32)                                 // [T, experts] fp32
        let partition = MLX.argPartition(routerLogits, kth: -k, axis: -1)       // [T, experts] int32
        let topIndices = partition[0..., (-k)...]                               // [T, k]
        let topValues = MLX.takeAlong(routerLogits, topIndices, axis: -1)
        let scores = MLX.softmax(topValues, axis: -1) / Float(k)                // [T, k] fp32

        // One gathered row per (token, top-k slot).
        let tokenIDs = MLXArray((0 ..< count).flatMap { Array(repeating: Int32($0), count: k) })
        let expertIDs = topIndices.reshaped([rows])
        let rowsWide = wide[tokenIDs]                                           // [rows, hidden] fp32

        let gateUpProj = layer.gateUpProj.asType(.float32)                      // [E, hidden, 2*inter]
        let gateUpBias = layer.gateUpProjBias.asType(.float32)                  // [E, 2*inter]
        // `gatherMM` output shape is `indices.shape + [a.shape(-2), b.shape(-1)]`, and a's default
        // lhs indices are its batch dims; the lhs must therefore be `[rows, 1, hidden]` (a unit M
        // axis) so the lhs and rhs index tensors are both `[rows]`. One extra unit axis here makes
        // them `[rows, 1]` vs `[rows]`, which broadcasts to `[rows, rows]` — a failed op.
        let gateUp = MLX.gatherMM(
            rowsWide.reshaped([rows, 1, hidden]), gateUpProj,
            rhsIndices: expertIDs).reshaped([rows, 2 * intermediate])
            + gateUpBias[expertIDs]

        // `[gate | up]`: the first half is the gate (clamped max only), the second the up half
        // (clamped both ways, then `+1`). Alpha 1.702, limit 7.0.
        let halves = gateUp.reshaped([rows, 2, intermediate])
        let gate = halves[0..., 0 ..< 1, 0...].squeezed(axis: 1)
        let up = halves[0..., 1 ..< 2, 0...].squeezed(axis: 1)
        let gateClamped = MLX.clip(gate, max: MLXArray(swigluLimit))
        let upClamped = MLX.clip(up, min: MLXArray(-swigluLimit), max: MLXArray(swigluLimit))
        let glu = gateClamped * MLX.sigmoid(MLXArray(swigluAlpha) * gateClamped)
        let gated = (upClamped + 1) * glu                                       // [rows, inter] fp32

        let downProj = layer.downProj.asType(.float32)                          // [E, inter, hidden]
        let downBias = layer.downProjBias.asType(.float32)                      // [E, hidden]
        let projected = MLX.gatherMM(
            gated.reshaped([rows, 1, intermediate]), downProj,
            rhsIndices: expertIDs).reshaped([rows, hidden])
            + downBias[expertIDs]                                               // [rows, hidden] fp32

        // Weighted sum over the top-k slots, fp32, then the reference's `* num_experts_per_tok`
        // (`MLP.forward`'s "additional scaling") before the cast back to bf16.
        let weighted = (projected * scores.reshaped([rows, 1]))
            .reshaped([count, k, hidden])
            .sum(axis: 1)                                                       // [T, hidden] fp32
        return (weighted * Float(k)).asType(normalized.dtype)
    }

    /// One encoder block: pre-norm attention with residual, then pre-norm MoE with residual
    /// (`OpenAIPrivacyFilterEncoderLayer`).
    static func encoderLayer(
        _ hidden: MLXArray,
        layer: PrivacyFilterWeights.Layer,
        rotary: Rotary,
        config: PrivacyFilterConfig,
        positionOffset: Int
    ) -> MLXArray {
        var x = hidden
        let afterAttention = x + selfAttention(
            rmsNorm(x, weight: layer.inputLayerNorm, eps: config.rmsNormEps),
            layer: layer, rotary: rotary, config: config, positionOffset: positionOffset)
        x = afterAttention + mixtureOfExperts(
            rmsNorm(afterAttention, weight: layer.postAttentionLayerNorm, eps: config.rmsNormEps),
            layer: layer, config: config)
        return x
    }
}
