//
//  Qwen35Attention.swift
//  MLXUI — Qwen3.5 (qwen3_5) hybrid GatedDeltaNet port, Slice QB1.
//
//  Port of `Qwen3_5Attention` from Blaizzy/mlx-vlm `models/qwen3_5/language.py`
//  (the standard gated full-attention layer used on every `full_attention_interval`-th
//  layer). Differences from stock `Qwen3.swift`:
//    - `q_proj` emits 2× width: the output splits into (query, gate); the attention
//      output is multiplied by `sigmoid(gate)` before `o_proj` (attn_output_gate).
//    - Per-head RMSNorm on q and k (head_dim).
//    - **Partial** rotary: only `head_dim * partial_rotary_factor` (= 64 of 256) dims
//      are rotated, interleaved, via multimodal RoPE.
//
//  ROPE CONTRACT: this slice applies rotation given precomputed `cos`/`sin`. Generating
//  them (3-axis interleaved MRoPE + position ids) is Slice QB2 — until QB2 lands nothing
//  constructs `cos`/`sin`, so this layer compiles but is not yet exercised. The interleave
//  convention here is validated against the reference in the QB8 parity test.
//

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

nonisolated final class Qwen35Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let rotaryDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: Qwen35TextConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.rotaryDim = Int((Double(config.headDim) * config.ropeParameters.partialRotaryFactor).rounded(.down))
        self.scale = pow(Float(config.headDim), -0.5)

        let bias = config.attentionBias
        let hidden = config.hiddenSize
        // q_proj is doubled: half query, half gate (attn_output_gate).
        self._wq.wrappedValue = Linear(hidden, numHeads * headDim * 2, bias: bias)
        self._wk.wrappedValue = Linear(hidden, numKVHeads * headDim, bias: bias)
        self._wv.wrappedValue = Linear(hidden, numKVHeads * headDim, bias: bias)
        self._wo.wrappedValue = Linear(numHeads * headDim, hidden, bias: bias)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    /// `x`: `[B, L, hidden]`. `cos`/`sin`: broadcastable to `[B, 1, L, rotaryDim/2]`
    /// (supplied by QB2's MRoPE). Returns `[B, L, hidden]`.
    func callAsFunction(
        _ x: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)

        // Query projection carries the gate in its second half.
        let qProj = wq(x).reshaped(B, L, numHeads, headDim * 2)
        let qParts = split(qProj, parts: 2, axis: -1)          // 2× [B,L,numHeads,headDim]
        var queries = qNorm(qParts[0]).transposed(0, 2, 1, 3)  // [B,numHeads,L,headDim]
        let gate = qParts[1].reshaped(B, L, -1)                // [B,L,numHeads*headDim]

        var keys = kNorm(wk(x).reshaped(B, L, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        queries = Qwen35RoPEApply.partialRotateHalf(queries, cos: cos, sin: sin, rotaryDim: rotaryDim)
        keys = Qwen35RoPEApply.partialRotateHalf(keys, cos: cos, sin: sin, rotaryDim: rotaryDim)

        let out = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        // Gated output projection: output * sigmoid(gate).
        return wo(out * Qwen35Math.sigmoid(gate))
    }
}

/// Applies rotary embedding to the first `rotaryDim` dims of the head axis, passing the
/// remainder through unchanged (partial rotary). Uses the NeoX half-split `rotate_half`
/// convention that Qwen3.5's `style="interleaved"` mrope resolves to (the "interleaved"
/// naming refers to the per-axis frequency selection in `Qwen35RoPE`, not the pairing).
/// `cos`/`sin` are width `rotaryDim` (freqs duplicated) and broadcast to `[B, 1, T, rotaryDim]`.
nonisolated enum Qwen35RoPEApply {
    static func partialRotateHalf(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, rotaryDim: Int
    ) -> MLXArray {
        let d = x.dim(-1)
        guard rotaryDim > 0, rotaryDim <= d else { return x }

        let rot = x[0..., 0..., 0..., 0 ..< rotaryDim]      // [B,H,L,rotaryDim]
        let half = rotaryDim / 2
        let x1 = rot[0..., 0..., 0..., 0 ..< half]           // first half
        let x2 = rot[0..., 0..., 0..., half ..< rotaryDim]   // second half
        let rotateHalf = concatenated([-x2, x1], axis: -1)   // [-x2, x1]

        let rotated = rot * cos + rotateHalf * sin
        if rotaryDim == d { return rotated }
        let passthrough = x[0..., 0..., 0..., rotaryDim ..< d]
        return concatenated([rotated, passthrough], axis: -1)
    }
}
