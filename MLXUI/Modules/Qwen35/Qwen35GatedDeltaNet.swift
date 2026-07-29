//
//  Qwen35GatedDeltaNet.swift
//  MLXUI — Qwen3.5 (qwen3_5) hybrid GatedDeltaNet port, Slice 1.
//
//  Port of the linear-attention layer from Blaizzy/mlx-vlm `models/qwen3_5`:
//    - `Qwen3_5RMSNormGated`  (gated RMSNorm)          -> `Qwen35RMSNormGated`
//    - `Qwen3_5GatedDeltaNet` (conv + delta-rule SSM)  -> `Qwen35GatedDeltaNet`
//    - `gated_delta_ops` / `_gated_delta_step_ops`     -> `Qwen35GatedDelta.ops(...)`
//
//  The reference ships hand-written Metal kernels (`gated_delta.py`) with pure-`mx`
//  ops fallbacks. mlx-swift has no ergonomic custom-kernel path, so this slice ports
//  the **ops fallback** — correctness-first, one sequential step per token. It matches
//  the reference recurrence exactly; the chunked / fused-kernel fast paths are a later
//  performance slice.
//
//  SCOPE (Slice 1): the forward math for a single prefill call with a zero initial
//  state. Integration with an incremental conv+SSM decode cache and the mlx-swift-lm
//  generate loop is a following slice (see RSI backlog "qwen3_5" group).
//

import Foundation
import MLX
import MLXFast
import MLXNN

/// Gated RMSNorm: RMS-normalise `x`, then (when a gate is supplied) multiply by
/// `silu(gate)` in float32 for precision. Mirrors `Qwen3_5RMSNormGated`.
nonisolated final class Qwen35RMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray? = nil) -> MLXArray {
        let normed = MLXFast.rmsNorm(x, weight: weight, eps: eps)
        guard let gate else { return normed.asType(x.dtype) }
        let g = Qwen35Math.silu(gate.asType(.float32))
        return (g * normed.asType(.float32)).asType(x.dtype)
    }
}

/// Small numerically-stable math helpers shared by the linear-attention path.
nonisolated enum Qwen35Math {
    /// silu(x) = x * sigmoid(x)
    static func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    /// sigmoid(x) = 1 / (1 + exp(-x))
    static func sigmoid(_ x: MLXArray) -> MLXArray { 1.0 / (1.0 + MLX.exp(-x)) }

    /// Numerically-stable softplus: max(x,0) + log1p(exp(-|x|)).
    static func softplus(_ x: MLXArray) -> MLXArray {
        MLX.maximum(x, 0) + MLX.log(1.0 + MLX.exp(-MLX.abs(x)))
    }
}

/// The delta-rule recurrence (ops reference). Kept free of any `Module` so it is
/// unit-testable in isolation against the Python `gated_delta_ops`.
nonisolated enum Qwen35GatedDelta {
    /// g = exp(-exp(A_log) * softplus(a + dt_bias)),  beta = sigmoid(b).
    /// - A_log, dt_bias: `[Hv]`;  a, b: `[B, T, Hv]`.
    static func computeGBeta(
        aLog: MLXArray, dtBias: MLXArray, a: MLXArray, b: MLXArray
    ) -> (g: MLXArray, beta: MLXArray) {
        let aLogF = aLog.asType(.float32)
        let g = MLX.exp(-MLX.exp(aLogF) * Qwen35Math.softplus(a + dtBias))
        return (g, Qwen35Math.sigmoid(b))
    }

    /// Sequential delta-rule scan over the time axis (the reference `gated_delta_ops`).
    ///
    /// Shapes:
    ///   - q, k: `[B, T, Hk, Dk]`   v: `[B, T, Hv, Dv]`
    ///   - g, beta: `[B, T, Hv]`    state: `[B, Hv, Dv, Dk]` (or nil → zeros)
    /// Returns y `[B, T, Hv, Dv]` and the final state `[B, Hv, Dv, Dk]`.
    static func ops(
        q: MLXArray, k: MLXArray, v: MLXArray,
        g: MLXArray, beta: MLXArray, state: MLXArray?
    ) -> (y: MLXArray, state: MLXArray) {
        let B = q.dim(0), T = q.dim(1), Hk = q.dim(2), Dk = q.dim(3)
        let Hv = v.dim(2), Dv = v.dim(3)

        // Broadcast key heads up to value heads (GQA on the linear path).
        var qh = q, kh = k
        let repeatFactor = Hv / Hk
        if repeatFactor > 1 {
            qh = repeated(q, count: repeatFactor, axis: 2)
            kh = repeated(k, count: repeatFactor, axis: 2)
        }

        var s = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)

        // Move time to axis 0 so each step is a clean first-axis index.
        let qT = qh.transposed(1, 0, 2, 3)   // [T,B,Hv,Dk]
        let kT = kh.transposed(1, 0, 2, 3)   // [T,B,Hv,Dk]
        let vT = v.transposed(1, 0, 2, 3)     // [T,B,Hv,Dv]
        let gT = g.transposed(1, 0, 2)        // [T,B,Hv]
        let betaT = beta.transposed(1, 0, 2)  // [T,B,Hv]

        var ys: [MLXArray] = []
        ys.reserveCapacity(T)
        for t in 0 ..< T {
            let qt = qT[t].reshaped(B, Hv, 1, Dk).asType(.float32)   // [B,Hv,1,Dk]
            let kt = kT[t].reshaped(B, Hv, 1, Dk).asType(.float32)   // [B,Hv,1,Dk]
            let vt = vT[t].asType(.float32)                          // [B,Hv,Dv]
            let gt = gT[t].reshaped(B, Hv, 1, 1).asType(.float32)    // [B,Hv,1,1]
            let bt = betaT[t].reshaped(B, Hv, 1).asType(.float32)    // [B,Hv,1]

            s = s * gt                                               // decay
            let kvMem = (s * kt).sum(axis: -1)                       // [B,Hv,Dv]
            let delta = (vt - kvMem) * bt                            // [B,Hv,Dv]
            s = s + kt * delta.reshaped(B, Hv, Dv, 1)                // rank-1 update
            let yt = (s * qt).sum(axis: -1)                          // [B,Hv,Dv]
            ys.append(yt)
        }
        let y = stacked(ys, axis: 1).asType(q.dtype)                 // [B,T,Hv,Dv]
        return (y, s)
    }

    private static func repeated(_ a: MLXArray, count: Int, axis: Int) -> MLXArray {
        MLX.repeated(a, count: count, axis: axis)
    }
}

/// The Qwen3.5 linear-attention block. Projects the hidden state into q/k/v (via a
/// depthwise causal conv) plus the gating scalars, runs the delta-rule scan, applies
/// the gated norm, and projects back. Mirrors `Qwen3_5GatedDeltaNet`.
nonisolated final class Qwen35GatedDeltaNet: Module {
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "norm") var norm: Qwen35RMSNormGated

    // Depthwise causal conv weight, sanitised to [convDim, kernelSize] at load
    // (HF ships [convDim, 1, kernelSize]).
    @ParameterInfo(key: "conv1d.weight") var convWeight: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    init(_ config: Qwen35TextConfig) {
        let hidden = config.hiddenSize
        self.numVHeads = config.linearNumValueHeads
        self.numKHeads = config.linearNumKeyHeads
        self.headKDim = config.linearKeyHeadDim
        self.headVDim = config.linearValueHeadDim
        self.keyDim = headKDim * numKHeads
        self.valueDim = headVDim * numVHeads
        self.convKernelSize = config.linearConvKernelDim
        self.convDim = keyDim * 2 + valueDim

        self._inProjQKV.wrappedValue = Linear(hidden, keyDim * 2 + valueDim, bias: false)
        self._inProjZ.wrappedValue = Linear(hidden, valueDim, bias: false)
        self._inProjB.wrappedValue = Linear(hidden, numVHeads, bias: false)
        self._inProjA.wrappedValue = Linear(hidden, numVHeads, bias: false)
        self._outProj.wrappedValue = Linear(valueDim, hidden, bias: false)
        self._norm.wrappedValue = Qwen35RMSNormGated(dimensions: headVDim, eps: config.rmsNormEps)

        self._convWeight.wrappedValue = MLXArray.zeros([convDim, config.linearConvKernelDim])
        self._dtBias.wrappedValue = MLXArray.ones([numVHeads])
        self._aLog.wrappedValue = MLXArray.zeros([numVHeads])
        super.init()
    }

    /// Prefill forward with a zero initial state (Slice 1). `inputs`: `[B, S, hidden]`.
    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        let B = inputs.dim(0), S = inputs.dim(1)

        let mixedQKV = inProjQKV(inputs)          // [B,S,convDim]
        let z = inProjZ(inputs).reshaped(B, S, -1, headVDim)
        let b = inProjB(inputs)                   // [B,S,Hv]
        let a = inProjA(inputs)                   // [B,S,Hv]

        // Causal depthwise conv with a zeroed left state, then silu.
        let convState = MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        let convInput = concatenated([convState, mixedQKV], axis: 1)  // [B, K-1+S, convDim]
        let convOut = Qwen35Math.silu(depthwiseCausalConv(convInput, length: S))

        // Split conv output into q / k / v and reshape to heads.
        let qFlat = convOut[0..., 0..., 0 ..< keyDim]
        let kFlat = convOut[0..., 0..., keyDim ..< (2 * keyDim)]
        let vFlat = convOut[0..., 0..., (2 * keyDim) ..< convDim]
        var q = qFlat.reshaped(B, S, numKHeads, headKDim)
        var k = kFlat.reshaped(B, S, numKHeads, headKDim)
        let vv = vFlat.reshaped(B, S, numVHeads, headVDim)

        // Per-head RMS-norm of q,k with the reference's inverse-scale factors.
        let invScale = pow(Float(headKDim), -0.5)
        q = (invScale * invScale) * rmsNormNoWeight(q)
        k = invScale * rmsNormNoWeight(k)

        let (g, beta) = Qwen35GatedDelta.computeGBeta(aLog: aLog, dtBias: dtBias, a: a, b: b)
        let (out, _) = Qwen35GatedDelta.ops(q: q, k: k, v: vv, g: g, beta: beta, state: nil)

        let normed = norm(out, gate: z)          // gated RMSNorm, [B,S,Hv,Dv]
        return outProj(normed.reshaped(B, S, -1))
    }

    /// Depthwise causal conv over `convInput` `[B, K-1+S, convDim]` → `[B, S, convDim]`.
    /// `convWeight` is `[convDim, K]`; output position s = Σ_j input[s+j] * w[:, j].
    private func depthwiseCausalConv(_ convInput: MLXArray, length S: Int) -> MLXArray {
        var acc = MLXArray.zeros([convInput.dim(0), S, convDim], dtype: .float32)
        let cin = convInput.asType(.float32)
        for j in 0 ..< convKernelSize {
            let slice = cin[0..., j ..< (j + S), 0...]                 // [B,S,convDim]
            let wj = convWeight[0..., j].asType(.float32).reshaped(1, 1, convDim)
            acc = acc + slice * wj
        }
        return acc.asType(convInput.dtype)
    }

    /// RMS-norm over the last axis with no learnable weight (reference passes `None`).
    private func rmsNormNoWeight(_ x: MLXArray) -> MLXArray {
        let d = x.dim(-1)
        let ones = MLXArray.ones([d]).asType(x.dtype)
        return MLXFast.rmsNorm(x, weight: ones, eps: 1e-6)
    }

    /// HF ships the conv weight as `[convDim, 1, kernelSize]`; collapse the singleton
    /// in-channel axis to `[convDim, kernelSize]` for the manual depthwise conv.
    static func sanitizeConvWeight(_ weights: [String: MLXArray], prefix: String) -> [String: MLXArray] {
        var out = weights
        let key = "\(prefix)conv1d.weight"
        if let w = out[key], w.ndim == 3 {
            out[key] = w.reshaped(w.dim(0), w.dim(2))
        }
        return out
    }
}
