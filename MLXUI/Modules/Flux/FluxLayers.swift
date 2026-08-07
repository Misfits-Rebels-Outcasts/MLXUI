import Foundation
import MLX

/// The small FLUX building blocks from `ml-explore/mlx-examples/flux/flux/layers.py`:
/// rotary position embeddings (`EmbedND`, `_rope`, `_apply_rope`), the fused rope attention
/// (`_attention`), and `timestep_embedding`. AM4d.
///
/// For `Flux-1.lite-8B` the config is `axes_dim: [16, 56, 56]`, `theta: 10000`, head dim
/// `hidden/heads = 3072/24 = 128` — so `EmbedND` produces a 128-dim rope (16+56+56).
nonisolated enum FluxLayers {

    // MARK: - RoPE

    /// `_rope(pos, dim, theta)`: builds the 4-part rope block `[[cos,-sin],[sin,cos]]` per
    /// half-dim. Output shape `(..., dim/2, 2, 2)`.
    static func rope(pos: MLXArray, dim: Int, theta: Float) -> MLXArray {
        let scale = MLXArray.arange(0, dim, step: 2) / dim
        let omega = 1.0 / pow(theta, scale)
        let x = pos.expandedDimensions(axis: -1) * omega  // (..., 1) * (dim/2)
        let cosx = cos(x)
        let sinx = sin(x)
        let pe = stacked([cosx, -sinx, sinx, cosx], axis: -1)          // (..., dim/2, 4)
        let halfDim = pe.shape[pe.shape.count - 2]
        let outerShape = pe.shape.dropLast(2)
        return pe.reshaped(Array(outerShape) + [halfDim, 2, 2])
    }

    /// `_apply_rope(x, pe)`: the fused `a*b + c*d` rotation of the last axis in pairs.
    static func applyRope(_ x: MLXArray, pe: MLXArray) -> MLXArray {
        let s = x.shape
        let reshaped = x.reshaped(Array(s.dropLast()) + [-1, 1, 2])
        let ab = reshaped[.ellipsis, 0] * pe[.ellipsis, 0]
        let cd = reshaped[.ellipsis, 1] * pe[.ellipsis, 1]
        return (ab + cd).reshaped(s)
    }

    /// `_attention(q, k, v, pe)`: apply rope to q/k, scaled dot-product attention, back to
    /// `(B, L, -1)`. `q/k/v` are `(B, H, L, D)`; `pe` is `(B, 1, L, D/2, 2, 2)` from `EmbedND`.
    static func attention(q: MLXArray, k: MLXArray, v: MLXArray, pe: MLXArray) -> MLXArray {
        let b = q.dim(0), l = q.dim(2)
        let qr = applyRope(q, pe: pe)
        let kr = applyRope(k, pe: pe)
        let d = Float(q.dim(3))
        let out = MLXFast.scaledDotProductAttention(
            queries: qr, keys: kr, values: v, scale: pow(d, -0.5), mask: nil)
        return out.transposed(0, 2, 1, 3).reshaped([b, l, -1])
    }

    /// `EmbedND`: concat the per-axis ropes along the (dim/2) axis, then add a head-axis
    /// singleton. Input `ids` is `(B, L, nAxes)` int; output `(B, 1, L, dim/2, 2, 2)`.
    static func embedND(ids: MLXArray, dim: Int, theta: Float, axesDim: [Int]) -> MLXArray {
        let pes = (0 ..< axesDim.count).map { i in
            rope(pos: ids[.ellipsis, i], dim: axesDim[i], theta: theta)
        }
        let pe = concatenated(pes, axis: -3)
        return pe.expandedDimensions(axis: 1)
    }

    // MARK: - Timestep embedding

    /// `timestep_embedding(t, dim, max_period, time_factor)` — the reference's variant:
    /// `freqs = exp(arange(0, half)/half * -log(max_period))`,
    /// then `[cos, sin]` of `time_factor * t * freqs`.
    static func timestepEmbedding(
        _ t: MLXArray,
        dim: Int,
        maxPeriod: Float = 10_000,
        timeFactor: Float = 1000.0
    ) -> MLXArray {
        let half = dim / 2
        let freqs = MLXArray.arange(0, half) / half
        let logFreqs = freqs * -log(maxPeriod)
        let expFreqs = exp(logFreqs)
        let x = (timeFactor * t).expandedDimensions(axis: -1) * expFreqs.expandedDimensions(axis: 0)
        return concatenated([cos(x), sin(x)], axis: -1).asType(t.dtype)
    }
}
