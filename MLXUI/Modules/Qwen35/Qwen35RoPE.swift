//
//  Qwen35RoPE.swift
//  MLXUI — Qwen3.5 (qwen3_5) hybrid GatedDeltaNet port, Slice QB2.
//
//  Multimodal RoPE (mrope) for the full-attention layers. Port of the
//  `style="interleaved"` path of `MRoPERotaryEmbedding` + `compute_mrope_frequencies`
//  in Blaizzy/mlx-vlm `models/rope_utils.py`, specialised to Qwen3.5:
//    - `inv_freq = 1 / base^(arange(0,dim,2)/dim)`, `dim = rotary_dim` (= head_dim * 0.25).
//    - "interleaved" here refers to how the 3 position axes (time/height/width) are
//      interleaved across frequencies via a position selector — the ROTATION itself is
//      the standard NeoX half-split (`rotate_half`), and `cos`/`sin` are width `rotary_dim`
//      (freqs duplicated: `concat([freqs, freqs])`).
//
//  Text generation uses 2-D position ids (one scalar position per token, shared across
//  all 3 mrope axes → the axis selector collapses). 3-D (per-axis) position ids arrive
//  with the vision tower (QB7); both layouts are handled here.
//

import Foundation
import MLX

/// Precomputes `inv_freq` and the interleaved position-axis selector, and produces the
/// `cos`/`sin` tensors consumed by `Qwen35RoPEApply` in the attention layer.
nonisolated final class Qwen35RoPE {
    let dim: Int              // rotary_dim (number of rotated head dims)
    let halfDim: Int          // rotary_dim / 2 (number of frequencies)
    let invFreq: MLXArray     // [halfDim]
    let positionSelector: MLXArray  // [halfDim] Int32, per-frequency mrope axis

    init(rotaryDim: Int, base: Double, mropeSection: [Int]) {
        self.dim = rotaryDim
        self.halfDim = rotaryDim / 2
        // inv_freq = 1 / base^(arange(0, dim, 2) / dim)
        let exponents = MLXArray(stride(from: 0, to: rotaryDim, by: 2).map { Float($0) }) / Float(rotaryDim)
        self.invFreq = 1.0 / MLX.pow(MLXArray(Float(base)), exponents)
        self.positionSelector = MLXArray(Self.interleavedSelector(mropeSection: mropeSection, freqDim: rotaryDim / 2))
    }

    /// Per-frequency axis assignment for the interleaved style (mirrors
    /// `_interleaved_position_selector`): frequency 0 → axis 0 (time); axes 1/2
    /// (height/width) are laced in at offsets 1 and 2, stride 3.
    static func interleavedSelector(mropeSection: [Int], freqDim: Int) -> [Int32] {
        var selector = [Int32](repeating: 0, count: freqDim)
        for (dim, offset) in [(1, 1), (2, 2)] {
            guard dim < mropeSection.count else { continue }
            var idx = offset
            let limit = min(mropeSection[dim] * 3, freqDim)
            while idx < limit {
                selector[idx] = Int32(dim)
                idx += 3
            }
        }
        return selector
    }

    /// Compute `cos`/`sin` for the given position ids, each shaped `[B, 1, T, rotaryDim]`
    /// (head axis already unsqueezed for broadcast in attention).
    ///
    /// - `positionIds` is either 2-D `[B, T]` (text) or 3-D `[3, B, T]` (per-axis).
    func cosSin(positionIds: MLXArray, dtype: DType) -> (cos: MLXArray, sin: MLXArray) {
        let freqs: MLXArray
        if positionIds.ndim == 2 {
            // Text: same scalar position across all axes → [B,T,1] * invFreq → [B,T,halfDim].
            let B = positionIds.dim(0), T = positionIds.dim(1)
            freqs = positionIds.asType(.float32).reshaped(B, T, 1) * invFreq
        } else {
            // Per-axis: pick the axis feeding each frequency, then [B,T,halfDim].
            let positions = MLX.take(positionIds, positionSelector, axis: 0)  // [halfDim,B,T]
            freqs = positions.transposed(1, 2, 0).asType(.float32) * invFreq  // [B,T,halfDim]
        }
        let emb = concatenated([freqs, freqs], axis: -1)   // [B,T,rotaryDim]
        let B = emb.dim(0), T = emb.dim(1)
        let cos = MLX.cos(emb).reshaped(B, 1, T, dim).asType(dtype)
        let sin = MLX.sin(emb).reshaped(B, 1, T, dim).asType(dtype)
        return (cos, sin)
    }

    /// Text-decode position ids `[1, length]` = `arange(offset, offset+length)`.
    static func textPositionIds(offset: Int, length: Int) -> MLXArray {
        MLXArray(Int32(offset) ..< Int32(offset + length)).reshaped(1, length)
    }
}
