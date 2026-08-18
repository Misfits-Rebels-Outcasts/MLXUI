import Foundation
import MLX
import MLXNN

// MARK: - 3D Rotary Position Embedding for WAN DiT

/// Build a 1-D sinusoidal frequency table. Returns [seqLen, dim/2].
nonisolated func wanBuildFreqs(seqLen: Int, dim: Int, base: Float = 10000.0) -> MLXArray {
    let halfDim = dim / 2
    let theta = 1.0 / pow(base, MLXArray(Int32(0) ..< Int32(halfDim)).asType(.float32) / Float(halfDim))
    let pos   = MLXArray(Int32(0) ..< Int32(seqLen)).asType(.float32)
    return pos.expandedDimensions(axis: 1) * theta.expandedDimensions(axis: 0) // [S, D/2]
}

/// Apply 1-D RoPE to x [..., D]. freqs [..., D/2] must be broadcastable to x.
/// Uses the interleaved (view-as-complex) convention: pair k = (x[2k], x[2k+1]).
/// This matches WAN 2.1's reference apply_rotary_emb which calls view_as_complex
/// on consecutive element pairs — NOT the split (first-half / second-half) variant.
nonisolated func wanApplyRoPE1D(_ x: MLXArray, freqs: MLXArray) -> MLXArray {
    let d = x.dim(-1)
    // Reshape so consecutive pairs become (real, imaginary): [..., D] → [..., D/2, 2]
    var pairShape = x.shape; pairShape[pairShape.count - 1] = d / 2; pairShape.append(2)
    let xp  = x.reshaped(pairShape)
    let xRe = xp[.ellipsis, 0]                 // [..., D/2] — even indices
    let xIm = xp[.ellipsis, 1]                 // [..., D/2] — odd indices
    let c   = cos(freqs).asType(x.dtype)
    let s   = sin(freqs).asType(x.dtype)
    let rRe = xRe * c - xIm * s
    let rIm = xRe * s + xIm * c
    // Interleave back: stack along a new last axis → [..., D/2, 2] → [..., D]
    let out = MLX.concatenated([rRe.expandedDimensions(axis: -1),
                                rIm.expandedDimensions(axis: -1)], axis: -1)
    return out.reshaped(x.shape)
}

/// Build the (t, h, w) position index arrays for a flattened sequence of nT×nH×nW tokens.
nonisolated func wanMakePositions(nT: Int, nH: Int, nW: Int)
    -> (tPos: MLXArray, hPos: MLXArray, wPos: MLXArray)
{
    let S = nT * nH * nW
    var ts = [Int32](); ts.reserveCapacity(S)
    var hs = [Int32](); hs.reserveCapacity(S)
    var ws = [Int32](); ws.reserveCapacity(S)
    for t in 0 ..< nT {
        for h in 0 ..< nH {
            for w in 0 ..< nW {
                ts.append(Int32(t)); hs.append(Int32(h)); ws.append(Int32(w))
            }
        }
    }
    return (MLXArray(ts), MLXArray(hs), MLXArray(ws))
}

/// Apply 3-D RoPE to Q or K of shape [B, S, nHeads, headDim].
/// headDim is split as: tDim = headDim - 2*dHW, hDim = wDim = dHW, where dHW = 2*(headDim/6).
nonisolated func wanApplyRoPE3D(
    _ x:     MLXArray,
    freqsT:  MLXArray,   // [maxT, tDim/2]
    freqsH:  MLXArray,   // [maxH, dHW/2]
    freqsW:  MLXArray,   // [maxW, dHW/2]
    tPos:    MLXArray,   // [S] int
    hPos:    MLXArray,   // [S] int
    wPos:    MLXArray    // [S] int
) -> MLXArray {
    let d   = x.dim(-1)
    let dHW = 2 * (d / 6)
    let dT  = d - 2 * dHW

    let xT = x[.ellipsis, 0 ..< dT]
    let xH = x[.ellipsis, dT ..< dT + dHW]
    let xW = x[.ellipsis, dT + dHW ..< d]   // explicit upper bound

    // Gather per-position freqs: [S, dim/2] → [1, S, 1, dim/2] for [B, S, H, D] broadcasting.
    let fT = freqsT[tPos].expandedDimensions(axis: 0).expandedDimensions(axis: 2)
    let fH = freqsH[hPos].expandedDimensions(axis: 0).expandedDimensions(axis: 2)
    let fW = freqsW[wPos].expandedDimensions(axis: 0).expandedDimensions(axis: 2)

    return MLX.concatenated([
        wanApplyRoPE1D(xT, freqs: fT),
        wanApplyRoPE1D(xH, freqs: fH),
        wanApplyRoPE1D(xW, freqs: fW),
    ], axis: -1)
}
