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
nonisolated func wanApplyRoPE1D(_ x: MLXArray, freqs: MLXArray) -> MLXArray {
    let d  = x.dim(-1)
    let x1 = x[.ellipsis, 0 ..< d / 2]
    let x2 = x[.ellipsis, d / 2 ..< d]    // explicit upper bound — PartialRangeFrom not supported
    let c  = cos(freqs).asType(x.dtype)
    let s  = sin(freqs).asType(x.dtype)
    return MLX.concatenated([x1 * c - x2 * s, x2 * c + x1 * s], axis: -1)
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
