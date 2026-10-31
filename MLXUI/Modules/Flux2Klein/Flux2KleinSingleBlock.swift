import Foundation
import MLX
import MLXNN

// FLUX.2 Klein single-stream ("parallel") transformer block (MP7-3f).
// Structure follows VincentGourbin/flux-2-swift-mlx @ 315c27e (MIT, © 2025 Vincent Gourbin) `Flux2SingleBlock` and `Flux2ParallelAttention`;
// the numbers are checked against diffusers 0.37.1 `Flux2SingleTransformerBlock` / `Flux2ParallelSelfAttention` (tiny-config fp32 golden,
// MLXUITests/Fixtures/Flux2Klein/g3f_dit_stack.safetensors, `mid.single_*`). diffusers is the arbiter wherever the two disagree.
//
// One fused input projection `attn.to_qkv_mlp_proj` (dim -> 3 * inner + 2 * mlpHidden) produces q, k, v AND the SwiGLU pair in one matmul; one fused output projection
// `attn.to_out` (inner + mlpHidden -> dim) consumes `[attention, mlp]`. Keys are the diffusers names, so the checkpoint loads without a remap:
// `attn.to_qkv_mlp_proj.weight`, `attn.to_out.weight` (a plain Linear here, NOT the `to_out.0` array of the double block), `attn.norm_q.weight`, `attn.norm_k.weight`.
// Everything is bias-free. The block's one LayerNorm has no affine parameters, so it adds no key.
//
// Order (diffusers `Flux2ParallelSelfAttnProcessor`): split the projection as [q | k | v] (3 * inner) then [gate | up] (2 * mlpHidden); q and k go through RMSNorm over head_dim
// (eps = the config eps) and the rotary table, then SDPA; the mlp half is `silu(gate) * up` (`x1, x2 = chunk(2)`, same as the double block's feed-forward); the output projection
// takes `concat([attention, mlp])` in that order. The block runs on the JOINT sequence [text, image] (text first) and the residual is `x + gate * out`.

nonisolated final class Flux2KleinParallelAttention: Module {
    let heads: Int
    let headDim: Int
    let mlpHidden: Int
    @ModuleInfo(key: "to_qkv_mlp_proj") var toQKVMLP: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm

    init(dim: Int, heads: Int, headDim: Int, mlpHidden: Int, eps: Float) {
        self.heads = heads
        self.headDim = headDim
        self.mlpHidden = mlpHidden
        let inner = heads * headDim
        _toQKVMLP.wrappedValue = Linear(dim, 3 * inner + 2 * mlpHidden, bias: false)
        _toOut.wrappedValue = Linear(inner + mlpHidden, dim, bias: false)
        _normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        _normK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        super.init()
    }

    /// `x` `[B, S, dim]` (already modulated), `cos`/`sin` `[S, headDim]` for the joint [text, image] sequence. Returns `[B, S, dim]`.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let b = x.dim(0), s = x.dim(1)
        let inner = heads * headDim
        let halves = MLX.split(toQKVMLP(x), indices: [3 * inner], axis: -1)        // [q k v | gate up]
        let qkv = MLX.split(halves[0], parts: 3, axis: -1)
        let q = normQ(qkv[0].reshaped(b, s, heads, headDim))
        let k = normK(qkv[1].reshaped(b, s, heads, headDim))
        let v = qkv[2].reshaped(b, s, heads, headDim)
        let qr = Flux2KleinRoPE.applyRotary(q, cos: cos, sin: sin).transposed(0, 2, 1, 3)
        let kr = Flux2KleinRoPE.applyRotary(k, cos: cos, sin: sin).transposed(0, 2, 1, 3)
        let o = MLXFast.scaledDotProductAttention(queries: qr, keys: kr, values: v.transposed(0, 2, 1, 3), scale: pow(Float(headDim), -0.5), mask: nil)
        let attention = o.transposed(0, 2, 1, 3).reshaped(b, s, inner)
        let pair = MLX.split(halves[1], parts: 2, axis: -1)
        let mlp = silu(pair[0]) * pair[1]
        return toOut(concatenated([attention, mlp], axis: -1))
    }
}

/// One of the 20 single-stream blocks. Weights are per block; the one modulation set comes from `Flux2KleinEmbedders.Conditioning.single[0]`.
nonisolated final class Flux2KleinSingleBlock: Module {
    let norm: LayerNorm
    let attn: Flux2KleinParallelAttention

    /// The four tensors a block owns, by their name inside the block.
    static let expectedTensorNames: Set<String> = [
        "attn.to_qkv_mlp_proj.weight", "attn.to_out.weight", "attn.norm_q.weight", "attn.norm_k.weight",
    ]

    init(dim: Int, heads: Int, headDim: Int, mlpRatio: Double, eps: Float) {
        norm = LayerNorm(dimensions: dim, eps: eps, affine: false)
        attn = Flux2KleinParallelAttention(dim: dim, heads: heads, headDim: headDim, mlpHidden: Int(Double(dim) * mlpRatio), eps: eps)
        super.init()
    }

    convenience init(_ c: Flux2KleinTransformerConfig) {
        self.init(dim: c.innerDim, heads: c.numAttentionHeads, headDim: c.attentionHeadDim, mlpRatio: c.mlpRatio, eps: Float(c.eps))
    }

    /// `x` is the joint `[B, St + Si, dim]` stream, text rows first; `mod` is `Conditioning.single[0]`; `cos`/`sin` from `Flux2KleinRoPE.jointCosSin`.
    func callAsFunction(_ x: MLXArray, mod: Flux2KleinModulationSet, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = norm(x) * (1 + mod.scale) + mod.shift
        return x + mod.gate * attn(h, cos: cos, sin: sin)
    }
}
