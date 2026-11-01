import Foundation
import MLX
import MLXNN

// FLUX.2 Klein double-stream transformer block (MP7-3e).
// Structure follows VincentGourbin/flux-2-swift-mlx @ 315c27e (MIT, © 2025 Vincent Gourbin) `Flux2TransformerBlock`, `Flux2Attention` and `Flux2FeedForward`;
// the numbers are checked against diffusers 0.37.1 `Flux2TransformerBlock` (tiny-config fp32 golden, MLXUITests/Fixtures/Flux2Klein/g3a_transformer.safetensors,
// `mid.double_0_*`). diffusers is the arbiter wherever the two disagree.
//
// Keys are the diffusers snake_case names, so the checkpoint loads without a remap: `attn.{to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_add_out}.weight`,
// `attn.to_out.0.weight` (a one-element array module, because diffusers keeps `to_out` as a ModuleList [Linear, Dropout]), `attn.{norm_q,norm_k,norm_added_q,
// norm_added_k}.weight` (RMSNorm over head_dim, eps = the config eps, 1e-6) and `{ff,ff_context}.linear_{in,out}.weight`. Everything is bias-free. The four block
// LayerNorms (norm1, norm1_context, norm2, norm2_context) have no affine parameters, so they add no keys.
//
// Order. The joint sequence is [text, image] everywhere: q/k/v concatenation, the rotary table (`Flux2KleinRoPE.jointCosSin`, text ids first) and the attention
// split. The block RETURNS (text, image), the order of diffusers' `Flux2TransformerBlock.forward` (`encoder_hidden_states, hidden_states`).
//
// Dtypes. No explicit float32 upcasts beyond what the primitives do: LayerNorm and RMSNorm (MLXFast) accumulate in float32 internally and return the input dtype,
// as torch does for bf16; the rotary step is float32 inside `Flux2KleinRoPE.applyRotary`; softmax runs inside the fused attention kernel.

/// SwiGLU feed-forward: `linear_out(silu(gate) * up)` where `linear_in` produces `[gate | up]` (diffusers `Flux2SwiGLU`: `x1, x2 = chunk(2); silu(x1) * x2`).
nonisolated final class Flux2KleinFeedForward: Module {
    @ModuleInfo(key: "linear_in") var linearIn: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    init(dim: Int, hidden: Int) {
        _linearIn.wrappedValue = Linear(dim, hidden * 2, bias: false)
        _linearOut.wrappedValue = Linear(hidden, dim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = MLX.split(linearIn(x), parts: 2, axis: -1)
        return linearOut(silu(parts[0]) * parts[1])
    }
}

/// Joint attention of the double block: image q/k/v from `to_q/to_k/to_v`, text q/k/v from `add_q_proj/add_k_proj/add_v_proj`.
nonisolated final class Flux2KleinJointAttention: Module {
    let heads: Int
    let headDim: Int
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "add_q_proj") var addQ: Linear
    @ModuleInfo(key: "add_k_proj") var addK: Linear
    @ModuleInfo(key: "add_v_proj") var addV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]          // one element: the key is `to_out.0`
    @ModuleInfo(key: "to_add_out") var toAddOut: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm

    init(dim: Int, heads: Int, headDim: Int, eps: Float) {
        self.heads = heads
        self.headDim = headDim
        let inner = heads * headDim
        _toQ.wrappedValue = Linear(dim, inner, bias: false)
        _toK.wrappedValue = Linear(dim, inner, bias: false)
        _toV.wrappedValue = Linear(dim, inner, bias: false)
        _addQ.wrappedValue = Linear(dim, inner, bias: false)
        _addK.wrappedValue = Linear(dim, inner, bias: false)
        _addV.wrappedValue = Linear(dim, inner, bias: false)
        _toOut.wrappedValue = [Linear(inner, dim, bias: false)]
        _toAddOut.wrappedValue = Linear(inner, dim, bias: false)
        _normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        _normK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        _normAddedQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        _normAddedK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        super.init()
    }

    /// `img` `[B, Si, dim]`, `txt` `[B, St, dim]` (both already modulated); `cos`/`sin` `[St + Si, headDim]`, text rows first. Returns `(txt, img)` attention outputs.
    func callAsFunction(img: MLXArray, txt: MLXArray, cos: MLXArray, sin: MLXArray) -> (txt: MLXArray, img: MLXArray) {
        let b = img.dim(0), si = img.dim(1), st = txt.dim(1)
        func heads(_ x: MLXArray, _ s: Int) -> MLXArray { x.reshaped(b, s, self.heads, headDim) }
        let q = concatenated([normAddedQ(heads(addQ(txt), st)), normQ(heads(toQ(img), si))], axis: 1)
        let k = concatenated([normAddedK(heads(addK(txt), st)), normK(heads(toK(img), si))], axis: 1)
        let v = concatenated([heads(addV(txt), st), heads(toV(img), si)], axis: 1)
        let qr = Flux2KleinRoPE.applyRotary(q, cos: cos, sin: sin).transposed(0, 2, 1, 3)
        let kr = Flux2KleinRoPE.applyRotary(k, cos: cos, sin: sin).transposed(0, 2, 1, 3)
        let o = MLXFast.scaledDotProductAttention(queries: qr, keys: kr, values: v.transposed(0, 2, 1, 3), scale: pow(Float(headDim), -0.5), mask: nil)
        let joint = o.transposed(0, 2, 1, 3).reshaped(b, st + si, self.heads * headDim)
        let parts = MLX.split(joint, indices: [st], axis: 1)
        return (toAddOut(parts[0]), toOut[0](parts[1]))
    }
}

/// One of the 5 double-stream blocks. Weights are per block; the modulation sets come from the shared `Flux2KleinEmbedders.Conditioning`.
nonisolated final class Flux2KleinDoubleBlock: Module {
    let norm1: LayerNorm
    let norm1Context: LayerNorm
    let norm2: LayerNorm
    let norm2Context: LayerNorm
    let attn: Flux2KleinJointAttention
    let ff: Flux2KleinFeedForward
    @ModuleInfo(key: "ff_context") var ffContext: Flux2KleinFeedForward

    init(dim: Int, heads: Int, headDim: Int, mlpRatio: Double, eps: Float) {
        norm1 = LayerNorm(dimensions: dim, eps: eps, affine: false)
        norm1Context = LayerNorm(dimensions: dim, eps: eps, affine: false)
        norm2 = LayerNorm(dimensions: dim, eps: eps, affine: false)
        norm2Context = LayerNorm(dimensions: dim, eps: eps, affine: false)
        attn = Flux2KleinJointAttention(dim: dim, heads: heads, headDim: headDim, eps: eps)
        let hidden = Int(Double(dim) * mlpRatio)
        ff = Flux2KleinFeedForward(dim: dim, hidden: hidden)
        _ffContext.wrappedValue = Flux2KleinFeedForward(dim: dim, hidden: hidden)
        super.init()
    }

    convenience init(_ c: Flux2KleinTransformerConfig) {
        self.init(dim: c.innerDim, heads: c.numAttentionHeads, headDim: c.attentionHeadDim, mlpRatio: c.mlpRatio, eps: Float(c.eps))
    }

    /// `img` `[B, Si, dim]`, `txt` `[B, St, dim]`; `imgMod` / `txtMod` are the two modulation sets (index 0 = attention, 1 = feed-forward) of
    /// `Conditioning.doubleImg` / `.doubleTxt`; `cos`/`sin` from `Flux2KleinRoPE.jointCosSin` (text ids first). Returns `(txt, img)`, as diffusers does.
    func callAsFunction(img: MLXArray, txt: MLXArray, imgMod: [Flux2KleinModulationSet], txtMod: [Flux2KleinModulationSet],
                        cos: MLXArray, sin: MLXArray) -> (txt: MLXArray, img: MLXArray) {
        precondition(imgMod.count == 2 && txtMod.count == 2, "a double block takes two modulation sets per stream")
        func modulate(_ x: MLXArray, _ m: Flux2KleinModulationSet) -> MLXArray { x * (1 + m.scale) + m.shift }

        let a = attn(img: modulate(norm1(img), imgMod[0]), txt: modulate(norm1Context(txt), txtMod[0]), cos: cos, sin: sin)
        var h = img + imgMod[0].gate * a.img
        var t = txt + txtMod[0].gate * a.txt

        h = h + imgMod[1].gate * ff(modulate(norm2(h), imgMod[1]))
        t = t + txtMod[1].gate * ffContext(modulate(norm2Context(t), txtMod[1]))
        return (t, h)
    }
}
