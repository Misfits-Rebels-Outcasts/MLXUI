// Empirical probe for the MLX-Swift gap analysis (Q9).
//
// Built against the pinned local mlx-swift checkout (0.32.3, bundled MLX C++ 0.32.2).
// Run as:  swift run -c release probe <check>
// Each check prints deterministic values as `name=comma,separated,floats` so the
// Python MLX side (mlx 0.31.2, repo .venv) can print the same lines and be diffed.

import Foundation
import MLX
import MLXNN
import MLXRandom

func out(_ name: String, _ a: MLXArray) {
    eval(a)
    let v = a.asType(.float32).asArray(Float.self)
    let s = v.map { String(format: "%.6f", $0) }.joined(separator: ",")
    print("\(name)=\(s)")
}

func fary(_ n: Int, _ f: (Int) -> Float) -> MLXArray {
    MLXArray((0 ..< n).map { f($0) })
}

func check(_ name: String) {
    switch name {
    case "version":
        print("mlxswift_probe_ok")
        print("device=\(Device.defaultDevice())")

    case "slice":
        // dit_model.py:16-21 _rotate_half, plus Python ellipsis/newaxis/stride slicing.
        let x = fary(8) { Float($0) / 7.0 }.reshaped(1, 1, 2, 4)
        let half = x.dim(-1) / 2
        let x1 = x[.ellipsis, 0 ..< half]
        let x2 = x[.ellipsis, half...]
        out("rotate_half", concatenated([-x2, x1], axis: -1))

        let y = fary(12) { Float($0) }.reshaped(2, 6)
        out("slice_partial_from", y[0..., 2...])          // python y[:, 2:]
        out("slice_stride2", y[0..., .stride(by: 2)])     // python y[:, 0::2]
        out("slice_index", y[1])                          // python y[1]
        let z = MLXArray([Float(1), 2, 3])
        print("slice_newaxis_shape=\(z[0..., .newAxis].shape)")  // python z[:, None]
        print("slice_neg_index=\(y[-1].shape)")           // python y[-1]

    case "mask":
        // dit_model.py:38-55 _create_sliding_window_mask (window 128 upstream, 2 here).
        let n = 8
        let seqLen = 8
        let window = 2
        let indices = MLX.arange(0, n)
        let diff = abs(indices[0..., .newAxis] - indices[.newAxis, 0...])
        let zero = MLXArray.zeros([seqLen, seqLen], dtype: .float32)
        let neg = MLXArray.full([seqLen, seqLen], values: MLXArray(-1e9 as Float), dtype: .float32)
        let mask = which(diff .<= window, zero, neg)
        out("window_mask8", mask)
        out("window_mask4x4", mask[0..., 0 ..< 4])

        // n-dtype: upstream builds the mask in the hidden-state dtype
        // (dit_model.py:588-599). -1e9 overflows float16.
        let m16 = mask.asType(.float16)
        let mb = mask.asType(.bfloat16)
        print("mask_f16_zero=\(m16[0, 0].item(Float.self))")
        print("mask_f16_neg1e9=\(m16[0, 3].item(Float.self))")
        print("mask_bf16_neg1e9=\(mb[0, 3].item(Float.self))")

    case "sdpa":
        // dit_model.py:237-239 fast.scaled_dot_product_attention(scale=..., mask=[1,1,L,L]).
        let shape = [1, 2, 8, 4]
        let q = fary(64) { Float($0 % 11) / 11.0 }.reshaped(shape)
        let k = fary(64) { Float(($0 + 3) % 13) / 13.0 }.reshaped(shape)
        let v = fary(64) { Float(($0 + 7) % 17) / 17.0 }.reshaped(shape)
        let indices = MLX.arange(0, 8)
        let diff = abs(indices[0..., .newAxis] - indices[.newAxis, 0...])
        let mask = which(
            diff .<= 2,
            MLXArray.zeros([8, 8], dtype: .float32),
            MLXArray.full([8, 8], values: MLXArray(-1e9 as Float), dtype: .float32)
        )[.newAxis, .newAxis]
        out("sdpa_nomask", scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.5, mask: nil))
        out("sdpa_mask", scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.5, mask: mask))
        // broadcast mask [8,8] without the leading [1,1] axes
        out("sdpa_mask_bcast", scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.5, mask: mask[0, 0]))
        // bf16 / float16 dtypes
        let qb = q.asType(.bfloat16), kb = k.asType(.bfloat16), vb = v.asType(.bfloat16)
        let ob = scaledDotProductAttention(queries: qb, keys: kb, values: vb, scale: 0.5, mask: mask.asType(.bfloat16))
        out("sdpa_bf16", ob)
        print("sdpa_bf16_dtype=\(ob.dtype)")
        let qh = q.asType(.float16), kh = k.asType(.float16), vh = v.asType(.float16)
        let oh = scaledDotProductAttention(queries: qh, keys: kh, values: vh, scale: 0.5, mask: mask.asType(.float16))
        out("sdpa_f16", oh)
        print("sdpa_f16_dtype=\(oh.dtype)")

    case "conv":
        // dit_model.py:469-475 proj_in Conv1d(patch_size, stride=patch_size)
        let x = fary(18) { Float($0) / 10.0 }.reshaped(1, 6, 3)
        let w = fary(36) { Float($0) / 7.0 - 2.0 }.reshaped(4, 3, 3)
        out("conv1d_s2p1", conv1d(x, w, stride: 2, padding: 1))
        out("conv1d_s1p0", conv1d(x, w, stride: 1, padding: 0))
        out("conv1d_s2p3d1", conv1d(x, w, stride: 2, padding: 3, dilation: 1))
        let c = Conv1d(inputChannels: 3, outputChannels: 4, kernelSize: 3, stride: 2, padding: 1)
        print("conv1d_module_weight_shape=\(c.weight.shape)")

    case "convtranspose":
        // vae_model.py:125-131 / dit_model.py:505-511 ConvTranspose1d
        let x = fary(12) { Float($0) / 8.0 }.reshaped(1, 4, 3)
        let w = fary(24) { Float($0) / 9.0 - 1.0 }.reshaped(4, 2, 3)  // [out, K, in]
        out("convT_s2p1", convTransposed1d(x, w, stride: 2, padding: 1))
        out("convT_s2p0", convTransposed1d(x, w, stride: 2, padding: 0))
        out("convT_s4p2", convTransposed1d(x, w, stride: 4, padding: 2))
        let ct = ConvTransposed1d(inputChannels: 3, outputChannels: 4, kernelSize: 2, stride: 2, padding: 1)
        print("convT_module_weight_shape=\(ct.weight.shape)")
        out("convT_module_out", ct(x))
        // de-conv used by dit_model.py:622 (proj_out)
        let pd = MLXArray.zeros([1, 4, 3], dtype: .float32)
        out("convT_zero", convTransposed1d(pd, w, stride: 2, padding: 1))

    case "quant":
        // upstream Python MLX port does NOT quantize; this probes whether an
        // mlx-swift port could (MLXNN.QuantizedLinear / quantizedMM layout).
        let w = fary(64 * 128) { Float($0 % 17) / 17.0 - 0.5 }.reshaped(64, 128)
        let (wq, scales, biases) = quantized(w, groupSize: 64, bits: 4)
        print("quant_wq_shape=\(wq.shape) scales_shape=\(scales.shape) biases_shape=\(biases?.shape ?? [])")
        print("quant_wq_dtype=\(wq.dtype) scales_dtype=\(scales.dtype)")
        let wd = dequantized(wq, scales: scales, biases: biases, groupSize: 64, bits: 4)
        out("dequant_row0", wd[0])
        print("dequant_dtype=\(wd.dtype)")
        let x = fary(2 * 128) { Float($0) / 128.0 }.reshaped(2, 128)
        out(
            "quantmm",
            quantizedMM(x, wq, scales: scales, biases: biases, groupSize: 64, bits: 4))
        out("dequant_matmul", matmul(x, wd.transposed(1, 0)))
        // 8-bit group 32
        let (wq8, s8, b8) = quantized(w, groupSize: 32, bits: 8)
        let wd8 = dequantized(wq8, scales: s8, biases: b8, groupSize: 32, bits: 8)
        out("dequant8_row0", wd8[0])
        // modules
        let ql = QuantizedLinear(128, 64, bias: false, groupSize: 64, bits: 4)
        print("qlinear_weight_shape=\(ql.weight.shape) scales_shape=\(ql.scales.shape) bias_nil=\(ql.bias == nil)")
        let qe = QuantizedEmbedding(embeddingCount: 8, dimensions: 128, groupSize: 64, bits: 4)
        print("qembedding_weight_shape=\(qe.weight.shape) scales_shape=\(qe.scales.shape)")

    case "quantbf16":
        // does quantize accept bf16 / f16 inputs? (Swift port loads bf16 safetensors)
        let w = fary(64 * 128) { Float($0 % 17) / 17.0 - 0.5 }.reshaped(64, 128)
        let (wqb, sb, bb) = quantized(w.asType(.bfloat16), groupSize: 64, bits: 4)
        let wdb = dequantized(wqb, scales: sb, biases: bb, groupSize: 64, bits: 4)
        print("quant_bf16_ok dtype=\(wdb.dtype)")
        out("dequant_bf16_row0", wdb[0])
        let (wqf, sf, bf) = quantized(w.asType(.float16), groupSize: 64, bits: 4)
        let wdf = dequantized(wqf, scales: sf, biases: bf, groupSize: 64, bits: 4)
        print("quant_f16_ok dtype=\(wdf.dtype)")

    case "rope":
        // dit_model.py uses a hand-rolled rotate_half RoPE, NOT mx.fast.rope.
        // This probes the fast.rope equivalent a Swift port would use.
        let x = fary(64) { Float($0) / 7.0 - 4.0 }.reshaped(1, 2, 4, 8)
        out("rope_nontrad_off3", RoPE(x, dimensions: 8, traditional: false, base: 10000, scale: 1, offset: 3))
        out("rope_trad_off3", RoPE(x, dimensions: 8, traditional: true, base: 10000, scale: 1, offset: 3))
        out("rope_nontrad_off0", RoPE(x, dimensions: 8, traditional: false, base: 10000, scale: 1, offset: 0))
        out(
            "rope_offset_array",
            RoPE(
                x, dimensions: 8, traditional: false, base: 10000, scale: 1,
                offset: MLXArray([3])))
        out(
            "rope_partial_dims4",
            RoPE(x, dimensions: 4, traditional: false, base: 10000, scale: 1, offset: 3))

    case "compile":
        // dit_generate.py:326-348 mx.compile(_raw_step) with 5 inputs;
        // mlx_vae_init.py:73-75 mx.compile of a bound model method.
        let a = fary(6) { Float($0) + 1 }.reshaped(2, 3)
        let f1 = compile { (x: MLXArray) -> MLXArray in x * 2.0 + 1.0 }
        out("compile_1arg", f1(a))
        // captured constant array (the compiled step closes over model params / mask)
        let c = MLXArray([Float(1), 2, 3])
        let fcap = compile { (x: MLXArray) -> MLXArray in x + c }
        out("compile_captured", fcap(a))
        // >3 inputs: use the [MLXArray] -> [MLXArray] overload like the DiT step
        let f5 = compile { (args: [MLXArray]) -> [MLXArray] in
            [args[0] * args[1] + args[2] + args[3] * args[4]]
        }
        let r5 = f5([
            a, MLXArray(2.0), MLXArray(0.5), a, MLXArray(0.25),
        ])
        out("compile_5args", r5[0])
        // typed 2-arg and 3-arg overloads (max typed arity is 3; >3 requires [MLXArray])
        let f2 = compile { (x: MLXArray, y: MLXArray) -> MLXArray in x * y }
        out("compile_2arg", f2(a, MLXArray(3.0)))
        let f3 = compile { (x: MLXArray, y: MLXArray, z: MLXArray) -> MLXArray in x * y + z }
        out("compile_3arg", f3(a, MLXArray(3.0), MLXArray(1.0)))

    case "random":
        // dit_generate.py:295-308 mx.random.key / mx.random.normal
        MLXRandom.seed(42)
        out("random_seed42", MLXRandom.normal([2, 3]))
        out("random_key42", MLXRandom.normal([2, 3], key: MLXRandom.key(42)))
        let (k1, k2) = MLXRandom.split(key: MLXRandom.key(7))
        out("random_split_k1", MLXRandom.normal([2, 2], key: k1))
        out("random_split_k2", MLXRandom.normal([2, 2], key: k2))
        let ks = MLXRandom.split(key: MLXRandom.key(7), into: 3)
        out("random_split3_k0", MLXRandom.normal([2, 2], key: ks[0]))
        out("random_key42_as_type_bf16", MLXRandom.normal([2, 3], dtype: .bfloat16, key: MLXRandom.key(42)))

    case "dtype":
        // dtype promotion + f16/bf16 range (affects the -1e9 mask and Snake exp).
        let f = fary(8) { Float($0) / 3.0 }.reshaped(2, 4)
        let b = f.asType(.bfloat16)
        let h = f.asType(.float16)
        out("bf16_values", b)
        out("f16_values", h)
        print("bf16_plus_f16_dtype=\((b + h).dtype)")
        print("bf16_plus_f32_dtype=\((b + f).dtype)")
        print("f16_plus_f32_dtype=\((h + f).dtype)")
        let n16 = MLXArray(-1e9 as Float).asType(.float16)
        let nb = MLXArray(-1e9 as Float).asType(.bfloat16)
        print("f16_neg1e9=\(n16.item(Float.self)) isInf=\(isInf(n16).item(Bool.self))")
        print("bf16_neg1e9=\(nb.item(Float.self))")
        // Snake1d exp overflow territory (vae_model.py:52-55)
        let big = MLXArray(Float(11.0)).asType(.float16)
        print("f16_exp11=\(exp(big).item(Float.self))")
        let bigb = MLXArray(Float(11.0)).asType(.bfloat16)
        print("bf16_exp11=\(exp(bigb).item(Float.self))")

    case "indexing":
        // chained / mixed subscripts used by the Python port's many slices
        let m = fary(64) { Float($0) }.reshaped(8, 8)
        print("A_newaxis2_shape=\(m[.newAxis, .newAxis].shape)")
        print("A2_single_newaxis_shape=\(m[.newAxis].shape)")
        print("A3_trailing_newaxis_shape=\(m[0..., .newAxis].shape)")
        print("A4_leading_newaxis_shape=\(m[.newAxis, 0...].shape)")
        print("A5_ellipsis_newaxis_shape=\(m[.ellipsis, .newAxis].shape)")
        print("B_newaxis2_then0_shape=\(m[.newAxis, .newAxis][0].shape)")
        print("C_newaxis2_then00_shape=\(m[.newAxis, .newAxis][0, 0].shape)")
        print("D_m00_shape=\(m[0, 0].shape)")
        print("E_full_shape=\(m[0..., 0...].shape)")
        print("F_newaxis_then0_shape=\(m[.newAxis][0].shape)")
        out("G_row_slice", m[0, 0 ..< 3])
        out("H_rowslice_cols", m[1 ..< 3, 0 ..< 2])
        out("I_ellipsis_cols", m[.ellipsis, 0 ..< 2])
        out("J_neg_range", m[0..., (-2) ..< 8][0])
        // Python: x[:, padding:-padding, :] pattern
        let t = fary(2 * 10 * 2) { Float($0) }.reshaped(2, 10, 2)
        out("K_trim", t[0..., 2 ..< 8, 0...])
        // Python: x[1:2] batch slice
        out("L_batch_slice", t[1 ..< 2])

    case "maskbcast":
        // does mlx-swift 0.32.2 SDPA accept a [L,L] / [H,L,L] mask (broadcast)?
        let shape = [1, 2, 8, 4]
        let q = fary(64) { Float($0 % 11) / 11.0 }.reshaped(shape)
        let k = fary(64) { Float(($0 + 3) % 13) / 13.0 }.reshaped(shape)
        let v = fary(64) { Float(($0 + 7) % 17) / 17.0 }.reshaped(shape)
        let indices = MLX.arange(0, 8)
        let diff = abs(indices[0..., .newAxis] - indices[.newAxis, 0...])
        let mask4 = which(
            diff .<= 2,
            MLXArray.zeros([8, 8], dtype: .float32),
            MLXArray.full([8, 8], values: MLXArray(-1e9 as Float), dtype: .float32)
        )
        print("mask4_shape=\(mask4.shape)")
        out("sdpa_mask_4d", scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.5, mask: mask4[.newAxis, .newAxis]))
        out("sdpa_mask_2d", scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.5, mask: mask4))
        out("sdpa_mask_3d", scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.5, mask: mask4[.newAxis]))
        // Recompute the 2D path after an explicit contiguous copy.
        out("sdpa_mask_2d_copy", scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.5, mask: mask4.reshaped(8, 8)))
        let squeezed = mask4[.newAxis, .newAxis][0, 0]
        print("squeezed_shape=\(squeezed.shape) squeezed_sum=\(squeezed.sum().item(Float.self))")
        out("sdpa_mask_squeezed", scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.5, mask: squeezed))

    case "extras":
        // MLXNN.RoPE module vs fast.rope (port could use either)
        let x = fary(64) { Float($0) / 7.0 - 4.0 }.reshaped(1, 2, 4, 8)
        let rope = RoPE(dimensions: 8, traditional: false, base: 10000, scale: 1)
        out("rope_module", rope(x, offset: 3))
        // cast every parameter dtype: parameters() -> mapValues -> update(parameters:)
        let lin = Linear(4, 4)
        let p16 = lin.parameters().mapValues { $0.asType(.float16) }
        _ = lin.update(parameters: p16)
        print("param_cast_dtype=\(lin.weight.dtype)")
        // MLXNN.quantize(model:) converts Linear -> QuantizedLinear in place
        let m = Linear(128, 64, bias: false)
        quantize(model: m, groupSize: 64, bits: 4)
        print("quantize_model_linear_dtype=\(m.weight.dtype)")

    case "quantmodule":
        // QuantizedLinear's packed layout vs raw quantize()+quantizedMM()
        let w = fary(64 * 128) { Float($0 % 17) / 17.0 - 0.5 }.reshaped(64, 128)
        let x = fary(2 * 128) { Float($0) / 128.0 }.reshaped(2, 128)
        let ql = QuantizedLinear(weight: w, bias: nil, groupSize: 64, bits: 4)
        print("ql_input_dims=\(ql.shape) weight_shape=\(ql.weight.shape) scales_shape=\(ql.scales.shape) biases_settings=\(ql.biases?.shape ?? [])")
        out("qlinear_out", ql(x))
        let (wq, s, b) = quantized(w, groupSize: 64, bits: 4)
        out("qmm_out", quantizedMM(x, wq, scales: s, biases: b, groupSize: 64, bits: 4))

    case "compile_random":
        // vae_model.py:302-303 randomness inside the compiled encode_sample
        let key = MLXRandom.key(0)
        let x = MLXArray.zeros([2, 3], dtype: .float32)
        let fk = compile { (a: MLXArray) -> MLXArray in a + MLXRandom.normal(a.shape, key: key) }
        out("compile_random_keyed", fk(x))
        out("compile_random_keyed_2nd", fk(x))
        let fu = compile { (a: MLXArray) -> MLXArray in a + MLXRandom.normal(a.shape) }
        out("compile_random_unkeyed", fu(x))
        out("compile_random_unkeyed_2nd", fu(x))

    case "evalcache":
        // mx.eval / mx.clear_cache / mx.async_eval usage in dit_generate.py & mlx_vae_decode_native.py
        let a = fary(4) { Float($0) }
        eval(a)
        asyncEval([a])
        eval(a)
        GPU.clearCache()
        print("eval_asyncEval_clearCache_ok")
        let before = Memory.snapshot()
        print("mem_before=\(before)")
        GPU.clearCache()
        Memory.clearCache()
        print("mem_after_clear=\(Memory.snapshot())")

    case "split":
        // dit_model.py:324 / 616 mx.split(x, n, axis=1)
        let m = fary(2 * 6 * 4) { Float($0) }.reshaped(2, 6, 4)
        let parts = split(m, parts: 6, axis: 1)
        print("split6_count=\(parts.count) shape0=\(parts[0].shape)")
        out("split6_part0", parts[0])
        out("split6_part5", parts[5])
        let two = split(m, parts: 2, axis: -1)
        print("split2_count=\(two.count) shape0=\(two[0].shape)")

    case "misc":
        // ops the upstream Python MLX port does not use but a Swift port may want
        out("pad_width2", padded(MLXArray([Float(1), 2, 3]), width: [0, 2]))
        out("erf", erf(MLXArray([Float(-1), 0, 1])))
        out("silu", silu(MLXArray([Float(-1), 0, 1])))
        out("tanh", tanh(MLXArray([Float(-1), 0, 1])))
        out("softmax", softmax(MLXArray([[Float(1), 2, 3]]), axis: -1))
        out("einsum", einsum("ij,jk->ik", MLXArray([Float(1), 2, 3, 4]).reshaped(2, 2), MLXArray([Float(5), 6, 7, 8]).reshaped(2, 2)))
        out("take_along", takeAlong(MLXArray([Float(10), 20, 30]), MLXArray([2, 0])))
        out("stack", stacked([MLXArray([Float(1), 2]), MLXArray([Float(3), 4])], axis: 0))
        out("repeat", repeated(MLXArray([Float(1), 2]), count: 2, axis: 0))
        out("broadcast", broadcast(MLXArray([Float(1), 2]), to: [2, 2]))
        out("minimum", minimum(MLXArray([Float(1), 5]), MLXArray([Float(3), 2])))
        out("power", pow(MLXArray([Float(2), 3]), MLXArray([Float(3), 2])))
        out("reciprocal", reciprocal(MLXArray([Float(2), 4])))
        out("zeros_like", MLXArray.zeros(like: MLXArray([Float(3), 4])))
        out("ones_like", MLXArray.ones(like: MLXArray([Float(3), 4])))
        // RMSNorm module parity with Python nn.RMSNorm / mx.fast.rms_norm
        let xr = fary(8) { Float($0) / 7.0 - 0.5 }.reshaped(2, 4)
        let wr = MLXArray([Float(1), 2, 3, 4])
        out("fast_rmsnorm", rmsNorm(xr, weight: wr, eps: 1e-6))
        out("fast_layernorm", layerNorm(xr, weight: wr, bias: MLXArray([Float(0), 1, 0, 1]), eps: 1e-6))

    default:
        print("unknown check: \(name)")
        exit(2)
    }
}

let args = CommandLine.arguments
let names: [String] =
    args.count > 1
    ? Array(args.dropFirst())
    : ["version"]
for n in names {
    print("### \(n)")
    check(n)
}
