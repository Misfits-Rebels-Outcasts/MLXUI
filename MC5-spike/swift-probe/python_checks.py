"""Python-MLX counterpart to the Swift probe (same check names, same output format).

Run with the repo venv (mlx 0.31.2):
  /private/tmp/mc5/ACE-Step-1.5/.venv/bin/python python_checks.py <check> ...
The lines it prints are diffed against `swift run -c release probe <check>`.
"""

import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def out(name, a):
    a = mx.array(a).astype(mx.float32)
    mx.eval(a)
    v = np.array(a).ravel()
    print(f"{name}=" + ",".join(f"{float(x):.6f}" for x in v))


def fary(n, f):
    return mx.array([f(i) for i in range(n)], dtype=mx.float32)


def check(name):
    if name == "version":
        print("mlx_python_version=" + getattr(mx, "__version__", "unknown"))
        print("device=" + str(mx.default_device()))

    elif name == "slice":
        x = fary(8, lambda i: i / 7.0).reshape(1, 1, 2, 4)
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        out("rotate_half", mx.concatenate([-x2, x1], axis=-1))
        y = fary(12, lambda i: float(i)).reshape(2, 6)
        out("slice_partial_from", y[:, 2:])
        out("slice_stride2", y[:, 0::2])
        out("slice_index", y[1])
        z = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        print("slice_newaxis_shape=" + str(list(z[:, None].shape)))
        print("slice_neg_index=" + str(list(y[-1].shape)))

    elif name == "mask":
        n, seq_len, window = 8, 8, 2
        indices = mx.arange(n)
        diff = mx.abs(indices[:, None] - indices[None, :])
        zero = mx.zeros(diff.shape, dtype=mx.float32)
        neg = mx.full(diff.shape, -1e9, dtype=mx.float32)
        mask = mx.where(diff <= window, zero, neg)
        out("window_mask8", mask)
        out("window_mask4x4", mask[:4, :4])
        m16 = mask.astype(mx.float16)
        mb = mask.astype(mx.bfloat16)
        print(f"mask_f16_zero={float(np.array(m16[0, 0].astype(mx.float32)))}")
        print(f"mask_f16_neg1e9={float(np.array(m16[0, 3].astype(mx.float32)))}")
        print(f"mask_bf16_neg1e9={float(np.array(mb[0, 3].astype(mx.float32)))}")

    elif name == "sdpa":
        shape = (1, 2, 8, 4)
        q = fary(64, lambda i: float(i % 11) / 11.0).reshape(shape)
        k = fary(64, lambda i: float((i + 3) % 13) / 13.0).reshape(shape)
        v = fary(64, lambda i: float((i + 7) % 17) / 17.0).reshape(shape)
        indices = mx.arange(8)
        diff = mx.abs(indices[:, None] - indices[None, :])
        mask = mx.where(
            diff <= 2,
            mx.zeros(diff.shape, dtype=mx.float32),
            mx.full(diff.shape, -1e9, dtype=mx.float32),
        )[None, None]
        out("sdpa_nomask", mx.fast.scaled_dot_product_attention(q, k, v, scale=0.5, mask=None))
        out("sdpa_mask", mx.fast.scaled_dot_product_attention(q, k, v, scale=0.5, mask=mask))
        out(
            "sdpa_mask_bcast",
            mx.fast.scaled_dot_product_attention(q, k, v, scale=0.5, mask=mask[0, 0]),
        )
        qb, kb, vb = q.astype(mx.bfloat16), k.astype(mx.bfloat16), v.astype(mx.bfloat16)
        ob = mx.fast.scaled_dot_product_attention(
            qb, kb, vb, scale=0.5, mask=mask.astype(mx.bfloat16)
        )
        out("sdpa_bf16", ob)
        print("sdpa_bf16_dtype=" + str(ob.dtype))
        qh, kh, vh = q.astype(mx.float16), k.astype(mx.float16), v.astype(mx.float16)
        oh = mx.fast.scaled_dot_product_attention(
            qh, kh, vh, scale=0.5, mask=mask.astype(mx.float16)
        )
        out("sdpa_f16", oh)
        print("sdpa_f16_dtype=" + str(oh.dtype))

    elif name == "conv":
        x = fary(18, lambda i: i / 10.0).reshape(1, 6, 3)
        w = fary(36, lambda i: i / 7.0 - 2.0).reshape(4, 3, 3)
        out("conv1d_s2p1", mx.conv1d(x, w, stride=2, padding=1))
        out("conv1d_s1p0", mx.conv1d(x, w, stride=1, padding=0))
        out("conv1d_s2p3d1", mx.conv1d(x, w, stride=2, padding=3, dilation=1))
        c = nn.Conv1d(3, 4, 3, stride=2, padding=1)
        print("conv1d_module_weight_shape=" + str(list(c.weight.shape)))

    elif name == "convtranspose":
        x = fary(12, lambda i: i / 8.0).reshape(1, 4, 3)
        w = fary(24, lambda i: i / 9.0 - 1.0).reshape(4, 2, 3)  # [out, K, in]
        out("convT_s2p1", mx.conv_transpose1d(x, w, stride=2, padding=1))
        out("convT_s2p0", mx.conv_transpose1d(x, w, stride=2, padding=0))
        out("convT_s4p2", mx.conv_transpose1d(x, w, stride=4, padding=2))
        ct = nn.ConvTranspose1d(3, 4, 2, stride=2, padding=1)
        print("convT_module_weight_shape=" + str(list(ct.weight.shape)))
        out("convT_module_out", ct(x))
        pd = mx.zeros((1, 4, 3), dtype=mx.float32)
        out("convT_zero", mx.conv_transpose1d(pd, w, stride=2, padding=1))

    elif name == "quant":
        w = fary(64 * 128, lambda i: float(i % 17) / 17.0 - 0.5).reshape(64, 128)
        wq, scales, biases = mx.quantize(w, group_size=64, bits=4)
        print(
            "quant_wq_shape=" + str(list(wq.shape))
            + " scales_shape=" + str(list(scales.shape))
            + " biases_shape=" + str(list(biases.shape))
        )
        print(f"quant_wq_dtype={wq.dtype} scales_dtype={scales.dtype}")
        wd = mx.dequantize(wq, scales, biases, group_size=64, bits=4)
        out("dequant_row0", wd[0])
        print("dequant_dtype=" + str(wd.dtype))
        x = fary(2 * 128, lambda i: i / 128.0).reshape(2, 128)
        out(
            "quantmm",
            mx.quantized_matmul(x, wq, scales, biases, group_size=64, bits=4),
        )
        out("dequant_matmul", x @ wd.T)
        wq8, s8, b8 = mx.quantize(w, group_size=32, bits=8)
        wd8 = mx.dequantize(wq8, s8, b8, group_size=32, bits=8)
        out("dequant8_row0", wd8[0])
        ql = nn.QuantizedLinear(128, 64, bias=False, group_size=64, bits=4)
        print(
            "qlinear_weight_shape=" + str(list(ql.weight.shape))
            + " scales_shape=" + str(list(ql.scales.shape))
            + f" bias_nil={getattr(ql, "bias", None) is None}"
        )
        qe = nn.QuantizedEmbedding(8, 128, group_size=64, bits=4)
        print(
            "qembedding_weight_shape=" + str(list(qe.weight.shape))
            + " scales_shape=" + str(list(qe.scales.shape))
        )

    elif name == "quantbf16":
        w = fary(64 * 128, lambda i: float(i % 17) / 17.0 - 0.5).reshape(64, 128)
        wqb, sb, bb = mx.quantize(w.astype(mx.bfloat16), group_size=64, bits=4)
        wdb = mx.dequantize(wqb, sb, bb, group_size=64, bits=4)
        print("quant_bf16_ok dtype=" + str(wdb.dtype))
        out("dequant_bf16_row0", wdb[0])
        wqf, sf, bf = mx.quantize(w.astype(mx.float16), group_size=64, bits=4)
        wdf = mx.dequantize(wqf, sf, bf, group_size=64, bits=4)
        print("quant_f16_ok dtype=" + str(wdf.dtype))

    elif name == "rope_handrolled":
        # upstream dit_model.py:16-35,71-84 hand-rolled RoPE vs mx.fast.rope
        head_dim, seq_len = 8, 4
        x = fary(2 * seq_len * head_dim, lambda i: i / 7.0 - 4.0).reshape(1, 2, seq_len, head_dim)
        base = 10000.0
        inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
        positions = mx.arange(seq_len).astype(mx.float32)
        freqs = positions[:, None] * inv_freq[None, :]
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(freqs)[None, None]
        sin = mx.sin(freqs)[None, None]

        def rot_half(z):
            h = z.shape[-1] // 2
            return mx.concatenate([-z[..., h:], z[..., :h]], axis=-1)

        hand = x * cos + rot_half(x) * sin
        out("rope_handrolled", hand)
        fast0 = mx.fast.rope(x, head_dim, traditional=False, base=base, scale=1.0, offset=0)
        out("rope_fast_nontrad0", fast0)
        fast1 = mx.fast.rope(x, head_dim, traditional=True, base=base, scale=1.0, offset=0)
        d0 = mx.max(mx.abs(hand - fast0))
        d1 = mx.max(mx.abs(hand - fast1))
        print(f"rope_handrolled_vs_fast_nontrad_maxdiff={float(np.array(d0))}")
        print(f"rope_handrolled_vs_fast_trad_maxdiff={float(np.array(d1))}")

    elif name == "rope":
        x = fary(64, lambda i: i / 7.0 - 4.0).reshape(1, 2, 4, 8)
        out(
            "rope_nontrad_off3",
            mx.fast.rope(x, 8, traditional=False, base=10000.0, scale=1.0, offset=3),
        )
        out(
            "rope_trad_off3",
            mx.fast.rope(x, 8, traditional=True, base=10000.0, scale=1.0, offset=3),
        )
        out(
            "rope_nontrad_off0",
            mx.fast.rope(x, 8, traditional=False, base=10000.0, scale=1.0, offset=0),
        )
        out(
            "rope_offset_array",
            mx.fast.rope(
                x, 8, traditional=False, base=10000.0, scale=1.0, offset=mx.array([3])
            ),
        )
        out(
            "rope_partial_dims4",
            mx.fast.rope(x, 4, traditional=False, base=10000.0, scale=1.0, offset=3),
        )

    elif name == "random":
        mx.random.seed(42)
        out("random_seed42", mx.random.normal((2, 3)))
        out("random_key42", mx.random.normal((2, 3), key=mx.random.key(42)))
        k1, k2 = mx.random.split(mx.random.key(7))
        out("random_split_k1", mx.random.normal((2, 2), key=k1))
        out("random_split_k2", mx.random.normal((2, 2), key=k2))
        ks = mx.random.split(mx.random.key(7), 3)
        out("random_split3_k0", mx.random.normal((2, 2), key=ks[0]))
        out(
            "random_key42_as_type_bf16",
            mx.random.normal((2, 3), dtype=mx.bfloat16, key=mx.random.key(42)),
        )

    elif name == "dtype":
        f = fary(8, lambda i: i / 3.0).reshape(2, 4)
        b = f.astype(mx.bfloat16)
        h = f.astype(mx.float16)
        out("bf16_values", b)
        out("f16_values", h)
        print("bf16_plus_f16_dtype=" + str((b + h).dtype))
        print("bf16_plus_f32_dtype=" + str((b + f).dtype))
        print("f16_plus_f32_dtype=" + str((h + f).dtype))
        n16 = mx.array(-1e9, dtype=mx.float16)
        nb = mx.array(-1e9, dtype=mx.bfloat16)
        print(
            f"f16_neg1e9={float(np.array(n16.astype(mx.float32)))} "
            f"isInf={bool(np.array(mx.isinf(n16)))}"
        )
        print(f"bf16_neg1e9={float(np.array(nb.astype(mx.float32)))}")
        big = mx.array(11.0, dtype=mx.float16)
        print(f"f16_exp11={float(np.array(mx.exp(big).astype(mx.float32)))}")
        bigb = mx.array(11.0, dtype=mx.bfloat16)
        print(f"bf16_exp11={float(np.array(mx.exp(bigb).astype(mx.float32)))}")

    elif name == "split":
        m = fary(2 * 6 * 4, lambda i: float(i)).reshape(2, 6, 4)
        parts = mx.split(m, 6, axis=1)
        print(f"split6_count={len(parts)} shape0={list(parts[0].shape)}")
        out("split6_part0", parts[0])
        out("split6_part5", parts[5])
        two = mx.split(m, 2, axis=-1)
        print(f"split2_count={len(two)} shape0={list(two[0].shape)}")

    elif name == "maskbcast":
        shape = (1, 2, 8, 4)
        q = fary(64, lambda i: float(i % 11) / 11.0).reshape(shape)
        k = fary(64, lambda i: float((i + 3) % 13) / 13.0).reshape(shape)
        v = fary(64, lambda i: float((i + 7) % 17) / 17.0).reshape(shape)
        indices = mx.arange(8)
        diff = mx.abs(indices[:, None] - indices[None, :])
        mask4 = mx.where(
            diff <= 2,
            mx.zeros(diff.shape, dtype=mx.float32),
            mx.full(diff.shape, -1e9, dtype=mx.float32),
        )
        print("mask4_shape=" + str(list(mask4.shape)))
        out("sdpa_mask_4d", mx.fast.scaled_dot_product_attention(q, k, v, scale=0.5, mask=mask4[None, None]))
        out("sdpa_mask_2d", mx.fast.scaled_dot_product_attention(q, k, v, scale=0.5, mask=mask4))
        out("sdpa_mask_3d", mx.fast.scaled_dot_product_attention(q, k, v, scale=0.5, mask=mask4[None]))
        out("sdpa_mask_2d_copy", mx.fast.scaled_dot_product_attention(q, k, v, scale=0.5, mask=mask4.reshape(8, 8)))

    elif name == "quantmodule":
        w = fary(64 * 128, lambda i: float(i % 17) / 17.0 - 0.5).reshape(64, 128)
        x = fary(2 * 128, lambda i: i / 128.0).reshape(2, 128)
        ql = nn.QuantizedLinear(w.shape[1], w.shape[0], bias=False, group_size=64, bits=4)
        ql.weight = w  # type: ignore[assignment]
        # rebuild scales/biases from the assigned weight
        wq, scales, biases = mx.quantize(w, group_size=64, bits=4)
        ql.weight, ql.scales, ql.biases = wq, scales, biases  # type: ignore[assignment]
        out("qlinear_out", ql(x))
        wq2, s2, b2 = mx.quantize(w, group_size=64, bits=4)
        out("qmm_out", mx.quantized_matmul(x, wq2, s2, b2, group_size=64, bits=4))

    elif name == "misc":
        out("pad_width2", mx.pad(mx.array([1.0, 2.0, 3.0]), [(0, 2)]))
        out("erf", mx.erf(mx.array([-1.0, 0.0, 1.0])))
        out("silu", nn.silu(mx.array([-1.0, 0.0, 1.0])))
        out("tanh", mx.tanh(mx.array([-1.0, 0.0, 1.0])))
        out("softmax", mx.softmax(mx.array([[1.0, 2.0, 3.0]]), axis=-1))
        out(
            "einsum",
            mx.einsum(
                "ij,jk->ik",
                mx.array([[1.0, 2.0], [3.0, 4.0]]),
                mx.array([[5.0, 6.0], [7.0, 8.0]]),
            ),
        )
        out("take_along", mx.take_along_axis(mx.array([10.0, 20.0, 30.0]), mx.array([2, 0]), axis=None))
        out("stack", mx.stack([mx.array([1.0, 2.0]), mx.array([3.0, 4.0])], axis=0))
        out("repeat", mx.repeat(mx.array([1.0, 2.0]), 2, axis=0))
        out("broadcast", mx.broadcast_to(mx.array([1.0, 2.0]), (2, 2)))
        out("minimum", mx.minimum(mx.array([1.0, 5.0]), mx.array([3.0, 2.0])))
        out("power", mx.power(mx.array([2.0, 3.0]), mx.array([3.0, 2.0])))
        out("reciprocal", mx.reciprocal(mx.array([2.0, 4.0])))
        out("zeros_like", mx.zeros_like(mx.array([3.0, 4.0])))
        out("ones_like", mx.ones_like(mx.array([3.0, 4.0])))
        xr = fary(8, lambda i: i / 7.0 - 0.5).reshape(2, 4)
        wr = mx.array([1.0, 2.0, 3.0, 4.0])
        out("fast_rmsnorm", mx.fast.rms_norm(xr, wr, 1e-6))
        out(
            "fast_layernorm",
            mx.fast.layer_norm(xr, wr, mx.array([0.0, 1.0, 0.0, 1.0]), 1e-6),
        )

    else:
        print("unknown check: " + name)
        sys.exit(2)


for name in sys.argv[1:]:
    print("### " + name)
    check(name)
