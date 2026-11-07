# Q9 — MLX Swift gaps: porting the ACE-Step 1.5 native-MLX path to `mlx-swift` 0.32.3

Status: spike finding · owner: swift-analyst · pinned upstream `ca1e85fe9430179831e6bc6be790c332190a3866`
Appendices: probe package [`/private/tmp/mc5/swift-probe`](../swift-probe) (builds against the pinned checkout by path).
Every claim below is cited: Python `path:line` @ the pinned commit; Swift `checkout path:line` @ mlx-swift 0.32.3 (`19601207e9a0`, bundled MLX C++ **0.32.2**, `Source/Cmlx/mlx/mlx/version.h:8-10`) or mlx-swift-lm 3.32.3 (`3b339ad6e3b3`).

## 0. Bottom line

**Every MLX operation and MLXNN module the upstream Python-MLX port actually uses has a direct `mlx-swift` equivalent, and the probe shows matching numerics.** The Python port is also much smaller than a generic diffusion port: it contains **no quantization, no conv2d, no `take_along_axis`/`put_along_axis`, no einsum, no pad, no custom metal kernels, no `mx.fast.rope`** (RoPE is hand-rolled) and no `mx.async_eval`.

The real Q9 risks are therefore **not op availability**. They are, in order:

1. **The Python "MLX path" is a DiT+VAE hybrid, not a whole pipeline.** It starts from `encoder_hidden_states`/`context_latents` already computed by PyTorch, and it loads weights through a PyTorch→numpy→MLX converter. A Swift app port must additionally own (or otherwise obtain) the **audio tokenizer (FSQ + AttentionPooler)**, the **LM / lyric / text encoders**, the **detokenizer**, and the **safetensors loading + weight-layout conversion** — none of which exists in Python MLX form to compare against. This is new work, not a gap in mlx-swift.
2. **Non-Haar DCW wavelets have no Swift/MLX implementation at all** — upstream bridges each step through PyTorch `pytorch_wavelets` (`acestep/models/mlx/dcw_correction_mlx.py:89-137`). Swift must implement the filter banks from primitives or ship Haar/pix only.
3. **`mx.compile` semantics differ**: mlx-swift has `compile` (max 3 *typed* `MLXArray` arguments; use the `[MLXArray] -> [MLXArray]` overload for the 5-argument DiT step) and compiled functions **freeze randomness** — same in Python (measured). Upstream compiles `encode_and_sample` (`acestep/core/generation/handler/mlx_vae_init.py:74`), which freezes its sampling noise; if that is a bug, Swift must pass the PRNG key as an input array instead.
4. **A silent `.newAxis`-in-subscript bug in mlx-swift 0.32.3** that drops inserted axes in some multi-index slices (gap table #13). The upstream idioms happen to avoid it, but port code must prefer `expandedDimensions(axis:)`.
5. **dtype + memory model decisions for a 2.4B DiT** (§6): the Python port runs the DiT in **float32** (weights converted via `.float()`, `dit_convert.py:39`), ~9.6 GB for 2.4B; Swift will load bf16 safetensors unless the port upcasts, and quantizing is new ground (upstream never quantizes the MLX path).

Risk labels used below: **low** = mechanical port; **medium** = needs a Swift-side decision or fixture; **high** = no supporting Python-MLX reference / needs new implementation.

## 1. Method and versions

| Item | Value |
|---|---|
| Python reference | `ACE-Step-1.5` `.venv`, **mlx 0.30.6** (not 0.32.x; upstream pin is `mlx>=0.25.2`, `pyproject.toml:46`) |
| Swift reference | `mlx-swift` **0.32.3** (`19601207e9a0`), bundled MLX C++ **0.32.2** |
| Toolchain | Xcode 27.2 / Swift 6.4.2 (swiftlang-6.4.2.1.9), macOS 27.2 arm64, M3 Pro (12 cores) |
| Probe | `/private/tmp/mc5/swift-probe`, SPM executable, `.package(path:)` → the pinned checkout, `swift build -c release -j 8` |
| Build cost | first build **≈4 min** (226 Cmlx C++ files + 43 Metal kernels); incremental 7–40 s |
| Run | `./.build/release/probe version slice mask sdpa conv convtranspose quant quantbf16 rope random dtype split misc maskbcast extras indexing quantmodule compile evalcache` |
| Raw evidence | `swift-probe/build.log`, `build2.log`…`build7.log`, `swift-out-final.txt`, `python-out.txt`, `python-out2.txt`, `python-rope-hand.txt`, `python-maskbcast.txt`, `python-quantmodule.txt`, `compare.py` |
| Warnings | harmless SPM warning `Invalid Exclude …/Cmlx/mlx/build`; no build errors |

Version caveat: the Python reference env runs mlx **0.30.6** while mlx-swift bundles C++ **0.32.2**; all ops exercised by the port agreed numerically (table below), but the port's oracle should eventually be a 0.32.x Python env (upstream's pin `mlx>=0.25.2` permits it).

The probe prints deterministic arrays in a `name=v1,v2,…` format; `python_checks.py` prints the same lines against Python MLX, and `compare.py` diffs them. Numeric agreement (max abs diff, f32 print precision):

| Probe area | Python vs mlx-swift |
|---|---|
| slice / `_rotate_half` (`dit_model.py:16-21`) | 0 |
| sliding-window mask (`dit_model.py:38-55`) | 0 |
| SDPA, f32 / f16 / bf16; masks 2-D, 3-D, 4-D (`dit_model.py:237-239`) | **0** |
| conv1d stride 1/2, padding, dilation (`dit_model.py:469-475`, `vae_model.py:71-109`) | 2e-6 |
| **convTransposed1d** stride 2/4, padding 0/1/2 (`vae_model.py:125-131`, `dit_model.py:505-511`) | 0–1e-6 |
| quantize/dequantize/quantizedMM, 4-bit g64, 8-bit g32, bf16/f16 inputs | 0 |
| `QuantizedLinear` vs `quantize()+quantizedMM` | **0 (identical packed layout)** |
| RoPE (hand-rolled vs `fast.rope`) | 1e-6 |
| random seed/key/split | **0 (bit-identical)** |
| dtype promotion, bf16/f16 rounding, `-1e9` mask | 0 |
| `split`/`stack`/`where`/`broadcast`/`pad`/`erf`/`silu`/`softmax`/`einsum`/`take_along`/RMSNorm/LayerNorm | 0–2e-6 |

## 2. Gap table (the load-bearing section)

Legend: *Avail* = mlx-swift availability (Swift `file:line` in the pinned checkout). *Match* = semantics agree with Python (probe-verified unless noted).

| # | Behaviour the Python port uses | Python use site | mlx-swift availability | Match | Workaround / note | Risk |
|---|---|---|---|---|---|---|
| 1 | `mx.compile` of the DiT step (5 array args) | `dit_generate.py:326-348, 356-357` | `compile(inputs:outputs:shapeless:_:)` `Source/MLX/Transforms+Compile.swift:287` (array overload), typed overloads 1–3 args :304/:327/:349, tuple overloads `Source/MLX/Transforms+CompileOverloads.swift:6…514` | yes (probe `compile_1arg/2arg/3arg/5args/captured`) | 5 DiT inputs → `compile { (args: [MLXArray]) -> [MLXArray] in … }`; captured params/masks are compile-time constants; pass state via `inputs:`/`outputs:` (`Updatable`) if it must change | **medium** — no typed >3-arg overload; compile freezes captured arrays |
| 2 | `mx.compile` of a bound VAE method incl. `mx.random.normal` | `mlx_vae_init.py:73-75`, `vae_model.py:302-303` | same | yes, but **random is frozen**: repeated calls return identical noise (probe `compile_random_*_2nd`, also true in Python 0.30.6) | do **not** compile `encode_and_sample`, or pass a fresh `MLXRandom.key` as an input array; `decode` (deterministic) can be compiled | medium |
| 3 | `mx.fast.scaled_dot_product_attention(q,k,v, scale=, mask=)` | `dit_model.py:237-239` | top-level `scaledDotProductAttention(queries:keys:values:scale:mask:)` `Source/MLX/MLXFast.swift:352`; namespace form `MLXFast.scaledDotProductAttention` :210 with `ScaledDotProductAttentionMaskMode` :139 (`.none`/`.array`/**`.causal`**); C++ `Source/Cmlx/mlx/mlx/fast.cpp:714` | yes (masks 2-D/3-D/4-D, f32/f16/bf16) | Swift name is camelCase; same scale/mask semantics; bool masks also supported (`fast.cpp:757-758`) | **low** (drop-in) |
| 4 | `mx.fast` namespace (rms_norm, layer_norm, rope, metal kernels) | hand-rolled RoPE `dit_model.py:16-84`; `nn.RMSNorm`; `nn.SiLU` | all fast ops moved into the main `MLX` module: `RoPE` `Source/MLX/MLXFast.swift:308` (+array offset :320), `rmsNorm` :371, `layerNorm` :389, `MLXFast.metalKernel` `Source/MLX/MLXFastKernel.swift:176/224`; module wrappers `MLXNN.RoPE` `Source/MLXNN/PositionalEncoding.swift:36`, `RMSNorm` `Source/MLXNN/Normalization.swift:127`, `SiLU` `Source/MLXNN/Activations.swift:762` | yes (1e-6) | `import MLXFast` is deprecated in 0.32.3 (`Source/MLXFast/MLXFast.swift:9-12`); use `MLX.*`. `MLXNN.RoPE(dimensions:traditional:base:)` module also fine for full-sequence, offset 0 | low |
| 5 | `nn.Conv1d` (patch embed, dilated VAE res-units, strided down) | `dit_model.py:469-475`; `vae_model.py:71-75,103-109,163-165,179,205-207,220-222` | `conv1d` `Source/MLX/Ops.swift:614`; `MLXNN.Conv1d` `Source/MLXNN/Convolution.swift:12` | yes (2e-6) | weight layout `[out, K, in]` both sides (`Convolution.swift:57-64` = Python `nn.Conv1d`); same `stride/padding/dilation`; `groups>1` only the default is documented supported (`Ops.swift:598`) — not used | low |
| 6 | **`nn.ConvTranspose1d`** (DiT `proj_out` de-patchify + VAE decoder upsampling) | `dit_model.py:505-511,622`; `vae_model.py:125-131,136-142` | `convTransposed1d` `Source/MLX/Ops.swift:835`; `MLXNN.ConvTransposed1d` `Source/MLXNN/ConvolutionTransposed.swift:13` | **yes** (probe `convT_s2p1/s2p0/s4p2`, zero case; 0–1e-6) | Swift class name is **`ConvTransposed1d`** (not `ConvTranspose1d`); weight `[out, K, in]` matches the Python MLX weight layout (`dit_convert.py:54-55`, `vae_convert.py:87-89`); `outputPadding` exists if ever needed | **low** (was flagged as risk; empirically closed) |
| 7 | `nn.Linear`, `nn.RMSNorm`, `nn.SiLU` | `dit_model.py:132-134,170-176,277-304,371-375,484,504` | `Linear` `Source/MLXNN/Linear.swift:70`, `RMSNorm` `Source/MLXNN/Normalization.swift:127`, `SiLU`/`silu` `Source/MLXNN/Activations.swift:762/212` | yes (`nn.silu`→`silu`, `nn.RMSNorm`→`RMSNorm(dimensions:eps:)`) | `RMSNorm` init label is `dimensions:` not positional | low |
| 8 | `mx.quantize` / `mx.dequantize` / `mx.quantized_matmul` — **not used by the Python MLX port** (no `quantiz*` symbol in `acestep/models/mlx/`; PyTorch path uses torchao only, `init_service_loader.py:79-100`) | — (future Swift option) | `quantized` `Source/MLX/Ops.swift:2473`, `dequantized` :1149, `quantizedMM` :2532 (`quantizedMatmul` deprecated alias :2498), model-level `quantize(model:)` `Source/MLXNN/Quantized.swift:65/91/114/139` | yes (quantize/dequantize/MM agree **exactly**; `QuantizedLinear` output == `quantized()+quantizedMM` **exactly**) | `QuantizedLinear` `Source/MLXNN/Quantized.swift:266` stores packed `weight [out, in·bits/32]`, `scales/biases [out, in/groupSize]` — **identical packed layout** to Python (`python` probe `qlinear_weight_shape=[64, 16] scales=[64, 2]`); group 64/32, bits 4/8, affine; bf16/f16 inputs round-trip supported | low for API; **high for quality risk** — upstream has no quantized-MLX reference output to compare |
| 9 | `mx.conv2d` | not used | `conv2d` `Source/MLX/Ops.swift:660`; `MLXNN.Conv2d` `Source/MLXNN/Convolution.swift:82` | n/a | avoid entirely | low |
| 10 | Additive sliding-window mask, `-1e9`, dtype of hidden states | `dit_model.py:38-55, 588-599` (mask built in `hidden_states.dtype`) | `where/which` `Source/MLX/Ops.swift:3369`; `abs` `Source/MLX/Ops+Array.swift:19`; `full` `Source/MLX/Factory.swift:897`; `arange` :403; `zeros` :25 | yes (mask values identical) | **f16 trap**: `-1e9` cast to float16 is `-inf`, cast to bf16 is `-998244352` (probe `mask_f16_neg1e9`, `mask_bf16_neg1e9`); if the Swift DiT runs f16, use a smaller `-inf`/mask value or add a float32 mask. bf16 is safe | medium (dtype-dependent) |
| 11 | SDPA mask broadcasting (`[1,1,L,L]`, also `[L,L]`, `[1,L,L]`) | `dit_model.py:237-239` with `_create_sliding_window_mask` `:38-55` | `fast.cpp:908-912` broadcasts mask to `[B, n_heads, L_q, L_kv]` then adds it (`fallback` in `fast.cpp:845-860`) | yes (probe `sdpa_mask_4d/3d/2d/2d_copy` all identical) | nothing special; keep mask in f32 or bf16 | low |
| 12 | `mx.arange`, `mx.abs`, `mx.zeros`, `mx.full`, `mx.where`, `mx.concatenate`, `mx.split`, `mx.stack`, `mx.expand_dims`, `mx.broadcast_to`, `mx.reshape`, `mx.transpose`, `mx.matmul`, `mx.cos/sin/exp/log/sqrt/power/reciprocal/minimum/ones_like/zeros_like`, `mx.pad`, `mx.erf`, `mx.take_along_axis`, `mx.einsum`, `mx.repeat` | `dit_model.py:49-54,71-78,184-186,324,392-397,566-579,616-618`; `vae_model.py:297-309`; `dit_generate.py:123-131,284-291,454-504`; `dcw_correction_mlx.py:44-62` | `arange` `Factory.swift:403`; `abs` `Ops+Array.swift:19`; `zeros` `Factory.swift:25`; `full` `Factory.swift:897`; `where/which` `Ops.swift:3369`; `concatenated` `Ops.swift:583`; `split` `Ops+Array.swift:1652/1681`; `stacked` `Ops.swift:2975`; `expandedDimensions` `MLXArray+Ops.swift:1902`; `broadcast` `Ops.swift:483`; `reshaped` `MLXArray+Ops.swift:2631`; `transposed` :3049; `matmul` `Ops+Array.swift:1054`; `cos/sin/exp/log/sqrt` `MLXArray+Ops.swift:1525/2708`, `Ops+Array.swift:429/1619/745/850/1718`; `pow` `Ops+Array.swift:1369`; `reciprocal` :1515; `minimum` `Ops.swift:2137`; `MLXArray.zeros(like:)` `Factory.swift:74`, `ones(like:)` :145; `padded` `Ops.swift:2294/2324`; `erf` :1316; `takeAlong` :3056/3074; `einsum` :1248/1260; `repeated` `Factory.swift:1294/1320` | yes (0–2e-6) | almost all are **renamed** (`mx.concatenate`→`concatenated`, `mx.stack`→`stacked`, `mx.broadcast_to`→`broadcast(_:to:)`, `mx.expand_dims`→`expandedDimensions(axis:)`, `mx.take_along_axis`→`takeAlong`, `mx.repeat`→`repeated`); Python's `.sum(axis=..., keepdims=...)` → Swift `sum(axis:keepDims:)` (`Source/MLX/MLXArray+Ops.swift:2909`), multi-axis `axes:` form :2887 | low |
| 13 | Array indexing/slicing: `x[..., :half]`, `indices[:, None]`, `indices[None, :]`, `mask[None,None,:,:]`, `x[:, 0::2, :]`, `x[:, padding:-padding, :]`, `xt[:bsz]`, `decoded[idx:idx+1]`, negative indices | `dit_model.py:19-21,49-55,82,184,324,373-374,616-617,625`; `vae_model.py:83-86,297-309`; `dcw_correction_mlx.py:44-62`; `mlx_vae_decode_native.py:47-48,120-130` | variadic subscript `Source/MLX/MLXArray+Indexing.swift:451`; ranges/partial ranges :1608-1640; `.stride(from:to:by:)` :1576; `.ellipsis`/`.newAxis` enum :1301/:1298; single-op fast path `getItem` `Source/MLX/MLXArray+Indexing.swift:530` | mostly yes — **one silent bug**: `getItemND`'s unsqueeze step is gated on `remainingIndices.count > src.ndim \|\| squeezeNeeded` (`MLXArray+Indexing.swift:712`), so a **multi-index subscript with `.newAxis` is silently dropped when total operations ≤ rank** (probe: `m[.newAxis,.newAxis]` → `[8,8]` not `[1,1,8,8]`; `m[0..., .newAxis]` → `[8,8]`; `m[.newAxis, 0...]` → `[8,8]`; but `z[0..., .newAxis]` on 1-D works, `m[.newAxis]` alone works, `m[.ellipsis, .newAxis]` works) | use `expandedDimensions(axis:)` (`MLXArray+Ops.swift:1902`) instead of `.newAxis` when rank matters; upstream's idioms (`indices[:, None]` on 1-D, `mask[None,None,:,:]` 4 ops > rank) are safe | **medium** (silent wrong shapes) |
| 14 | Hand-rolled RoPE (`rotate_half` + duplicated cos/sin halves, base 1e6, f32 tables, full sequence no offset) | `dit_model.py:16-35,62-92,228-230` | `RoPE(_:dimensions:traditional:base:scale:offset:)` `Source/MLX/MLXFast.swift:308`, array-offset :320; module `MLXNN.RoPE` `Source/MLXNN/PositionalEncoding.swift:36` | **`traditional: false` == upstream hand-rolled RoPE to f32 precision** (Python check `rope_handrolled_vs_fast_nontrad_maxdiff=4.77e-07`; `traditional: true` differs by 8.39). Swift `fast.rope` matches Python `fast.rope` at 1e-6 | replace `MLXRotaryEmbedding` with `RoPE(x, dimensions: head_dim, traditional: false, base: 1_000_000, scale: 1, offset: 0)`; no offset needed for full-sequence DiT; table cache/`materialize_static_buffers` (`dit_model.py:86-92`) becomes unnecessary | low |
| 15 | FSQ (finite scalar quantization), levels `[8,8,8,5,5,5]`, dim 2048 | **not in `acestep/models/mlx/` at all**; PyTorch only: `modeling_acestep_v15_base.py:1187-1229` (tokenizer), `:1202` `ResidualFSQ(dim=2048, levels=…, num_quantizers=1)`, config `configuration_acestep_v15.py:151-153`; math in `vector_quantize_pytorch` (`finite_scalar_quantization.py:159-176`, `residual_fsq.py:169-266`) | **no MLX/Swift counterpart**; but it is composed of basic ops only: `Linear(2048→6)`/`Linear(6→2048)` projections + `tanh` + `floor`/`round` + residual subtract + scales (`levels**-i`) + mixed-radix indices | n/a — must be written and fixture-matched | implement from MLX primitives: `Linear`, `tanh`, `floor` (or `round` STE-equivalent at inference), `clip`, `matmul`/einsum for mixed-radix index; the 6-level case is tiny. Verify against PyTorch `ResidualFSQ` outputs on real latents (there is **no Python-MLX reference**) | **high (effort/parity)** |
| 16 | Pooling/attention-pooler (Qwen3-style encoder with CLS special token) | PyTorch only: `modeling_acestep_v15_base.py:740-830` (`AttentionPooler`), used at `:1200`; `pool_window_size=5` (`configuration_acestep_v15.py:175`) | no direct module in mlx-swift core; mlx-swift-lm has Qwen3 building blocks (`Libraries/MLXLLM/Models/Qwen3.swift`, 3.32.3) | n/a — basic ops (Linear/RMSNorm/RoPE/SDPA + prepend token + mask) | port as a small Module (or reuse mlx-swift-lm layer pieces); needs its own numerical fixture | high (new work) |
| 17 | `mx.random.key`, `mx.random.normal(shape, key=)`, `mx.random.split`, global `mx.random.seed` | `dit_generate.py:295-308, 482-483`; `vae_model.py:302-303` | `MLXRandom.key` `Source/MLX/Random.swift:839`, `split(key:into:)` :847, `split(key:)` :855, `normal` :964/:994, `seed` :830; module namespace also `MLXRandom` | **bit-identical** to Python 0.30.6 for seed 42, key 42, splits (probe `random_*`) | use `MLXRandom.normal(shape, key: k)`; to reproduce a Python reference exactly, seed/key values line up — do **not** rely on the global key in compiled or concurrent code | low |
| 18 | `mx.eval` per step, `mx.clear_cache`, `mx.async_eval` (unused) | `dit_generate.py:439,456,464-466,471,490,504,514,519`; `mlx_vae_decode_native.py:40-56,120-130` (clear_cache), `dit_model.py:92,532`; `mlx_vae_init.py:63` (`mx.eval(parameters())`) | `eval` `Source/MLX/Transforms+Eval.swift:122/134`, `asyncEval` :147/:251, `Memory.clearCache` `Source/MLX/Memory.swift:356` (`GPU.clearCache` deprecated `Source/MLX/GPU+Metal.swift:191`), `Module: Evaluatable` `Source/MLXNN/Module.swift:1032` so `eval(model)` works | yes (probe `evalcache`) | same laziness as Python: **must** call `eval` per diffusion step to bound graph/memory | low |
| 19 | Weight conversion PyTorch→MLX (`swapaxes(1,2)`, `transpose(1,2,0)`, weight-norm fusion, snake squeeze) | `dit_convert.py:38-61`; `vae_convert.py:62-132`; load `dit_convert.py:82-83`, `vae_convert.py:148-149` (`load_weights`, `mx.eval(parameters())`) | `loadArrays(url:)` `Source/MLX/IO.swift:125`; `Module.update(parameters:)` `Source/MLXNN/Module.swift:406`; helper `loadWeights` in mlx-swift-lm `Libraries/MLXLMCommon/Load.swift:368/418` | n/a (Swift has no PyTorch) | read safetensors → `[String: MLXArray]`, rename keys (`proj_in.1.`→`proj_in.`, `proj_out.1.`→`proj_out.`), apply the same axis permutations and `weight_norm` fusion (`w = g·v/‖v‖`), drop `rotary_emb` buffers; then `model.update(parameters: ModuleParameters.unflattened(...))` | medium (mechanical, needs a golden-checked conversion script) |
| 20 | `mx.array(numpy)` / `np.array(mx)` bridges (handlers, DCW torch bridge) | `dit_generate.py:265-274,529`; `mlx_vae_decode_native.py:40-56`; `dcw_correction_mlx.py:122-137` | `MLXArray` inits `Source/MLX/MLXArray+Init.swift:550` (Sequence), `asArray` `Source/MLX/MLXArray+Bytes.swift:123` | n/a — Swift has no numpy/torch | host conversion via `[Float]`/`Data` (or keep everything in `MLXArray`, which is preferable); **no torch bridge** — see #21 | low |
| 21 | Non-Haar DCW wavelets (db2/db4/sym4/sym8/coif2) via `pytorch_wavelets` | `dcw_correction_mlx.py:70-137` (torch bridge; MLX only implements native Haar `:31-62`) | **none** | n/a | implement multi-level DWT/IDWT with `conv1d`/slicing from primitives, or restrict the Swift port to `haar` + `pix` DCW modes and ship the rest later (**recommended for v1**; `haar` is the default and is exact) | **high** if full parity needed |
| 22 | Repaint host-side mask math (`np.nonzero`, `np.linspace`, `np.array`) | `dit_generate.py:142-165, 508-519` | Swift stdlib/Accelerate (no MLX op needed) | n/a | port loops to Swift (per-batch `mask_np` rows are tiny); then `where`/arithmetic in MLX | low |
| 23 | `mx.issubdtype(dtype, mx.floating)` + `tree_map` to f16 + `update(parameters)` | `mlx_vae_init.py:56-70` | `DType.isFloatingPoint` `Source/MLX/DType.swift:83`; `parameters()` `Source/MLXNN/Module.swift:355`, `mapValues` `Source/MLX/Nested.swift:346`, `update(parameters:)` :406 | yes (probe `param_cast_dtype=float16`) | `model.parameters().mapValues { $0.asType(.float16) }` → `model.update(parameters:)` | low |
| 24 | `mx.Dtype` selection (f16/f32) | `mlx_vae_init.py:48-56,63-70` | `DType` `Source/MLX/DType.swift`; `asType` `Source/MLX/MLXArray.swift:495` | yes (probe: bf16+f16→f32, bf16+f32→f32, f16+1e9→-inf etc. — same as Python) | pass explicit `.float32` where upstream is f32 | low |

## 3. `mx.compile` — detail

* Python compiles a **bound method** or closure (`mx.compile(mlx_vae.decode)`, `mx.compile(_raw_step)` where `_raw_step` closes over `mlx_decoder`), and the compiled function has **5 array arguments** in the DiT case and **1** in the VAE case (`dit_generate.py:326-336`, `mlx_vae_init.py:73-75`).
* mlx-swift's typed overloads go up to **3 `MLXArray` arguments** (`Transforms+Compile.swift:304/327/349`; tuple-returning variants in `Transforms+CompileOverloads.swift`). Arbitrary arity is available through the untraced `([MLXArray]) -> [MLXArray]` overload (`Transforms+Compile.swift:287`). Probe: `compile_5args` works and matches the uncompiled result.
* Compiled closures may **capture `MLXArray`s** (model weights, sliding-window mask). Probe `compile_captured` works. Those are constants; if a parameter changes (e.g. LoRA/quantize switch) the compiled function must be re-created. `inputs:`/`outputs:` (`[any Updatable]`, `Module: Updatable` `Source/MLXNN/Module.swift:1032`) is the supported way to pass mutable state.
* **Randomness inside `compile` freezes.** Measured on both sides: Python 0.30.6 `mx.compile(lambda a: a + mx.random.normal(a.shape))` returns the *same* noise on every call, and mlx-swift behaves the same for both keyed (captured key) and unkeyed draws (probe `compile_random_keyed*`, `compile_random_unkeyed*`). Consequence: upstream `_mlx_compiled_encode_sample` (`mlx_vae_init.py:74`) silently makes VAE sampling deterministic. In Swift, compile `decode` only, or feed the key as a compiled-function input.
* There is `compile(enable:)` (`Transforms+Compile.swift:368`) as a global kill switch — useful for an A/B parity flag.

## 4. The `mx.fast` namespace and masks — detail

| Python | mlx-swift | Note |
|---|---|---|
| `mx.fast.scaled_dot_product_attention` | `scaledDotProductAttention` (main `MLX` module, `Source/MLX/MLXFast.swift:352`) | same C++ op (`fast.cpp:714`); SDPA softmax is f32 internally; mask broadcast to `[B,H,Lq,Lk]` (`fast.cpp:908-912`), additive fp mask or bool mask. **Causal** masks are also exposed (`MLXFast.ScaledDotProductAttentionMaskMode.causal`, `Source/MLX/MLXFast.swift:139-167`, C++ `mask_mode="causal"`, `fast.cpp:745-749`) — upstream DiT uses fully bidirectional + sliding-window additive masks, so a port needs no causal mask for the DiT, but a full port's LM (Qwen3) can use it |
| `mx.fast.rms_norm` | `rmsNorm` (`Source/MLX/MLXFast.swift:371`) | `nn.RMSNorm` uses it (`Normalization.swift:143-145`) |
| `mx.fast.layer_norm` | `layerNorm` (`Source/MLX/MLXFast.swift:389`) | not used by upstream |
| `mx.fast.rope` | `RoPE` (`Source/MLX/MLXFast.swift:308`, array-offset :320) | upstream does **not** use it; probe proves `traditional:false` equals the hand-rolled version |
| `mx.fast.metal_kernel` | `MLXFast.metalKernel` (`Source/MLX/MLXFastKernel.swift:176/224`) | not used by upstream; available for perf work |

Sliding-window mask construction (`dit_model.py:38-55`) maps 1:1; the result is a **materialized, cached** tensor keyed by `(seq_len, dtype)` (`dit_model.py:517-534`), which must likewise be materialized once with `eval` and cached in Swift. Window is 128 (`dit_model.py:445`), `layer_types` alternating sliding/full (`:456-460`). The mask is `[1,1,L,L]`, built in the hidden-state dtype — see the f16 `-1e9 → -inf` trap in the table (#10).

## 5. FSQ / audio tokenizer / attention pooler — the "no counterpart" cluster

This is the most important Q9 finding after the op table: **the Python MLX port does not contain the audio-tokenization front-end**. `MLXDiTDecoder.__call__` starts at `encoder_hidden_states`/`context_latents` (`dit_model.py:536-566`); those are produced in PyTorch by `prepare_condition` (`modeling_acestep_v15_base.py:1657`) using the `AceStepAudioTokenizer` (`:1187-1229`), which is:

1. `audio_acoustic_proj = nn.Linear(64 → hidden)` (`:1199`),
2. `AttentionPooler` (`:740-830`, `:1200`) — Qwen3-style encoder over `pool_window_size=5` patches with a learned `special_token` prepended,
3. `ResidualFSQ(dim=2048, levels=[8,8,8,5,5,5], num_quantizers=1)` (`:1202-1207`; config `configuration_acestep_v15.py:151-153`).

FSQ is **not a special MLX op**: with `codebook_dim = len(levels) = 6` and `dim = 2048` the module is `Linear(2048→6)` + per-level `tanh`/`floor` bound (`finite_scalar_quantization.py:159-176`, symmetry-preserving variant used: `preserve_symmetry=True`) + residual scales `levels**-i` (`residual_fsq.py:76-79`) + `Linear(6→2048)` (`residual_fsq.py:76-80`, forward `:169-266`). All primitives exist in mlx-swift (`Linear`, `tanh`, `floor`, `clip`, arithmetic). So it is **trivially portable in op terms, but has no MLX reference implementation and no fixtures** → medium/high effort-to-parity, not a blocker of kind "op missing".

Same for the LM / lyric encoder / text encoders / detokenizer: PyTorch only. A complete Swift app port needs them from mlx-swift-lm (Qwen3 family) or new Swift modules; this is outside the Python-MLX reference and therefore outside what Q9 can verify. **Flag for the plan as separate workstreams.**

## 6. dtype promotion and memory model for a 2.4B DiT

**dtype promotion matches.** MLX Core type promotion is shared C++ code; probe confirms: `bf16 + f16 → f32`, `bf16 + f32 → f32`, `f16 + f32 → f32`; Python weak-scalar arithmetic (`normed * (1.0 + scale)`) keeps the array dtype on both sides.

**But the Python port is float32 by construction, and a Swift port will not be unless it asks.** `dit_convert.py:39` (`value.detach().cpu().float().numpy()`) and `vae_convert.py:81-82,107,116,124` convert every weight to **f32**; `dit_model.py` then runs f32, and only the VAE can opt into f16 via `ACESTEP_MLX_VAE_FP16` (`mlx_vae_init.py:48-56`). Consequences for 2.4B:

* f32 DiT weights ≈ **9.6 GB** + activations; bf16 ≈ 4.8 GB but bf16 arithmetic differs from the f32 Python reference (accept small numeric drift; unit-test logits).
* Upstream's `Snake1d` comments document f16 overflow (`vae_model.py:40-51`): `exp(alpha)` overflows f16 at alpha > ~11 and they reverted to f32-safe code (`:52-55`). If the Swift VAE runs f16, keep `alpha/beta` f32 or clip; probe confirms `exp(11)` f16 = 59872, bf16 = 59904.
* f16 sliding-window mask turns `-1e9` into `-inf` (probe; Python same) — use bf16 or rewrite mask arithmetic.
* **`mx.eval` per step is load-bearing** (`dit_generate.py:439/456/490/504`). Swift is equally lazy; missing `eval` per step will blow memory/graph.
* Memory APIs exist: `Memory.clearCache` (`Memory.swift:356`) and the VAE tile loop (`mlx_vae_decode_native.py:100-130`, chunk 512 latents / 64 overlap, `clear_cache` every 4 chunks) port directly.
* 4-bit gives ~1.2 GB DiT but is **unvalidated** — no upstream MLX-quantized reference; would need quality fixtures (§2 #8).

## 7. No-MLX-Swift-equivalent list (design changes required)

1. **PyTorch tensor ↔ MLX bridge** (`dit_generate.py:144-165,529`, `mlx_vae_decode_native.py:40-56`, `dcw_correction_mlx.py:89-137`) — no equivalent (and no need): Swift should keep arrays in `MLXArray`/`[Float]`/`Data` end-to-end.
2. **`pytorch_wavelets` basis set** for DCW (`dcw_correction_mlx.py:70-137`) — none; implement filter banks or restrict to `haar`/`pix`.
3. **`vector_quantize_pytorch.ResidualFSQ` + `AttentionPooler` + `AceStepAudioTokenizer`** — none; new Swift modules (primitives only).
4. **LM / lyric / text encoders / detokenizer** — none in the Python MLX path; use mlx-swift-lm or new modules.
5. **torchao weight-only quantization** (`init_service_loader.py:79-100`) — not MLX; use `MLXNN.quantize(model:)` (new feature, unvalidated vs upstream).
6. **`mx.fast` *as a namespace*** — no 1:1 namespace; fast ops are in the main `MLX` module in 0.32.3 and `import MLXFast` is deprecated (`Source/MLXFast/MLXFast.swift:9-12`).
7. **`mx.compile` > 3 typed args / named-Python-callable compile** — no; use the `[MLXArray]` overload.
8. **Negative-index/slice-set/scatter conveniences**: `putAlong`/`ArrayAt` exist (`Source/MLX/Ops.swift:2417`, `Source/MLX/ArrayAt.swift:20`) but there is no full numpy-style fancy indexing parity; upstream does not need it.
9. **`mx.eval` streaming of a module's parameter tree** — available indirectly via `Module: Evaluatable` (`Module.swift:1032`) + `eval(_ values: Any...)` (`Transforms+Eval.swift:176/189`).

## 8. Probe package (repro)

```
/private/tmp/mc5/swift-probe/
  Package.swift                  # .package(path: <pinned mlx-swift checkout>), products MLX/MLXNN/MLXRandom
  Sources/probe/main.swift       # dispatch: version slice mask sdpa conv convtranspose quant quantbf16
                                 #   rope random dtype split misc maskbcast extras indexing quantmodule
                                 #   compile compile_random evalcache
  python_checks.py               # same checks against Python MLX (repo .venv, mlx 0.30.6)
  compare.py                     # diffs the two outputs
  build*.log, swift-out-final.txt, python-*.txt
```

```
cd /private/tmp/mc5/swift-probe
swift build -c release -j 8                     # first ~4 min, then 7-40 s
./.build/release/probe version slice mask sdpa conv convtranspose quant rope random dtype split misc \
                       maskbcast extras indexing quantmodule compile compile_random evalcache
/private/tmp/mc5/ACE-Step-1.5/.venv/bin/python python_checks.py <same names>
python3 compare.py
```

Key probe outputs quoted above: `rope_handrolled_vs_fast_nontrad_maxdiff=4.76837158203125e-07`; `convT_s2p1` matched to 1e-6; `qlinear_out == qmm_out` exactly; `random_seed42`/`random_key42` bit-identical; `mask_f16_neg1e9=-inf`; `sdpa_mask_2d/3d/4d` identical; `A_newaxis2_shape=[8, 8]` (bug) vs `A2_single_newaxis_shape=[1, 8, 8]`.

## 9. Residual risks to carry into the port plan

1. **No upstream Swift/LM reference for the full pipeline** — parity must be established per-stage (DiT step, VAE decode, FSQ, pooler) against Python fixtures, not end-to-end first.
2. **Quantization is net-new** — decide f32 vs bf16 vs 4-bit before sizing; measure with the pinned Metal stack; keep a `compile(enable:)` + dtype switch for A/B.
3. **`MLXNN.CompiledFunction` + captured weights**: if the plan wants VAE decode compiled, verify the capture doesn't fight a later dtype/quantize change (recompile after mutating parameters).
4. **`.newAxis` bug** means ported slicing must be reviewed line-by-line; add a Swift unit test that asserts shapes for every translated slice from `dit_model.py`/`vae_model.py`.
5. **Thread model**: upstream materializes static buffers (`dit_model.py:86-92, 521-523`, `mx.eval` of RoPE tables + mask) before reusing the decoder from Gradio worker threads — mlx-swift's `MLXArray`/`Stream` handling should be pinned to one `GPU` stream; no equivalent of Python's implicit default-stream-per-thread assumptions.
6. **mlx-swift-lm is required for weight loading** (`loadWeights` lives in `MLXLMCommon`, `Libraries/MLXLMCommon/Load.swift:368/418`), not just mlx-swift core — keep the app's existing 3.32.3 pin.

## Appendix A — Python op inventory (grep-verified, `acestep/models/mlx/` + `handler/mlx_*.py`)

`mx.array`(97) · `mx.random.normal`(31) · `mx.eval`(26) · `mx.float*` dtypes(23) · `mx.concatenate`(16) · `mx.compile`(11) · `mx.abs`(11) · `mx.full`(8) · `mx.zeros`(7) · `mx.exp`(6) · `mx.expand_dims`(5) · `mx.sqrt`(4) · `mx.split`(4) · `mx.sin`(4) · `mx.arange`(4) · `mx.where`(3) · `mx.clear_cache`(3) · `mx.broadcast_to`(3) · `mx.reciprocal`(2) · `mx.random.key`(2) · `mx.power`(2) · `mx.ones_like`(2) · `mx.minimum`(2) · `mx.cos`(2) · `mx.zeros_like`(1) · `mx.stack`(1) · `mx.log`(1) · `mx.issubdtype`(1) · `mx.fast.scaled_dot_product_attention`(1) · `mx.all`(1)
Modules: `nn.Module`, `nn.Linear`, `nn.RMSNorm`, `nn.SiLU`, `nn.Conv1d`, `nn.ConvTranspose1d`; load: `nn.Module.load_weights`, `parameters/update(tree_map)`.
**Not present in the Python-MLX port** (and therefore not a porting dependency; Swift equivalents exist if wanted): `mx.quantize`, `mx.quantized_matmul`, `mx.dequantize`, `conv2d`, `mx.pad`, `mx.erf`, `mx.take_along_axis`, `mx.put_along_axis`, `mx.einsum`, `mx.repeat`, `mx.fast.rope`, `mx.fast.rms_norm`, `mx.fast.layer_norm`, `mx.fast.metal_kernel`, `mx.async_eval`, fast-SDPA causal/`sliding_window` mask modes, and scatter/index-update (`at[]`) — mlx-swift has `putAlong`/`ArrayAt` (`Source/MLX/Ops.swift:2417`, `Source/MLX/ArrayAt.swift:20`) if a port needs them.

## Appendix B — Swift file:line index (pinned checkout)

`Source/Cmlx/mlx/mlx/version.h:8-10` (C++ 0.32.2) · `Source/MLX/Transforms+Compile.swift:287,304,327,349,368` · `Source/MLX/Transforms+CompileOverloads.swift:6+` · `Source/MLX/MLXFast.swift:308,320,352,371,389` · `Source/MLX/MLXFastKernel.swift:176,224` · `Source/MLX/Ops.swift:598,614,660,835,1149,1248,2294,2417,2473,2532,2975,3056,3369` · `Source/MLX/Ops+Array.swift:19,429,745,850,1054,1369,1515,1619,1652,1681,1718` · `Source/MLX/MLXArray+Ops.swift:2887,1885/1902,2631,3049` · `Source/MLX/MLXArray+Indexing.swift:451,574,712,1576,1608-1640` · `Source/MLX/MLXArray+Init.swift:550` · `Source/MLX/MLXArray+Bytes.swift:123` · `Source/MLX/MLXArray.swift:495` · `Source/MLX/DType.swift:83` · `Source/MLX/Factory.swift:25,74,145,403,897,1294` · `Source/MLX/IO.swift:125` · `Source/MLX/Random.swift:830,839,847,855,964,994` · `Source/MLX/Transforms+Eval.swift:122,134,147,251` · `Source/MLX/Memory.swift:356` · `Source/MLX/Nested.swift:346` · `Source/MLX/Protocols.swift:13,28` · `Source/MLXNN/Convolution.swift:12,82` · `Source/MLXNN/ConvolutionTransposed.swift:13` · `Source/MLXNN/Linear.swift:70` · `Source/MLXNN/Normalization.swift:127` · `Source/MLXNN/Activations.swift:212,762` · `Source/MLXNN/PositionalEncoding.swift:36` · `Source/MLXNN/Quantized.swift:39,65,162,266` · `Source/MLXNN/Module.swift:355,406,1032` · `Source/Cmlx/mlx/mlx/fast.cpp:714,757-763,845-860,908-912` · mlx-swift-lm `Libraries/MLXLMCommon/Load.swift:368,418`.
