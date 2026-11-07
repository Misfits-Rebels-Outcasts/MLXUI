# Q2 — Weights inventory, module map, static buffers, layout conversions

**Scope.** The four safetensors files of `ACE-Step/Ace-Step1.5` @ `19671f406d603126926c1b7e2adc169acbcade22`
(plus `config.json`, `silence_latent.pt`, and the tokenizer files), measured from the actual
downloaded bytes. Code references are to the pinned checkout
`/private/tmp/mc5/ACE-Step-1.5` @ `ca1e85fe9430179831e6bc6be790c332190a3866` (verified
`git rev-parse HEAD`). Where the code that *actually executes* differs, it is the copy shipped
inside the HF repo and loaded by `trust_remote_code` — cited as `HF:acestep-v15-turbo/…` and
called out explicitly.

**Method.** `scripts/weights/inventory_safetensors.py` parses the safetensors header
(8-byte little-endian JSON length + JSON header + data offsets) with the Python standard library
only — no torch, no numpy, no `safetensors` — and validates every tensor's
`data_offsets` span against `dtype × prod(shape)`, so an incomplete or corrupt file fails loudly.
`scripts/weights/group_totals.py` rolls the same parse up to top-level groups. Both run under
`python3 -I`. Raw outputs: `findings/inventory/inventory_report.txt`,
`findings/inventory/modules_<name>.txt`, `findings/inventory/names_<name>.txt` (full tensor lists).

Everything below is **measured**, not inferred from the config.

---

## 1. Per-file totals

| File | Bytes (measured) | Tensors | Params | Dtypes present |
|---|---|---|---|---|
| `acestep-v15-turbo/model.safetensors` | 4,787,825,604 | 677 | 2,393,872,518 | **BF16 only** |
| `acestep-5Hz-lm-1.7B/model.safetensors` | 3,708,521,528 | 310 | 1,854,243,840 | **BF16 only** |
| `Qwen3-Embedding-0.6B/model.safetensors` | 1,191,586,416 | 310 | 595,776,512 | **BF16 only** |
| `vae/diffusion_pytorch_model.safetensors` | 337,431,388 | 365 | 168,695,426 | **BF16 only** |
| **total** | **10,025,364,936** | **1,662** | **5,012,588,296** | |

All four files carry `__metadata__ = {"format": "pt"}` and are **single-file, unsharded, unquantized**
(no `.scales`/`.biases`, no `model.safetensors.index.json`). A Swift loader needs one plain
safetensors read per file — no shard merging, no quantized-tensor reconstruction.

### f32 / bf16 mix

**Inside the four safetensors: none — every tensor is BF16.** The mix that matters is outside them:

- `acestep-v15-turbo/silence_latent.pt` — 3,841,215 B PyTorch zip checkpoint; the tensor is
  **F32** (`FloatStorage`, 960,000 elements, size `(1, 64, 15000)`, stride `(960000, 15000, 1)`,
  parsed from `silence_latent/data.pkl`), then `.transpose(1, 2)` → **`[1, 15000, 64]` f32**, and
  cast to the runtime dtype (`bf16`) at load:
  `acestep/core/generation/handler/init_service_loader.py:211-216`.
- The upstream MLX VAE runs **f32 by default** and only casts to f16 when
  `ACESTEP_MLX_VAE_FP16=1`: `acestep/core/generation/handler/mlx_vae_init.py:53-57`. So "bf16 on
  disk" does not mean the reference implementation computes the VAE in bf16 — it upcasts.
- MLXEmbedders' `Qwen3Model` has no dtype constraint, but its `RoPE` is computed in f32 and the
  q/k norms are RMSNorm over bf16 activations; keep the VAE path f32 (see §6).

---

## 2. Module map — which module each tensor group loads into

The single top-level object in `acestep-v15-turbo/model.safetensors` is
**`AceStepConditionGenerationModel`** (config `architectures[0]`, and `auto_map.AutoModel` →
`modeling_acestep_v15_turbo.AceStepConditionGenerationModel`). Its `__init__` is
`acestep/models/turbo/modeling_acestep_v15_turbo.py:1593` (pinned repo) /
`HF:…/modeling_acestep_v15_turbo.py:1554`.

| Tensor prefix (measured) | Tensors | Params | Loads into (class, file:line) |
|---|---|---|---|
| `decoder.*` | 476 | 1.575 B (65.8 %) | `AceStepDiTModel` — repo `:1243`, HF `:1237` |
| `encoder.*` | 140 | 608.37 M (25.4 %) | `AceStepConditionEncoder` — repo `:1513`, HF `:1506` |
| `tokenizer.*` | 32 | 105.03 M (4.4 %) | `AceStepAudioTokenizer` — repo `:1184`, HF `:1178` |
| `detokenizer.*` | 28 | 105.01 M (4.4 %) | `AudioTokenDetokenizer` — repo `:865`, HF `:859` |
| `null_condition_emb` | 1 | 2,048 | `AceStepConditionGenerationModel.null_condition_emb` — repo `:1622`, HF `:1572` |

Sub-module map (all measured shapes for one representative instance are in
`findings/inventory/modules_acestep-v15-turbo.txt`):

```
decoder  (AceStepDiTModel, repo:1243)
├── proj_in.1.{weight [2048,192,2], bias [2048]}        Sequential(Lambda, Conv1d, Lambda)  repo:1267
├── time_embed.{linear_1,linear_2,time_proj}            TimestepEmbedding  repo:203
├── time_embed_r.{…}                                    TimestepEmbedding  (2nd copy)
├── condition_embedder.{weight [2048,2048], bias}       Linear, bias=True  repo:1286
├── layers.0..23                                        AceStepDiTLayer  repo:446
│   ├── self_attn_norm / cross_attn_norm / mlp_norm     Qwen3RMSNorm [2048]
│   ├── self_attn.{q,k,v,o}_proj  {q,k}_norm            AceStepAttention  repo:259
│   ├── cross_attn.{q,k,v,o}_proj {q,k}_norm
│   ├── mlp.{gate,up,down}_proj                          Qwen3MLP
│   └── scale_shift_table [1,6,2048]
├── norm_out.weight [2048]                              Qwen3RMSNorm
├── proj_out.1.{weight [2048,64,2], bias [64]}          Sequential(Lambda, ConvTranspose1d, Lambda)  repo:1291
└── scale_shift_table [1,2,2048]
encoder  (AceStepConditionEncoder, repo:1513)
├── text_projector.weight [2048,1024]  (bias=False)     Linear  repo:1525
├── lyric_encoder  (AceStepLyricEncoder, repo:580)      8 layers
│   ├── embed_tokens.{weight [2048,1024], bias}
│   ├── layers.0..7                                     AceStepEncoderLayer  repo:377
│   └── norm.weight
└── timbre_encoder (AceStepTimbreEncoder, repo:1000)    4 layers, same layer shape
    ├── embed_tokens.{weight [2048,64], bias}
    ├── layers.0..3 , norm.weight , special_token [1,1,2048]
tokenizer  (AceStepAudioTokenizer, repo:1184)
├── audio_acoustic_proj.{weight [2048,64], bias}
├── attention_pooler (AttentionPooler, repo:737)        2 layers + embed_tokens + norm + special_token
└── quantizer.{project_in [6,2048]+bias, project_out [2048,6]+bias}   ResidualFSQ, repo:1199
detokenizer (AudioTokenDetokenizer, repo:865)
├── embed_tokens.{weight,bias}
├── special_tokens [1,5,2048]
├── layers.0..1 , norm , proj_out.{weight [64,2048], bias}   (proj_out = plain Linear, no conv)
```

Cross-checks against the config: 24 DiT layers ✓ `num_hidden_layers`, 8 lyric-encoder layers ✓
`num_lyric_encoder_hidden_layers`, 4 timbre layers ✓, 2 pooler/detokenizer layers ✓
`num_attention_pooler_hidden_layers`, `mlp` width 6144 ✓ `intermediate_size`, GQA 16 q-heads /
8 kv-heads × head_dim 128 ✓ (`q_proj` [2048,2048], `k/v_proj` [1024,2048]).

### Is the condition encoder inside `acestep-v15-turbo/model.safetensors`?

**Yes.** The condition encoder is the `encoder.*` prefix — 140 tensors / 608.37 M params
(25.4 % of the file): `encoder.text_projector.weight`,
`encoder.lyric_encoder.{embed_tokens,layers.0-7,norm}`,
`encoder.timbre_encoder.{embed_tokens,layers.0-3,norm,special_token}`. There is **no separate
condition-encoder checkpoint** anywhere in the 28-file repo; the 5 Hz LM (`acestep-5Hz-lm-1.7B`)
is a different thing (a Qwen3-derived planner that emits audio codes / captions, not the
`encoder.*` DiT conditioner). So one 4.79 GB read gives you the DiT *and* its conditioner; the
text encoder (`Qwen3-Embedding-0.6B`) is a separate file and a separate model.

### Two model files are the same shape as a Qwen3 decoder

`acestep-5Hz-lm-1.7B/model.safetensors` (1.854 B params, 28 layers) and
`Qwen3-Embedding-0.6B/model.safetensors` (595.78 M, 28 layers) have **identical tensor
structure**: `embed_tokens.weight` + `layers.{0..27}.{input_layernorm, post_attention_layernorm,
self_attn.{q,k,v,o}_proj, self_attn.{q,k}_norm, mlp.{gate,up,down}_proj}` + `norm.weight`,
310 tensors each, **no `lm_head`** (both configs say `tie_word_embeddings: true`). The only
differences are widths: 1024/3072/16h/8kv (embedding) vs 2048/6144/16h/8kv (1.7 B LM), and the
LM's `vocab_size` 217,204 vs the embedding model's 151,669. Both are plain `Qwen3Model`
(`architectures: ["Qwen3Model"]`, `model_type: qwen3`), i.e. **no LM head, no pooler, no
classifier** — a Swift port must not look for one.

> Consequence for the port: the 5 Hz LM is architecturally *just* a Qwen3 decoder. It can reuse
> the same Qwen3 module shape as the text encoder (MLXLLM's `Qwen3Model` for sampling, or a
> shared MLXEmbedders-style trunk) — one fewer architecture to hand-write. (The sampling/planner
> logic is a separate question, owned elsewhere.)

---

## 3. Static buffers / tensors derived at load time

The safetensors are **not** a complete description of the runtime state. Five things are built or
loaded outside them. The first two are the load-path ones a Swift loader must reproduce exactly.

| # | Buffer | Where it is built / loaded | In safetensors? |
|---|---|---|---|
| 1 | **`silence_latent`** | `torch.load(<ckpt>/silence_latent.pt, weights_only=True).transpose(1,2)` → `[1,15000,64]` f32, cast to model dtype — `acestep/core/generation/handler/init_service_loader.py:211-216`. Required; the loader **raises `FileNotFoundError`** if absent (`:212-213`). Sliced `[:750]` as the silent timbre reference — `acestep/core/generation/handler/conditioning_embed.py:47`. | **NO** — separate `.pt`, download it too. |
| 2 | **RoPE tables** | `Qwen3RotaryEmbedding` from `transformers.models.qwen3.modeling_qwen3` (imported `HF:…:33-36`), instantiated 5× as `decoder.rotary_emb` (repo `:1254`), `lyric_encoder.rotary_emb` (`:594`), `timbre_encoder.rotary_emb` (`:1015`), `attention_pooler.rotary_emb` (`:751`), `detokenizer.rotary_emb` (`:878`). `inv_freq` is a *non-persistent* buffer → recomputed from `rope_theta=1e6`, `head_dim=128`. | **NO** — and correct: 0 `rotary_emb` / `inv_freq` tensors in the file (measured). |
| 3 | **FSQ levels** | `ResidualFSQ(dim=2048, levels=[8,8,8,5,5,5], num_quantizers=1)` — repo `:1199-1203`, HF `:1193-1196`. `levels` come from **config**, not the checkpoint. Only `project_in` [6,2048] / `project_out` [2048,6] are learned (measured). Levels-based FSQ has **no codebook**. | **NO** (levels); project_in/out **YES** |
| 4 | **Sliding-window / causal attention masks** | Built per forward pass: the 4-way mask builder at `HF:…:58-102` (`is_causal` × `is_sliding_window`), selected per layer from `config.layer_types` (`HF:…:284`, `:652`, `:674-688`); MLX reference caches them in `MLXDiTDecoder._get_sliding_mask` (`acestep/models/mlx/dit_model.py:525`) and `_create_sliding_window_mask` (`:38`). | **NO** — per-request / cached derived state |
| 5 | **`timbre_fix_frame` = 750** | A config field (`vae/config`… no: main `config.json`, and `acestep/models/common/configuration_acestep_v15.py:183,222`) that is **not read anywhere in the code** — the value is *hard-coded* as the literal `750` in `conditioning_embed.py:47`, and as the dummy reference shape `torch.randn(3, 750, 64)` in the modeling self-tests. 750 latent frames = 30 s @ 25 Hz. | **NO** — a number to hard-code |

Additional derived state found in the load path (not tensors, but behaviour a port must match):

- **`condition_dim`**: repo has `condition_dim = config.encoder_hidden_size or config.hidden_size`
  before `condition_embedder` (`repo:1285-1286`); the HF copy has the simpler
  `nn.Linear(inner_dim, inner_dim)` (`HF:1279`). Same [2048,2048] for the 1.5 turbo; matters only
  for XL (4 B) checkpoints.
- **FSQ `pad` path**: `tokenize()` pads with `silence_latent[:1,:pad_len]` before the
  `pool_window_size` reshape (`HF:…:1628-1634`) — another hard dependency on buffer #1.
- `_sync_alignment_config()` (`init_service_loader.py:194`, defined
  `lyric_alignment_common.py:18`) only fixes up *lyric-alignment head indices*; it does not
  materialise weights.
- `self.model = AutoModel.from_pretrained(path, trust_remote_code=True, …)` at
  `init_service_loader.py:173-179` — the code itself is remote-loaded, but the **weights** are the
  four local safetensors; no weights are downloaded or generated at load.

---

## 4. Layout conversions the Swift loader must perform

### 4.1 DiT decoder — `acestep/models/mlx/dit_convert.py` (84 lines total)

The converter walks `pytorch_model.decoder.state_dict()` (`:32`) and applies exactly three rules:

| Rule | Code | Tensor-name pattern | Action | Tensors hit (measured) |
|---|---|---|---|---|
| **Conv1d** weight, after dropping a `nn.Sequential` Lambda wrapper | `:44-49` (`proj_in.1.` → `proj_in.`, `swapaxes(1,2)`) | `decoder.proj_in.1.weight` | `[out,in,K]` `[2048,192,2]` → `[out,K,in]` `[2048,2,192]` | 1 |
| **ConvTranspose1d** weight, same Sequential unwrap | `:50-56` (`proj_out.1.` → `proj_out.`, `transpose(1,2,0)`) | `decoder.proj_out.1.weight` | `[in,out,K]` `[2048,64,2]` → `[out,K,in]` `[64,2,2048]` | 1 |
| **Rotary buffers dropped** | `:58-60` (`if "rotary_emb" in key: continue`) | `*.rotary_emb.*` | skipped, recomputed | 0 present |
| everything else | `:62` | all remaining decoder tensors | copied unchanged | **474** |

Notes a Swift loader must not miss:

- **The `.1.` indices are real, on disk.** The inventory shows the literal names
  `decoder.proj_in.1.weight` and `decoder.proj_out.1.weight`. The rename to `proj_in.weight` /
  `proj_out.weight` is required because the MLX reference replaces the 3-element
  `Sequential(Lambda, Conv, Lambda)` with a **bare** `nn.Conv1d` / `nn.ConvTranspose1d`
  (`acestep/models/mlx/dit_model.py:469` and `:505`). There is no parameter at index `0` or `2`
  (Lambdas are parameterless), so nothing else is dropped.
- **`dit_convert.py` converts the decoder only.** `convert_decoder_weights` reads
  `pytorch_model.decoder` (`:31`) and `convert_and_load` loads an `MLXDiTDecoder` (`:76-84`).
  The upstream MLX path is DiT-decoder + VAE **only** — `_init_mlx_dit`
  (`acestep/core/generation/handler/mlx_dit_init.py:9-30`) and `_init_mlx_vae`
  (`mlx_vae_init.py:34-48`) — with graceful PyTorch fallback
  (`acestep/models/mlx/__init__.py:1-2`). **There is no upstream MLX implementation of
  `encoder.*`, `tokenizer.*` or `detokenizer.*`**, so those three groups have *no reference
  converter*; the Swift port writes them from the PyTorch code with no transposes needed (they
  are all plain `nn.Linear` on `[out,in]` weights — MLX `Linear` uses the same `[out,in]`, so
  **zero layout change** for 139 of the 140 encoder tensors and all 32 tokenizer / 28
  detokenizer tensors; the only exception is `encoder.timbre_encoder.special_token` and the other
  `special_token`/`special_tokens` parameters, which are raw `[1,1,2048]` / `[1,5,2048]`
  parameters, copied as-is).
- **No transposed Linear anywhere.** Every `nn.Linear` in this model (q/k/v/o, gate/up/down,
  text_projector, condition_embedder, embed_tokens, audio_acoustic_proj, detokenizer proj_out,
  FSQ project_in/out) has PyTorch `[out_features, in_features]` weight, which is exactly what
  MLX's `Linear` expects. Unlike a `nn.Linear`→`Conv1d` rewrite, nothing needs transposing.
- The converter casts to **f32** (`value.detach().cpu().float().numpy()`) before building the MLX
  array (`:41`). The Swift loader must **not** copy that: transpose the bf16 tensor directly
  (MLX `transposed`/`swappedAxes` keeps dtype), or you double the transient memory of a 4.8 GB
  load. Also `mx.array(np_val)` on a 25 M-element f32 tensor is a per-tensor copy — do the reshape
  on-device.

### 4.2 VAE — `acestep/models/mlx/vae_convert.py` (150 lines total)

The VAE is a `diffusers.AutoencoderOobleck` (`vae/config.json` `_class_name`), and the conversion
is **much more involved than the DiT's** — it fuses `weight_norm` before transposing:

| Rule | Code | Name pattern | Action | Tensors hit (measured) |
|---|---|---|---|---|
| **weight_norm fusion** `w = g·v/‖v‖`, norm along dim 0 | `:69-97`, `_fuse_weight_norm` at `:18-34` | every `*.weight_g` + `*.weight_v` pair | fuse into a single `*.weight`; `weight_g`/`weight_v` disappear | **74 pairs** (148 tensors → 74 weights) |
| **ConvTranspose1d** axis order, when the *fused* weight's parent is `conv_t1` | `:86-88` (`transpose(1,2,0)`) | `*.conv_t1.weight_{g,v}` | `[in,out,K]` → `[out,K,in]` | **5 pairs** (10 tensors; one per decoder block) |
| **Conv1d** axis order (all other fused convs) | `:89-91` (`swapaxes(1,2)`) | every other `*.weight_{g,v}` | `[out,in,K]` → `[out,K,in]` | **69 pairs** (138 tensors) |
| **Snake1d activation** `alpha`/`beta` | `:106-110` (`.squeeze()`) | `*.alpha`, `*.beta` | `[1,C,1]` → `[C]` | **144 tensors** (72 Snake1d modules) |
| **bias** | `:115-119` | `*.bias` | unchanged | 73 |
| catch-all pass-through | `:124-127` | anything left | unchanged | 0 |

Notes / gotchas:

- The fuse direction differs between the two conv families and the code relies on the **shapes in
  the file**, not on a table: `weight_g` is `[out,1,1]` for a `Conv1d` and `[in,1,1]` for a
  `ConvTranspose1d` (measured: `decoder.block.0.conv_t1.weight_g` is `[2048,1,1]` with
  `weight_v` `[2048,1024,20]`; `decoder.block.0.res_unit1.conv1.weight_g` is `[1024,1,1]` with
  `weight_v` `[1024,1024,7]`). A Swift loader can dispatch on `weight_g.shape[1] == 64`
  no — dispatch on the **module name** (`conv_t1` vs everything else), exactly as the Python does,
  and assert the shape.
- Both the *encoder* and *decoder* halves are in the file (182 + 183 tensors, 84.41 M / 84.28 M
  params). **Inference only needs the decoder** (`vae.decode`), but the file has both and the
  shipping app may want encode (cover/repaint). Decide whether to load half and skip 84 M params.
- The fuse must happen **before** the transpose, in f32 (`:81-83` upcasts), then the result is
  passed through as an f32 MLX array (`:94`) — the reference then runs the VAE in **f32**
  (`mlx_vae_init.py:53-57`). Do the same: fuse in f32, keep f32, *don't* round-trip bf16.
- **`diffusion_pytorch_model.safetensors` has no `__metadata__` other than `{"format":"pt"}`** and
  no `_class_name`; the architecture must come from `vae/config.json`.
- Nothing in the VAE file is a static buffer — no RoPE, no masks, no silence latent. It is a pure
  weight file (but see the licence note in `licence.md`: `AutoencoderOobleck` is Stability AI's
  Oobleck VAE architecture).

### 4.3 Qwen3 files — no conversion at all

`Qwen3-Embedding-0.6B/model.safetensors` and `acestep-5Hz-lm-1.7B/model.safetensors` need **zero
layout work**: HF `model.embed_tokens.weight` / `model.layers.N…` already match MLX `Embedding` /
`Linear` `[out,in]` conventions, and both have **no `lm_head`** (tied embeddings). The only
transform is the key prefix: MLXEmbedders' `Qwen3Model.sanitize(weights:)` prepends `model.` and
drops any `lm_head` / `rotary_emb.inv_freq`
(`mlx-swift-lm/Libraries/MLXEmbedders/Models/Qwen3.swift:363-384`). For the 5 Hz LM under MLXLLM
the equivalent sanitize is inside MLXLLM's `Qwen3Model`.

### 4.4 Conversion summary table for the Swift loader

| Group | Tensors | Transpose? | Fuse? | Reshape? | Skip? |
|---|---|---|---|---|---|
| `decoder.*` self/cross-attn, mlp, norms, `scale_shift_table`, `time_embed*`, `condition_embedder`; + `null_condition_emb` | **475** (474 + 1) | no | no | no | — |
| `decoder.proj_in.1.weight` | 1 | yes `swapaxes(1,2)` | no | rename `proj_in.1.`→`proj_in.` | — |
| `decoder.proj_out.1.weight` | 1 | yes `transpose(1,2,0)` | no | rename `proj_out.1.`→`proj_out.` | — |
| `encoder.*` / `tokenizer.*` / `detokenizer.*` | 200 | no | no | no | — |
| VAE `*.weight_g` + `*.weight_v` | 148 (74 pairs) → 74 | yes (5 conv_t1 vs 69 Conv1d) | **yes**, f32, dim-0 norm | rename to `*.weight` | — |
| VAE `*.alpha`/`*.beta` | 144 | no | no | `squeeze [1,C,1]→[C]` | — |
| VAE `*.bias` | 73 | no | no | no | — |
| RoPE / masks / FSQ levels / silence_latent | 0 | — | — | — | **synthesise** (§3) |
| Qwen3 ×2 | 620 | no | no | prefix `model.` | `lm_head`, `rotary_emb.inv_freq` (absent anyway) |

**Order matters** for the VAE: fuse → transpose → strip. Fusing after a transpose silently
produces the wrong norm axis; the Python's sorted-key iteration (`:60`) makes the `_g`/`_v`
pairing order-independent, and a Swift version should key off the `weight_g` name only.

---

## 5. Things the converters skip or handle specially (the traps)

1. **`dit_convert.py` never touches `encoder.*` / `tokenizer.*` / `detokenizer.*`.** The upstream
   MLX port is deliberately DiT-only. Do not assume the upstream MLX reference proves those
   groups are convertible — they are simply not converted there, and (as measured here) they need
   no conversion anyway (plain Linears + raw parameters).
2. **`nn.Sequential` indices are load-bearing.** Only index `1` exists; the rename must be exact
   or `loadWeights(verify: .all)` will fail with an unmatched key.
3. **`proj_out` in the DiT is a ConvTranspose1d (`[2048,64,2]`), but `proj_out` in the
   detokenizer is a plain Linear (`[64,2048]`).** Same leaf name, two different rules. Do not key
   a conversion rule on `proj_out` alone — key on the *prefix*.
4. **Rotary is absent and must be recomputed**, including the `traditional: false` rotate-half
   convention. The MLX reference builds `[max_len, head_dim]` cos/sin at construction
   (`dit_model.py:65-77`) and slices `[:seq_len]` per call — with `max_position_embeddings: 32768`
   × 128 × 2 (cos+sin) × 4 B ≈ 33.5 MB if you precompute the whole table in f32. Prefer building
   only the needed length, or cache lazily.
5. **The VAE `weight_g`/`weight_v` pairs are the single most error-prone part** — 74 pairs, two
   axis conventions, a norm axis that is dim 0 for both but a `g` shape that differs between the
   conv families.
6. **`NULL` / empty tensors: none.** 1,662 tensors, all non-empty, all validated.

---

## 6. Memory footprint a port must plan for

Measured file bytes are the *floor*: 4.79 + 3.71 + 1.19 + 0.34 GB = **10.03 GB of bf16 weights**.
Add:

- `silence_latent.pt` 3.84 MB.
- The VAE should run **f32** (reference default) → its 168.70 M params become **0.675 GB**, not
  0.337 GB.
- Text encoder (1.19 GB) is needed once per prompt (≤ 256 tokens) and could be loaded, run, and
  released before the diffusion loop, or kept resident.
- The 5 Hz LM (3.71 GB) is a separate planner phase; the turbo model needs it to produce music
  (`mlx-community/ACE-Step1.5-MLX` README: "This is required for the turbo model to produce
  music") but not on every diffusion step.

This matches the app's existing rule that diffusion-style entries need **measured** RAM numbers in
their notes, not the 4-bit-LLM `paramCount × bits/8 × 1.5` heuristic (CLAUDE.md, "Formulas").
