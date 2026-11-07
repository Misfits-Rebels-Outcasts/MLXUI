This is by an automated agent, not a human.

# MC5-1 spike — MLX Swift port plan for ACE-Step 1.5 — **part 2/2: Q8–Q13, method, evidence**

Part 1/2 (Q1–Q7, pinned inputs, machine) is the previous comment. This part carries the
measurements, the MLX-Swift gap analysis, the slice plan, the licence audit and the artifact links.

---

## Q8 — Memory and time (measured on this Mac)

**Machine: Apple M3 Pro, 12 cores, 18 GiB RAM, macOS 27.0.** Turbo checkpoint, 8 steps, shift 1.0,
ODE/Euler, DCW on (turbo default), seed 12345, instrumental prompt, batch 1 (LM runs: see note),
`ACESTEP_MLX_VAE_FP16=1`. Each row is a fresh process running the pinned upstream code with the
two documented measurement patches (bf16 one-copy MLX DiT + staged free; see "Method and caveats").
`phys_footprint` is sampled every 50 ms; `gen wall` excludes process start-up/imports/init.

### bf16 / lm=0.6B

| dur | init wall | text enc s/peak | condition s/peak | DiT 8 steps s/peak | VAE decode s/peak | gen wall (model) | process peak | MLX peak | torch@exit | (all memory in GiB) |
|---:|---:|---|---|---|---|---:|---:|---:|---:|
| 30 s | 7.8 s | 0.28 s / 10.97 GiB | 0.38 s / 11.13 GiB | 3.13 s / 12.72 GiB | 4.46 s / 16.80 GiB | 16.1 s | 16.80 GiB | 7.1 GiB | 7.7 GiB |
| 120 s | 7.2 s | 0.26 s / 11.20 GiB | 0.38 s / 11.36 GiB | 12.09 s / 13.18 GiB | 12.06 s / 16.82 GiB | 40.8 s | 16.82 GiB | 7.2 GiB | 7.7 GiB |
| 240 s | 7.6 s | 0.22 s / 11.59 GiB | 0.39 s / 11.75 GiB | 23.95 s / 13.62 GiB | 21.62 s / 16.87 GiB | 79.2 s | 16.87 GiB | 7.2 GiB | 7.8 GiB |

### bf16 / lm=none

| dur | init wall | text enc s/peak | condition s/peak | DiT 8 steps s/peak | VAE decode s/peak | gen wall (model) | process peak | MLX peak | torch@exit | (all memory in GiB) |
|---:|---:|---|---|---|---|---:|---:|---:|---:|
| 30 s | 7.9 s | 0.09 s / 10.63 GiB | 0.88 s / 10.78 GiB | 3.02 s / 12.28 GiB | 3.70 s / 15.99 GiB | 8.7 s | 15.99 GiB | 5.9 GiB | 6.4 GiB |
| 120 s | 13.2 s | 0.08 s / 10.79 GiB | 0.49 s / 10.90 GiB | 10.33 s / 12.75 GiB | 11.40 s / 16.45 GiB | 24.3 s | 16.45 GiB | 5.9 GiB | 6.5 GiB |
| 240 s | 7.2 s | 0.08 s / 10.79 GiB | 1.02 s / 10.88 GiB | 22.39 s / 12.78 GiB | 21.49 s / 16.60 GiB | 49.8 s | 16.60 GiB | 6.0 GiB | 6.6 GiB |

### bf16-4bit / lm=none

| dur | init wall | text enc s/peak | condition s/peak | DiT 8 steps s/peak | VAE decode s/peak | gen wall (model) | process peak | MLX peak | torch@exit | (all memory in GiB) |
|---:|---:|---|---|---|---|---:|---:|---:|---:|
| 30 s | 9.0 s | 0.07 s / 11.38 GiB | 1.05 s / 11.53 GiB | 2.70 s / 12.22 GiB | 3.84 s / 16.28 GiB | 8.6 s | 16.28 GiB | 3.8 GiB | 4.3 GiB |
| 120 s | 7.5 s | 0.08 s / 11.53 GiB | 0.60 s / 11.67 GiB | 9.98 s / 13.43 GiB | 11.67 s / 16.45 GiB | 24.1 s | 16.45 GiB | 3.8 GiB | 4.4 GiB |
| 240 s | 8.8 s | 0.09 s / 11.55 GiB | 0.67 s / 11.71 GiB | 22.72 s / 13.42 GiB | 21.64 s / 16.62 GiB | 49.0 s | 16.62 GiB | 3.9 GiB | 4.5 GiB |

**How to read the peaks (important).** The whole-process peaks are dominated by **torch MPS
allocator retention**, which is measurement overhead: (a) upstream's macOS path is a torch+MLX
hybrid and, after the bf16 one-copy patch, still holds a ~3.2 GB torch condition-encoder/text/VAE
set; (b) on this Mac torch does not return freed MPS memory to the OS (controlled check: allocate
and free 2 GiB → footprint stays ~2.1 GiB higher; `torch.mps.empty_cache()` does not help). In the
table, `MLX peak` is MLX's own high-water mark and `torch@exit` is the retained torch MPS driver
allocation at process end. A native Swift port has neither: its process peak is
`(MLX high-water) + app baseline`, which on this machine is the `MLX peak` column plus the app's
usual ~1.5 GB host footprint. **That projection — not the whole-process number — is what decision
4's 12 GB cap has to be compared against.**

**What the measurement says about decision 4's 12 GB cap.**

- `bf16` no-LM: MLX high-water **5.91 / 5.94 / 6.01 GiB** at 30 s / 2 min / 4 min; whole-process
  peak 16.0–16.6 GiB (≈ 6.4–6.6 GiB of it retained torch memory). **Port projection = MLX high-water
  + the app's ~1.5 GiB host baseline ≈ 7.4–7.6 GiB** — inside the 12 GB cap with ~4 GiB headroom at
  4 min. Diffusion grows ~quadratically: 3.0 s → 10.3 s → 22.4 s (8 steps); VAE decode
  3.7 / 11.4 / 21.5 s.
- `bf16 + 0.6B LM`: MLX high-water **7.14 / 7.18 / 7.25 GiB**; port projection ≈ **8.6–8.8 GiB** —
  still inside the cap at 4 min. The LM's own stage is time-heavy: 0.6B load 27 s (cold tokenizer
  12 s dominates), generation 6.2 s at 30 s / 13.8 s at 2 min / 26.4 s at 4 min (the token count is
  forced to `duration×5`). The whole-process peaks (16.8–16.9 GiB) include the retained torch memory.
- `4-bit DiT` (quantized by us from the official bf16 weights with
  `mlx.nn.quantize(group_size=64, bits=4)`; 888 MB packed vs 3,151 MB bf16): MLX high-water
  **3.80 / 3.84 / 3.91 GiB**; port projection ≈ **5.3–5.4 GiB**. Diffusion is *not* faster on this
  GPU at these sizes (2.7 s vs 3.0 s at 30 s; 22.7 s vs 22.4 s at 4 min) — 4-bit buys memory, not
  time. Quality is the owner's listen call — see the listening note below.
- **What fits decision 4 on this 18 GiB Mac:** all three measured configurations fit the **12 GB
  process cap** in the port projection, with 4 min measured as the longest duration; the tightest
  is bf16 + 0.6B LM at ≈8.8 GiB. What does **not** fit is a with-LM v1 under the app's
  installed-size booking: bundling the 1.7B LM makes the installed-files basis 10.09 GB × 1.10 ≈
  **11.1 GB booked before any working set** (Q12), and the 1.7B LM's memory was not measured. The
  0.6B LM (1.37 GB separate repo, not in the v1 file set) is the only plausible with-LM option.
- **Longest duration measured: 4 min.** 480–600 s were not run (they are outside the recommended
  v1 cap and the 12 full-attention layers make them quadratic — Q10).
- The upstream **as-shipped** macOS path (float32 torch DiT + float32 MLX copy, never released) is
  ~19.2 GB (f32) for the DiT alone and **cannot run on 18 GiB without swapping**: our attempt reached
  16.5 GB footprint with 7 GB of swap *during model load* and was stopped. That is the strongest
  argument for the port, and it must be stated plainly: the "full bf16 + LM" expectation in the
  issue does not match what upstream actually does on MPS; the measured numbers above are the
  bf16/one-copy variant, with the patches documented in `MC5-spike/scripts/mc5_measure.py`.

**Listening note — bf16 vs 4-bit.** I cannot listen; the owner must. The attached WAV pair
(`samples/mc5-samples.zip`, same prompt/seed 12345/30 s, no LM) is the listening material, and
`results/audio_compare.json` records objective comparisons (log-mel cosine, log-spectral distance,
envelope correlation, byte identity). Read them as "how far apart are two renders", not as a
quality verdict: the same weights at the same seed are deterministic (DiT only), so the pair's
difference is entirely the 4-bit weight quantization. Summary of the objective result is in the
table below (log-mel cosine **0.872**, log-spectral distance 3.27, short-time envelope correlation 0.42, RMS 0.092 vs 0.079 (peak-normalised to the same −1 dBFS ceiling)).


### Sample settings (for the attached WAVs)

Common: turbo checkpoint, 8 steps, shift 1.0, ODE/Euler, DCW on (turbo default), bf16 one-copy
DiT, f16 chunked MLX VAE, no quantize, 30 s, `ACESTEP_MLX_VAE_FP16=1`, machine as above.
No-LM rows: `batch_size=1`. LM rows: `batch_size=2, seeds=[20251106, 20251107]` (upstream's
single-item LM path has no seeding; the batch path seeds per item — sample 0 is attached). Q8
bf16-vs-4-bit pair: no-LM, seed 12345, `batch_size=1`; the 4-bit row quantizes the MLX DiT with
`mlx.nn.quantize(group_size=64, bits=4)` at load.

| File in zip | caption (style prompt) | lyrics | vocal language |
|---|---|---|---|
| `Q4_noLM_instrumental_30s_seed20251106.wav` | upbeat electronic dance music with heavy bass, punchy drums, festival energy, 128 BPM | `[Instrumental]` | unknown |
| `Q4_noLM_english_vocal_30s_seed20251106.wav` | warm acoustic singer-songwriter ballad, intimate male vocal, fingerpicked guitar, soft strings, 84 BPM | English verse/chorus (see `scripts/prompts.json`) | en |
| `Q4_noLM_chinese_vocal_30s_seed20251106.wav` | gentle Mandarin pop ballad, female vocal, piano and strings, emotional, 72 BPM | Mandarin verse/chorus (see `scripts/prompts.json`) | zh |
| `Q4_0.6B_{instrumental,english_vocal,chinese_vocal}_30s_seed20251106.wav` | same three prompts | same | same |
| `Q4_1.7B_{instrumental,english_vocal,chinese_vocal}_30s_seed20251106.wav` | same three prompts | same | same |
| `Q8_bf16_instrumental_30s_seed12345.wav` | same instrumental prompt | `[Instrumental]` | unknown |
| `Q8_4bit_instrumental_30s_seed12345.wav` | same instrumental prompt | `[Instrumental]` | unknown |

The full prompt/lyrics strings are in `MC5-spike/scripts/prompts.json`; the exact resolved
parameters for every run are in the `params` block of each `results/json/*.json`.

**Objective A/B summary** (same prompt/seed/30 s, from `results/audio_compare.json`; a *verdict*
on the LM needs the owner's ears). The no-LM vs LM pairs differ substantially — expected, because
LM-on adds the FSQ code hint *and* (with the default `use_cot_caption`) replaces the caption with
the LM's rewritten one. The 0.6B duplicate row is the reproducibility check: it is **not**
byte-identical because the LM's CoT metadata phase is unseeded (C# minor vs D major in the two
runs).

- instrumental: no-LM vs 0.6B: log-mel cosine **0.617**, LSD 5.44, envelope corr 0.28, identical bytes: False
- instrumental: no-LM vs 1.7B: log-mel cosine **0.587**, LSD 5.83, envelope corr 0.21, identical bytes: False
- instrumental: 0.6B vs 1.7B: log-mel cosine **0.807**, LSD 3.79, envelope corr 0.73, identical bytes: False
- English vocal: no-LM vs 0.6B: log-mel cosine **0.641**, LSD 5.00, envelope corr 0.27, identical bytes: False
- Chinese vocal: no-LM vs 0.6B: log-mel cosine **0.644**, LSD 5.52, envelope corr 0.35, identical bytes: False
- 0.6B duplicate run (same settings): log-mel cosine **0.764**, LSD 3.96, envelope corr 0.47, identical bytes: False

---

## Q9 — MLX Swift gaps

**Bottom line: every MLX op and MLXNN module the upstream Python-MLX path actually uses has a
direct `mlx-swift` 0.32.3 equivalent, and the probe shows matching numerics.** The Python port is
also smaller than expected: it contains no quantization, no conv2d, no einsum/pad, no custom metal
kernels and no `mx.fast.rope` (RoPE is hand-rolled). The real gaps are architectural, not ops.

Probe: `MC5-spike/swift-probe/` — an SPM executable built against the pinned `mlx-swift` checkout
(`swift build -c release`, first build ≈4 min), with a Python-MLX counterpart and a diff script.
21 checks; numeric agreement vs Python mlx 0.30.6:

| Area | Max abs diff |
|---|---|
| conv1d (stride 1/2, pad, dilation) | 2e-6 |
| **convTransposed1d** (stride 2/4, pad 0/1/2) | **0–1e-6** (the feared gap is closed: `MLXNN.ConvTransposed1d`) |
| `fast.scaledDotProductAttention` f32/f16/bf16, 2-D/3-D/4-D masks | 0 |
| quantize/dequantize/quantized matmul 4-bit g64, 8-bit g32; `QuantizedLinear` packed layout | exact (same layout as Python) |
| hand-rolled RoPE vs `RoPE(traditional: false)` | 4.8e-7 (`traditional: true` differs by 8.4) |
| random key/split/normal; dtype promotion; f16 `-1e9` | bit-identical / identical / both give `-inf` |

The gap table (`findings/swift-gaps.md`, 24 rows with Python `path:line` ↔ Swift `file:line`) is
mostly renames: `mx.concatenate`→`concatenated`, `mx.expand_dims`→`expandedDimensions`, SDPA lives
in the main `MLX` module (not `MLXFast`), `compile` typed overloads stop at 3 array args (use the
`[MLXArray]→[MLXArray]` overload for the 5-arg DiT step). The real findings:

1. **No Swift counterpart at all:** (a) non-Haar DCW wavelets — upstream bridges through
   `pytorch_wavelets` per step (`dcw_correction_mlx.py:70-137`); implement filter banks or ship
   Haar (+`pix`) only, which is the turbo default. (b) FSQ `ResidualFSQ` + `AttentionPooler` +
   audio tokenizer + LM/lyric/text/encoders/detokenizer — these are **not in the Python MLX path**
   at all (it starts from PyTorch-computed `encoder_hidden_states`/`context_latents`), so they are
   new Swift modules + new fixtures. FSQ is op-trivial (Linear/tanh/floor/round, 6 levels); the
   work is parity, not availability.
2. **`mx.compile` freezes captured randomness in both runtimes** (measured). Upstream compiles the
   VAE `encode_and_sample` (`mlx_vae_init.py:73-75`), which silently makes its sampling
   deterministic — don't compile the encode path, or pass a fresh key as an input.
3. **A silent `mlx-swift` bug:** a multi-index subscript with `.newAxis` drops the inserted axes
   when the total op count ≤ rank (`m[.newAxis,.newAxis]` → `[8,8]`). Use `expandedDimensions(axis:)`
   in ported slicing; upstream's own idioms happen to avoid it.
4. **dtype/memory decision is the port's, not the library's:** the Python MLX DiT runs **float32**
   (converter does `.float()`, `dit_convert.py:39`) ≈ 9.6 GB; the Swift port will load bf16
   (3.15 GB) unless it deliberately upcasts. 4-bit via `MLXNN.quantize` is API-verified but has
   no upstream MLX reference output — hence the listening gate.
5. `mx.eval` per step is load-bearing (graph memory); no equivalent problem in Swift, same rule.

---

## Q10 — Durations and shapes

```
latent frames T      = round(seconds, 1) × 25        # 48000/1920 = 25 Hz, 64 channels
latent tensor         [1, T, 64] float32
DiT tokens            = ceil(T / 2)                  # patch_size 2
context channels      = 128 = 64 src_latents + 64 broadcast chunk mask
DiT input             = cat([context, noise], -1) → [B, T, 192] → Conv1d(k=2,s=2)
samples               = seconds × 48000, 2 channels
LM codes (LM on)      = int(duration × 5)            # 5 Hz FSQ indices, EOS forced
```

Constraints: hard API/UI range **10–600 s** (`constants.py:58-59`); `silence_latent` holds exactly
600 s and tiles silently beyond; architectural ceiling from `max_position_embeddings=32768` tokens
× patch 2 = 2621 s, so 600 s is not an arch limit; odd `T` is padded by 1 then cropped; the timbre
reference is a fixed 30 s (750 frames) regardless of output duration; VAE latent frame rate 25 Hz
(hardcoded `1920` at `conditioning_target.py:64,84,90`).

| seconds | latent `[T,64]` | DiT tokens | WAV 48 kHz stereo f32 | LM codes |
|---:|---:|---:|---:|---:|
| 30 | 187.5 KB | 375 | 11.52 MB | 150 |
| 60 | 375 KB | 750 | 23.04 MB | 300 |
| 120 | 750 KB | 1500 | 46.08 MB | 600 |
| 240 | 1.5 MB | 3000 | 92.16 MB | 1200 |
| 600 | 3.75 MB | 7500 | 230.4 MB | 3000 |

Latents are tiny; **cost is attention**: 12 full-attention layers are O((T/2)²) (60 s→750²,
600 s→7500² — 100× the attention work for 10× audio), 12 sliding-window layers are linear; a
600 s layer's MLP intermediate is ~184 MB. Per-step working set is the memory driver, not the
latent. **Recommendation for v1: offer 10–240 s (measured at 30/120/240 above), default 60 s,
with a hard cap at 4 min**; 480–600 s would need a FlashAttention-style kernel, not just a faithful
port. This matches the owner's 12 GB process cap for the no-LM configuration.

---

## Q11 — Parity plan (summary; full tables in `findings/parity-plan.md`)

The reference oracle is the **pinned upstream handler path** (torch/MPS text+condition encoders,
native MLX DiT/VAE) at the pinned commits; PyTorch classes are the specification. Two tiers, both
already established in this app:

- **Committed tiny tier** (follows `MLXUITests/Fixtures/RIFE/`): deterministic tiny configs with
  randomised weights (`TINY_QWEN`, `TINY_COND`, `TINY_DIT`, `TINY_VAE`), generated once by a
  committed script, SHA-256-pinned, **total ≈3 MB (hard ceiling 10 MB)**. G0 prompt/token ids are
  exact; every other fixture is numerical.
- **env-var-gated real-weight goldens** for the owner/CI machine, skipped by default (same pattern
  as the app's existing real-load tests).

Per-slice fixtures and tolerances (each captures inputs, every stage tap, and the output):

| Slice | Reference tensors | Input | Tolerance |
|---|---|---|---|
| 3c tokenizer/format | prompt/lyrics strings + ids + masks | the three Q4 prompts | exact bytes |
| 3c text encoder + projector | `last_hidden_state [1,L,1024]`, projected `[1,L,2048]` | 256-token prompt | cos ≥ 0.9995 (f32), ≥ 0.995 (bf16) |
| 3d condition encoder | lyric/timbre encoder taps, pooled tokens, packed `[1,S,2048]`, masks | tiny + real prompt/lyrics/silence | max abs Δ ≤ 1e-5 tiny; cos ≥ 0.9995 real |
| 3e DiT block + full forward | per-block out, `temb`, timestep proj, `patch_in`, cross K/V layer 0, pred at two σ | one tiny forward; real forward at T=250 | max abs Δ ≤ 1e-5 tiny; per-block norm ratio ≤ 2 % bf16 |
| 3f 8-step denoise (+ DCW on/off) | injected noise, x_k/v_k/dcw taps, final x0 | fixed noise (RNG itself is not parity-testable), schedule shift 1.0 | max abs Δ ≤ 1e-5 tiny; x8 cos ≥ 0.99 real |
| 3g VAE decode/encode | decoded waveform, decoder-block taps, incl. one chunked decode | 2 s stereo, real latents | max abs Δ ≤ 1e-5 tiny; cos ≥ 0.999, max Δ ≤ 0.02 real |
| 3h end-to-end | one 10 s no-LM WAV + all inter-stage hand-offs | fixed prompt/seed/settings | tiny max abs Δ ≤ 1e-5; real per-stage |

Corrections the plan makes vs the issue text: reference `shift` is **1.0**, not 3.0; the duration
floor is 10 s (so the end-to-end reference is 10 s, not 8 s); noise is injected into the fixture
(RNG is not a parity surface); reference DiT dtype must be recorded (f32 in upstream, bf16 in our
harness) because it changes the tolerance.

---

## Q12 — Slice plan and install shape (summary; full plan in `findings/slice-plan.md`)

**Catalog/install: one card, official repo, LM excluded.** The official HF repo is already a
component-subfolder layout, which `ModelFileSelector.componentDirs` supports. The v1 card is
`ACE-Step/Ace-Step1.5` **minus `acestep-5Hz-lm-1.7B/`**:

| Component | What ships | Size |
|---|---|---:|
| `Qwen3-Embedding-0.6B/` | model.safetensors 1,191.59 MB + tokenizer files 15.88 MB | 1,207.47 MB |
| `acestep-v15-turbo/` | model.safetensors 4,787.83 MB + `silence_latent.pt` 3.84 MB + config/code 0.11 MB | 4,791.78 MB |
| `vae/` | diffusion_pytorch_model.safetensors 337.43 MB + config | 337.43 MB |
| **total (v1)** | | **6,336.68 MB = 6.34 GB = 5.90 GiB** |

Three selector edits are required and named exactly: (1) an **exact-name exception** for
`acestep-v15-turbo/silence_latent.pt` (the generic `.pt` skip at `ModelFileSelector.swift:150-153`
fires before the component branch at `:163`; the file is mandatory), (2) an **excluded-components**
switch for the 1.7B LM subfolder, (3) no other metadata work — every tokenizer/config name the
components need is already in `metadataNames` (`ModelFileSelector.swift:18-69`). No per-component
install markers are needed (the root `.installed` marker covers it). If a later with-LM variant is
wanted, the 0.6B planner is a separate 1.37 GB repo; the 1.7B LM is 3.75 GB and excluded.

**Slices `MC5-3a…3i`** (≈16–21 engineer-days + 1–2 owner days; ordering is a dependency chain):

| Slice | What | Days | Main risk |
|---|---|---:|---|
| 3a | Catalog card, curated manifest, bridge row, ModelSupport gap, selector edits | 0.5–1 | `.pt` skip order |
| 3b | Selective safetensors reader (encoder.*/decoder.* from the one 4.79 GB file) + `.pt` reader | 1.5–2 | **highest**: naive load peaks 4.8 GB and defeats staging; torch-pickle format |
| 3c | Tokenizer (subfolder), SFT prompt, text encoder via MLXEmbedders, lyric embedding table | 1.5–2 | subfolder tokenizer; private `embed_tokens` |
| 3d | Condition encoder (lyric/timbre/pooler/pack) | 2–3 | 4-D masks, sliding/full alternation |
| 3e | DiT block + full forward | 3–4 | conv layout, AdaLN chunk order, GQA repeat |
| 3f | 8-step sampler + DCW | 1.5–2 | DCW silently skipped |
| 3g | VAE decoder (weight-norm fusion, Snake, tiling) | 2–3 | 365 tensors, silent layout mistakes |
| 3h | Engine, staging lifecycle, module registration, Run UI, `.music` row | 3–4 | same-file encoder/DiT split |
| 3i | Real-weights parity + owner listening gate | owner 1–2 | "sounds right" |

**`.music` memory row.** Today `.music` falls to `default` at `ModelWeightCharge.swift:115` (books
`max(files×1.10, catalog ramGB)`, unmeasured). The future edit is a measured `case .music:` before
`default:` (`:79-117`) with `factor 1.10` and
`workingSetBytes = ceil_to_0.05GB( max_stage_peak(recommended config) − F×1.10 )`, where
`F = 6,316,843,408 B` (the three components' safetensors) so `F×1.10 ≈ 6.95 GB`; if the measured
stage peak is below that, the working set is 0 and the row books 6.95 GB (safe over-book). The
measurement file for that decision is `results/measurements.json` in this deliverable. Note the
port projection (MLX-only) is materially below the whole-process peaks in Q8; the row should be
set from the port's own post-3h measurement, with the config and duration cap recorded as
provenance (same convention as the SAM-Audio row).

**Workflow wiring:** `Generate Sound` already exists (text→audio,
`engines.diffusion.generate_sound`, kind `.music`, pool `["MusicGen"]`), and `TaskModels.swift:160`
already serves the prefix. A later implementation adds the curated manifest + `CatalogBridge` row,
**appends** the ACE-Step display name to the pool (`TaskModels.swift:257`, append-last;
MusicGen stays), and must add **two new `StageConfig` fields — `duration` and `lyrics`** — plus
their parsing in `RealExecutor.stageConfig`, otherwise every flow run falls back to 120 s and can
never carry lyrics. That vocabulary addition needs a `catflow-mlx` `SPEC_QUESTIONS.md` entry first
(the curated manifests are ported verbatim; inventing settings is catalog semantics).

---

## Q13 — Licence and provenance

| Component | Declared | Actually governed by | Commercial/App Store |
|---|---|---|---|
| ACE-Step 1.5 code (repo) | MIT, © 2026 ACEStep | MIT | ✅ |
| `acestep-v15-turbo/model.safetensors` | MIT (card) | MIT | ✅ |
| `Qwen3-Embedding-0.6B/*` | bundled in an MIT repo | **Apache-2.0** (Qwen weights keep Qwen's licence) | ✅ with Apache notice |
| `acestep-5Hz-lm-1.7B` (excluded from v1) | MIT card | **Apache-2.0** (Qwen3-1.7B derivative) | ✅ with notice |
| `vae/diffusion_pytorch_model.safetensors` | MIT card, no provenance notice | **undetermined** — `AutoencoderOobleck` is Stability's architecture, and `stable-audio-open-1.0` is gated under the Stability AI Community Licence | ⚠️ **verify before shipping** |
| `silence_latent.pt`, tokenizers | MIT / Apache-2.0 | as above | ✅ |

**The one real licence question is the VAE.** The repo declares MIT on the card, but the VAE is an
`AutoencoderOobleck` (Stability AI's stable-audio VAE architecture) and the repo ships no
third-party notice; `stabilityai/stable-audio-open-1.0` is gated (`license: other` = Stability AI
Community Licence). I could not diff the tensors because the Stability repo needs a gated token
(10-minute check once one is available; 48 kHz-vs-44.1 kHz and different channel counts are weak
evidence it was re-trained). **If it is a Stability derivative, the community licence's
acceptable-use terms and revenue cap attach, and a paid App Store build is in restricted
territory** — the app already renders that licence specially
(`DataNormalizer.swift:19`). Everything else is permissive: no copyleft, no field-of-use limits,
no source-availability trigger (a Swift port runs no remote code; the `trust_remote_code=True` in
upstream is a Python trust boundary for the *reference*, not for the app). Attribution is the
follow-up: label the entry `mit` + "includes Apache-2.0 Qwen3 weights" and surface both notices;
don't ship ACE-Step/StepFun trademarks as ours. The cross-check 4-bit repos
(`MLXCreator/...` labels Apache-2.0, `mlx-community/ACE-Step1.5-MLX` declares no licence) are
**not** usable as licence references.

---

## Method and caveats (so the numbers can be read honestly)

- Everything ran on the **18 GiB M3 Pro** named above; the issue's 32 GB guidance was not available.
  The machine reports 18.0 GiB to `sysctl` and 13.3 GiB to the upstream MPS tier heuristic
  (`torch.mps.recommended_max_memory()`), so upstream's own policy puts it in tier5
  (480 s with LM / 600 s without) — that tier table was written for CUDA VRAM and is only a hint.
- **Upstream's macOS path was not modified in place.** Measurement patches are applied at runtime in
  `MC5-spike/scripts/mc5_measure.py` and are printed in every result JSON ("patches" key); with
  them the pipeline is the pinned upstream code. The `upstream` configuration (unpatched) was
  attempted and does not fit (see Q8).
- LM runs use `batch_size=2, seeds=[s,s+1]`: upstream's **single-item LM path has no seeding at
  all**, and even the batch path only seeds the codes phase — the once-per-batch CoT metadata phase
  is unseeded. The duplicate run confirms the consequence: same seed, different caption/keyscale,
  log-mel cosine 0.764. A fixed seed is therefore only fully meaningful in the no-LM path.
- The Q4 no-LM and LM WAVs are the listening material; objective comparisons are in
  `results/audio_compare.json`. I cannot judge musical quality; the owner's listen rules on the LM
  and on bf16-vs-4-bit.
- The legacy `silence_latent.pt` is a torch pickle; the port should convert it once to
  safetensors/raw f32 rather than implement a pickle reader.
- Time spent: **≈ 45 minutes of agent wall time** for this spike (setup + weight/venv downloads,
  four parallel research tracks, the measurement matrix, the Swift probe build, and this report).
  The slice estimates (16–21 engineer-days) are for a human implementer working through Xcode.

## Evidence (ticket branch, no PR)

Attached per the issue's fallback rule (the agent account has no gist scope for binary uploads, and
GitHub issue comments cannot carry files). No weight files are committed.

- Tree: <https://github.com/Misfits-Rebels-Outcasts/MLXUI/tree/agent/ace-step-spike-f8c22e1e4b0d/MC5-spike>
- README (method, reproduction): <https://github.com/Misfits-Rebels-Outcasts/MLXUI/blob/agent/ace-step-spike-f8c22e1e4b0d/MC5-spike/README.md>
- Scripts: <https://github.com/Misfits-Rebels-Outcasts/MLXUI/tree/agent/ace-step-spike-f8c22e1e4b0d/MC5-spike/scripts> — measurement harness, VAE probe, analysis, matrix runner, stdlib safetensors inventory
- Findings (Q1–Q13 backing notes with full `file:line` citations): <https://github.com/Misfits-Rebels-Outcasts/MLXUI/tree/agent/ace-step-spike-f8c22e1e4b0d/MC5-spike/findings>
- Swift probe (builds against pinned mlx-swift): <https://github.com/Misfits-Rebels-Outcasts/MLXUI/tree/agent/ace-step-spike-f8c22e1e4b0d/MC5-spike/swift-probe>
- Raw results: <https://github.com/Misfits-Rebels-Outcasts/MLXUI/tree/agent/ace-step-spike-f8c22e1e4b0d/MC5-spike/results>
- WAV set (Q4's 9 + bf16/4-bit pair): <https://github.com/Misfits-Rebels-Outcasts/MLXUI/raw/agent/ace-step-spike-f8c22e1e4b0d/MC5-spike/samples/mc5-samples.zip>

**Not delivered / open:** the VAE-vs-Stability tensor diff (gated token needed), and the LM-in-v1
and bf16-vs-4-bit calls (owner listening). No PR against `main` was opened from this issue.
