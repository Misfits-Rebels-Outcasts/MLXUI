This is by an automated agent, not a human.

# MC5-1 spike — MLX Swift port plan for ACE-Step 1.5 — **part 1/2: Q1–Q7**

Report for #10. Part 2/2 is the next comment (Q8–Q13, measurements, slice plan, licence).
Full scripts, probe package, raw result JSONs and the WAV set are committed to the ticket branch
(links at the end of part 2).

## Pinned inputs and machine (apply to both parts)

| Item | Value |
|---|---|
| Official weights | `ACE-Step/Ace-Step1.5` @ `19671f406d603126926c1b7e2adc169acbcade22` — 10,092,102,593 B, 28 files |
| Upstream code | `github.com/ace-step/ACE-Step-1.5` @ `ca1e85fe9430179831e6bc6be790c332190a3866` (MIT); all `path:line` below are to this commit |
| Swift packages | `mlx-swift` 0.32.3 (`1960120`), `mlx-swift-lm` 3.32.3 (`3b339ad`), `mlx-audio-swift` `5c4c12b` (app pins) |
| Upstream Python env | `uv sync --frozen` from the pinned `uv.lock`: Python 3.12.10, `mlx` **0.30.6**, `mlx-lm` 0.29.1, `torch` 2.10.0, `transformers` 4.57.6, `safetensors` 0.7.0 (recorded because the MLX Swift pin is 0.32.x) |
| Machine | **Apple M3 Pro, 12 cores, 18 GiB RAM** (`hw.memsize` = 19,327,352,832 B), macOS 27.0, Xcode 27.2 / Swift 6.4.2 |
| Disk | 38 GiB free at start; all work under `/private/tmp` |
| Date | 2026-11-07 |

Two things found while setting up change how the issue should be read:

1. **Upstream is not "no MLX on macOS" — it is a partial native-MLX port.** At this commit the DiT
   decoder (`acestep/models/mlx/dit_model.py`), the VAE encode/decode (`vae_model.py`) and the 5 Hz
   LM (via `mlx-lm`) are native MLX and are the *default* on Apple Silicon
   (`acestep/gpu_config.py:846-867`, `init_service_setup.py:114-146`). What is still PyTorch is the
   **text encoder + the condition encoder stack** (text projector, lyric encoder, timbre encoder,
   attention pooler, FSQ tokenizer/detokenizer). So the port is "port what is still torch", not
   "port everything".
2. **The issue's "full bf16 path" is not what upstream runs on MPS.** `initialize_service` sets the
   torch dtype to **float32** for any non-CUDA/XPU device
   (`init_service_orchestrator.py:99-105`) and the MLX decoder is built by
   `value.detach().cpu().float().numpy()` (`models/mlx/dit_convert.py:39`), i.e. **a second f32
   copy**, with the torch decoder never released. That is ~19 GB of DiT alone and does not fit
   18 GiB. For the measurements in part 2 we ran upstream's code with two documented,
   measurement-only patches (bf16 one-copy + staged free); the measured numbers and the exact
   patches are in Q8/§Method in part 2. Nothing was changed in the app repo.

---

## Q1 — Component map: prompt text → WAV

| # | Stage | Upstream location | macOS today |
|---|---|---|---|
| 1 | Prompt/metadata/lyrics string formatting (SFT prompt `# Instruction / # Caption / # Metas`) | `acestep/constants.py:160-168`; metadata block `metadata_utils.py`; tokenize `conditioning_text.py:127-146` (`max_length=256` prompt, `2048` lyrics) | CPU (HF tokenizer) |
| 2 | **Text encoder** `Qwen3-Embedding-0.6B` (last hidden state, 1024-d) | load `init_service_loader_components.py:68-102`; forward `conditioning_embed.py:71-74` | **PyTorch/MPS f32** |
| 3 | **5 Hz planner LM** (optional): 0.6B/1.7B/4B Qwen3-derivative | `llm_inference.py` class `:48`, `initialize` `:499`; MLX load `:2933-3018` | **native MLX** (`mlx-lm`) |
| 4 | **Condition encoder** = `text_projector` + `AceStepLyricEncoder` (8 L) + `AceStepTimbreEncoder` (4 L) + `pack_sequences` | `models/turbo/modeling_acestep_v15_turbo.py:1513-1558`, `:580-735`, `:1000-1182`, `:141-173` | **PyTorch/MPS** |
| 5 | **Audio tokenizer** = `audio_acoustic_proj` → `AttentionPooler` (2 L) → `ResidualFSQ` (levels `[8,8,8,5,5,5]`, dim 2048) | `modeling_acestep_v15_turbo.py:1184-1226`, pooler `:737-863` | **PyTorch/MPS** |
| 6 | **Detokenizer** (FSQ index → 25 Hz latent): 2 encoder layers, 5-patch expansion | `modeling_acestep_v15_turbo.py:865-998` | **PyTorch/MPS** |
| 7 | `prepare_condition`: runs 4+5+6, returns `encoder_hidden_states [B,S,2048]`, mask, and `context_latents [B,T,128]` = `cat([src_latents(64), chunk_mask(64)])` | `modeling_acestep_v15_turbo.py:1654-1699`; called at `service_generate_execute.py:176-194` **even when the sampler is MLX** | PyTorch/MPS |
| 8 | **DiT** — 24 layers, 2048 hidden, 16/8 GQA heads, head_dim 128, alternating sliding(128)/full attention, patch 2, in 192→64 | torch ref `:1243-1509`; **MLX** `models/mlx/dit_model.py:421+` | **native MLX** |
| 9 | **Sampler** — 8-step Euler ODE + DCW (Haar) | **MLX** `models/mlx/dit_generate.py:168-535`; torch fallback `:1852-2226`; dispatch `service_generate_execute.py:197-277` | **native MLX** |
| 10 | **VAE decode** — `AutoencoderOobleck`, 64→2 ch, ×1920, 48 kHz | **MLX** `models/mlx/vae_model.py:237-330`, tiled wrapper `mlx_vae_decode_native.py:27-140`; torch fallback `vae_decode*.py` | **native MLX** |
| 11 | Peak-normalise, save WAV | `generate_music_decode.py:199-203`, `inference.py` payload | CPU |

The torch→numpy→MLX seams are exactly two: DiT (`diffusion.py:121-191`) and VAE
(`mlx_vae_decode_native.py:42-72`). A Swift port collapses both into one MLX graph. The condition
encoder runs in torch **on every generation**, including no-LM runs, because FSQ tokenisation of the
target latents lives there (`modeling_acestep_v15_turbo.py:1688-1695`).

---

## Q2 — Weights: tensor groups, module map, layout conversions, static buffers

Measured directly from the downloaded bytes with a stdlib safetensors-header parser
(`scripts/weights/inventory_safetensors.py`, `python3 -I`, validates every `data_offsets` span):

| File | Bytes | Tensors | Params | Dtypes |
|---|---:|---:|---:|---|
| `acestep-v15-turbo/model.safetensors` | 4,787,825,604 | 677 | 2,393,815,672 | **BF16 only** |
| `acestep-5Hz-lm-1.7B/model.safetensors` | 3,708,521,528 | 310 | 1,854,132,992 | BF16 |
| `Qwen3-Embedding-0.6B/model.safetensors` | 1,191,586,416 | 310 | 595,783,424 | BF16 |
| `vae/diffusion_pytorch_model.safetensors` | 337,431,388 | 365 | 168,695,426 | BF16 |
| **total** | **10,025,364,936** | 1,662 | **5,012,588,296** | |

All four are single-file, unsharded, unquantised (no `.scales`/`.biases`, no index json). **There is
no f32 tensor in any safetensors.** The only f32 file is `silence_latent.pt` (Q7).

**Module map** (one `AceStepConditionGenerationModel`, `models/turbo/modeling_acestep_v15_turbo.py:1593`):

| Tensor prefix | Tensors | Params | Loads into |
|---|---:|---:|---|
| `decoder.*` | 476 | 1.575 B (65.8 %) | `AceStepDiTModel` (`:1243`) |
| `encoder.*` | 140 | 608.37 M (25.4 %) | `AceStepConditionEncoder` (`:1513`) — *yes, the condition encoder is inside this one file* |
| `tokenizer.*` | 32 | 105.03 M (4.4 %) | `AceStepAudioTokenizer` (`:1184`) |
| `detokenizer.*` | 28 | 105.01 M (4.4 %) | `AudioTokenDetokenizer` (`:865`) |
| `null_condition_emb` | 1 | 2,048 | model-level null embedding |

Layers: 24 DiT (12 sliding window 128 / 12 full, 16 q-heads / 8 kv-heads, head_dim 128, MLP 6144),
8 lyric-encoder, 4 timbre-encoder, 2 pooler / detokenizer layers — all match `config.json`.
Full per-module shape lists: `findings/inventory/modules_acestep-v15-turbo.txt`.

**Layout conversions a Swift loader must do**

- **DiT (only 2 tensors!):** `decoder.proj_in.1.weight [2048,192,2]` → rename `proj_in.*`, `swapaxes(1,2)`
  → `[2048,2,192]`; `decoder.proj_out.1.weight [2048,64,2]` → rename `proj_out.*`, `transpose(1,2,0)`
  → `[64,2,2048]`; drop `rotary_emb` buffers (recomputed). Everything else: plain `[out,in]` Linear,
  no conversion. Reference: `models/mlx/dit_convert.py:11-66`.
- **Condition encoder/tokenizer/detokenizer (all 200 tensors): zero layout conversion** — plain
  Linear/Embedding layouts (the reference leaves them untouched).
- **VAE is the trap:** 74 **weight-norm pairs** must be fused (`w = g·v/‖v‖` along dim 0) *before*
  transposing; 5 `conv_t1` are ConvTranspose1d (`[in,out,K]→[out,K,in]`) and 69 are Conv1d
  (`[out,in,K]→[out,K,in]`); 144 Snake `alpha`/`beta` `[1,C,1]→[C]`. Reference: `models/mlx/vae_convert.py:36-200`.
  Getting the fuse/transpose order wrong is silent numeric corruption.
- MLX is channels-last `[B,L,C]` while torch is `[B,C,L]`; every conv boundary and the VAE decode
  does a transpose (`mlx_vae_decode_native.py:42-56`).

**Static buffers / load-time-derived values** — five, all accounted for:

1. `silence_latent` — separate `acestep-v15-turbo/silence_latent.pt`, hard-required (Q7).
2. RoPE tables — recomputed, **not stored** (0 tensors) in either DiT (`dit_model.py:62-92`) or VAE.
3. FSQ level/codebook constants — from config (`ResidualFSQ`, `:1199-1207`); codebook size 64,000.
4. Sliding-window additive masks — built per sequence length at forward time and cached
   (`dit_model.py:38-55,517-534`); the port must cache them too.
5. `timbre_fix_frame=750` — **dead config**: the literal `750` (30 s × 25 Hz) is hardcoded at
   `conditioning_embed.py:44-48`, and a 30 s zero reference clip is synthesised
   (`conditioning_batch.py:70`). Do not read it from config.

---

## Q3 — Text encoder: exact tensor, prompt, and whether MLX Swift can produce it

Upstream takes `Qwen3Model.last_hidden_state` — **post-final-RMSNorm, causal, un-pooled**,
`[B,L,1024]` (`conditioning_embed.py:71-74`; HF `Qwen3Model` with `input_ids` and **no
attention_mask** → full causal attention). The prompt is the SFT template
`# Instruction\n{DEFAULT_DIT_INSTRUCTION}\n\n# Caption\n{caption}\n\n# Metas\n{metadata}<|endoftext|>`
(`constants.py:160-168`), tokenised with **`max_length=256`** (`conditioning_text.py:127-135`), no
chat template, no BOS/EOS beyond the literal `<|endoftext|>`. Lyrics are **not** run through the
transformer at all: `embed_tokens` lookup only (`conditioning_embed.py:76-79`). The output is
projected by a bias-free `Linear(1024→2048)` (`text_projector`, `modeling_...:1525-1527`).

**Can MLX Swift give this tensor through a public API? Yes — do not copy the forward.** In the pinned
`mlx-swift-lm` 3.32.3:
`EmbedderTypeRegistry.shared` registers `qwen3` → `Qwen3Model`
(`Libraries/MLXEmbedders/ModelFactory.swift:37`); `Qwen3Model.callAsFunction` is public and returns
`EmbeddingModelOutput.hiddenStates` = `norm(h)` (`Models/Qwen3.swift:343-356`, `:294`), i.e. exactly
the post-norm last hidden state; `loadContainer` + `container.perform` is the same public entry point
the app already uses (`MLXUI/Modules/Embedding/EmbeddingEngine.swift:44-47`). Recommendation:
**use `MLXEmbedders`, call `context.model(...)` and read `.hiddenStates`** — do not call `encode`
(that pools; Qwen3's pooling strategy is `.last`). `MLXLLM`'s qwen3 is **not** usable (it only
surfaces logits, and the LM head is tied/absent). One real API gap: `Qwen3Model.model.embed_tokens`
is private, so the **lyric embedding table** must be loaded as a separate
`Embedding(151669, 1024)` from the same `model.safetensors` tensor (155 MB) — no fork needed.
Full evidence: `findings/text-encoder.md` §2.

---

## Q4 — The planner LM: what it is, how it feeds the DiT, and the measured A/B

**Mechanism (the important correction).** The LM's text never reaches the DiT. The 5 Hz LM is a
Qwen3-architecture causal LM (`vocab_size 64003`) that emits, in two constrained phases
(`llm_inference.py:1282-1560`): (1) a YAML-ish metadata block — `bpm`, `caption`, `duration`,
`keyscale`, `language`, `timesignature` — under an FSM
(`constrained_logits_processor.py:56-78,204-230`); (2) exactly `int(duration·5)` audio-code tokens
`<|audio_code_N|>` where `N ∈ [0,63999]` (`constrained_logits_processor.py:1616-1628`,
`MAX_AUDIO_CODE = 63999`). The codes are then **FSQ indices decoded to a 25 Hz latent**
(`audio_codes.py:50-72`: `quantizer.get_output_from_indices` → `detokenizer`, 5× expansion) and used
to **replace `src_latents`** in `prepare_condition`
(`modeling_acestep_v15_turbo.py:1685-1698`). So the LM contributes an *initial latent + metadata*,
not a prompt prefix. `audio_codes` also sets `is_covers = True` for any task
(`conditioning_masks.py:69-71`).

**Is the LM required for text-to-music? No.** Gate: `inference.py:640-655`; `text2music` is not in
`DIRECT_CONDITIONING_TASKS` (`:36-41`). LM-off happens when `llm_handler` is absent/uninitialised,
or `thinking=False` **and** all `use_cot_*` are False. Without the LM: `audio_codes=""` → no hint →
`src_latents` = `silence_latent` slice and `context_latents` is a constant; all conditioning comes
from the text/lyric/timbre encoders. Note the API defaults are LM-on (`thinking=True`, all
`use_cot_*` True, `inference.py:203-211`), so a no-LM caller must set all four (that is what the
harness does).

**Measured A/B** (M3 Pro 18 GiB, 30 s, seed 20251106, turbo 8 steps, all three prompts; WAVs in the
attached zip; settings in part 2). The LM streamed its planned metadata/codes (recorded in the
result JSONs). Objective comparison of the audio pairs is in `results/audio_compare.json` and
summarised in part 2. Because the single-item LM path is **not seeded** at all
(`llm_inference.py:985/3397/3720`; seeds exist only on batch paths), the LM runs were done at
`batch_size=2` with `seeds=[s,s+1]` so the fixed seed is meaningful; sample 0 is attached. A
duplicate run of the 0.6B/instrumental case is included to show whether that reproducibly applies.
The owner should decide "LM in v1" by listening to the 9 WAVs; the code fact is that no-LM is a
complete, deterministic text2music path, and the LM is a quality/planning add-on.

---

## Q5 — Sampler

**Turbo schedule** (`models/turbo/modeling_acestep_v15_turbo.py:1914-1987`; MLX copy
`models/mlx/dit_generate.py:31-97` — same constants): infer-steps path
`raw = [1 − i/n]`, `n = min(infer_steps,20)`, then `t = shift·t/(1+(shift−1)·t)` when `shift≠1`.
Default `inference_steps=8, shift=1.0` → `[1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]`; the
final `t=0` step is the direct x0 jump (`xt − v·t_curr`), not in the list. `shift∈{1,2,3}` is
discretised when no explicit step count is given; explicit `timesteps` are snapped to
`VALID_TIMESTEPS` in torch (the MLX copy does **not** snap — a divergence to keep in mind for
parity). **`timestep_mu=-0.4` / `timestep_sigma=1.0` are training-only** (sampled in `forward` at
`:1755`); the inference samplers never read them.

**CFG/APG/DCW on turbo:** CFG is forced off (`generate_music.py:309-318`), `use_adg` is
accept-and-ignore for turbo; **APG code exists in the MLX sampler but is unreachable for turbo**
(it needs `guidance_scale>1` + `null_condition_emb`, `dit_generate.py:281,369-377`). **DCW is ON by
default for turbo** (`dcw_enabled=None → is_turbo_model()` `generate_music.py:181-189`; Haar, scalers
0.05/0.02, applied after each non-final step and at x0, `dit_generate.py:492-513`). Non-Haar wavelet
modes bridge through PyTorch — out of scope for v1 (Q9).

**Noise / reproducibility:** torch `Generator(device).manual_seed(seed)` → `randn((B,T,64))`
(`:1780-1817`); MLX `mx.random.key(seed)` → `mx.random.normal` (`dit_generate.py:295-308`),
deterministic for a key. End-to-end fixed-seed reproducibility also needs
`GenerationConfig.use_random_seed=False` + `seeds=[N]` (default `use_random_seed=True` *overwrites*
your seed, `inference.py:298-330,576-606`), `infer_method="ode"` **not "sde"** (SDE renoise uses the
global RNG, `dit_generate.py:470`), and **the LM off** (or batch-seeded, above). The DiT itself is
deterministic for a fixed key on the ODE/euler path.

---

## Q6 — VAE

`AutoencoderOobleck` (`vae/config.json`): output **48 kHz, 2 ch, float32** (peak-normalised to ≤1,
`generate_music_decode.py:199-203`; `handler.py:121`). Latent rate **25 Hz** (`prod(downsampling_ratios)
= 2·4·4·6·10 = 1920`, `48000/1920 = 25`; hardcoded as `1920` at `conditioning_target.py:64,84,90`),
**64 channels**. (The `vae_model.py:242-244` docstring says hop 2048 — it is wrong;
`from_pytorch_config` overrides with the real `[2,4,4,6,10]`.)

**Decode must be chunked.** MLX path: `mlx_chunk = max(192, self.mlx_vae_chunk_size)`, overlap 64,
stride `chunk−128`; `mlx_vae_chunk_size` comes from a memory heuristic (`gpu_config.py:779-806`):
this Mac reports 13.3 GiB to that heuristic → **chunk 256, stride 192 = 7.68 s of audio per step**.
The torch MPS path is worse: chunk forced to 32/overlap 8 (`vae_decode.py:13-14,60-77`), which is
why the MLX VAE exists.

**Measured decode cost on this Mac** (isolated probe, `scripts/mc5_vae_probe.py`; MLX VAE only,
torch model/text freed first; process footprint includes remaining torch overhead so compare
*deltas*):

| Audio | latent frames | f32 MLX VAE | f16 MLX VAE (`ACESTEP_MLX_VAE_FP16=1`) |
|---:|---:|---|---|
| 10 s | 250 | 0.82 s | **1.01 s**, process peak 10.08 GB, Δ0.33 GB |
| 60 s | 1500 | 6.82 s | **5.03 s**, peak 9.91 GB |
| 120 s | 3000 | 13.87 s | — |
| 240 s | 6000 | **aborted**: process reached 13.84 GB with 5.3 GB swap after ~60 s | **20.03 s**, peak 11.25 GB, Δ4.20 GB |

So decode is ~5 s per minute of audio in f16 (≈0.11× real time), roughly linear; memory scales with
the chunk size plus a few GB of transient conv buffers. **Recommendation: ship f16 MLX VAE decode
(the env switch exists in upstream) and chunk at ≤256 latent frames** — f32 decode at 4 min did not
fit this machine in a reasonable time.

---

## Q7 — `silence_latent`

A precomputed **latent of silence** (the VAE encoding of near-silence), used as `src_latents`/context
for text2music without LM codes and as padding for LM hints. It is a **file in the official HF repo**:
`acestep-v15-turbo/silence_latent.pt`, 3,841,215 B, loaded at
`init_service_loader.py:211-216` — **mandatory** (init raises `FileNotFoundError` without it).
Verified by parsing the pickle: stored tensor is **f32, `[1,64,15000]`**, transposed at load to
`[1,15000,64]` = 15,000 latent frames = **exactly 600 s at 25 Hz** (the API max duration); the
helper tiles it if asked for longer (`conditioning_target.py:18-33`), it does not error. Text2music
**always** needs it: LM-off → it *is* the context; LM-on → it pads short code hints
(`conditioning_text.py:34-46`); and the "no reference audio" timbre input is
`silence_latent[:, :750, :]` — 30 s at 25 Hz (`conditioning_embed.py:44-48`, hardcoded `750`).
Ship it as a 3.84 MB asset, converted once to safetensors/raw f32 (a Swift-side torch-pickle parser
is not worth writing).

---

*Part 2 follows: Q8 (measured memory/time), Q9 (MLX Swift gaps), Q10 (durations/shapes),
Q11 (parity), Q12 (slice plan + install shape), Q13 (licence), method/caveats, and the attachment
links.*
