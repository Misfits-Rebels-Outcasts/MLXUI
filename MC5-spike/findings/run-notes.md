# ACE-Step 1.5 — headless run notes for the measurement harness

Repo pinned `ca1e85fe9430179831e6bc6be790c332190a3866`.
Weights root used below: `/private/tmp/mc5/Ace-Step1.5` (HF `ACE-Step/Ace-Step1.5`).
All `path:line` relative to the repo root.

**Download state check (verified at time of writing):** the weights root already contains
`acestep-v15-turbo/model.safetensors` (4.79 GB), `acestep-v15-turbo/silence_latent.pt`,
`acestep-5Hz-lm-1.7B/model.safetensors` (3.71 GB), a bonus `acestep-5Hz-lm-0.6B/` (1.3 GB),
`Qwen3-Embedding-0.6B/model.safetensors` (1.19 GB), `vae/diffusion_pytorch_model.safetensors`
(337 MB). That is all four `MAIN_MODEL_COMPONENTS` (`model_downloader.py:307-315`), so the
auto-download gate will be satisfied and **no network is needed**.

---

## 1. The canonical headless invocation

The repo ships one: **`run_generate_test.py`** at the repo root. It is exactly the programmatic
entry point (no Gradio, no server) and it is the file to copy. Its imports are `run_generate_test.py:18-20`; `main()` at `:31`, `GenerationParams` at `:75`, `GenerationConfig` at `:90`, the call at `:96`:

```python
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
```

Minimal working script (adapting `run_generate_test.py:31-120`):

```python
import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"       # handler.py:9 does this anyway
os.environ["ACESTEP_CHECKPOINTS_DIR"] = "/private/tmp/mc5/Ace-Step1.5"
os.environ["HF_HUB_OFFLINE"] = "1"                    # belt-and-braces: never hit the network
CWD = "/private/tmp/mc5/ACE-Step-1.5"                 # repo root; get_project_root() uses cwd
os.chdir(CWD)
sys.path.insert(0, CWD)

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

CKPT = "/private/tmp/mc5/Ace-Step1.5"

# ---- DiT / VAE / text encoder ----
dit = AceStepHandler()
status, ok = dit.initialize_service(
    project_root=CWD,          # only used if ACESTEP_CHECKPOINTS_DIR is unset
    config_path="acestep-v15-turbo",
    device="auto",             # -> "mps" on Apple Silicon (init_service_setup.py:14-24)
    offload_to_cpu=False,      # see §6 memory note before choosing True
    offload_dit_to_cpu=False,
    use_mlx_dit=True,          # native MLX DiT (default)
    quantization=None,         # torchao; unsupported on MPS, gpu_config forces None
    compile_model=False,       # torch.compile unsupported on MPS; MLX uses mx.compile
)
assert ok, status

# ---- 5 Hz LM (omit entirely for a no-LM run) ----
lm = LLMHandler()
status, ok = lm.initialize(
    checkpoint_dir=CKPT,                 # joined with lm_model_path, must exist
    lm_model_path="acestep-5Hz-lm-1.7B", # or "acestep-5Hz-lm-0.6B" (faster, in the same repo)
    backend="mlx",                       # -> mlx-lm native (llm_inference.py:678-697)
    device="auto",
    offload_to_cpu=False,
    dtype=None,                          # forced float32 on mps (llm_inference.py:568-583)
)
assert ok, status

params = GenerationParams(
    task_type="text2music",
    caption="warm lo-fi hip hop, mellow piano, vinyl crackle, 85 bpm",
    lyrics="[Instrumental]",
    instrumental=True,
    duration=30.0,             # seconds; 10..600 (constants.py:58-59); -1 => LM/auto
    vocal_language="en",
    inference_steps=8,         # turbo default
    shift=1.0,                 # 1.0/2.0/3.0 discretised; see code-map Q5
    infer_method="ode",        # "sde" is NOT reproducible
    guidance_scale=1.0,        # forced to 1.0 for turbo anyway
    seed=1234,
    # ---- no-LM run: uncomment the next four ----
    # thinking=False,
    # use_cot_metas=False, use_cot_caption=False, use_cot_language=False,
)
config = GenerationConfig(
    batch_size=1,
    use_random_seed=False,     # CRITICAL: default True discards `seed`
    seeds=[1234],              # takes precedence over params.seed
    audio_format="wav",
)

result = generate_music(dit, lm, params=params, config=config,
                        save_dir="/private/tmp/mc5/out")
if not result.success:
    raise SystemExit(result.error)
for a in result.audios:
    print(a["path"], a["sample_rate"], tuple(a["tensor"].shape))   # [2, samples] float32 CPU
print(result.extra_outputs["time_costs"])
```

### Signatures (verbatim)

- `AceStepHandler.initialize_service(project_root, config_path, device="auto",
  use_flash_attention=False, compile_model=False, offload_to_cpu=False,
  offload_dit_to_cpu=False, quantization=None, prefer_source=None, use_mlx_dit=True,
  vae_checkpoint=None) -> (status_msg: str, success: bool)`
  — `acestep/core/generation/handler/init_service_orchestrator.py:48-61`
- `LLMHandler.initialize(checkpoint_dir, lm_model_path, backend="vllm", device="auto",
  offload_to_cpu=False, dtype=None) -> (status_msg, success)`
  — `acestep/llm_inference.py:499-507`
- `generate_music(dit_handler, llm_handler, params, config, save_dir=None, progress=None)
  -> GenerationResult` — `acestep/inference.py:514-521`

`AceStepHandler()` takes no arguments (`acestep/handler.py:103`) — everything is set by
`initialize_service`.
`LLMHandler()` accepts only `persistent_storage_path` (`llm_inference.py:56`) — not needed.

---

## 2. Pointing at a pre-downloaded checkpoints root (no auto-downloader)

Resolution order inside `initialize_service` (`init_service_orchestrator.py:119-131`):

```python
env_ckpt = os.environ.get("ACESTEP_CHECKPOINTS_DIR")
if env_ckpt:
    checkpoint_dir = str(get_checkpoints_dir())        # -> env var wins
elif project_root:
    checkpoint_dir = os.path.join(project_root, "checkpoints")
else:
    checkpoint_dir = str(get_checkpoints_dir())
```

and `get_checkpoints_dir()` itself resolves `ACESTEP_CHECKPOINTS_DIR` → `<get_project_root()>/checkpoints`,
where `get_project_root()` is `ACESTEP_PROJECT_ROOT` or **`os.getcwd()`**
(`model_downloader.py:329-359`).

So the recommended, unambiguous setup:

```bash
export ACESTEP_CHECKPOINTS_DIR=/private/tmp/mc5/Ace-Step1.5
cd /private/tmp/mc5/ACE-Step-1.5          # so get_project_root() is sane
```

Then the layout `<root>/<config_path>` must exist, i.e.
`/private/tmp/mc5/Ace-Step1.5/acestep-v15-turbo`, and the loader reads these exact paths:

| Component | Joined path | Code |
|---|---|---|
| DiT | `<root>/<config_path>` | `init_service_orchestrator.py:161` |
| VAE | `<root>/vae` (`resolve_vae_path`, variant `"official"`) | `init_service_loader_components.py:46-49` |
| text encoder | `<root>/Qwen3-Embedding-0.6B` | `init_service_loader_components.py:90-95` |
| `silence_latent.pt` | `<root>/<config_path>/silence_latent.pt` — **hard failure if absent** | `init_service_loader.py:211-213` |
| LM | `LLMHandler.initialize(checkpoint_dir=<root>, lm_model_path="acestep-5Hz-lm-1.7B")` → `<root>/acestep-5Hz-lm-1.7B` | `llm_inference.py:589-592` |

**No download will fire** iff every one of `MAIN_MODEL_COMPONENTS`
(`model_downloader.py:307-315`: `acestep-v15-turbo`, `vae`, `Qwen3-Embedding-0.6B`,
`acestep-5Hz-lm-1.7B`) contains one of the weight filenames accepted by
`_contains_model_weights` (`model.safetensors`, `model.safetensors.index.json`,
`pytorch_model.bin`, `diffusion_pytorch_model.safetensors`, …) — `model_downloader.py:362-386`.
The current tree satisfies this. For a no-LM run you still need `acestep-5Hz-lm-1.7B` present for
the precheck to pass, because the precheck is LM-agnostic; it costs nothing (no load).

**Watch out — the loader writes into the checkpoint dir.** `_sync_model_code_if_needed`
(`init_service_downloads.py:76-87`) hashes `acestep/models/turbo/*.py` against
`<root>/acestep-v15-turbo/*.py` and **copies the repo versions over the downloaded ones** on any
mismatch (`model_downloader.py:96-136`). So the weights dir must be writable on first init, and
expect `modeling_acestep_v15_turbo.py` / `configuration_acestep_v15.py` timestamps to change.
That is desirable (it keeps the HF `trust_remote_code` copies in lockstep with the pinned repo) —
just don't be surprised by it, and don't hash-compare the tree afterwards.

`prefer_source="huggingface"|"modelscope"` only affects the downloader and is irrelevant here.

---

## 3. Disabling the LM entirely (no-LM run)

Two equivalent ways; use both for safety.

1. **Don't create / don't pass an LLM handler.**
   `use_lm = (params.thinking or need_lm_for_cot) and llm_handler is not None and
   llm_handler.llm_initialized and not skip_lm` — `inference.py:655`.
   Passing `llm_handler=None` short-circuits to `use_lm = False` and the whole Phase-1/Phase-2 block
   is skipped (`inference.py:637-843`). `init_llm false` in the Gradio/CLI/API layer means exactly
   this ("don't construct LLMHandler"); it is **not** a parameter of `initialize_service`
   (`acestep_v15_pipeline.py:287,523-545`; `acestep/api/startup_llm_init.py:29-47`;
   `acestep/api/llm_readiness.py:49-50`, env `ACESTEP_INIT_LLM=0`).

2. **Turn every CoT switch off**, because `GenerationParams` defaults them on
   (`inference.py:203-211`: `thinking=True`, `lm_temperature=0.85`, `lm_cfg_scale=2.0`,
   `use_cot_metas=True`, `use_cot_caption=True`, `use_cot_language=True`):

   ```python
   params = GenerationParams(..., thinking=False,
                             use_cot_metas=False, use_cot_caption=False,
                             use_cot_language=False)
   ```

With the LM off the DiT conditions on: caption+lyrics+metadata (as text → condition encoder) and
`silence_latent` as the source/context latent. **Duration must be explicit** (`params.duration`),
otherwise the DiT default path falls back to a 60 s-class target. Details in code-map Q4.3.

If the LM is wanted but `params.audio_codes` is supplied by the caller, the LM runs Phase 1 only
(`infer_type="dit"`, `inference.py:705`) — useful for measuring LM metadata cost without codes.

---

## 4. `GenerationParams` / `GenerationConfig` fields that matter

`GenerationParams` — `acestep/inference.py:56-233`; defaults verified at the cited lines.

| Field | Default | Line | Notes |
|---|---|---|---|
| `task_type` | `"text2music"` | 111 | turbo set: text2music/repaint/cover/cover-nofsq (`constants.py:82`) |
| `instruction` | `"Fill the audio semantic mask based on the given conditions:"` | 112 | |
| `caption` | `""` | 122 | DiT text prompt |
| `lyrics` | `""` | 124 | use `"[Instrumental]"` |
| `instrumental` | `False` | 125 | |
| `vocal_language` | `"unknown"` | 128 | |
| `bpm` / `keyscale` / `timesignature` | `None` / `""` / `""` | 129-131 | go into the `# Metas` block |
| `duration` | `-1.0` | 132 | **seconds; set this explicitly.** <0/None ⇒ auto |
| `latent_shift` / `latent_rescale` | `0.0` / `1.0` | 141-142 | latent post-process before VAE |
| `inference_steps` | `8` | 145 | |
| `seed` | `-1` | 146 | only used if `config.seeds` unset |
| `guidance_scale` | `7.0` | 147 | forced to 1.0 on turbo (`generate_music.py:309-318`) |
| `use_adg` | `False` | 148 | base models only; no-op on turbo |
| `cfg_interval_start/end` | `0.0` / `1.0` | 149-150 | unused on turbo |
| `shift` | `1.0` | 151 | discretised to 1/2/3 |
| `infer_method` | `"ode"` | 152 | `"sde"` adds unseeded noise |
| `sampler_mode` | `"euler"` | 153 | `"heun"` doubles model evals |
| `velocity_norm_threshold` / `velocity_ema_factor` | `0.0` | 154-155 | optional stabilisers |
| `dcw_enabled` | `None` | 159 | `None` ⇒ turbo=True (`generate_music.py:181-189`) |
| `dcw_mode/scaler/high_scaler/wavelet` | `"double"/0.05/0.02/"haar"` | 164-167 | |
| `timesteps` | `None` | 170 | **overrides `inference_steps` and `shift`** |
| `audio_cover_strength` | `1.0` | 199 | <1 ⇒ 2nd `prepare_condition` pass |
| `cover_noise_strength` | `0.0` | 200 | |
| `thinking` | **`True`** | 203 | |
| `lm_temperature` | `0.85` | 204 | |
| `lm_cfg_scale` | `2.0` | 205 | forced 1.0 during CoT phase |
| `lm_top_k` / `lm_top_p` | `0` / `0.9` | 206-207 | |
| `lm_negative_prompt` | `"NO USER INPUT"` | 208 | dropped unless "meaningful" (`llm_inference.py:305-307`) |
| `use_cot_metas/caption/language` | `True`/`True`/`True` | 209-211 | |
| `use_constrained_decoding` | `True` | 213 | keep on; the FSM is what makes the format valid |

`GenerationConfig` — `inference.py:236-268`:

| Field | Default | Line | Notes |
|---|---|---|---|
| `batch_size` | `2` | 254 | **set 1 for measurements** |
| `allow_lm_batch` | `False` | 255 | |
| `use_random_seed` | **`True`** | 256 | **must be False for a fixed seed** |
| `seeds` | `None` | 257 | list; takes precedence over `params.seed` |
| `lm_batch_chunk_size` | `8` | 258 | LM chunking over batch |
| `constrained_decoding_debug` | `False` | 259 | |
| `audio_format` | `"flac"` | 260 | `"wav"` / `"wav32"` / `"mp3"` / `"opus"` / `"aac"` |
| `mp3_bitrate` / `mp3_sample_rate` | `"128k"` / `48000` | 261-262 | |

### Output shape of `GenerationResult`

`GenerationResult` = `inference.py:278-300`. `result.audios[i]` is a dict built at
`inference.py:1059-1066`:

```python
{"path": <file or "">, "tensor": <torch.FloatTensor [2, samples] CPU float32>,
 "key": <uuid hash of params>, "sample_rate": 48000, "params": {...}}
```

`result.extra_outputs["time_costs"]` holds the per-stage timings, prefixed in
`inference.py:1072-1094`:

- `dit_encoder_time_cost`, `dit_diffusion_time_cost`, `dit_diffusion_per_step_time_cost`,
  `dit_vae_decode_time_cost`, `dit_total_time_cost`, `dit_offload_time_cost`
- `lm_phase1_time`, `lm_phase2_time`, `lm_total_time`
- `pipeline_total_time` = `lm_total_time + dit_total_time_cost`

That is exactly what the measurement harness should log (per duration, with `batch_size=1`).
`result.extra_outputs["lm_metadata"]` holds the LM's CoT dict (or `None`).

---

## 5. Environment variables (only the ones that actually affect a library run)

Verified readers in the pinned tree:

| Env var | Read at | Effect |
|---|---|---|
| `ACESTEP_CHECKPOINTS_DIR` | `init_service_orchestrator.py:119-122`; `init_service_catalog.py:20-21` | **checkpoint root override — use this** |
| `ACESTEP_PROJECT_ROOT` | `model_downloader.py:329-335` | base for `<root>/checkpoints` when the above is unset; defaults to **cwd** |
| `ACESTEP_VAE_CHECKPOINT` | `init_service_orchestrator.py:127-131` | VAE variant id (`official`, `scragvae`) or absolute dir |
| `ACESTEP_MLX_VAE_FP16` | `mlx_vae_init.py:48-52` | `1` ⇒ MLX VAE weights float16 (halves VAE RAM) |
| `ACESTEP_MLX_VAE_CHUNK` | `gpu_config.py:779-800` | MLX VAE decode chunk in latent frames (floor 192) |
| `ACESTEP_VAE_ON_CPU` | `generate_music_decode.py:145` | force torch VAE decode on CPU |
| `ACESTEP_VAE_DECODE_CHUNK_SIZE` | `memory_utils.py:81-90` | torch VAE decode chunk |
| `ACESTEP_MPS_DECODE_OFFLOAD` | `memory_utils.py:118-124` | keep decoded wav chunks on CPU |
| `ACESTEP_DISABLE_TQDM` | `handler.py:138`; `llm_inference.py:66` | no progress bars (also auto-off when stderr isn't a tty) |
| `ACESTEP_DEBUG_STATS` | `handler.py:139` | extra timing logs |
| `ACESTEP_SKIP_VRAM_PREFLIGHT` | `generate_music.py:400-412` | skip the CUDA-era VRAM pre-flight (which is meaningless on MPS) |
| `ACESTEP_SAVE_MEMORY` | `gpu_config.py:27`, apply at `:1572` | memory-conservative tier overrides |
| `ACESTEP_ROCM_DTYPE` | `init_service_orchestrator.py:36-45` | ROCm only, irrelevant |
| `ACESTEP_TMPDIR` | `api/lifespan_runtime.py:61-73` | API server only |
| `ACESTEP_DEVICE`, `ACESTEP_LM_BACKEND`, `ACESTEP_LM_DEVICE`, `ACESTEP_LM_MODEL_PATH`, `ACESTEP_INIT_LLM`, `ACESTEP_OFFLOAD_TO_CPU`, `ACESTEP_OFFLOAD_DIT_TO_CPU`, `ACESTEP_USE_FLASH_ATTENTION`, `ACESTEP_COMPILE_MODEL`, `ACESTEP_CONFIG_PATH`, `ACESTEP_API_*` | `acestep/api/*`, `acestep/cli_args.py`, `acestep/acestep_v15_pipeline.py` | **only the CLI / API-server / Gradio entry points read these.** A direct library import ignores them — pass arguments instead |

`ACESTEP_DTYPE` **does not exist** in the code: it appears only inside a log string at
`generate_music_decode.py:79`. If the issue text (or upstream docs) references it, that is wrong.
Likewise `ACESTEP_MLX_VAE_CHUNK`/`ACESTEP_MLX_VAE_FP16` are real but **there is no
`ACESTEP_MLX_DIT_FP16`** — the MLX DiT is always float32 (`dit_convert.py:39`
`.detach().cpu().float().numpy()`).

---

## 6. Gotchas that will bite the harness

1. **Fixed seed needs two switches, not one.**
   `GenerationConfig.use_random_seed=False` *and* `seeds=[N]`. With the default
   `use_random_seed=True`, `prepare_seeds` overwrites your seed with `random.randint(0, 2**32-1)`
   (`task_utils.py:26-28`; called from `inference.py:576-600`, `:903-905`). `params.seed=N` alone is
   ignored when `config.seeds` is unset only in the narrow sense of being a fallback
   (`inference.py:597-606`).
2. **With the LM on, the run is not reproducible at `batch_size=1`.** The single-item LM decoders
   have no seeding at all (`llm_inference.py:985`, `:3397`, `:3720`); seeds are only applied on
   batch paths (`:1186-1191`, `:3298`, `:4043-4044`) and `generate_with_stop_condition` only builds
   a seed list when `batch_size > 1` (`:1355-1362`, docstring `:1318-1320`
   "Only used when batch_size > 1. TODO: not used yet"). For an A/B measurement, either run the
   no-LM path or accept per-run variation in the LM output and average.
   (`batch_size>1` + MLX batch-native codes path *is* seeded per item per step — `:3296-3298`.)
3. **`infer_method="sde"` is never reproducible** — its renoise draws
   `mx.random.normal(xt.shape)` / `torch.randn_like` outside any generator (`dit_generate.py:482`,
   `modeling_acestep_v15_turbo.py:2150-2156`). Use `"ode"`.
4. **Memory: the repo keeps two copies of the DiT.** This is the biggest risk on an 18 GiB Mac:
   - `initialize_service` sets `self.dtype = torch.float32` on MPS (`init_service_orchestrator.py:101`
     — bfloat16 only for CUDA/XPU). The 4.79 GB bf16 checkpoint therefore becomes ~9.6 GB float32.
   - The MLX DiT is a **second, independent float32 copy** (`dit_convert.py:12-71` converts
     `pytorch_model.decoder` via `.float().numpy()`), and `self.model` is never released —
     `prepare_condition` still needs it. So `use_mlx_dit=True` (the default) holds ~19 GB of DiT
     alone before the 2.4 GB float32 Qwen3 encoder, ~0.3 GB VAE, and activations.
   - With `use_mlx_dit=False` you get one float32 DiT (~9.6 GB) + encoder (~2.4 GB) ≈ 12 GB —
     right at the stated cap. `offload_to_cpu=True, offload_dit_to_cpu=True` moves the torch DiT to
     CPU between calls (`init_service_offload_context.py:13-80`) but `_load_model_context("model")`
     pulls all 9.6 GB back to MPS around **every** `prepare_condition`, so the peak is unchanged.
   - Practical consequence: **measure the pure-PyTorch-MPS path (`use_mlx_dit=False`) and the
     no-LM path first**, and treat "MLX DiT + torch DiT both resident" as the thing the *Swift port
     is supposed to remove*. These are arithmetic from the code, not measurements — confirm with
     RSS/`mx.get_active_memory()` before trusting them.
   - On an 18 GiB machine `get_gpu_memory_gb()` reports ~`recommended_max_memory()` (≈75 % of RAM;
     `memory_utils.py:67-79`), which lands in tier6a (16-20 GB): MLX VAE chunk = 512
     (`gpu_config.py:779-806`), `max_duration_with_lm=480`, `max_duration_without_lm=600`
     (`gpu_config.py:409-427`).
5. **`offload_to_cpu` semantics are tri-state.** `offload_to_cpu=False` ⇒ everything on MPS.
   `offload_to_cpu=True, offload_dit_to_cpu=False` ⇒ DiT is *persistently* on MPS once first used
   (it moves it lazily, `init_service_offload_context.py:17-33`). `offload_to_cpu=True,
   offload_dit_to_cpu=True` ⇒ DiT goes back to CPU after each use. Choose deliberately.
6. **Batch default is 2.** `GenerationConfig.batch_size=2` and `AceStepHandler.batch_size=2`
   (`handler.py:127`) — always pass `batch_size=1` for per-clip timings, otherwise you measure two.
7. **VAE dtype differs from DiT dtype.** On MPS the torch VAE runs **float16**
   (`memory_utils.py:199-216` `_get_vae_dtype`), DiT float32, LM float32. Don't assume one dtype.
8. **First `initialize_service` call may edit the weight files** — see §2
   (`_sync_model_code_if_needed`).
9. **`silence_latent.pt` is a `torch.save` zip, not safetensors.** Any Swift/MLX port needs a
   one-time conversion to raw float32 `[15000, 64]` (3.84 MB) or `.safetensors`.
10. **`to_py`/`tensor` is CPU float32 already** (`inference.py:1074-1080`), so saving WAV is just
    `a["tensor"].numpy().T` at `a["sample_rate"]`. Waveform peak is normalised to ≤1.0 by the
    decoder (`generate_music_decode.py:199-203`) and again by `normalize_audio` if
    `enable_normalization` (default True) and `normalization_db<=0` (default -1.0)
    (`inference.py:999-1011`).
11. **`GenerationResult.success` can be True with an empty `audios` list** if `save_dir` is None
    (files are only written when `save_dir` is given, `inference.py:1042-1045`) — the in-memory
    tensor is still populated, so read `a["tensor"]` for measurements.

---

## 7. Suggested measurement matrix (what the code supports cheaply)

| Axis | Values that are cheap | Why |
|---|---|---|
| LM on/off | `llm_handler=None` vs 0.6B vs 1.7B | isolates LM cost; 0.6B is in the same repo |
| DiT backend | `use_mlx_dit=False` vs `True` | pure-torch baseline vs MLX (watch the ×2 RAM) |
| duration | 10, 30, 60, 120 s | quadratic attention kicks in above ~120 s |
| steps | 8 (default) | anything else changes the schedule, not just cost |
| pipeline stages | `dit_encoder_time_cost` / `dit_diffusion_time_cost` / `dit_vae_decode_time_cost` | already emitted; no instrumentation needed |

Do **not** compare across `shift` values: `shift` selects a different `t` schedule, so the outputs are
not comparable audio, only comparable cost.
