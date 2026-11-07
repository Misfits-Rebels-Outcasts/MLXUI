# ACE-Step 1.5 — code map for the MLX Swift port (MC5)

Upstream: `Misfits-Rebels-Outcasts/ACE-Step-1.5` pinned `ca1e85fe9430179831e6bc6be790c332190a3866`
(verified `git rev-parse HEAD`). All `path:line` are relative to the repo root at that commit.
Official weights: HF `ACE-Step/Ace-Step1.5` @ `19671f406d603126926c1b7e2adc169acbcade22`
(local mirror `/private/tmp/mc5/Ace-Step1.5`).

Two facts up front, because they change the shape of the port:

1. **Upstream already contains a partial native-MLX port on macOS.** The DiT *decoder*, the VAE
   (encode + decode) and the 5 Hz LM all have MLX implementations at this commit, and on
   Apple Silicon they are the *default* (`acestep/gpu_config.py:866` sets
   `recommended_backend="mlx"`; `init_service_setup.py:114-146` enables the MLX DiT/VAE).
   The **condition encoder stack (text projector, lyric encoder, timbre encoder, attention
   pooler, FSQ tokenizer/detokenizer) has no MLX implementation** and runs in PyTorch on MPS;
   its outputs are handed to MLX as numpy arrays.
2. So the remaining Swift work is not "port everything" — it is "port what is still torch"
   (text encoder + condition encoder) plus re-implement the handler orchestration. The PyTorch
   graph is the reference for those.

---

## Q1. Component map: prompt text → WAV

### 1.1 Text/prompt tokenization (CPU, HF `tokenizers`)

| What | Where | Runtime today |
|---|---|---|
| DiT prompt template | `acestep/constants.py:160-168` `SFT_GEN_PROMPT` | pure string fmt |
| caption/lyric tokenize | `acestep/core/generation/handler/conditioning_text.py:127-135` (DiT text prompt, `max_length=256`), `:138-146` (lyrics, `max_length=2048`); SFT-lego branch re-tokenizes at `:171-179` | HF tokenizer (Rust, CPU) |
| LM prompt build | `acestep/llm_inference.py:1668-1800` `build_formatted_prompt*` | CPU strings |

The DiT prompt is exactly `"# Instruction\n{instruction}\n\n# Caption\n{caption}\n\n# Metas\n{metas}<|endoftext|>\n"`
(`constants.py:160-168`), where `{metas}` is the YAML-ish `bpm:`/`duration:`/`keyscale:`/`language:`/
`timesignature:` block (`metadata_utils.py`). This is the whole text conditioning surface for the
DiT — there is no "style prompt" beyond it.

### 1.2 Text encoder — Qwen3-Embedding-0.6B

- Loaded from `<checkpoints>/Qwen3-Embedding-0.6B`:
  `acestep/core/generation/handler/init_service_loader_components.py:68-102`
  (`AutoTokenizer.from_pretrained` + `AutoModel.from_pretrained`, moved to device, `eval()`).
- Inference: `conditioning_embed.py:78-81` → `self.text_encoder(input_ids=...).last_hidden_state`.
  Hidden size **1024** (`text_hidden_dim`, config), frozen embedding model, not causal.
- **Runtime today: PyTorch on MPS, float32.** `initialize_service` sets `self.dtype = torch.float32`
  for any non-CUDA/XPU device (`init_service_orchestrator.py:101` region: `else: self.dtype = torch.bfloat16 if resolved_device == "xpu" else torch.float32`).
  There is no MLX text-encoder path in this repo.

### 1.3 (Optional) 5 Hz planner LM — `acestep-5Hz-lm-1.7B`

- Handler: `acestep/llm_inference.py` (4253 lines), class `LLMHandler:48`,
  `initialize(checkpoint_dir, lm_model_path, backend, device, offload_to_cpu, dtype)` at `:499`.
- **Runtime today: native MLX via `mlx-lm`** on Apple Silicon. Backend choice:
  `gpu_config.py:843-867` → `recommended_backend="mlx"`, `lm_backend_restriction="pt_mlx_only"`;
  MLX load at `llm_inference.py:2933-3018` (`mlx_lm.utils.load`, with a weight-key remap fallback
  because the checkpoint omits the `model.` prefix, `:2955-2996`).
- PyTorch/MPS path exists as fallback: `llm_inference.py:401-418` `_load_pytorch_model`.
- Full mechanics in Q4 below.

### 1.4 Condition encoder — `AceStepConditionEncoder` + audio tokenizer/detokenizer

This is the stack that is **still PyTorch** and is the hardest part of the port.

| Sub-module | Where | Shape / role |
|---|---|---|
| `AceStepConditionEncoder` | `acestep/models/turbo/modeling_acestep_v15_turbo.py:1513-1558` | orchestrates the three encoders |
| `text_projector` | `:1527` `nn.Linear(text_hidden_dim=1024 → hidden_size=2048, bias=False)` | caption emb → 2048 |
| `AceStepLyricEncoder` | `:580-735` | 8 layers (`num_lyric_encoder_hidden_layers=8`), bidirectional, embeds lyric token ids |
| `AceStepTimbreEncoder` | `:1000-1182` | 4 layers (`num_timbre_encoder_hidden_layers=4`), 64-dim acoustic latents in (`timbre_hidden_dim=64`), outputs one CLS-ish token per reference clip |
| `pack_sequences` | `:141-173` | concatenates lyric‖timbre‖text into one cross-attn sequence |
| `AttentionPooler` | `:737-863` | pools 25 Hz latents by `pool_window_size=5` → 5 Hz |
| `AceStepAudioTokenizer` (+`ResidualFSQ`) | `:1184-1226` | `audio_acoustic_proj`(64→2048) → `AttentionPooler` → `ResidualFSQ(dim=2048, levels=[8,8,8,5,5,5], num_quantizers=1)`; from `vector_quantize_pytorch`, imported `:42` |
| `AudioTokenDetokenizer` | `:865-998` | 2 `AceStepEncoderLayer`s; expands each 5 Hz token into `pool_window_size=5` patches and projects `hidden→audio_acoustic_hidden_dim=64` → back to 25 Hz |

`ResidualFSQ` levels `[8,8,8,5,5,5]` ⇒ codebook size `8*8*8*5*5*5 = 64000`, matching
`MAX_AUDIO_CODE = 63999` (`acestep/constrained_logits_processor.py:47`). One code = 5 latent
frames = 200 ms ⇒ **5 codes per second** (stated at `llm_inference.py:1310-1311`).

**Where `prepare_condition` happens:** `acestep/models/turbo/modeling_acestep_v15_turbo.py:1654-1699`,
and it is *called by the handler*, not by the sampler:
`acestep/core/generation/handler/service_generate_execute.py:176-194`
(`self.model.prepare_condition(...)`, inside `torch.inference_mode()` and
`self._load_model_context("model")`). It runs **even when the DiT sampler is MLX**. A second,
non-cover copy is computed at `service_generate_execute.py:196-229` when
`audio_cover_strength < 1.0`.

It returns three tensors:
- `encoder_hidden_states` `[B, S_enc, 2048]` + `encoder_attention_mask` `[B, S_enc]` (cross-attn K/V input)
- `context_latents` `[B, T, 128]` = `cat([src_latents (64 ch), chunk_masks broadcast to 64 ch], -1)`
  (`modeling_acestep_v15_turbo.py:1698`; broadcast done in `conditioning_embed.py:100-101`)

### 1.5 DiT — `AceStepDiTModel` (24 layers) — **runs in MLX**

- Torch reference: `modeling_acestep_v15_turbo.py:1243-1509` (`forward` at `:1307`).
- MLX: `acestep/models/mlx/dit_model.py:421+` (`MLXDiTDecoder`), converted at init from
  **only** `pytorch_model.decoder` weights — `acestep/models/mlx/dit_convert.py:12-71`
  (`convert_decoder_weights`); `init_service_setup.py:123-136` → `mlx_dit_init.py:8-40`.
- Config: 24 layers alternating `sliding_attention`/`full_attention` (12 full, sliding window 128),
  16 heads / 8 KV heads, head_dim 128, hidden 2048, intermediate 6144, `patch_size=2`,
  `in_channels=192`, `audio_acoustic_hidden_dim=64`, RoPE theta 1e6,
  `max_position_embeddings=32768` (`acestep-v15-turbo/config.json`).
- Input construction (torch `:1351-1364`, mirrored in MLX):
  `x = cat([context_latents(128), hidden_states/noise(64)], -1)` → `[B,T,192]` (line 1351) →
  pad T to a multiple of `patch_size=2` (`:1355-1360`) → `Conv1d(192→2048, k=2, s=2)` (`:1363`) →
  `[B, T/2, 2048]`. So **DiT sequence length = latent_frames / 2**.
- Cross-attention to `encoder_hidden_states` (after `condition_embedder` `Linear(2048→2048)`, `:1364`);
  `timestep` and `timestep_r` both fed, and their embeddings summed (`:1344-1349`); output
  `ConvTranspose1d(2048→64)` then cropped to the original length (`:1495-1506`).
- Per-step cross-attn KV is cached (`EncoderDecoderCache`, torch `:2107`, `:2133-2140`;
  MLX `MLXCrossAttentionCache` in `dit_generate.py:345`).

### 1.6 Sampler — dual implementation

- **MLX sampler:** `acestep/models/mlx/dit_generate.py:168-535` `mlx_generate_diffusion`
  (schedule at `:49-97`, loop at `:422-527`).
- **Torch sampler (reference/fallback):** `modeling_acestep_v15_turbo.py:1852-2226` `generate_audio`.
- Entry point / dispatch: `service_generate_execute.py:197` (MLX if
  `self.use_mlx_dit and self.mlx_decoder is not None`), else torch `self.model.generate_audio`
  at `:277`. An MLX exception silently falls back to torch (`:272-274`).

### 1.7 VAE decode — **runs in MLX**

- MLX model: `acestep/models/mlx/vae_model.py:237-330` `MLXAutoEncoderOobleck`
  (decoder `:190-232`), converted from `self.vae` weights in `vae_convert.py:36-200`
  (weight-norm fusion + Conv axis swaps). Init: `mlx_vae_init.py:14-100`.
- MLX decode wrapper (tiling): `acestep/core/generation/handler/mlx_vae_decode_native.py:27-140`.
- Torch fallback/tiling: `vae_decode.py:16-86`, `vae_decode_chunks.py:13-160`.
- Call site: `generate_music_decode.py:107-240` `_decode_generate_music_pred_latents`; latents are
  `[B,T,64]` → `.transpose(1,2)` → `[B,64,T]` before decode (`:132`); MLX branch selected at `:141`,
  tiled decode dispatch at `:173`.

### 1.8 What actually crosses torch → numpy → MLX

Only two boundaries on the MLX path (plus the LM, which is fully MLX):

1. **DiT**: `acestep/core/generation/handler/diffusion.py:121-132` converts
   `encoder_hidden_states`, `context_latents` (+ non-cover pair, repaint mask / clean src) to
   numpy; `:155-188` calls `mlx_generate_diffusion(...)`; `:190-191` converts the result back with
   `torch.from_numpy(target_np)`. Inside MLX the result leaves as numpy (`dit_generate.py:529`).
2. **VAE decode**: `mlx_vae_decode_native.py:42-46` torch `[B,64,T]` → numpy → transpose to NLC →
   `mx.array`; `:59-72` `np.array(decoded)` → `torch.from_numpy`. VAE encode is symmetric:
   `mlx_vae_encode_native.py:42-43`, `:75`, `:85-87`.
3. **Silence latent never crosses**: it stays a torch tensor (`self.silence_latent`), used only by
   torch code (`prepare_condition`, masks, timbre padding).

Design note for the port: a Swift implementation would collapse all three boundaries into one
MLX graph; the numpy round-trip here exists only because upstream keeps the condition encoder
in torch.

### 1.9 Handler files involved (`acestep/core/generation/handler/`)

Composition root: `acestep/handler.py:62-100` (`AceStepHandler`, 36 mixins); mixin exports at
`acestep/core/generation/handler/__init__.py:3-40`.

| File | Responsibility |
|---|---|
| `service_generate.py:21` | `service_generate` — top-level latent generation orchestration |
| `service_generate_request.py` | input normalization (captions/lyrics/keys/seeds/steps) |
| `batch_prep.py:63-120` | `prepare_batch_data`, `create_target_wavs` (duration→silence wav), `_encode_audio_to_latents` |
| `conditioning_batch.py:25-155` | `_prepare_batch` — assembles the whole conditioning dict |
| `conditioning_text.py:20-120` | caption/lyric tokenization, `_prepare_precomputed_lm_hints` |
| `conditioning_embed.py:78-133` | text/lyric embedding inference, `preprocess_batch` |
| `conditioning_masks.py:14-107` | repaint chunk masks, `src_latents`, `is_covers` |
| `conditioning_target.py:18-144` | target latents from duration, silence-latent slicing/tiling |
| `diffusion.py:18-196` | `_mlx_run_diffusion` — the torch↔numpy↔MLX bridge |
| `service_generate_execute.py:146-284` | **`prepare_condition` call + MLX/torch sampler dispatch** |
| `service_generate_outputs.py` | attaches payload to the result dict |
| `generate_music.py:195-540` | `generate_music` — public handler API |
| `generate_music_request.py:83-300` | runtime params, seeds, reference/src audio, service inputs |
| `generate_music_execute.py` | service_run → decode handoff |
| `generate_music_decode.py:100-240` | VAE decode + peak normalize |
| `generate_music_payload.py` | success payload / time costs |
| `vae_decode.py`, `vae_decode_chunks.py`, `vae_encode.py`, `vae_encode_chunks.py` | PyTorch VAE tiling |
| `mlx_vae_init.py`, `mlx_vae_decode_native.py`, `mlx_vae_encode_native.py` | MLX VAE |
| `mlx_dit_init.py` | MLX DiT init |
| `init_service_orchestrator.py:48-215` | `initialize_service` (public) |
| `init_service_setup.py:14-146` | device resolution, MLX backend enable |
| `init_service_downloads.py:16-87` | checkpoint presence / auto-download gate |
| `init_service_loader.py:150-217` | load DiT, silence_latent |
| `init_service_loader_components.py:19-102` | load VAE, text encoder |
| `init_service_memory*.py`, `init_service_offload_context.py`, `memory_utils.py` | memory/offload |
| `audio_codes.py:18-120` | `<|audio_code_N|>` parse, `_decode_audio_codes_to_latents`, `convert_src_audio_to_codes` |
| `task_utils.py:19-125` | `prepare_seeds`, `create_target_wavs`, instruction selection |
| `io_audio.py`, `metadata_utils.py`, `prompt_utils.py`, `padding_utils.py`, `progress.py` | I/O, metadata, prompts, padding, progress |
| `lyric_score.py`, `lyric_timestamp.py`, `lyric_alignment_common.py` | lyric alignment (optional) |
| `lora_manager.py` | LoRA adapters |
| `repaint_step_injection.py`, `repaint_waveform_splice.py` | repaint helpers |
| `service_generate_flow_edit.py`, `service_generate_flow_edit_source.py` | flow-edit overlay (#1156) |
| `training_preset.py` | training presets (not inference) |

---

## Q4-mechanism. What the 5 Hz LM produces and how it reaches the DiT

### 4.1 The LM is a text model. It emits *no* latents and *no* audio.

`acestep-5Hz-lm-1.7B` is a Qwen3-architecture causal LM (config `architectures: Qwen3ForCausalLM`,
`vocab_size: 64003`) loaded through `mlx-lm` (`llm_inference.py:2933-2996`). All of its output is
**text**. Two phases, driven by `generate_with_stop_condition`
(`llm_inference.py:1282-1560`, phases split at `:1478`):

**Phase 1 — CoT metadata** (`llm_inference.py:1367-1435`). A single sample for the batch. The
constrained FSM forces the output to be exactly (`constrained_logits_processor.py:56-78`,
class docstring `:81-100`):

```
<think>
bpm: 30..300
caption: <text, no code blocks>
duration: 10..max_dur
keyscale: <A-G[#/b] major|minor>
language: <VALID_LANGUAGES>
timesignature: 2|3|4|6
</think>
```

Enforced token-by-token by `MetadataConstrainedLogitsProcessor` (FSM states
`constrained_logits_processor.py:53-78`; numeric prefix trees
`:204-230`; audio-code token mask `:548-572`). So the LM's "musical understanding" is emitted as
**YAML-ish text metadata**, not embeddings. `parse_lm_output`
(`llm_inference.py:2792-2880`) turns it into a dict, and
`_update_metadata_from_lm` (`inference.py:343-430`) only fills fields the *user* left unset.

**Phase 2 — audio semantic codes** (`llm_inference.py:1478-1560`). Still text: the model emits a
run of special tokens `<|audio_code_N|>`, `N ∈ [0, 63999]`, nothing else
(token mask built at `constrained_logits_processor.py:529-572`; `MAX_AUDIO_CODE = 63999` at `:47`).
The count is **forced**: `target_codes = int(duration * 5)` and EOS is blocked until that many codes
(`constrained_logits_processor.py:1616-1628`; `set_target_duration` `:1285-1300`;
`_compute_max_new_tokens` `llm_inference.py:238-303`). So **5 codes = 1 second**, i.e. a 5 Hz
semantic token stream. Example prompt/format: `parse_lm_output` docstring `llm_inference.py:2794-2806`.

### 4.2 How those codes are fed in: decode→25 Hz latent hint, not a token stream

`<|audio_code_N|>` is a *text* serialization of an FSQ index. The DiT never sees text. The path is:

1. LM returns `audio_codes` as a string of `<|audio_code_N|>` tokens
   (`inference.py:790-800`; result assembled `:787-800`).
2. String → int list: `acestep/core/generation/handler/audio_codes.py:27-48`
   `_parse_audio_code_string` (clamps to 63999).
3. Int indices → FSQ codebook vectors → 25 Hz latents:
   `audio_codes.py:50-72` `_decode_audio_codes_to_latents`:
   `quantizer.get_output_from_indices(indices)` then `detokenizer(quantized)`; the detokenizer
   (`modeling_acestep_v15_turbo.py:865-998`) expands each 5 Hz token into `pool_window_size=5`
   patches → `[1, 5*N, 64]`, i.e. **5× denser than the codes** (25 Hz, 64 ch).
4. Pad/truncate to the requested latent length: `conditioning_text.py:20-55`
   `_prepare_precomputed_lm_hints`; short hints are padded with `silence_latent`, giving
   `precomputed_lm_hints_25Hz: [B, T_latent, 64]`.
5. In `prepare_condition` the hints **replace `src_latents`** when `is_covers` is true:
   `modeling_acestep_v15_turbo.py:1685-1697`
   ```
   if precomputed_lm_hints_25Hz is not None:
       lm_hints_25Hz = precomputed_lm_hints_25Hz[:, :src_latents.shape[1], :]
   ...
   src_latents = torch.where(is_covers.unsqueeze(-1).unsqueeze(-1) > 0, lm_hints_25Hz, src_latents)
   context_latents = torch.cat([src_latents, chunk_masks.to(dtype)], dim=-1)   # :1698
   ```
   Note `is_covers` includes "has audio codes" for *any* task:
   `conditioning_masks.py:69-71` (`is_cover = (task_type == "cover") or has_code_hint`), and
   `task_utils.py:101-116` `determine_task_type` also flips `is_cover_task` true when codes exist.
6. Also the *target* latents for the batch are the same decoded codes; `target_wavs` is zeroed
   (`conditioning_target.py:76-85`). So with LM on, the "source audio" the DiT conditions on is
   the LM's semantic-code latent, and the DiT's job is to render it into 25 Hz acoustic latents.

**So: text + lyrics + metadata go to the condition encoder (cross-attention); the FSQ codes go to
the DiT as the initial/context latent `src_latents` (channel-concatenated with the chunk mask).**
They are two different conditioning channels, and this is the single most important thing for the
port: the LM's contribution is an *initial latent*, not a prompt prefix.

### 4.3 Is the LM required for text2music? **No.**

The gate is `inference.py:640-655`:

```python
need_lm_for_cot = params.use_cot_caption or params.use_cot_language or params.use_cot_metas
skip_lm = params.task_type in skip_lm_tasks or morph_on_text2music
use_lm = (params.thinking or need_lm_for_cot) and llm_handler is not None \
         and llm_handler.llm_initialized and not skip_lm
```

`skip_lm_tasks = DIRECT_CONDITIONING_TASKS = {cover, cover-nofsq, repaint, extract, complete, lego}`
(`inference.py:36-41`) — **`text2music` is not in it**.

LM-off happens when any of:
- `llm_handler is None` (just don't construct/pass one — this is what `init_llm false` means at the
  Gradio/CLI/API level; it is not a parameter of `initialize_service`), or
- `llm_handler.llm_initialized` is False (LM init failed / was never called), or
- `thinking=False and use_cot_metas=False and use_cot_caption=False and use_cot_language=False`.

caveat: **`GenerationParams` defaults are `thinking=True`, `use_cot_metas/caption/language=True`**
(`inference.py:203-211`), so the LM is ON by default. A no-LM run must set all four to False, or
pass `llm_handler=None`.

What the DiT gets in the LM-off path:
- `audio_code_string_to_use = params.audio_codes` = `""` → `audio_code_hints = None` →
  `is_covers = False` (or true only for real cover tasks).
- `_prepare_target_latents_and_wavs` sees a silent target wav → `target_latents` is the
  **silence_latent slice** (`conditioning_target.py:86-92`), so `src_latents` is silence
  (`conditioning_masks.py:95` appends `silence_latent_tiled`).
- Hence `context_latents = cat([silence_latent(T), chunk_mask(T)])` — a *constant* context.
- All information then flows through `encoder_hidden_states` (text projector + lyric encoder + timbre
  encoder): caption, lyrics, and the metadata block built from `params.bpm/keyscale/timesignature/
  duration/vocal_language` (`batch_prep.py:88-107`, title block from `constants.py:160-168`).
- `audio_duration` must be supplied explicitly (or infer_type uses 60 s default); nothing
  generates a duration for you. With LM on, duration may come from the CoT metadata
  (`inference.py:1415-1432`).

What changes on the LM-on path:
- `thinking=True` → `infer_type="llm_dit"` (`inference.py:705`) → Phase 1 + Phase 2 (codes).
  `infer_type="dit"` (only when `need_audio_codes` is False, i.e. the user supplied
  `params.audio_codes`) runs Phase 1 only and returns metadata.
- `use_cot_metas` → allow Phase 1 at all (`llm_inference.py:1367`: `if not has_all_metas and use_cot_metas`).
  If `has_all_metas(user_metadata)` (bpm+keyscale+timesignature+duration all set,
  `llm_inference.py:1245-1254`) the CoT phase is **skipped even when requested** and the
  user metadata is formatted into the CoT text instead (`llm_inference.py:1423-1429`).
- `use_cot_caption` / `use_cot_language` → set `skip_caption` / `skip_language` in the FSM
  (`llm_inference.py:1393-1394`), and afterwards override the DiT caption/language with the LM's
  (`inference.py:825-827`).
- `lm_cfg_scale` (default 2.0) → LM-side classifier-free guidance via a second unconditional forward
  pass and a negative prompt (`llm_inference.py:2618-2700` `_generate_with_cfg_custom`;
  unconditional prompt built at `:374-398`). **It is forced to 1.0 during Phase 1**
  (`llm_inference.py:1386-1391`: "CFG must not be applied during CoT"), so it only affects code
  generation.
- `lm_negative_prompt` (default `"NO USER INPUT"`) → only used when "meaningful"
  (`llm_inference.py:305-307`); it replaces the caption in the unconditional prompt.
- `lm_temperature` (0.85), `lm_top_k` (0), `lm_top_p` (0.9), `repetition_penalty` → sampling knobs.
- The LM also costs time and memory: it is a **separate 1.7B-parameter model**.

Practical consequence for v1: **you can ship text2music with no LM at all** (that is the "pure DiT"
path), and add the LM later as an optional quality/duration-consistency feature. The quality delta is
empirical — the code only tells us the mechanisms differ (silence context vs LM-code context).

### 4.4 Issue-text check

The upstream code does **not** feed LM tokens to the DiT as a prefix; it decodes them to 25 Hz
latents that replace `src_latents`. Any plan that says "condition the DiT on the LM's text" is
wrong. Also, `audio_codes` is not only a cover feature — any task with codes sets `is_covers`.

---

## Q5. Sampler

### 5.1 The turbo timestep schedule (three code paths, same numbers)

Constants in **both** implementations, byte-identical:
`models/turbo/modeling_acestep_v15_turbo.py:1914-1936` and `models/mlx/dit_generate.py:31-46`.

```python
VALID_SHIFTS = [1.0, 2.0, 3.0]
VALID_TIMESTEPS = [1.0, 0.9545454545454546, 0.9333333333333333, 0.9, 0.875,
                   0.8571428571428571, 0.8333333333333334, 0.7692307692307693, 0.75,
                   0.6666666666666666, 0.6428571428571429, 0.625, 0.5454545454545454,
                   0.5, 0.4, 0.375, 0.3, 0.25, 0.2222222222222222, 0.125]
SHIFT_TIMESTEPS = {
    1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    2.0: [1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693,
          0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222],
    3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75,
          0.6428571428571429, 0.5, 0.3],
}
```

Schedule resolution order (`dit_generate.py:49-97` = the cleaner copy; torch `:1937-1987`):

1. **Explicit `timesteps`** (list, trailing zeros stripped, >20 truncated to 20). Each value is
   snapped to the nearest entry of `VALID_TIMESTEPS` (`torch :1937-1964`; the MLX copy does **not**
   snap — a divergence worth knowing, `dit_generate.py:72-86`).
2. **`infer_steps` given** (`>0`): `raw = [1 - i/n for i in range(n)]`, `n = min(infer_steps, 20)`;
   then if `shift != 1.0`, `raw = [shift*t / (1 + (shift-1)*t) for t in raw]`
   (`torch :1966-1979`, `dit_generate.py:87-90`).
3. **Neither**: round `shift` to the nearest of `[1,2,3]` and use `SHIFT_TIMESTEPS[shift]`
   (`torch :1981-1987`, `dit_generate.py:91-96`).

Defaults and what a default turbo run actually does:
`GenerationParams.inference_steps = 8`, `shift = 1.0` (`inference.py:145,151`) → path 2 fires with
`n=8, shift=1.0` → `[1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]`, numerically identical to
`SHIFT_TIMESTEPS[1.0]`. For the `turbo-shift3` checkpoint family, `shift=3.0` gives the 8-step list
above. **The final `t=0` step is not in the list**: the last scheduled step is handled specially by
predicting x0 directly (`torch :2140-2147`, MLX `:452-456`).

The loop step (turbo, Euler ODE) is `xt = xt - vt * (t_curr - t_next)`; the final step is
`xt = xt - vt * t_curr` (i.e. `x0 = x_t - v*t`), matching flow-matching parameterisation
`x_t = t*noise + (1-t)*x_data` (`torch :2093-2213`; `get_x0_from_noise` `:1819-1820`).

### 5.2 `timestep_mu` / `timestep_sigma`

`timestep_mu=-0.4`, `timestep_sigma=1.0`, `data_proportion=0.5` are config fields
(`acestep-v15-turbo/config.json`) used **only for training** timestep sampling:
`sample_t_r(batch_size, device, dtype, data_proportion, timestep_mu, timestep_sigma, use_meanflow)` at
`modeling_acestep_v15_turbo.py:175-201`, called from `forward` (training) at `:1755`.
**They are not read by either inference sampler** — grep `timestep_mu` in the sampler paths: absent
from `generate_audio` and from `mlx_generate_diffusion`. They matter to the port only as proof that
inference uses the *fixed* schedule above, not a logit-normal draw.

### 5.3 CFG / APG / DCW on turbo

**CFG: off, structurally.** Turbo is CFG-distilled:
- The handler forces `guidance_scale = 1.0` for turbo before anything else:
  `generate_music.py:309-318` ("Turbo models bake guidance into the distillation process and do not
  use CFG"). `GenerationParams.guidance_scale` defaults to 7.0, so this override is load-bearing.
- `generate_audio` accepts the CFG kwargs but declares them no-ops and logs a warning when a
  non-default value arrives (`modeling_acestep_v15_turbo.py:1892-1912`).
- `use_adg` (Adaptive **Dual** Guidance) is likewise accepted-and-ignored on turbo. `use_adg` exists
  for the *base* models only (`inference.py:59` docstring: "only works for base model";
  `modeling_acestep_v15_base.py` has the ADG path). It reaches the turbo sampler only as an unused kwarg.

**APG / ADG: they belong to the base model, not turbo, and are unreachable here.**
Adaptive Projected Guidance (`apg_forward`) and Adaptive Dual Guidance (`adg_forward`) live in
`acestep/models/common/apg_guidance.py:33-106` and `:107-190`; they are imported and used only by the
**base** model (`models/base/modeling_acestep_v15_base.py:48-51`, applied at `:2054-2075` when
`do_cfg_guidance`, with `use_adg` selecting ADG over APG). There is no `apg` symbol in
`models/turbo/modeling_acestep_v15_turbo.py` (grep: 0 hits) — the turbo sampler never runs a twin
unconditional forward pass at all (`:1892-1912`).

The MLX loop nonetheless carries a copy: `_mlx_apg_forward` (`dit_generate.py:100-131`), reached via
`_apply_cfg` (`:369-377`) **only when `do_cfg` is true**, i.e.
`do_cfg = guidance_scale > 1.0 and null_condition_emb_np is not None` (`dit_generate.py:281`). On
turbo the handler passes `diffusion_guidance_scale=1.0` (`service_generate_execute.py:243`), so
`do_cfg` is False and the CFG/APG branch is dead code. **APG is a base/SFT-model feature**; do not
port it into a turbo-only v1.

**DCW: ON by default for turbo.** DCW = "Differential Correction in Wavelet domain" (CVPR 2026,
arXiv:2604.16044, referenced `generate_music.py:490-494`, `dit_generate.py:1-15`).
- Resolution: `dcw_enabled=None` → `_resolve_dcw_enabled` → `self.is_turbo_model()`
  (`generate_music.py:181-189`; `service_generate_execute.py:64-72`). So **turbo ⇒ DCW enabled**,
  base/SFT ⇒ disabled, unless the caller overrides.
- Defaults: `dcw_mode="double"`, `dcw_scaler=0.05` (low band), `dcw_high_scaler=0.02` (high band),
  `dcw_wavelet="haar"` (`inference.py:156-167`).
- Applied **after every non-final ODE step** and on the final x0, using
  `denoised = x_before - v*t_curr`; strength decays with `t_curr` so it is identity at t=0
  (torch `:2200-2210`, final-step `:2140-2147`; MLX `:492-513`; implementation
  `acestep/models/mlx/dcw_correction_mlx.py`, torch `acestep/models/common/dcw_correction.py`).
- MLX implements only `haar` natively; other wavelets bridge through PyTorch wavelets
  (`dit_generate.py:392-412`, and the `dcw_correction_mlx` module).
- Scope: DCW is an *optional post-step filter*, not part of the distilled model. It can be disabled
  (`dcw_enabled=False`) with no shape/logic change — a useful A/B lever for the port.

### 5.4 Noise seeding and reproducibility

**DiT noise (torch):** `prepare_noise` (`modeling_acestep_v15_turbo.py:1780-1817`) builds a
`torch.Generator(device=device).manual_seed(int(seed))` and draws
`torch.randn((B, T, 64), generator=generator)`; a per-item `List[int]` seed loops one generator per
sample, `-1`/`None` means fresh `torch.randn`. Shape `(B, T, context.shape[-1]//2)` = `(B, T, 64)`.
It is created once and reused for the whole loop (repaint/SDE reuse `noise`, `:2182-2210`).

**DiT noise (MLX):** `_draw_noise` (`dit_generate.py:295-308`) —
`mx.random.key(int(seed))` → `mx.random.normal((B,T,C), key=key)`; list seeds draw per-sample.
**Key-based, deterministic for a given seed.**

**Other RNG in the DiT loop:**
- SDE renoise is *not* seeded from the generator: `mx.random.normal(xt.shape)` (MLX `:470`) uses the
  global RNG; the torch SDE path calls `renoise(pred_clean, next_timestep)` with `noise=None` →
  `torch.randn_like` (torch `:2150-2156`, `renoise` `:1822-1829`). So **`infer_method="sde"` breaks
  fixed-seed reproducibility**; `infer_method="ode"` (the default) does not.
- VAE encode sampling (only cover/repaint reference audio) draws `mx.random.normal`
  (`vae_model.py:296-303`) / the torch equivalent — another reason cover/repaint is not bit-reproducible.

**End-to-end reproducibility at a fixed seed** requires all of:
1. `GenerationConfig.use_random_seed = False` and `GenerationConfig.seeds = [<seed>]`
   (`inference.py:298-330`); `use_random_seed` defaults to **True**, in which case
   `prepare_seeds` overwrites your seed with `random.randint(0, 2**32-1)` —
   `task_utils.py:26-28`, used by `inference.py:576-600` and `:903-905`.
   `params.seed` is only a *fallback* when `config.seeds` is unset (`inference.py:597-606`).
2. `infer_method="ode"` and `sampler_mode="euler"` (or heun — deterministic, but heun is another
   2×N forward passes), not `"sde"`.
3. **The LM must be off, or its sampling must be deterministic — which it is not on the single-item
   path.** This is the important, non-obvious finding:

   The LM's own token sampling is seeded only on *batch* code paths:
   `torch.manual_seed(seeds[i])` / `torch.mps.manual_seed` at `llm_inference.py:1186-1191`
   (inside `_run_pt`, batch), `mx.random.seed(item_seed_bases[i] + step*1000003)` at
   `llm_inference.py:3298` (`_run_mlx_batch_native`), and `mx.random.seed(seeds[i])` at `:4043-4044`
   (`_run_mlx` batch fallback). The **single-item** LM paths (`_run_pt_single:985`,
   `_run_mlx_single:3720`, `_run_mlx_single_native:3397`) contain **no seeding at all**, and
   `generate_with_stop_condition` only builds a seed list when `batch_size > 1`
   (`llm_inference.py:1355-1362`, and the docstring at `:1318-1320` says outright
   "Only used when batch_size > 1. TODO: not used yet").

   Consequence: with the LM on and `batch_size=1`, two runs at the same seed produce **different
   captions/metadata and different audio codes**, hence different audio. With `thinking=False`
   (no LM), the pipeline is deterministic given a fixed DiT seed (ODE/Euler), modulo GPU
   non-determinism in conv/attention kernels.
4. Same `shift`, `infer_steps`, `timesteps`, `guidance_scale`, and `dcw_enabled`. Note DCW scaler is
   *not* random, so it does not affect reproducibility.

For the Swift port this is a design opportunity: seed a single `mlx.random.key` and thread
counter-based keys through sampler + VAE so that fixed-seed runs are reproducible by construction.

---

## Q6. VAE

- Config: `/private/tmp/mc5/Ace-Step1.5/vae/config.json`
  (`_class_name: AutoencoderOobleck`, `audio_channels: 2`, `sampling_rate: 48000`,
  `decoder_input_channels: 64`, `downsampling_ratios: [2,4,4,6,10]`, `channel_multiples: [1,2,4,8,16]`,
  `decoder_channels: 128`, `encoder_hidden_size: 128`). Loaded as
  `diffusers.models.AutoencoderOobleck` (`init_service_loader_components.py:46-66`).
- **Output: 48 kHz, stereo (2 channels), float32** (the decode path force-casts
  `pred_wavs.float()` and peak-normalises to ≤1.0, `generate_music_decode.py:199-203`).
  `self.sample_rate = 48000` (`handler.py:121`).
- **Latent frame rate: 25 Hz.** `prod(downsampling_ratios) = 2*4*4*6*10 = 1920`; `48000/1920 = 25`
  frames/s. The code hardcodes this as `1920` everywhere: `conditioning_masks.py:55-56`,
  `conditioning_target.py:64,84,90`, and the inverse `expected_latent_length =
  current_wav.shape[-1] // 1920`. The 5 Hz LM operates on `25/pool_window_size(5) = 5` Hz.
- Latent channels **64** (`decoder_input_channels`; `audio_acoustic_hidden_dim: 64` in the DiT config).
  The 128-channel `context_latents` are `64 latent + 64 broadcast chunk-mask` (Q1).
- **Decode must be chunked for long audio.** Two independent chunkers:
  - **MLX path** (`mlx_vae_decode_native.py:88-140`): `mlx_chunk = max(192, mlx_vae_chunk_size)`,
    `mlx_overlap = 64`; if `latent_frames <= mlx_chunk` it decodes in one shot, else
    `stride = mlx_chunk - 2*overlap`, overlap-discard per chunk. `mlx_vae_chunk_size` is auto-tuned
    from unified memory (`gpu_config.py:779-806`): `≤16 GB → 256`, `≤36 GB → 512`, `≤64 GB → 1024`,
    else `2048`; env override `ACESTEP_MLX_VAE_CHUNK`; floor 192. **An 18 GiB Mac gets 512** →
    stride 384 latent frames = 15.36 s of audio per step.
  - **PyTorch path** (`vae_decode.py:16-86`, `vae_decode_chunks.py:13-86`): auto chunk size from
    `memory_utils.py:81-118` (`VAE_DECODE_MAX_CHUNK_SIZE = 512`), then **on MPS it is forcibly
    reduced to 32 with overlap 8** (`vae_decode.py:13-14, 60-74`) because
    "MPS Conv1d has a hard output-size limit during temporal upsampling". That makes the torch-MPS
    decode path ~10-30× more chunk-steps than MLX — the main reason MLX VAE exists.
  - Both paths do overlap-discard: `trim_start/trim_end` computed from the measured
    `upsample_factor = audio_len/latent_len` rather than assumed (`vae_decode_chunks.py:103-117`,
    `mlx_vae_decode_native.py:117-127`).
- **Scaling per minute of audio** (arithmetic, not measured):
  - latent tensor: `60 s × 25 = 1500` frames × 64 ch × 4 B = **384 kB / minute** (float32; 375 KiB).
  - waveform: `60 s × 48000 × 2 ch × 4 B = 23.04 MB / minute` (float32 in memory);
    a 16-bit WAV is 11.52 MB/min, 24-bit 17.28 MB/min.
  - decode work is linear in latent frames (conv stack), i.e. linear in duration; peak memory is
    bounded by `mlx_chunk` (512 frames ≈ 20.5 s of latent ⇒ ~7.9 MB of waveform per chunk before
    trimming) plus the model (the VAE checkpoint `diffusion_pytorch_model.safetensors` is ~350 MB,
    and float32 MLX copy ≈ 2× that; `ACESTEP_MLX_VAE_FP16=1` halves it).
  - MLX decode is `mx.eval`-ed per chunk with `mx.clear_cache()` every 4 chunks
    (`mlx_vae_decode_native.py:83-86`, `:134-135`).
- One correctness detail for the port: the MLX VAE class docstring claims default
  `downsampling_ratios = [2,4,4,8,8]` (hop 2048) at `vae_model.py:242-244`, but
  `from_pytorch_config` (`:326-338`) overrides with the real `[2,4,4,6,10]` (hop 1920). The docstring
  is **wrong**; trust 1920/25 Hz.

---

## Q7. `silence_latent`

**What it is.** A precomputed **latent of silence** — the VAE encoding of a silent (or near-silent)
stereo clip, stored once so the pipeline never has to run the VAE encoder for "no source audio".
It is the `src_latents` / context for every text2music generation without LM codes, and the padding
value for LM hints.

**Where upstream gets it.** From a file next to the DiT weights, **computed offline** (not at load):
`init_service_loader.py:211-216`

```python
silence_latent_path = os.path.join(model_checkpoint_path, "silence_latent.pt")
if not os.path.exists(silence_latent_path):
    raise FileNotFoundError(f"Silence latent not found at {silence_latent_path}")
self.silence_latent = torch.load(silence_latent_path, weights_only=True).transpose(1, 2)
self.silence_latent = self.silence_latent.to(silence_latent_device).to(self.dtype)
```

It is **mandatory**: init hard-fails without it (no fallback synthesis).

**File / shape / dtype** — verified by parsing the pickle in the official repo
(`/private/tmp/mc5/Ace-Step1.5/acestep-v15-turbo/silence_latent.pt`, 3,841,215 bytes):
- HF file name: **`silence_latent.pt`**, inside the **`acestep-v15-turbo/`** subfolder of
  `ACE-Step/Ace-Step1.5` (not the repo root).
- Stored tensor: `FloatStorage`, **`torch.float32`**, **CPU**, shape `(1, 64, 15000)`,
  strides `(960000, 15000, 1)`; raw data 3,840,000 B.
- After the `.transpose(1, 2)` in the loader: **`[1, 15000, 64]` float32** —
  15000 latent frames = **600 s at 25 Hz** = exactly `DURATION_MAX`.
- Saved with the new zip-based `torch.save` format (`silence_latent/data.pkl`), so a Swift loader can
  read it as a plain safetensors-like blob only after conversion; simplest is to convert once to
  `.safetensors` or raw `Float32` (3.84 MB) in the port's download step.

**Access helper.** `conditioning_target.py:18-33` `_get_silence_latent_slice(length)`:
returns `self.silence_latent[0, :length, :]`; if `length > 15000` it **tiles** the tensor
(`repeat(ceil(length/available))`) — so durations >600 s would silently repeat the latent rather
than error.

**Who needs it.**
- **text2music: YES — always.** With LM off it *is* the `src_latents`/context
  (`conditioning_masks.py:95`); with LM on it pads short code hints to the requested length
  (`conditioning_text.py:34-46`). It is also the fallback "no reference audio" timbre input:
  `conditioning_embed.py:44-49` uses `silence_latent[:, :750, :]` (30 s) when the reference clip is
  all zeros, and `conditioning_batch.py:70` supplies exactly `zeros(2, 30*sample_rate)` when no
  reference audio is given.
- **repaint**: `conditioning_masks.py:88-92` overwrites the *non-repainted* region of the source
  latents with silence (except for `lego`), i.e. silence latent is used as the "keep nothing" fill.
- **cover / cover-nofsq**: the source latent is the reference-audio encoding, but silence is still
  used for the non-cover branch when `audio_cover_strength < 1.0`
  (`modeling_acestep_v15_turbo.py:2014-2034`, `service_generate_execute.py:205-229`) and for padding.
- **extract / lego / complete**: silence for padding / non-source spans (`conditioning_masks.py:88-95`).
- So: not optional. Ship it as a 3.84 MB asset (or regenerate it once from the VAE by encoding
  digital silence — but the official values are the trained expectation; use the file).

---

## Q10. Durations and shapes

### 10.1 The arithmetic

```
latent_frames   T = round(seconds,1) * 25         # 48000/1920 = 25 Hz
latent tensor     = [1, T, 64] float32            # 64 acoustic channels
DiT tokens        = ceil(T / patch_size=2)        # 2 consecutive frames -> 1 token
context channels  = 128  (= 64 src_latents + 64 broadcast chunk mask)
DiT input         = cat([context(128), noise(64)], -1) -> [B, T, 192] -> conv k=2,s=2
samples           = seconds * 48000, 2 channels   # VAE upsample x1920
LM codes          = int(effective_duration * 5)   # 5 Hz, EOS forced
```

Derivation of T from the code:
- `create_target_wavs`: `frames = int(duration * 48000)`, `torch.zeros(2, frames)`
  (`task_utils.py:117-125`).
- `expected_latent_length = wav.shape[-1] // 1920` (`conditioning_target.py:90`) ⇒ `T = duration*25`.
- `max_latent_length = max(128, max(T_i))` — a **floor of 128 frames (5.12 s)**
  (`conditioning_target.py:121`).
- Repaint spans use the same factor: `start_latent = int(sec * sample_rate // 1920)`
  (`conditioning_masks.py:55-56`).

### 10.2 Max duration and constraints

| Constraint | Value | Where |
|---|---|---|
| Hard API/UI max duration | **600 s** | `constants.py:58-59` (`DURATION_MIN=10`, `DURATION_MAX=600`) |
| Min duration | 10 s | same |
| `silence_latent` capacity | 600 s (15000 frames); longer is **tiled**, not rejected | `conditioning_target.py:18-33` |
| Arch ceiling from positions | `max_position_embeddings=32768` tokens × `patch_size=2` = 65536 frames = **2621.4 s** (RoPE position ids = `arange(seq_len)`, torch `:1365-1373`) | config; so 600 s is not an arch limit |
| LM duration clamp | `DURATION_MIN ≤ d ≤ min(gpu_config.max_duration_with_lm, 600)`; codes `= d*5` | `llm_inference.py:268-292`, `constrained_logits_processor.py:1285-1300` |
| GPU-tier max duration (LM on / off) | tier6a (16–20 GB): **480 s / 600 s**; tier6b (20–24 GB): 480 / 480 | `gpu_config.py:409-447` |
| `patch_size` | 2 — odd `T` is zero-padded up by 1 frame, then cropped back | torch `:1355-1360`, `:1505-1506` |
| VAE stride | 1920 samples (25 Hz) | `conditioning_target.py:64,84,90` |
| `timbre_fix_frame` | 750 latent frames = **30 s at 25 Hz**, fixed regardless of output duration. Declared as config default 750 (`models/common/configuration_acestep_v15.py:183,222`) but the *code* hardcodes `750` (`conditioning_embed.py:47`) and a 30 s zero clip (`conditioning_batch.py:70`). Independent of `T`. | as cited |
| `pool_window_size` | 5 — 25 Hz → 5 Hz codes; `T` must be divisible by 5 for exact code count, padding handled in `tokenize` (`modeling_acestep_v15_turbo.py:1628-1638`) | config |
| sliding window | 128 tokens on 12 of 24 layers (local), full attention on the other 12 | config `layer_types` |
| batch | tier6a: 4 with LM / 8 without; `GenerationConfig.batch_size` default 2 | `gpu_config.py:415-416`; `inference.py:291` |

An 18 GiB Apple Silicon Mac lands in **tier6a** (16–20 GB) by `get_gpu_tier`'s ≤20 GB rule, so the
upstream *policy* limits it to 480 s with the LM, 600 s without — but that tier table was written for
CUDA VRAM and is applied to unified memory too. Treat it as a hint, not a measurement.

### 10.3 Per-duration sizes (arithmetic, float32)

| seconds | latent frames T | latent `[T,64]` | DiT tokens (T/2) | samples/ch | wav `2×samples` f32 | LM codes (5 Hz) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 250 | 64 kB | 125 | 480,000 | 3.84 MB | 50 |
| 30 | 750 | 192 kB | 375 | 1,440,000 | 11.52 MB | 150 |
| 60 | 1500 | 384 kB | 750 | 2,880,000 | 23.04 MB | 300 |
| 120 | 3000 | 768 kB | 1500 | 5,760,000 | 46.08 MB | 600 |
| 180 | 4500 | 1.15 MB | 2250 | 8,640,000 | 69.12 MB | 900 |
| 240 | 6000 | 1.54 MB | 3000 | 11,520,000 | 92.16 MB | 1200 |
| 300 | 7500 | 1.92 MB | 3750 | 14,400,000 | 115.2 MB | 1500 |
| 480 | 12000 | 3.07 MB | 6000 | 23,040,000 | 184.32 MB | 2400 |
| 600 | 15000 | 3.84 MB | 7500 | 28,800,000 | 230.4 MB | 3000 |

(`2×samples f32` = 384 KB/s; a 16-bit WAV is half that.)

### 10.4 What actually scales the cost — the part the RAM cap cares about

The latent tensors are tiny; what grows is **attention**. `DiT tokens = T/2`, so:
- 12 full-attention layers cost `O((T/2)²)`: 60 s → 750² = 0.56 M, 600 s → 7500² = 56.25 M
  (**100× more attention work for 10× the audio**). The 12 sliding-window layers cost `O(T/2 * 128)`,
  i.e. linear.
- Cross-attention KV cache is `24 layers × 2 (k,v) × S_enc × 8 kv_heads × 128 dim × dtype`
  (`:2104-2116`); with `S_enc` ≈ a few hundred tokens this is tens of MB, independent of duration.
- Per-step working set: `[B, T, 192]` activations plus `[B, T/2, 6144]` MLP intermediates per layer
  (streamed). At 600 s, `T/2 = 7500`, one layer's `[B, 7500, 6144]` f32 is 184 MB — this is why
  10-minute generations are memory-heavy even though the latents are 3.75 MB.

Practical read for a ~12 GB process cap on an 18 GiB Mac (Lead's call, but the code supports this
range): 60–120 s is comfortable; 180–300 s is the plausible band for a single generation with
sequential VAE chunks; 480–600 s will be dominated by the quadratic full-attention cost and is where
the port would need FlashAttention-style kernels, not just a faithful port.


---

## Appendix. Issue-text checks and open gaps

I did not have the GitHub issue text locally (it is not in the repo and the app checkout had no copy),
so this is limited to the claims I was handed and could test against the code.

Claims that check out:
- "5 Hz planner LM" — correct: 5 codes/s (`llm_inference.py:1310-1311`,
  `constrained_logits_processor.py:1285-1300`).
- "condition encoder (lyric / timbre / pooler / projection)" — correct set of components, all in
  `AceStepConditionEncoder` (`modeling_acestep_v15_turbo.py:1513-1558`) plus the
  tokenizer/detokenizer (`:865-1226`).
- "`prepare_condition`" — correct name and it is the torch→MLX seam
  (`service_generate_execute.py:176-194`).
- "VAE decode" — correct, and it is already MLX (`vae_model.py`, `mlx_vae_decode_native.py`).

Things a port plan must not get wrong (each is a place where a natural paraphrase of the pipeline is
false):
1. **The LM's text is not fed to the DiT.** Its FSQ codes are decoded to 25 Hz latents that *replace
   `src_latents`*. (§Q4.2)
2. **CFG does not apply to turbo, and neither does APG/ADG.** Only DCW does (on by default).
   (§Q5.3)
3. **`timestep_mu`/`timestep_sigma` are training-only.** Inference uses a fixed 8-step schedule
   unless you override `inference_steps`/`timesteps`. (§Q5.2)
4. **The MLX DiT is float32 and coexists with the torch DiT** — the code has two copies. An 18 GiB
   Mac cannot hold both; that duplication is the strongest argument for the Swift port. (§Run-notes 6.4)
5. **`silence_latent.pt` is mandatory and is `[15000, 64]` float32 = exactly 600 s**, and it tiles
   silently past 600 s rather than erroring. (§Q7)
6. **Fixed-seed reproducibility is broken by the LM at `batch_size=1`** and by `infer_method="sde"`.
   (§Q5.4)

Drift found in the tree itself (each is a trap for a reader):
- `vae_model.py:242-244` documents `downsampling_ratios = [2,4,4,8,8]` (hop 2048); the real config is
  `[2,4,4,6,10]` (hop 1920, 25 Hz) and `from_pytorch_config` uses the real one.
- `llm_inference.py:570` says "The DiT and VAE use float16 on MPS"; the DiT is float32 on MPS
  (`init_service_orchestrator.py:101`) and only the VAE is float16 (`memory_utils.py:212-213`).
- `modeling_acestep_v15_turbo.py:1690-1691` contains
  `lm_hints_5Hz = self.tokenize.quantizer.get_output_from_indices(audio_codes)` —
  `self.tokenize` is a **bound method**, so `.quantizer` would raise `AttributeError`. It is
  unreachable from the handler because the handler always passes `precomputed_lm_hints_25Hz`
  instead of `audio_codes` (`service_generate_execute.py:176-194`). Dead/broken branch — do not
  port it; port `_decode_audio_codes_to_latents` (`audio_codes.py:50-72`) instead.
- `ACESTEP_DTYPE` is referenced in a log message (`generate_music_decode.py:79`) but read nowhere.

Open gaps I could not close in code (need measurement or an upstream question):
- Actual peak RSS / `mx.get_active_memory()` per duration on 18 GiB — the code gives only arithmetic.
- Whether the `is_covers`-driven `src_latents` replacement is *intended* for `text2music` when codes
  are present (it is a side effect of `has_code_hint`, `conditioning_masks.py:69-71`) or an artifact
  of the cover path. It is load-bearing for LM-on quality.
- Text-encoder choice is fixed in code (`<root>/Qwen3-Embedding-0.6B`, `init_service_loader_components.py:90`);
  there is no config to swap it, so the Swift port must ship that exact encoder.
- No MLX implementation of the lyric/timbre/FSQ stack exists anywhere in the tree — that is the
  largest genuinely-new porting surface.
