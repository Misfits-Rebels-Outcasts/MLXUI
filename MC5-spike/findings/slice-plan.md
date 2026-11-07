# Q12 — Slice plan, catalog/install shape and the `.music` memory row (MC5)

Owner: plan-analyst (teammate). Read-only spike. Q11 fixtures/tolerances live in
`parity-plan.md`; this file is the delivery plan. App-repo citations are `path:line`;
upstream citations are paths under `/private/tmp/mc5/ACE-Step-1.5` @ `ca1e85f`.

**Decisions at a glance**

1. **One catalog card**, v1 = **no-LM**, official repo `ACE-Step/Ace-Step1.5`, download
   **6.34 GB** (5.90 GiB). The 5 Hz LM subfolder is *excluded at install* (it would add
   3.71 GB for a feature v1 does not run, and it breaks the memory row — §2.1).
2. **bf16 download + quantize the DiT at load** (Klein precedent). Pre-quantized community
   MLX repos do exist (`mlx-community/ACE-Step1.5-MLX-4bit`, 5.29 GB; `MLXCreator/…-4bit`,
   same sizes), but they keep the Qwen3 text encoder in f32 and declare no conversion licence,
   so they are listed as alternatives (§2.1/§2.3), not the recommendation. A *published* 4-bit
   repo of our own is out of scope for this spike.
3. **8 slices `MC5-3a…3h`** ≈ **16–21 engineer-days**, each independently shippable into the
   module template, plus a 1–2 day owner-run `MC5-3i` real-weights/listening gate.
4. **`.music` memory row**: add `case .music` to `ModelLoadProfile.entry(for:)` with the same
   `files × 1.10 + working set` form as `.separate`, the working set filled from
   `/private/tmp/mc5/out/measurements.json` (formula in §3). Until that file exists the row
   **must not be added** (no invented number): today `.music` falls to the `default` row,
   `ModelWeightCharge.swift:115`, and books `max(files × 1.10, catalog ramGB)` (`:132-138`).
5. **MusicGen stays.** `Generate Sound` already has a served executor path; ACE-Step is
   appended to the pool **last**, after the owner's listening review. No removal proposal.
6. **Two new `StageConfig` fields are required** (`duration`, `lyrics`) before a flow row can
   drive ACE-Step; otherwise the curated manifest cannot express the two settings that define
   the output. Flagged as open question OQ-3.

---

## 1. What the app already gives us (and what is missing)

### 1.1 The module template

A port is `Modules/<Name>/` with `XConfig`, `XEngine`, `XSDK: ModelSDK`, `XStage`,
`XUI`, `XModule` and one registration line:

- `ModelModule` / `ModelSDK` / `ModelUI` / `ClaimScore` — `MLXUI/Core/ModelRegistry.swift:7-75`;
  `bestModule(for:)` `:98-110`; `claimableModelIDs` is exactly
  `catalog.filter { registry.bestModule(for: $0) != nil }` — `MLXUI/State/AppState.swift:564`,
  so **a card is runnable the moment an SDK claims it**.
- registration list — `MLXUI/App/ModelModules.swift:39` (`MusicGenModule.self`);
  MusicGen's own wiring is the smallest end-to-end precedent:
  `MusicGenModule.swift:25-27`, `MusicGenSDK.swift:10-19`, `MusicGenStage.swift:6-47`,
  `MusicGenEngine.swift:12-123`.
- Run UI — `MLXUI/Modules/MusicGen/MusicGenUI.swift` is the visual/controls precedent.
- staged multi-component diffusion with *never two resident* and an observable lifecycle —
  `MLXUI/Modules/Flux2Klein/Flux2KleinEngine.swift:4-22` (stage order), `:30-38`
  (`Flux2KleinLifecycleEvent`), `:60-75` (injectable noise/embeddings), `:126-158`
  (own budget = 0.6 × physical RAM, busiest stage, 1.15 margin);
  `Flux2KleinStage.swift:94-95` (`LeaseSized`) and `Flux2KleinSDK.swift:92-116`
  (`Flux2KleinBooking`) are how a staged engine books from the shared budget.
- Engine cache / budget — `MLXUI/FlowKit/EngineCache.swift:187-199` (`bytesNeeded`),
  `:241-249` (`footprint(of:)`, with the Klein staged special case at `:246`),
  `MLXUI/Core/MemoryBudget.swift:64-74` (`max(4 GB, RAM × 0.6)`).

### 1.2 The flow wiring

- `Generate Sound` exists as a task: `TaskCatalog.swift:188` —
  `text → audio`, `.model`/`.engine`, refName `engines.diffusion.generate_sound`.
- `TaskModels.taskKinds["Generate Sound"] = .music` — `TaskModels.swift:82`.
- Its refName prefix is **already served**: `TaskModels.swift:160`
  (`"engines.diffusion.generate_sound"` inside `servedRefNamePrefixes`, `:152-170`), so no
  allow-list change is needed for a *second* music model (the allow-list is per refName,
  not per model).
- The model pool: `TaskModels.taskModels["Generate Sound"] = ["MusicGen"]` — `:257`.
  Order is preference/seed only; the repo's standing rule is **append last** unless the owner
  reorders (MoC-2-4 discipline, documented at `TaskModels.swift:280-287` and the
  `llmModels` comment `:288-315`).
- `RealExecutor.runModel`'s no-input branch takes the row's first bare setting as the text
  prompt for any `engines.diffusion.generate_*` row — `RealExecutor.swift:544-556` — and
  `stageConfig`'s diffusion branch parses `seed`/`width`/`height`/`steps`/`scale`/`quantize`
  and nothing else — `RealExecutor.swift:1181-1215`.
- Curated manifest is the settings authority, ported verbatim from catflow-mlx:
  `Resources/CatFlow/models/` (`musicgen-small.json` is the music precedent, `:1-39`). A model
  with no reference manifest is MLXUI-authored from a sibling shape — precedent
  `qwen3.5-9b-4bit.json` (`CatalogBridge.swift:214-224` comments), and that is the shape
  ACE-Step needs.
- Bridge row maps the `.cat` display name to an installable card:
  `CatalogBridge.swift:82` (`entries`), MusicGen row `:135-140` (`.substitute`), catalog
  candidate must exist — `everyCandidateExistsInBrowserCatalog` is a golden already
  (`CatalogBridge.swift:205-216` comment).

---

## 2. Catalog and install shape

### 2.1 One card or several? — one, and the LM must be excluded by the selector

The official repo is a **component-subfolder layout** (same shape Kokoro/Qwen3-TTS/FLUX
already use — that is *why* the selector was generalised):

```
ACE-Step/Ace-Step1.5/
  config.json                              1,968 B   (duplicate of the DiT config; the loader reads the subfolder)
  Qwen3-Embedding-0.6B/   config.json, tokenizer.json(11.42 MB), tokenizer_config.json,
                          vocab.json(2.78), merges.txt(1.67), added_tokens.json,
                          special_tokens_map.json, chat_template.jinja,
                          model.safetensors (1,191.59 MB)                       ← text encoder
  vae/                    config.json, diffusion_pytorch_model.safetensors (337.43 MB)
  acestep-v15-turbo/      config.json, configuration_acestep_v15.py, modeling_acestep_v15_turbo.py,
                          silence_latent.pt (3.84 MB), model.safetensors (4,787.83 MB)
  acestep-5Hz-lm-1.7B/    model.safetensors (3,708.52 MB) + 4 tokenizer/config files (45.23 MB)  ← excluded in v1
  README.md, .gitattributes                                                              ← filtered
```
(sizes from the HF API, 2026-11-07, to 0.01 MB; loader paths: DiT at
`<checkpoints>/acestep-v15-turbo` — `core/generation/handler/init_service_orchestrator.py:155`,
VAE at `<checkpoints>/vae` — `init_service_loader_components.py:41-44`, text encoder at
`<checkpoints>/Qwen3-Embedding-0.6B` — `:90-94`; silence latent at
`<DiT>/silence_latent.pt` — `init_service_loader.py:214`.)

**Why one card and not two:**
- `InstallManager` installs `model.variants?.first?.hfModelId` (`State/InstallManager.swift:240-242`);
  variants are not a component-picker, so they cannot express "±LM".
- `ModelFileSelector.companionRepo(for:)` (`Core/ModelFileSelector.swift:94-103`) adds a second
  repo's files **unconditionally** for a card (`InstallManager.swift:260-296` plans them in and
  puts them in a fixed subfolder). A companion is "always bundled", so it cannot express
  "only if the user wants the LM".
- Therefore: **v1 card = the main repo minus `acestep-5Hz-lm-1.7B/`.** The LM-assisted mode
  (the 5 Hz LM produces the audio-code hint that is fed as `precomputed_lm_hints_25Hz`,
  `modeling_acestep_v15_turbo.py:1660-1690`) is a later feature that needs its own install
  decision (the LM can come from `ACE-Step/acestep-5Hz-lm-0.6B`, 1,325.80 MB model +
  24.32 MB tokenizer ≈ **1.37 GB**, a separate repo listed in upstream
  `model_downloader.py:303-313`). Do not bundle it in v1.

**Alternative source repos that already exist on HF (verified live, 2026-11-07).** These change
the options but not the recommendation:

| Repo | Layout | Total download | Notes |
|---|---|---|---|
| `ACE-Step/Ace-Step1.5` (official, recommended) | 3 components + LM subfolder, **bf16** | **6.34 GB** excluding LM (10.09 GB with it) | MIT declared on the model card; the only repo whose provenance/licence is stated; the one the upstream code loads |
| `mlx-community/ACE-Step1.5-MLX` | root `model.safetensors` + `Qwen3-Embedding-0.6B/` + `vae/`, `silence_latent.npy`, no LM | **12.65 GB** | an **f32** MLX conversion (text 2.38 GB, model 9.58 GB, VAE 0.67 GB) made with `mlx-audio`; **no licence declared** (`licence.md` §cross-check); bigger than the official bf16 repo — not a download win |
| `mlx-community/ACE-Step1.5-MLX-4bit` | same as above, DiT 4-bit | **5.29 GB** | f32 text encoder (2.38 GB) + 4-bit model (2.21 GB) + f32 VAE (0.67 GB); MLX-branded (matches the app's `source: mlx` rule) but **no licence declared**, quality unverified, and its README claims the 5 Hz LM "is required for the turbo model to produce music" — a claim the official no-LM path contradicts (`out/smoke.json` ran `lm: none`) |
| `MLXCreator/MLXCreator-ACEStep1.5-4bit` | same shape/sizes as the 4-bit row above | 5.29 GB | community conversion, card says Apache-2.0, 25 downloads; not `mlx-community`, lowest provenance |

Recommendation stands: **official repo, bf16, LM excluded**, DiT quantized at load. It is the
smallest *provenance-safe* download, and it is the exact tree the upstream reference runs. The
mlx-community 4-bit repo is the only alternative that would let the card be `mlx-community/...`
by name; it saves 1.05 GB (16 %) but carries an f32 text encoder and an unstated licence, so it
is an **open question, not the plan** (OQ-8a). The selector would need `silence_latent.npy`
(rather than `.pt`) for those repos — add it to the exact-name exceptions below if ever chosen.

### 2.2 What `ModelFileSelector` does with this shape today

`Core/ModelFileSelector.swift`:

- **`metadataNames`** (`:18-69`) is a flat basename allow-list applied at *any depth*:
  `config.json`, `tokenizer.json`, `tokenizer_config.json`, `preprocessor_config.json`,
  `processor_config.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt`,
  `added_tokens.json`, `tokenizer.model`, `chat_template.json`, `chat_template.jinja`,
  `t5_config.json`, … — everything ACE-Step's two tokenizer subfolders need is already here
  (no new names required).
- **`componentDirs`** (`:137-144`) is computed from the *lowercased sibling list*: a subfolder
  is a component if it contains **any** `*.safetensors` **or** a file whose basename is in
  `metadataNames` — keyed by the prefix including the trailing `/`.
- The final filter (`:146-194`): top-level metadata names always come down (`:159`); inside a
  component dir, its `*.safetensors` **and** its metadata basenames come down (`:163-166`);
  top-level `model.safetensors`/shards via the standard branches (`:169-171`); the diffusers
  `model_index.json` layout rule (`:127-128`, `:190-191`) is not triggered (no
  `model_index.json` in this repo), so the fallback is irrelevant.
- **What this means for ACE-Step as-is:** `Qwen3-Embedding-0.6B/`, `vae/`,
  `acestep-v15-turbo/`, and `acestep-5Hz-lm-1.7B/` are *all* component dirs → the current
  selector would download **all 28 files including the 3.71 GB LM**. Three concrete gaps:
  1. **No way to exclude a component** (`silence_latent.pt` and the LM are the two cases).
     `silence_latent.pt` is *skipped*, not downloaded: `.pt` is in the hard skip at
     `:150-153` and that branch returns **before** the component branch at `:163`, so it is
     unreachable regardless of `metadataNames`. It is required at run time (the timbre encoder
     gets `silence_latent[:, :750, :]` as the no-reference-audio input,
     `core/generation/handler/conditioning_embed.py:44-48`; the file is `[1,64,15000]` fp32,
     `init_service_loader.py:214` transposes to `[1,15000,64]`).
  2. **No way to scope a card to a component subset** (LM exclusion).
  3. `model.safetensors` is one 4.79 GB file holding **both** `encoder.*` (≈0.8 B params) and
     `decoder.*` (≈1.6 B params) — this is an engine-side staging problem (§4, slice 3b), not
     an install problem, but the plan must not pretend the two can be fetched separately.

**Exact future edits (implementation, not this spike):**

| Where | Change |
|---|---|
| `Core/ModelFileSelector.swift:150-153` | before the generic `.pt` skip, allow the exact nested name `acestep-v15-turbo/silence_latent.pt` (an exact-name exception, like the `joint_head.safetensors` precedent at `:173-183`; **not** a general "allow .pt in component dirs" rule — Wan's `.pth` and Kokoro's `.pt` voices are why the skip exists). If a converted repo is chosen instead, the exception is `silence_latent.npy` (top-level, same reason: it matches no existing branch) |
| `Core/ModelFileSelector.swift:94-103` | add `static func excludedComponents(for hfModelId:) -> Set<String>` (or `excludedSubfolders`), returning `["acestep-5hz-lm-1.7b/"]` for `"ACE-Step/Ace-Step1.5"`; apply it in `filesToDownload` **and** at the componentDirs/final-filter step (`:163-166`). Mirror the `companionRepo` switch exactly (one hfModelId, one small table) so it stays reviewable |
| `Core/ModelFileSelector.swift:118-145` (signature) | `filesToDownload(siblings:)` stays pure; the exclusion is passed in (`filesToDownload(siblings:exclude:)`) so the unit test can assert both the v1 set and the "full repo" set |
| `Resources/browser.json` music domain (`:2264-2332`) | add the card: `modelType: "music"`, `source: "mlx"`, `hfModelId: "ACE-Step/Ace-Step1.5"`, `id: "ACE-Step--Ace-Step1.5"`, `format: "mlx-bf16"`, `ramGB`/`downloadSizeGB` from §2.3 + §3, `variants[0]` matching |
| `Resources/CatFlow/models/ace-step-1-5.json` (new) | MLXUI-authored curated manifest (shape of `musicgen-small.json`, `:1-39`): `kind "local"`, `engine "mlx-diffusion"`, `tasks ["Generate Sound"]`, settings `seed`, `steps` (default 8, 1…20), `duration` (maps_to a new field, OQ-3), `lyrics`/`caption` (see OQ-3); `capabilities.variant "acestep-v1.5-turbo"`; resources from §2.3; `source https://huggingface.co/ACE-Step/Ace-Step1.5`; licence MIT (`ACE-Step/Ace-Step1.5` README/model card) — **confirm with `licence.md`** |
| `FlowKit/CatalogBridge.swift:82` (after the MusicGen row `:135-140`) | `BridgeEntry(display: <manifest display>, pinnedID: "ACE-Step/Ace-Step1.5", candidates: ["ACE-Step/Ace-Step1.5"], equivalence: .same, manifestFile: "ace-step-1-5.json")` |
| `FlowKit/TaskModels.swift:257` | append the display name **last** to `["MusicGen"]` (the music analogue of the MoC-2-4/`llmModels` append rule `:280-315`) |
| `App/ModelModules.swift:39` area | register `ACEStepModule.self` |
| `Core/ModelSupport.swift:17-34` | while 3a lands before any engine exists, add a `gaps` entry (`idFragment: "ace-step"`) so Browse says "no runner yet"; remove it in 3h. Precedent: every previous port's gap entry is listed/removed there (`:18-33`) |
| `Resources/browser.json` `domainIndex`/`sidebarSections` | **no change**: `music` domain and the `music` section slot already exist (`:49`, `:87-88`, `:2264-2331`) |

### 2.3 Download arithmetic (v1 recommended = no-LM, bf16)

| Component | Files | Size |
|---|---|---|
| `Qwen3-Embedding-0.6B/` | model.safetensors 1,191.59 + tokenizer.json 11.42 + vocab 2.78 + merges 1.67 + configs/template 0.01 | **1,207.47 MB** |
| `acestep-v15-turbo/` | model.safetensors 4,787.83 + silence_latent.pt 3.84 + 2 `.py` 0.11 + config 0.00 | **4,791.78 MB** |
| `vae/` | diffusion_pytorch_model.safetensors 337.43 + config 0.00 | **337.43 MB** |
| **v1 total** | | **6,336.68 MB = 6.34 GB = 5.90 GiB** |
| *(excluded) `acestep-5Hz-lm-1.7B/`* | LM 1.7B | *3,753.59 MB (3.75 GB, 3.50 GiB)* |
| *(not in v1) LM 0.6B from its own repo* | `ACE-Step/acestep-5Hz-lm-0.6B` | *1,372.68 MB (1.37 GB, 1.28 GiB)* |

Options for v1 (one card designed to allow a later variant card without a data migration —
the installed folder is the same three components):

| Option | Download | Resident weights after load (weights.md measured params) | Fit vs 12 GB process cap |
|---|---|---|---|
| **A. no-LM, bf16 + DiT int8 at load (recommended)** | **6.34 GB** | text enc 1.19 + cond enc 1.22 (608 M params) + DiT 1.58 (int8 of 1.575 B) + VAE 0.68 (f32, 168.7 M), **staged one at a time** | fits with room (§3.4) |
| B. no-LM, bf16 + DiT int4 at load | 6.34 GB | DiT ~0.79 | fits; quality is the owner's call |
| C. no-LM, bf16 resident (no quantize) | 6.34 GB | DiT 3.15 (bf16) + cond enc 1.22, staged | fits **if truly staged** (selective load, 3b); fails if the whole 4.79 GB checkpoint is held while another stage runs |
| D. with LM 1.7B (whole official repo as-is) | 10.09 GB | + LM 1.85 B params = 3.71 GB bf16 (or ~2 GB 4-bit) | **incompatible with the plain kind-row booking** (10.09 × 1.10 ≈ 11.1 GB before working set) — needs §3.3's staged booking; RAM-wise only if the LM is released before DiT |
| E. with LM 0.6B | 7.71 GB | + LM ~1.37 GB bf16 (separate repo `ACE-Step/acestep-5Hz-lm-0.6B`) | fits staged; needs a second-install decision |
| F. `mlx-community/ACE-Step1.5-MLX-4bit` | 5.29 GB | text 2.38 (f32!) + model 2.21 (4-bit) + VAE 0.67 (f32) | fits; **no licence declared**, quality unverified, and it drops the official-repo provenance — open question OQ-8a |
| G. `mlx-community/ACE-Step1.5-MLX` (f32) | 12.65 GB | text 2.38 + model 9.58 + VAE 0.67 | download alone nearly the whole cap; not recommended |

**There *is* a 4-bit download** — two community MLX conversions (F above and
`MLXCreator/MLXCreator-ACEStep1.5-4bit`, same sizes) — but neither declares a licence for the
conversion, both keep the Qwen3 text encoder in **f32** (2.38 GB, 45 % of their download), and
the app's A2 pattern does not need one: quantizing at load is the established path (FLUX.2 Klein's
card is bf16 and the engine quantizes the DiT at load — `Flux2KleinEngine.swift:40-46`, catalog
`mlx-community/FLUX.2-klein-4B-bf16`). Recommendation: **do not** switch v1's source to a
converted repo on the basis of 1.05 GB; revisit only if the owner rules the official repo's
provenance less important than `mlx-community/...` naming (OQ-8a). Publishing a new 4-bit repo is
the Lead's call and is out of this spike's scope.

### 2.4 Per-subfolder metadata each component needs (all already covered by `metadataNames`)

| Component | Required at load | Covered today? |
|---|---|---|
| `Qwen3-Embedding-0.6B/` | `config.json` (Qwen3 dims), `tokenizer.json` (BPE), `tokenizer_config.json`, `vocab.json`/`merges.txt` fallback, `added_tokens.json`, `special_tokens_map.json`, `chat_template.jinja` (not used for embeddings, harmless) | yes — `metadataNames` `ModelFileSelector.swift:18-69` + component branch `:163-166`; the loader in the app is `HFTokenizerLoader` (`Core/HFTokenizerLoader.swift`) — a tokenizer **inside a subfolder** has no precedent in `HFTokenizerLoader`; check its folder argument (open question OQ-4) |
| `vae/` | `config.json` (Oobleck: `decoding_ratios`, channels, `sampling_rate 48000`) + `diffusion_pytorch_model.safetensors` | yes |
| `acestep-v15-turbo/` | `config.json` (the AceStepConfig — 24 layers, 2048, 16/8 heads, patch 2, timbre dims, `sliding_window 128`, task vocabulary) + `silence_latent.pt` | config yes; `silence_latent.pt` **no** → §2.2 edit 1 |
| root `config.json` | duplicate of the DiT config; the loader uses the subfolder (`init_service_orchestrator.py:155`) so it is optional — include it (it is already a top-level `metadataNames` file) | yes |

No per-component `.installed` markers are needed: install is all-or-nothing per model
directory and the atomic marker is `.installed` at the model root
(`CLAUDE.md` "Install & storage"; `InstallManager.swift:246-296` writes the same plan).

---

## 3. The `.music` memory row

### 3.1 How the mechanism works (exact lines)

- `ModelWeightCharge` is the **LLM** rule: safetensors on disk × `loadFactor 1.10`
  (`Core/ModelWeightCharge.swift:20-43`).
- `ModelLoadProfile` is the **per-kind** table for everything else
  (`Core/ModelWeightCharge.swift:46-142`), keyed by `RunnerKind` (`:79-117`):
  `case .llm/.ocr/.asr/.decision/.diarization/.enhance/.vision/.interpolate/.separate` are
  measured; **every other kind — including `.music` — falls to `default` at `:115`**
  (`Entry(factor: 1.10, workingSetBytes: 0, measured: false)`), and the doc comment says so:
  "A kind with **no** measurement (TTS, diffusion, **music**, segmentation, …) books
  `max(files × 1.10, catalog ramGB)`" (`:62-65`).
- `charge(kind:directory:catalogFallback:)` (`:132-138`) sums **all** safetensors *under* the
  directory (subfolders included, `:119-128` — it walks recursively on purpose for models with
  component subfolders), multiplies by the row's factor; a **measured** row returns exactly
  that booked figure, an unmeasured one floors it at the catalog `ramGB`.
- `workingSetBytes(for:)` (`:141`) returns the row's per-call extra.
- Callers: `EngineCache.bytesNeeded` (`FlowKit/EngineCache.swift:187-199`) —
  `weights + workingSet` for models whose engine books its own weights, and `EngineCache.footprint(of:)`
  (`:241-249`) with a **staged-engine special case** for Klein at `:246`. The flow path routes
  through `AppFlowExecutorFactory.make` (`Views/Flows/AppFlowExecutorFactory.swift:18-44`,
  including `footprintOverride: Flux2KleinBooking.leaseOverride(...)` at `:31`); the Run window
  path books through the same cache (SC-5a/b).
- The measured-row precedent and its test: `.separate` (SAM-Audio) `:105-114` and
  `MLXUITests/SAMAudioCatalogTests.swift:188-235` — the test builds a sparse fixture folder with
  the real file sizes, asserts the row is measured, asserts the booked bytes, and asserts the
  working set equals `measured peak − files×1.10` rounded up to the next 0.05 GB. That is the
  exact pattern a `.music` row must copy.
- Cross-checks that must agree: Klein's `measuredFootprintGB` equals its catalog `ramGB` (a test
  pins that — `Modules/Flux2Klein/Flux2KleinSDK.swift:95-99`); SAM's `browser.json` `ramGB`
  14.2 equals the measured peak (browser.json `:3364`-area entry; the description says
  "RAM figure is measured: process peak (phys_footprint) 13.21 GiB (14.19 GB, listed as 14.2)").

### 3.2 The exact future edits (no code now)

| # | File:line | Edit |
|---|---|---|
| 1 | `MLXUI/Core/ModelWeightCharge.swift:79-117` | add `case .music:` **before** `default:` at `:115`: `return Entry(factor: ModelWeightCharge.loadFactor, workingSetBytes: <W>, measured: true)` — same form as `.separate` (`:114`) and `.enhance` (`:98`); W from `measurements.json` (§3.3) |
| 2 | `MLXUI/Core/ModelWeightCharge.swift:46-65` (comment table) | add the `.music` line to the doc table and name the measurement run + config (the table is the provenance; every other row has it) |
| 3 | `MLXUI/Resources/browser.json` music card (`:2264-2332`) | `ramGB` = the **booked** figure from §3.3 (`P`, decimal GB; `variants[0].ramGB` the same) — provenance in the card's `description` like SAM's (`:3364`), and `downloadSizeGB` 6.34; the test at #4 pins `ramGB` to the booked value (Klein precedent) |
| 4 | new `MLXUITests/ACEStepCatalogTests.swift` (or extend an ACE-Step parity file) | mirror `SAMAudioCatalogTests.swift:188-235`: sparse fixture folder with the **three v1 components' real sizes** (1,191,586,416 + 4,787,825,604 + 337,431,388 B; `ModelWeightCharge.safetensorsBytes(underneath:)` must include the subfolders), assert the row is `measured`, assert `factor == ModelWeightCharge.loadFactor`, assert `booked == files×1.10` and that the working set is `peak − booked` rounded up; assert the booked figure equals the catalog `ramGB` (Klein precedent) |
| 5 | `MLXUI/FlowKit/EngineCache.swift:246` (only if the staged engine books *less* than files×1.10 — see §3.3) | a `ACEStepBooking.bytes(for:)`/`leaseOverride(...)` special case next to `Flux2KleinBooking`; the Run-window and flow paths both read it (`EngineCache.swift:246`, `AppFlowExecutorFactory.swift:31`) |

No other file needs to change: `runnerKind` already maps `.music`
(`Core/RunnerKind.swift:30`), `.music` is already a `RunnerKind`
(`RunnerKind.swift:8`), and install/RAM gating uses the catalog `ramGB`
(`Core/StageFactory.swift:100-106`).

### 3.3 The row formula, and the dependency on the Lead's measurement file

Let `F` = sum of `*.safetensors` for the **installed v1** folder (the three components above;
`ModelLoadProfile.safetensorsBytes(underneath:)` picks them all up), and `P(config)` = the
**busiest stage's incremental cost** from the Lead's measurement (`out/measurements.json` /
the per-run `out/json/q8-*.json`; take the max over durations and stages of the recommended
config, then add a small app baseline because the reference's base includes its own co-resident
models — see the interpretation note below):

```
row.workingSetBytes = ceil_to_0.05GB( P(recommendedConfig) − F × 1.10 )   (0 if negative)
row.factor          = 1.10
row.measured        = true
booked              = F × 1.10 + row.workingSetBytes  ≈ P(recommendedConfig)
```
with `F = 1,191,586,416 + 4,787,825,604 + 337,431,388 = 6,316,843,408 B` (the three components'
`*.safetensors`, measured in `weights.md` §1; tokenizer/config files are not counted by
`ModelWeightCharge`), so `F × 1.10 = 6,948,527,749 B ≈ 6.95 GB`.

**What the file must contain** (the Lead's; do not invent it): per config
(`bf16-noLM`, `bf16+0.6B-LM`, `4bit-DiT`) and per stage (text encoder, condition encoder, DiT
denoise, VAE decode) × duration (30 s, 2 min, 4 min) → `peak_phys_footprint_bytes` and
`wall_seconds`. The matrix was still running when this was written: `out/json/q8-bf16-nolm-30s.json`
and `q8-bf16-nolm-120s.json` exist, `q8-bf16-nolm-240s` was in flight, `measurements.json` not yet
consolidated. **If the file is absent the row stays unadded** — `.music` keeps the `default`
fallback (`:115`), so the card would book `max(6.95 GB, ramGB)`.

**Measured so far (bf16, no-LM, `staged_free: true`, `batch 1`, shift 1.0 — verbatim from
`out/json/`):**

| Run | whole-process peak | text.encode Δ | condition.prepare Δ | DiT Δ | VAE decode Δ | wall |
|---|---|---|---|---|---|---|
| 30 s | **16,377.1 MB** | 11.2 | 141.4 | 1,534.5 | **5,869.8** | 8.72 s |
| 120 s | **16,848.0 MB** | 13.8 | 116.5 | 1,880.5 | **7,058.4** | 24.26 s |

Two things this shows for the row:

- **The whole-run peak is dominated by the reference's own persistent base (~10.5–11 GB: it holds
  the torch condition-encoder model, the text encoder, and both a torch and an MLX DiT).** Booking
  that number for the *Swift* staged engine would over-book by >2× and make the model always run
  alone — SAM's accepted precedent, but not what the Swift port will do.
- **The port's own incremental cost is the stage delta**: the DiT is ~1.5–1.9 GB above an idle
  baseline and the VAE decode (which includes the VAE weights and chunk buffers) is the busiest at
  ~5.9–7.1 GB. That is the number the `.music` row should represent, with an app baseline (a few
  hundred MB, not the reference's 10 GB) added.
- **The 240 s / 4 min numbers are not in when this was written**; the VAE delta is the one that
  grows with duration (chunked decode) and it decides the working set. Fill the formula only from
  the consolidated file, and record the config + duration the peak came from in the row's comment
  (the `ModelWeightCharge.swift:46-65` table carries provenance for every row).

**Interpretation caveat (flag to the Lead, do not decide silently):** the table's documented
semantics are "what every **non-LLM** load books … while resident and running" — for SAM the number
is a whole-process `phys_footprint` peak of the real engine path. ACE-Step's reference whole-process
peak includes components the Swift staged engine never co-hosts, so using it verbatim would
misrepresent the port. The recommendation here is the stage-delta + app-baseline reading; the
conservative alternative is booking the reference whole-process peak (~16.8 GB for 120 s), which
keeps the model permanently over-budget like SAM. Decide with the Lead; the formula supports both
by changing what `P` means.

**Alternative if the measured peak is materially below files×1.10** (likely under the stage-delta
reading): give ACE-Step its own staged booking like Klein — `ACEStepBooking.bytes(for:)`/
`leaseOverride(...)` returning the busiest-stage peak × 1.15, wired at `EngineCache.swift:246` and
`AppFlowExecutorFactory.swift:31` — and *keep* the `.music` kind row unmeasured (no row) so the
special case is the only booking. The two are mutually exclusive: a `measured: true` `.music` row
changes every music-kind model (MusicGen's 2.5 GB folder too), which the table's own comment
anticipates ("Each measured row is **one model**; the kind's other models inherit it" — `:63-65`).
**Recommendation: add the row (the assignment's ask), and revisit only if a measurement shows a
real problem.**

### 3.4 Decision 4's 12 GB process cap — which configs fit

The cap is about the **whole process peak**, which for the Swift staged engine is
`max over stages (resident weights + that stage's activations + host buffers)` while the app is
otherwise idle. Two facts frame the comparison:

1. **The reference Python pipeline does not fit.** The Lead's runs record whole-process peaks of
   **16.38 GB at 30 s** and **16.85 GB at 120 s** (bf16, no-LM, `out/json/q8-bf16-nolm-*.json`),
   18.84 GB at the earlier 10 s smoke (`out/smoke.json`), with `init` alone at 10.5 GB and
   `vae.decode.mlx` deltas of 5.87 GB / 7.06 GB. That is expected: the upstream keeps a torch DiT
   **and** an MLX DiT (`run-notes.md` §6.4), runs the text/condition encoders in f32, and its
   staging (`staged_free: true`) still keeps the torch conditioner/text encoder resident. The
   measurement is not a verdict on the model — it is the number the Swift staged port exists to
   beat.
2. **The full matrix is not consolidated yet.** `out/json/` has the bf16 no-LM 30 s and 120 s runs
   (240 s in flight when this was written; `out/smoke.json` is the earlier 10 s smoke); the
   30 s / 2 min / 4 min × {bf16 no-LM, bf16 + 0.6B LM, 4-bit DiT} `measurements.json` does not
   exist yet. The `.music` row value (§3.3) and the final cap verdict depend on it. Until it
   lands, this table is arithmetic from measured weights (`weights.md` §1/§6), plus the per-stage
   deltas quoted above.

| Config | Download | Arithmetically largest staged stage | Verdict vs 12 GB |
|---|---|---|---|
| no-LM, bf16, DiT int8 at load | 6.34 GB | text enc f32 2.38 / bf16 1.19; cond enc bf16 1.22; **DiT int8 1.58** + activations; **VAE f32 0.68 + one chunk** (chunk 512 latent frames ≈ 20.5 s ⇒ ~8 MB waveform before trim + conv workspace) | **fits** (expected busiest ≈ 2.5–4 GB) — recommended |
| no-LM, bf16, DiT int4 | 6.34 GB | DiT int4 0.79 + activations | fits; tightest quality risk |
| no-LM, bf16 resident | 6.34 GB | DiT bf16 3.15 + activations | fits **only with a truly staged loader** (3b); loading the whole 4.79 GB checkpoint while the text encoder/condition encoder runs is what pushes a naive port over |
| + LM 1.7B, bf16 | 10.09 GB | LM 3.71 bf16 + KV/logits for a 1.854 B Qwen3; then DiT 1.58–3.15 if released in between | **blocked by the plain kind-row booking** (10.09 × 1.10 ≈ 11.1 GB > 12 GB before working set, before the app); RAM-wise it fits only if the LM is fully released before the DiT loads and the LM's KV is bounded. Needs the §3.3 staged booking *and* a measurement first |
| + LM 0.6B | 7.71 GB | LM 1.37 bf16 + KV | fits staged; still a second-install decision |
| `mlx-community/...-4bit` (F) | 5.29 GB | text 2.38 f32 (the f32 text encoder is the biggest single piece) + model 2.21 + VAE 0.67 | fits, but licence/quality unanswered (OQ-8a) |

**Largest configuration that fits under the cap with the plain kind-row booking:
no-LM (6.34 GB installed, 6.95 GB booked at ×1.10), DiT quantized at load (int8 default, int4 if
the owner accepts it), VAE decoded in chunks.** The **with-LM 1.7B configuration is the one that
does not fit the row's own arithmetic** — call that out explicitly in the issue comment, and state
that it would need the staged-booking override plus a measurement. The 0.6B-LM option is the only
LM variant with a plausible fit; it is not recommended for v1 because v1 does not run an LM at all
(OQ-8b is the quality question that decides whether any LM is needed — see §6).

---

## 4. Proposed slices

Each slice lands in the module template, adds one module file group + one test suite, and
changes no flow behaviour until 3h. Sizes are engineer-days (single implementer, Xcode +
`xcode-tools`, tests as described); every slice's parity test is in `parity-plan.md` §3.

### MC5-3a — Catalog + install shape (0.5–1 d) — no engine
Ports: the browser.json card, the curated manifest, the CatalogBridge row, the ModelSupport
gap, and the three `ModelFileSelector` changes of §2.2 (silence_latent exception, LM
exclusion, v1 file set).
Done-when: a unit test calls `ModelFileSelector.filesToDownload` on the official repo sibling
list and gets **exactly** the v1 set (asserts the 3 components' files, the absence of
`acestep-5Hz-lm-1.7B/*`, the presence of `acestep-v15-turbo/silence_latent.pt`, and the
`downloadSizeGB` arithmetic 6.34 GB); `Resources/browser.json` decodes (`validate_browser_json.py`
+ the decode test), the bridge candidate golden passes, and Browse shows the card dimmed with
the "no runner yet" reason.
Risk: medium-low — the selector is pure and unit-tested; the only trap is the `.pt` skip order
(`:150` before `:163`) and `HFRemoteFile.filename` keeping the nested path (it does:
`InstallManager.swift:313-318` creates intermediate dirs).

### MC5-3b — Weight plumbing: selective safetensors reader (1.5–2 d)
Ports: a small `SafetensorsReader` (parse the 8-byte header length + JSON header, memory-map or
`Data(contentsOf:options:.mappedIfSafe)`, slice a tensor's byte range, `MLXArray(data, shape,
dtype:)` — the initialiser exists in mlx-swift `Source/MLX/MLXArray+Init.swift:668-676`) so the
engine can load `encoder.*` and `decoder.*` **separately from the single 4.79 GB file**; a
`ACEStepWeights` facade with per-component load reports (bytes read, tensors loaded); a
`silence_latent.pt` reader (the only PyTorch-authored file: `torch.load` format — check with
`weights.md`; it is a single fp32 tensor `[1,64,15000]`, so a tiny `.pt` parser or a one-time
conversion is needed — flag).
Done-when: (a) a committed tiny safetensors fixture (two prefixes, bf16 + fp32) loads only the
requested prefix and the load report proves the other prefix was never materialised; (b) the
real file in an env-var test loads `decoder.*` and `encoder.*` with the expected tensor counts
and dtypes; (c) unit tests for header parsing edge cases (metadata key, empty tensor list,
misaligned offsets).
Risk: **highest technical risk in the plan** — a 4.79 GB single-file checkpoint means naive
`MLX.loadArrays` peaks at ~4.8 GB per stage and defeats the staging; 64-bit offsets, bf16 dtype
mapping, and the `silence_latent.pt` (torch pickle) are the three traps. Alternative if the
reader proves brittle: a one-time `prepare` step that splits the checkpoint into
`condition_encoder.safetensors` / `dit.safetensors` / `tokenizer.safetensors` on first load
(extra 4.8 GB of disk transiently, precedent for a prepare step: `FlowKit/BundledWorkspaces.swift`).

### MC5-3c — Tokenizer, prompt formatting, text encoder (1.5–2 d)
Ports: **wire MLXEmbedders' public `Qwen3Model`** (`EmbedderModelFactory.loadContainer(from:
<dir>/Qwen3-Embedding-0.6B, using: HFTokenizerLoader())` → `context.model(...)` →
`.hiddenStates`, the same call `Modules/Embedding/EmbeddingEngine.swift:44-47` already makes —
`text-encoder.md` §2-§3), `HFTokenizerLoader` for the **subfolder** tokenizer with
`addSpecialTokens: false`, the SFT prompt + lyric header formatting
(`constants.py:160-168`, `prompt_utils.py:38-40`), the 256/2048 truncation, the 1024→2048
`text_projector`, and the **lyric embedding table** (a separate `Embedding(151669, 1024)` loaded
from the same safetensors tensor — the one API gap, `text-encoder.md` §3).
Done-when: parity `g0` (exact strings/ids) + `g1` (projector/lyric wiring) pass; env-var real
parity `real_text` passes at §3.2's f32 tolerance; a 256-token prompt is tokenized in <1 s and
the encoder forward runs; `claimableModelIDs` still excludes the card (no SDK yet).
Risk: low — no forward is hand-written. The unfamiliar parts are the tokenizer living in a
model subfolder (OQ-4) and exposing the private `embed_tokens` as a second table; both are named,
small changes. This is the slice text-encoder.md's recommendation most changes (it removes the
1–2 days a hand-written Qwen3 trunk would have cost).

### MC5-3d — Condition encoder (2–3 d)
Ports: `text_projector`; `lyric_encoder` (8 bidirectional layers, 4-D mask, full/sliding
alternation, RoPE); `timbre_encoder` (4 layers over the shipped silence latent, first-token
pooling, no special token); `pack_sequences` (lyric→timbre→text); the masks.
Done-when: parity `g2` (tiny) passes at every tap; env-var real `real_cond` passes (cos ≥ 0.9995
on the packed sequence); the packed sequence's length/order is asserted exactly.
Risk: medium — the 4-D bidirectional masks and the sliding/full alternation are where a subtle
mask bug hides; per-tap fixtures exist precisely for this.

### MC5-3e — DiT block + full forward (3–4 d)
Ports: `AceStepDiTLayer` (AdaLN self-attn, AdaLN cross-attn with K/V cache, AdaLN MLP,
`scale_shift_table` chunking), `TimestepEmbedding` ×2 (sinusoidal + `t`, `t−t_r`), `proj_in`
Conv1d with the context‖x concat and patch padding, `condition_embedder`, output AdaLN +
`proj_out` ConvTranspose1d, RoPE, sliding/full 4-D masks, the cross-attention K/V cache across
steps.
Done-when: parity `g3` tiny passes per-block (max|Δ| ≤ 1e-5) incl. `alt.*` at a second σ; env-var
`real_dit` passes per-block norm ratios within 2% (bf16) / §3.4's int8/int4 limits for the
quantized builds; the cross K/V for layer 0 matches.
Risk: high — conv layout (PT `[out,in,K]` ↔ MLX `[out,K,in]`), AdaLN chunk order
(`shift, scale, gate, c_shift, c_scale, c_gate`), the `t−t_r` embedding, and the GQA
repeat direction are four independent opportunities for a "runs but wrong" port. The per-block
fixtures are the mitigation.

### MC5-3f — 8-step sampler + DCW (1.5–2 d)
Ports: the schedule (`1−i/8` then `3t/(1+2t)`, `:1970-1979`), Euler ODE (`:2126-2131`), the
final-step direct x0 (`:2122-2130`), DCW (`common/dcw_correction.py`, haar 1-level, scalers
0.05/0.02 decaying with t, on by default for turbo), and the noise injection seam.
Done-when: parity `g4` tiny passes (max|Δ| ≤ 1e-5 for dcw on and off, schedule exact); env-var
`real_denoise` passes x_8 cosine ≥ 0.99 and per-step cosines printed; "same seed twice →
byte-identical audio" test passes.
Risk: medium — the formula is small; the risk is DCW (a wavelet correction with `t`-dependent
scalers) being silently skipped, which passes per-stage tests and fails end-to-end (OQ-2).
Dumping both dcw on/off makes the decision cheap.

### MC5-3g — VAE decoder (2–3 d)
Ports: `AutoencoderOobleck` decoder (weight_norm fusion g·v/‖v‖, conv axis swap, Snake1d
α/β reshape, Oobleck residual units + upsampling blocks, 48 kHz stereo output, the reference's
`peak > 1 → divide` normalisation), 1-D tiling optional (not needed at 8 s).
Done-when: parity `g5` tiny passes (max|Δ| ≤ 1e-5, incl. every decoder-block tap); env-var
`real_vae` passes (2 s decode, cos ≥ 0.999, max|Δ| ≤ 0.02); output is exactly
`[2, T×1920]` at 48 kHz.
Risk: medium — 365 reference tensors with `weight_g`/`weight_v`; the conversion is mechanical
but silent shape/layout mistakes only show in the first tap that's checked per block
(fixtures included). The MLX port's `vae_convert.py` is a usable second opinion.

### MC5-3h — Engine assembly, staging lifecycle, module registration, Run UI (3–4 d)
Ports: `ACEStepEngine` (stage order text encoder → condition encoder → DiT denoise → VAE
decode, **never two of the big components resident**, `Memory.cacheLimit` saved/capped/restored
on every exit like `Flux2KleinCacheLimitGuard` `Flux2KleinEngine.swift:10-12`), `ACEStepSDK`
(claim), `ACEStepStage` (`.text → .audio`), `ACEStepUI` (duration/lyrics/steps/seed controls —
MusicGenUI precedent), `ACEStepModule` + `App/ModelModules.swift` registration; remove the
ModelSupport gap; the `.music` row if `measurements.json` exists by then (§3).
Done-when: tiny end-to-end `g6` passes (max|Δ| ≤ 1e-5 through the whole stage order); the
lifecycle test observes exactly one of {text/cond, DiT, VAE} resident at a time (Klein's test
shape); a real-weights smoke run produces a non-empty 8 s WAV; the card is claimable
(`AppState.claimableModelIDs`, `AppState.swift:564`).
Risk: medium-high — this is the slice where the module template + budget integration lives;
the staging constraint is already proven by Klein, but ACE-Step adds the *same-file* encoder/DiT
split from 3b.

### MC5-3i — Real-weights parity + listening gate (1–2 d, owner-run)
Runs `make_aces_step_real_goldens.py`, then `real_*` suites + the end-to-end 8 s reference
(Q11 §3.7); owner listens once and decides (a) whether the output is shippable, (b) DCW on/off,
(c) int8 vs int4 default, (d) whether `Generate Sound`'s pool gets the ACE-Step display name.
Done-when: the owner says so; MusicGen unchanged either way.
Risk: this is the only slice where "sounds right" can fail while every fixture passes — it is
deliberately last and owner-gated.

**Total: ≈ 16–21 engineer-days + 1–2 owner days.** Ordering is a dependency chain:
3a → 3b → (3c, 3d in parallel) → 3e → 3f → (3g in parallel with 3f) → 3h → 3i.
Every slice before 3h is invisible to users and revertable on its own; 3a can ship alone as
soon as the owner wants the card visible-but-dimmed.

---

## 5. Workflow wiring constraint (`Generate Sound`) — what a later implementation changes

Current state (all verified, no change is needed to *serve* the task — only to add a second
model and its settings):

| Element | Today | For ACE-Step |
|---|---|---|
| `TaskCatalog` row | `Generate Sound`, text→audio, refName `engines.diffusion.generate_sound` — `TaskCatalog.swift:188` | unchanged |
| executor path | prefix already in `servedRefNamePrefixes` — `TaskModels.swift:160` | unchanged (allow-list is per refName) |
| task→kind | `.music` — `TaskModels.swift:82` | unchanged |
| pool | `["MusicGen"]` — `TaskModels.swift:257` | **append the display name last** (owner-visible order; append-last discipline `:280-315`); do not remove MusicGen |
| catalog bridge | MusicGen row `:135-140` | new `BridgeEntry` with `equivalence: .same` (same official repo) + new manifest |
| curated manifest | `musicgen-small.json:1-39` | new `ace-step-1-5.json`, MLXUI-authored (no reference manifest exists; precedent `qwen3.5-9b-4bit.json`, `CatalogBridge.swift:214-224`) |
| settings delivery | `RealExecutor.stageConfig`'s diffusion branch parses seed/width/height/steps/scale/quantize only — `RealExecutor.swift:1181-1215`; `StageConfig` has no duration/lyrics — `StageFactory.swift:6-34` | **new `StageConfig.duration: Double?` and `lyrics: String?`**, parsed in that branch, `maps_to` names declared by the manifest; `duration` → `T = Int(duration × 25)` latents (upstream `conditioning_target.py:64`, `padding_utils.py:79-90`); lyrics → the lyric header string. Without this the manifest's only usable knobs are `seed`/`steps` and every run is the 120 s fallback (`padding_utils.py:96-100`) — a product bug, not a port detail |
| prompt | a no-input `engines.diffusion.generate_*` row uses its first bare setting — `RealExecutor.swift:544-556` | unchanged for the caption; **lyrics need a quoted `lyrics="…"` setting** (the parser handles quotes/newlines — `FlowSettings.unquote`, same machinery the rerank query uses `RealExecutor.swift:1226-1245`) |
| module registration | `App/ModelModules.swift:39` | add `ACEStepModule.self`; remove the ModelSupport gap (`Core/ModelSupport.swift:17-34`) |
| Run UI | MusicGenUI (`Modules/MusicGen/MusicGenUI.swift`) | ACE-Step Run UI with duration/lyrics/steps/seed/duration-cap controls; the stage refuses settings it cannot honour by name (the `FluxSDK`/`Flux2KleinSDK`/`RealExecutor.stageConfig` rule, CLAUDE.md "Doc drift") |

**MusicGen stays until the owner's listening review.** It is the incumbent `Generate Sound`
model: same display name, same task, same kind. Nothing in this plan removes, demotes or
replaces it; the ACE-Step card is additive and appears in the Model menu only after 3h. The
only ordering rule: **append**, never prepend (the existing seed for a new flow row must keep
resolving to MusicGen until the owner decides otherwise).

---

## 6. Open questions (Q12)

- **OQ-1 (install shape):** is an exact-name exception for `acestep-v15-turbo/silence_latent.pt`
  acceptable in `ModelFileSelector` (`:150-153`), or should the `.pt` be converted to
  `silence_latent.safetensors` at conversion time? The app cannot convert without a converter,
  so the exception (or a dedicated `silence_latent` name) is the conservative reading.
- **OQ-2 (DCW):** part of the turbo reference default; include in 3f or ship off and document
  the divergence? Recommendation: implement (small, deterministic), fixtures pin both.
- **OQ-3 (settings vocabulary):** `duration`/`lyrics` are not `StageConfig` fields and the
  curated manifest is "the settings authority … ported verbatim from catflow-mlx". An
  MLXUI-authored manifest that invents `duration` is inventing catalog semantics — this needs a
  `SPEC_QUESTIONS.md` entry in `catflow-mlx` (per CLAUDE.md's "if the spec is ambiguous")
  before 3h, not a silent addition. Recommended answer: add the two settings in the Python
  runtime too (both are first-class in the upstream `generate_music` API:
  `audio_duration`/`lyrics`, `handler/generate_music.py:198-215`). Ranges are upstream's:
  **duration 10…600 s** (`constants.py:58-59`), steps 1…20 with 8 the turbo default
  (`inference.py:145`). A row that omits `duration` hits the reference's 120 s fallback
  (`padding_utils.py:96-100`) — the manifest must carry a default, and the stage must refuse a
  missing duration rather than silently making a 2-minute song.
- **OQ-4 (tokenizer in a subfolder):** `HFTokenizerLoader` has been used for top-level and
  `tokenizer_2/`-style folders (FLUX T5 `spiece.model`, `ModelFileSelector.swift:47-50`), but
  not for a full Qwen3 folder inside `Qwen3-Embedding-0.6B/`. Verify its folder argument in 3c;
  if it only reads the model root, that is a small, named change.
- **OQ-5 (licence — `licence.md` is the authority):** the code is **MIT** (`LICENSE`,
  `Copyright (c) 2026 ACEStep`) and the weights repo's card declares **`license: mit`** (no
  `LICENSE` file in the repo, public, not gated). Two components are **Apache-2.0** and must be
  attributed, not relabelled: the bundled `Qwen3-Embedding-0.6B` (Qwen3 weights) and (if ever
  shipped) the 5 Hz LM's Qwen3-1.7B base. The **VAE is the open one**: `AutoencoderOobleck` is
  Stability AI's Oobleck VAE from Stable Audio Open; `licence.md` §1.5 says its provenance is
  unresolved and to **not silently label the VAE MIT** — before shipping, the owner decides
  (diff the weights / ship with the derivation question in the card's notes). Catalog `license`
  = `mit` with those notes; do not hide it.
- **OQ-6 (`ramGB` provenance):** the product rule is "measured numbers with their provenance in
  the entry's notes" (CLAUDE.md "Formulas"). If `measurements.json` is not ready when 3a lands,
  the card ships with what? Recommendation: do **not** ship the card until 3a has either the
  measurement or an explicitly-temporary `ramGB` marked provisional (SAM's MC1-12 did the
  measured number in the same release; there is no precedent for a placeholder diffusion row).
- **OQ-7 (process cap vs LM):** decision 4's 12 GB cap makes the with-LM config the one to
  justify in the issue; under the kind-row booking it does not fit (10.09 GB × 1.10 ≈ 11.1 GB
  before working set). If the owner wants LM quality, the staged-booking override (§3.3) is the
  only honest route, and it makes ACE-Step a special case in `EngineCache` like Klein. Flag,
  don't decide.
- **OQ-8a (source repo):** official `ACE-Step/Ace-Step1.5` (MIT declared, 6.34 GB bf16, proven by
  the reference) vs `mlx-community/ACE-Step1.5-MLX-4bit` (5.29 GB, MLX-branded, **no licence
  declared**, quality unverified, its README claims the LM is required). Recommendation: official
  repo for v1; revisit only on an owner ruling.
- **OQ-8b (does v1 need the LM at all?):** the official no-LM path runs and produced audio in the
  Lead's smoke (`lm: "none"`), but the mlx-community card says the 5 Hz LM is "required for the
  turbo model to produce music", and the LM is also where caption/lyric rewriting (CoT) happens.
  That is a **quality** question for the owner's listening review (MC5-3i), and the answer
  decides between configs D/E and the cap analysis. The plan's v1 (A/B) is the smallest thing
  that can be judged; do not pre-empt the review.
