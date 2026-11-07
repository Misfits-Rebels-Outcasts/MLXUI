# Q13 — Licence audit of every component a port would ship

**Method.** Read the pinned upstream `LICENSE`; read the downloaded repo's `README.md` and
`config.json`s; queried the Hugging Face API (`https://huggingface.co/api/models/<id>`, which
returns `cardData.license` and the file list) and fetched model-card `README.md` files at the
pinned revision; cross-checked the app's existing licence-display table. **No `gh api` was needed**
and nothing was downloaded beyond KB-sized metadata. Where a file is gated I say so rather than
guessing.

Recorded URLs (all verified live on 2026-11-07):

| Component | URL |
|---|---|
| upstream code repo | `https://github.com/ace-step/ACE-Step-1.5` (pinned `ca1e85f…`) |
| weights repo | `https://huggingface.co/ACE-Step/Ace-Step1.5` (pinned `19671f4…`) |
| text encoder (original) | `https://huggingface.co/Qwen/Qwen3-Embedding-0.6B` |
| 5 Hz LM base | `https://huggingface.co/Qwen/Qwen3-1.7B` |
| cross-check MLX conversion | `https://huggingface.co/mlx-community/ACE-Step1.5-MLX` |
| cross-check relabel | `https://huggingface.co/MLXCreator/MLXCreator-ACEStep-1.5-4bit` |

---

## 1. Component-by-component

### 1.1 Upstream code — `ACE-Step/ACE-Step-1.5`

- File: `/private/tmp/mc5/ACE-Step-1.5/LICENSE` (21 lines).
- **SPDX: `MIT`.** Line 1 `MIT License`, line 3 **`Copyright (c) 2026 ACEStep`**.
- `README.md:293-295`: "This project is licensed under [MIT](./LICENSE)", followed by a
  **non-binding** responsible-use disclaimer (verify originality of generated works, disclose AI
  involvement, respect artist rights). That paragraph is prose guidance, **not** an acceptable-use
  term in the licence grant — the grant is plain MIT.
- MIT code is App-Store-compatible (permissive, no field-of-use restriction, no source-disclosure
  duty). Obligation is the standard one: ship the copyright + permission notice (§ "The above
  copyright notice and this permission notice shall be included in all copies or substantial
  portions of the Software").

### 1.2 Weights — `ACE-Step/Ace-Step1.5` @ `19671f4`

- HF API `cardData`: **`{"library_name": "transformers", "license": "mit", "pipeline_tag": "text-to-audio", …}`**;
  tag `license:mit`.
- **There is no `LICENSE` file inside the HF repo.** The 28 sibling files are only `.gitattributes`,
  `README.md`, `config.json`, the four model dirs, and `vae/`. So the weights' licence is declared
  **only** by the model-card YAML front-matter (`license: mit`) and the README's
  "**License: [MIT]**" line.
- Repo is **public, not gated** (`gated: false`) — the app can fetch it with a plain HF token or
  none.
- The vendor's commercial claim is explicit in `README.md`: "💰 **Commercial-Ready:** … You can
  strictly use the generated music for **commercial purposes**", and the training data is
  described as licensed / royalty-free / synthetic.

### 1.3 Text encoder — `Qwen3-Embedding-0.6B`

- The copy in the ACE-Step repo is byte-identically the upstream Qwen model (same 12-file layout,
  same `config.json` with `architectures: ["Qwen3Model"]`, `vocab_size: 151669`,
  `hidden_size: 1024`).
- Original repo: **`Qwen/Qwen3-Embedding-0.6B`** — API `cardData.license` = **`apache-2.0`**,
  tag `license:apache-2.0`, **not gated**. Base model listed as `Qwen/Qwen3-0.6B-Base`.
- Verified by measurement that the file in `Ace-Step1.5/Qwen3-Embedding-0.6B/model.safetensors`
  is a **full, standalone 595.78 M-param Qwen3 trunk** (`findings/weights.md` §2) — i.e. Qwen
  weights are being redistributed inside an MIT-labelled repo.
- **Apache-2.0 is compatible with the MIT labelling** (Apache-2.0 §4 permits derivative works to
  be distributed under other terms provided the Apache notices are retained and changes stated),
  so this is **not a licence conflict** — but it is a **mislabel**: the correct statement is
  *"the ACE-Step repo is MIT; the Qwen3-Embedding-0.6B weights it bundles remain Apache-2.0 and
  carry Qwen's copyright/NOTICE."* Apache-2.0 is commercial-use and App-Store-friendly
  (no field-of-use limit; requires licence + NOTICE + change statement if redistributed).

### 1.4 5 Hz LM — `acestep-5Hz-lm-1.7B`

- The README's own model-zoo table (`README.md`, "LM Models") says **"Pretrain from: Qwen3-1.7B"**.
- **`Qwen/Qwen3-1.7B` is `license: apache-2.0`, not gated** (verified via API).
- So the 5 Hz planner is an **Apache-2.0 derivative** of Qwen3-1.7B, fine-tuned by ACE-Step.
  It inherits Apache-2.0 obligations (keep Qwen's copyright + NOTICE, state modifications), which
  the MIT-only card does not record. Same conclusion as §1.3 — compatible in practice, incomplete
  as labelled.
- Measured: it really is a Qwen3-shaped trunk (28 layers, 2048 hidden, 310 tensors, no `lm_head`,
  tied embeddings) — consistent with "Qwen3-1.7B pretrain", so treat the derivation as fact.

### 1.5 VAE — `vae/diffusion_pytorch_model.safetensors` ⚠️ **the one real licence question**

- `vae/config.json` declares **`_class_name: "AutoencoderOobleck"`** — that is
  **Stability AI's Oobleck VAE**, the autoencoder from `stable-audio-tools` used in
  **Stable Audio Open 1.0**. `_name_or_path: "/root/data/repo/gongjunmin/ACE-Step-1.5/checkpoints/vae/"`
  is a local training path, not a Stability path.
- Upstream code loads it as a stock diffusers class:
  `AutoencoderOobleck.from_pretrained(vae_checkpoint_path)`
  (`acestep/core/generation/handler/init_service_loader_components.py:46,53`).
- **The code side is fine**: the `AutoencoderOobleck` implementation is Apache-2.0 in `diffusers`
  and MIT in `stable-audio-tools`; an architecture is not the licensed artefact.
- **The weight side is not evidenced either way:**
  - ACE-Step declares MIT for the whole repo and ships **no third-party notice, no NOTICE file,
    no provenance statement** for `vae/` (grep for `oobleck|stability|stable.audio|third.party|NOTICE`
    across `*.py|*.md|*.txt|*.toml` returns only code comments and the diffusers import — nothing
    licence-related).
  - **Counter-evidence to derivation:** this VAE is **48 kHz** (`vae/config.json`
    `"sampling_rate": 48000`), whereas Stable Audio Open 1.0 is a 44.1 kHz model. A retrained
    VAE at a different sample rate is the natural reading.
  - **I could not compare against Stable Audio Open's own `vae/config.json` or `LICENSE.md`,
    because `stabilityai/stable-audio-open-1.0` is gated** (`gated: auto`, `license: other`) and
    the raw files returned "Access to model … is restricted". So the shape-by-shape comparison
    is **not done** and I am not asserting non-derivation.
- **Why it matters:** Stable Audio Open is released under the **Stability AI Community License**
  (`license: other`; the family has non-commercial / revenue-capped terms and acceptable-use
  restrictions). If the ACE-Step VAE is fine-tuned from Stability weights, then (a) the ACE-Step
  MIT label is wrong for `vae/`, and (b) a **paid / commercial App Store build could be
  restricted** — exactly the class of problem the app already tracks with its
  `stability-ai-community-license` display entry
  (`MLXUI/Normalization/DataNormalizer.swift:19`).
- **Verdict: open.** Recommended action before shipping (cheap): (1) diff the ACE-Step VAE
  tensors against `stable-audio-open-1.0`'s `vae/` safetensors with a gated token — identical
  values ⇒ derivative, licence follows Stability; (2) otherwise treat it as ACE-Step's own
  (MIT) but record the derivation question in the model entry's notes. Do **not** silently label
  the VAE MIT.

### 1.6 `silence_latent.pt`

- Part of the same HF repo and the same `license: mit` card. 3.84 MB f32 tensor
  (`findings/weights.md` §3, buffer #1). Nothing to audit beyond the checkpoint's own licence;
  it must be shipped/downloaded alongside.

### 1.7 Tokenizer files (`Qwen3-Embedding-0.6B/{tokenizer,merges,vocab,added_tokens,...}`)

- These are Qwen3 tokenizer assets, Apache-2.0, same provenance as §1.3. The
  `chat_template.jinja` ships with it but is **not used** by ACE-Step inference (see
  `text-encoder.md` §1.3) — so a port need not ship it.

### 1.8 Third-party Python dependencies (not shipped, for completeness)

`pyproject.toml` pulls `diffusers`, `transformers`, `vector_quantize_pytorch`, `mlx`, `mlx-lm`,
`peft`, `lycoris-lora`, `modelscope`, `nano-vllm`, `pytorch-wavelets`… **None of this ships in a
Swift port**; only the *weights* and the *algorithm* do. `vector_quantize_pytorch` matters only
because the FSQ *levels* come from it — levels are arithmetic (no code copied), and its licence
(MIT) is irrelevant once reimplemented. Note for the port plan: the FSQ quantizer must be
reimplemented from the levels list; do not vendor its code without checking its licence first.

---

## 2. Cross-check checkpoint — `MLXCreator/MLXCreator-ACEStep-1.5-4bit` (for the record only)

Measured via HF API + raw model card:

| Field | Value |
|---|---|
| `cardData.license` | **`apache-2.0`** |
| tag | `license:apache-2.0` |
| gated / private | `false` / `false` |
| siblings | 13 files |
| model card body | "**Based on:** mlx-community/ACE-Step1.5-MLX" — *"This is an MLX repackaging/conversion; the original credit above is retained per its license."* |
| `config.json` | `architectures: ["AceStepConditionGenerationModel"]`, same `auto_map`, same `fsq_input_levels`, `hidden_size 2048`, `in_channels 192` — i.e. it is a **4-bit repack of the same ACE-Step 1.5 turbo graph**, not a new model |

And the repo it credits:

| `mlx-community/ACE-Step1.5-MLX` | Value |
|---|---|
| `cardData` | `{"library_name": "mlx", "tags": ["mlx", "music-generation", "ace-step", "audio", "text-to-music"], "base_model": "ACE-Step/ACE-Step1.5"}` |
| licence field | **absent — no `license` in `cardData`, no `license:*` tag, no LICENSE file in the 13 siblings** |
| files | `model.safetensors`, `vae/diffusion_pytorch_model.safetensors`, `Qwen3-Embedding-0.6B/model.safetensors`, **`silence_latent.npy`** (note: `.npy`, not `.pt`), configs, tokenizers |
| model card | "MLX-converted weights for ACE-Step/ACE-Step1.5, a 3.5B parameter flow-matching … Converted using mlx-audio"; **no licence section at all** |

**What this tells us (chain of labels):**

> upstream **MIT** → mlx-community conversion: **no licence declared** → MLXCreator: **Apache-2.0**

- The MLXCreator Apache-2.0 label is **not a violation** in itself: the upstream is MIT, and MIT
  explicitly permits sublicensing, so re-releasing a conversion under Apache-2.0 is allowed —
  **provided the upstream copyright + permission notice is retained**. Its card does carry an
  attribution line to mlx-community, but **not** the ACE-Step copyright notice, and the
  mlx-community link it points to has **no licence to retain**.
- For our purposes: **do not treat either MLX community repo as the licensing reference.** Take
  the licence from `ACE-Step/Ace-Step1.5` (MIT) + `Qwen/Qwen3-Embedding-0.6B` and
  `Qwen/Qwen3-1.7B` (Apache-2.0). The MLX repos are useful as an **independent implementation
  reference** (and `mlx-community/ACE-Step1.5-MLX` is served by `Blaizzy/mlx-audio`, a Python MLX
  port we can read) — not as a licence source.
- Also worth noting for the port: the mlx-community card repeats the same functional claim our
  plan depends on — *"The turbo model uses a 5Hz Language Model planner by default (`use_lm=True`)
  … This is required for the turbo model to produce music."* So the 5 Hz LM is a **required**
  component, not optional; its Apache-2.0 obligations therefore apply to any shipped runtime.

---

## 3. Summary table

| Component | Declared licence | Actually governed by | Commercial / App Store OK? |
|---|---|---|---|
| ACE-Step 1.5 code (upstream repo) | MIT (`Copyright (c) 2026 ACEStep`) | MIT | ✅ yes |
| `acestep-v15-turbo/model.safetensors` | MIT (card only, no LICENSE file) | MIT | ✅ yes |
| `acestep-5Hz-lm-1.7B/model.safetensors` | MIT (card only) | **Apache-2.0** (Qwen3-1.7B derivative) | ✅ yes, with Apache notice |
| `Qwen3-Embedding-0.6B/*` | MIT by association (bundled) | **Apache-2.0** (Qwen) | ✅ yes, with Apache notice |
| `vae/diffusion_pytorch_model.safetensors` | MIT (card only) | **UNDETERMINED** — Oobleck architecture; possibly Stability AI Community License | ⚠️ **needs verification** |
| `silence_latent.pt` | MIT | MIT | ✅ yes |
| ACE-Step tokenizer assets | MIT | MIT (GPT-2-style BPE) | ✅ yes |
| Qwen3 tokenizer assets | bundled | Apache-2.0 | ✅ yes |

---

## 4. What would block an App Store build

**Nothing categorically blocks it today, but three items need action.**

1. **VAE provenance (§1.5) — the only *potential* blocker.** If the VAE turns out to be a
   Stability-weights derivative, the Stability AI Community Licence's acceptable-use terms and
   revenue cap would attach, and a **paid** App Store product would be in restricted territory
   (this is precisely why the app already renders that licence specially:
   `MLXUI/Normalization/DataNormalizer.swift:19`). Resolve before shipping; it is a 10-minute
   tensor comparison once a gated token is available. Everything else in the table is permissive.
2. **Attribution, not compatibility.** MIT requires the copyright + permission notice in
   "substantial portions"; Apache-2.0 requires the licence text + NOTICE + a statement of changes.
   The app does not currently show per-model licence text, only a label
   (`DataNormalizer.swift:12-21`). Since the app **downloads** the weights from HF at runtime
   rather than bundling them, the redistribution trigger is lighter — but the safest, cheap
   reading (and the habit already in the catalog) is: for each of the ACE-Step entries store
   `license: "mit"` **plus** a note naming Qwen3-Embedding-0.6B / Qwen3-1.7B as Apache-2.0
   components, and surface both in the model detail view.
3. **Trademark / branding.** The model card carries the ACE-Step name and logo, and the project is
   "co-led by ACE Studio and StepFun". MIT grants no trademark rights. Do **not** ship the
   ACE-Step or StepFun logos/wordmarks as if they were ours, and do not name the app after the
   model. Using the name to *describe* a model entry is normal nominative use.

Non-issues, explicitly:

- **No "no-source remote code" problem.** The upstream inference path uses
  `AutoModel.from_pretrained(..., trust_remote_code=True)`
  (`acestep/core/handler/init_service_loader.py:173-179`), which is a *Python* trust boundary and
  a supply-chain smell — but a **Swift port executes no remote code at all**: it reads four
  safetensors + one `.pt` and runs our own MLX graph. The remote Python file
  (`acestep-v15-turbo/modeling_acestep_v15_turbo.py`) is **reference source we read**, not code
  the app would run. Nothing about App Review's "no downloaded executable code" concern attaches
  to model weights (they are data, not code) — the same position the app already takes for every
  other MLX model it downloads.
- **The README disclaimer is not a licence restriction** (§1.1) — it is advisory prose, unlike an
  acceptable-use policy embedded in a licence (e.g. RAIL++-M, OpenRAIL). Do not treat it as
  binding, but do keep the generated-audio provenance guidance in the app's copy for the same
  reason the vendor wrote it.
- **No copyleft, no share-alike, no field-of-use limits** in MIT / Apache-2.0 → App Store
  distribution and sandboxing are unaffected. No new entitlement, no network behaviour, no
  encryption-export question (weights are not cryptography).

---

## 5. Open questions to resolve (owner / next cycle)

1. **VAE provenance** (§1.5) — compare `vae/diffusion_pytorch_model.safetensors` against
   `stabilityai/stable-audio-open-1.0/vae/` with a gated token, or ask upstream to state it. This
   is the only item that could change the shipping decision.
2. **Apache-2.0 bookkeeping for the Qwen3 components** — decide whether to (a) label the catalogue
   entries `mit` + an "includes Apache-2.0 Qwen3 weights" note, or (b) label the *bundle*
   `apache-2.0`, which is the more accurate but less useful display. Recommend (a), and add the
   Qwen NOTICE text to whatever credits surface the app grows.
3. Whether the app will **bundle** any of these weights (it should not — 10 GB) or always
   download them; if it ever bundles, the NOTICE obligations above become hard requirements
   inside the app bundle rather than just UI copy.
