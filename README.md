# MLXUI &middot; Local AI Browser

*Run MLX models visually — no terminal, no Python, no command-line flags.*

[![Platform](https://img.shields.io/badge/platform-macOS%2014.0+-blue)](https://developer.apple.com/macos/)
[![Swift](https://img.shields.io/badge/Swift-5.9+-orange)](https://swift.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-required-333333)](https://developer.apple.com/macos/)

<a href="https://apps.apple.com/us/app/mlxui-local-llm-ai-browser/id6787626220">
  <img src="https://tools.applemediaservices.com/api/badges/download-on-the-mac-app-store/black/en-us?size=250x83" alt="Download on the Mac App Store" height="60">
</a>

MLXUI is the **SwiftUI user interface layer for MLX models** — a native macOS app built on
[`mlx-swift`](https://github.com/ml-explore/mlx-swift) that lets you browse, install, and
run local AI models on Apple Silicon. It ships a curated catalog of MLX-optimized models
from Hugging Face's [`mlx-community`](https://huggingface.co/mlx-community), downloads
them with one click, and provides a purpose-built Run UI for each model type.

> **The mission:** become the UI layer for every MLX model on Hugging Face — the place
> where anyone on a Mac can discover, download, and try out local AI without ever opening
> a terminal.

---

## ✨ What's New

<!-- Items marked *(hidden in-app)* are built but switched off behind an in-app flag right
     now — pruning candidates if you don't want to announce them before they're switched on. -->

### AI Workflows — automate your models with a visual, numbered flow

A brand-new way to chain your installed models together. Reached from **Automate → AI
Workflows**, a flow is a simple numbered list — each row picks a task (and a model, if it
needs one) — that runs top to bottom, with live progress dots and cancel. Comes with a
gallery of ready-made sample flows, plus save, open, import and export for your own.

- **~40 built-in row types** — text tools (split, filter, dedupe, template), image tools
  (resize, crop, convert, watermark), record/table tools, document retrieval (build and
  search an index), OCR and image-description rows, and more
- **Per-row output caching** — re-running a flow only redoes the rows that actually changed
- **Human-in-the-loop rows** — a flow can pause and wait for you to type an answer before
  continuing, with an optional timeout
- **Remote model providers** — a row can call Claude, GPT, DeepSeek, or a LAN endpoint using
  your own API key (stored in Keychain), or run certain tasks fully on-device via Apple
  Intelligence; every remote row is clearly marked before it can send anything off-device
  *(the Settings tab for entering provider keys is hidden in-app right now)*
- **Web Search row** — search the web from inside a flow with your own Tavily or Brave key
- **Workspaces** — group related flows in one folder that share files; pair a "build" flow
  (indexes your documents) with an "ask" flow (queries them) for a one-card document-Q&A
  setup, with a bundled "Ask Your Docs" starter
- **Missing-file warnings** — a flow now tells you up front if a file a row needs isn't
  there, instead of failing partway through the run
- **In-flow file picker** — reuse a file already in the flow, add a new one, or create a
  folder, instead of a bare system file dialog
- **A real Settings window** — tabs for Models (storage usage, clear cache), Providers,
  Tools (every agent capability, with an audit log), and Privacy (a plain-language summary
  of what can leave your Mac) *(the Providers and Privacy tabs are hidden in-app right now)*

### SAM3 — image segmentation

Click points on an image and SAM3 produces a mask you can save as a PNG. Also usable as a
"Segment" step inside AI Workflows.

### WAN 2.1 — text-to-video generation *(hidden in-app)*

Type a prompt, get a short generated video, exported to MP4 with a built-in player.

### SeedVR2 — image upscaling *(rolling out)*

A 3B-parameter upscaling model, added as a new Browse entry.

### New models in the catalog

- **GLM-OCR** — a new OCR model, usable in Browse and in AI Workflows
- **Qwen3.5-9B** and **Qwen3.5-9B Vision** — new chat and vision models
- **Rerank / Qwen3-Reranker-0.6B** *(hidden in-app)* — reorders search results inside
  retrieval flows for better relevance
- **Faster, cheaper installs** — if two catalog entries share the same underlying model
  files, installing one no longer re-downloads it for the other

### MusicGen — type a description, get music back

MusicGen-small is now in the catalog. Type a musical description (e.g. "happy rock with drums"),
pick Short / Medium / Long, and the model generates a WAV file entirely on-device using Apple
Silicon. Playback is inline; you can also Save WAV to keep the file.

The engine is a bespoke Swift/MLX port of the MusicGen autoregressive pipeline: a T5-base text
encoder conditions a 24-layer causal transformer that predicts 4 interleaved EnCodec codebook
streams, which are then decoded by a GPU-accelerated EnCodec decoder back to 32 kHz mono audio.
The whole run — weights, encode, decode loop, audio decode — stays on-device.

Model: [`jasonvassallo/mlx-musicgen-small`](https://huggingface.co/jasonvassallo/mlx-musicgen-small)
(≈ 2.5 GB install including the bundled 32 kHz EnCodec weights, CC-BY-NC-4.0,
non-commercial use only).

### SDXL-Turbo — a second image-generation engine, 4-step on-device diffusion

Stability AI's SDXL-Turbo joins FLUX.1 in the Image Generation category. It's a from-reference
Swift/MLX port of the SDXL UNet, dual CLIP text encoders (CLIP-L + OpenCLIP ViT-G/14), and the
AutoencoderKL VAE, driven by a 4-step DDIM sampler with no classifier-free guidance — a text
prompt becomes a 512×512 image in seconds, entirely on-device. Like Flux, it lives in its own
isolated module folder (`Modules/StableDiffusion/`) so it can evolve independently of every
other model type.

### FLUX.1 Image Generation — text prompt → on-device image, no Python

FLUX.1-Lite-8B (8-bit) is now in the catalog with a full native Run UI. Type a prompt,
hit **Generate**, and watch a 512×512 image appear — CLIP + T5 text encoding, a native
Flux transformer, and VAE decode all running on Apple Silicon via MLX. No Python
environment, no API keys, no round-trips to the cloud.

The image module follows the same isolated-module pattern as Whisper and Kokoro: the
Flux folder is self-contained and can be updated independently of every other model type.

### Cancel Install — mid-download cancel now reliably cleans up

Canceling a model install while it was downloading could leave stale files in the
temp `downloads/` folder. That's fixed: canceling now atomically removes the partial
download so disk space is always reclaimed correctly.

### Agentic Chat — Local AI that can actually *do* things

Chat models can now call tools mid-conversation. A single local model can browse the web,
run calculations, transcribe audio, read images, and work with files you grant — all on-device,
all private. Every tool call is shown transparently; anything with side-effects waits for
your explicit approval before it runs.

| Tool | What it does | Requires approval |
|------|-------------|:-----------------:|
| `fetch_url` | Fetch any webpage and return its text | — |
| `calculator` · `datetime` · `run_javascript` | Math, dates, and JS evaluation | — |
| `embed_text` · `summarize` · `semantic_search` | Use your installed embedding and LLM models as tools | — |
| `transcribe_audio` | Transcribe an audio URL with any installed Whisper model | — |
| `ocr_image` | Extract text from an image URL using any installed OCR model | — |
| `read_file` · `list_directory` · `search_files` | Browse and read inside folders you explicitly grant | — |
| `write_file` | Write files inside granted folders | ✅ |
| `write_clipboard` | Copy text to the clipboard | ✅ |
| `run_shell` *(Direct build only)* | Run any shell command on your Mac | ✅ |

### Ternary-Bonsai-27B — 27B parameters in ~8.4 GB

Prism ML's [Ternary-Bonsai-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-mlx-2bit) — a
2-bit hybrid GatedDeltaNet model — is now in the catalog. Download it in one click and run a full
27B-parameter chat model in roughly 8.4 GB of RAM.

### Two builds, one codebase

| | App Store | Direct Distribution |
|---|---|---|
| **Distribution** | Mac App Store | Notarized standalone `.app` |
| **Sandbox** | ✅ Full sandbox | — |
| **File tools** | Scoped to folders you grant | Full filesystem (denylist-guarded) |
| **Shell tool** | — | ✅ `run_shell`, approval-gated |

---

## What this is

**The ultimate native SwiftUI cockpit for local AI, built exclusively for MLX and Apple Silicon.**

There are thousands of MLX models on Hugging Face. Using them today means finding the
right repo, reading the README, managing Python dependencies, figuring out CLI flags,
and converting formats when things don't match.

MLXUI replaces all of that with a single app. Browse by category (Chat & Text, Vision,
OCR, Speech-to-Text, Text-to-Speech, Embeddings), filter by what fits your hardware,
install with a click, and run each model in a purpose-built interface — chat for LLMs,
a microphone and transcript for ASR, a voice picker for TTS, an image dropwell for OCR.

Under the hood, MLXUI wraps Apple's `mlx-swift` and `mlx-libraries` (MLXLLM, MLXVLM,
MLXWhisper, MLXAudioTTS, etc.) in a modular SwiftUI architecture. Each model type gets
its own isolated module folder — updating Whisper never touches Kokoro, and a
contributor adding image-generation support doesn't need to understand the chat pipeline.

---

## Screenshots

<!-- TODO: add screenshots once you have them -->

| Browse | Model Detail |
|--------|-------------|
| ![Browse](docs/screenshots/MLXUI_Local_AI_Browser.png) | ![Detail](docs/screenshots/MLX_UI.png) | 

---

## Features

- **Browse models** organized by category — Chat & Text, Vision, OCR, Speech, Embeddings
- **Hardware-aware filtering** — see which models fit your Mac's RAM and chip bandwidth
  before you install
- **One-click install** — downloads model files directly from Hugging Face with progress
  tracking and atomic install markers
- **Comparison mode** — compare models side-by-side by RAM, speed, and quality
- **Dedicated Run UIs** — each model type gets a purpose-built interface, not a generic
  wrapper:
  - LLMs → chat
  - ASR (Whisper, Voxtral) → microphone + live transcript
  - TTS (Kokoro, MLXAudioTTS, Qwen3-TTS, Chatterbox, Orpheus) → voice picker + audio player
  - Vision (VLM) → image upload + Q&A
  - OCR (PaddleOCR, DeepSeek-OCR, dots.ocr, olmOCR) → image dropwell + extracted text
  - Embeddings → text input + vector output
  - Image generation (Flux, SDXL-Turbo) → text prompt + on-device generated image
  - Music generation (MusicGen) → text description + on-device generated WAV
- **Pipeline runner** — chain models together (transcribe → summarize → speak) in a
  single workflow
- **Command palette (⌘K)** — find and run any model instantly
- **100% local** — everything runs on-device. No data leaves your machine.
- **No API keys, no per-token pricing** — models download once and run forever

---

## Supported Models

### browser.json — MVP catalog (30 models)

The app ships `browser.json`, a hand-curated set of 30 models across 8 categories.
Every model listed here is downloadable and has a working Run UI.

| Category | Models | Sizes |
|----------|--------|-------|
| **Chat & Text** (9) | Gemma 3, Ministral 3, Qwen3, Llama 3.1, Gemma 2, Qwen2.5, GPT-OSS, Devstral-Small 2, **Ternary-Bonsai-27B** | 1B – 27B |
| **Vision** (4) | LFM2-VL, Gemma 3, Qwen3-VL | 1.6B – 12B |
| **OCR** (4) | PaddleOCR-VL, DeepSeek-OCR-2, dots.ocr, olmOCR 2 | 255M – 7B |
| **Speech-to-Text** (4) | Whisper tiny/small/large-v3, Voxtral-Mini | 37M – 1.5B |
| **Text-to-Speech** (4) | Kokoro, Qwen3-TTS, Chatterbox, Orpheus | 82M – 3B |
| **Embeddings** (4) | all-MiniLM-L6, embeddinggemma, ModernBERT-embed, bge-m3 | 18M – 568M |
| **Image Generation** (2) | FLUX.1-Lite-8B, **SDXL-Turbo** | 3.5B – 8B |
| **Music Generation** (1) | **MusicGen-small** | 615M (float32, ~2.5 GB) |

The full `browser.json` catalog tracks **~435 models** from mlx-community. New models
and Run UIs are added with every release.

### Engine coverage

| Category | Type | Browse | Install | Run UI | Engine |
|----------|------|:---:|:---:|:---:|--------|
| Text | LLM (Llama, Qwen, Gemma, Phi, ...) | ✅ | ✅ | ✅ | MLXLLM |
| Speech | ASR — WhisperKit | ✅ | ✅ | ✅ | WhisperKit |
| Speech | ASR — MLX Whisper | ✅ | ✅ | ✅ | MLXWhisper |
| Speech | ASR — Voxtral | ✅ | ✅ | ✅ | Voxtral |
| Speech | TTS — Kokoro | ✅ | ✅ | ✅ | kokoro-swift |
| Speech | TTS — MLXAudioTTS (Qwen3-TTS, Chatterbox, Orpheus) | ✅ | ✅ | ✅ | mlx-audio-swift |
| Vision | VLM (Gemma, Qwen VL, LFM2-VL) | ✅ | ✅ | ✅ | MLXVLM |
| Vision | OCR (PaddleOCR, DeepSeek-OCR, dots.ocr, olmOCR) | ✅ | ✅ | ✅ | MLXVLM + native |
| Embeddings | Text embeddings | ✅ | ✅ | ✅ | MLX + ModernBERT |
| Image | Diffusion (Flux) | ✅ | ✅ | ✅ | Flux (native MLX) |
| Image | Diffusion (SDXL-Turbo) | ✅ | ✅ | ✅ | SDXL-Turbo (native MLX) |
| Audio | Music generation (MusicGen) | ✅ | ✅ | ✅ | MusicGen (native MLX) |

✅ = built &emsp; ⬜ = available for contribution

---

## Architecture

MLXUI uses a **module system** where each model type lives in its own isolated folder.
Adding a new model type touches only that model's folder — updating Whisper never
touches Kokoro, and a contributor adding image-generation support doesn't need to
understand the chat pipeline.

```
Modules/
  Whisper/              ← one model type, one folder
    WhisperModule.swift   registers the module with the ModelRegistry
    WhisperEngine.swift   wraps the MLX inference
    WhisperSDK.swift      factory that creates the pipeline stage
    WhisperUI.swift       the SwiftUI Run view
    WhisperStage.swift    pipeline stage (input/output types for chaining)
```

**The rule:** each module may import `Core` (shared types, protocols) and its own
external package — never another model's module. All cross-model knowledge lives in
`Core` + the `ModelRegistry`, which are small and rarely change.

Full spec: [`Design/aisdk-aiui-architecture.md`](Design/aisdk-aiui-architecture.md)

See also: [`StandAloneRunner.md`](StandAloneRunner.md) for per-model file isolation,
storage layout, and engine-to-model dispatch.

---

## Getting Started

### Requirements

- macOS 14.0 (Sonoma) or later
- Apple Silicon (M1 or later)
- Xcode 15.0+

### Build

```bash
git clone https://github.com/your-org/MLXUI.git
cd MLXUI
open PipelineStudio.xcodeproj
```

Two schemes are available:

| Scheme | Distribution | Sandbox |
|--------|-------------|:-------:|
| **MLXUI** | Mac App Store | ✅ sandboxed |
| **MLXUI-Direct** | Notarized standalone | — |

Select your preferred scheme, pick **My Mac**, and press ⌘R.

### First run

1. The app opens with a catalog of ~28 models organized by category
2. Browse or search with ⌘K
3. Click **Install** on any model — it downloads from Hugging Face
4. Once installed, click **Run** to open the model's dedicated interface
5. Installed models live under `~/Library/Application Support/AI Browser/models/`

---

## Contributing

The core contribution path: **add a Run UI for a model type that doesn't have one yet.**
Every model in the catalog should eventually have a working Run button.

### How to add a new model type

1. Pick an unsupported row from the [Engine coverage](#engine-coverage) table
   above (the ⬜ rows)
2. Read the module architecture spec:
   [`Design/aisdk-aiui-architecture.md`](Design/aisdk-aiui-architecture.md)
3. Use an existing module as a template — Whisper is the simplest complete example
4. Create a new folder under `Modules/` with these files:

   | File | What it does |
   |------|-------------|
   | `XModule.swift` | Registers the module: declares model family, provides SDK + UI |
   | `XEngine.swift` | Wraps the MLX inference — loads model, runs it, returns output |
   | `XSDK.swift` | Factory that creates a `PipelineStage` for pipeline chaining |
   | `XUI.swift` | The SwiftUI Run view — what the user sees when they click Run |
   | `XStage.swift` | Pipeline stage — declares input/output types for chaining |

5. Register your module in [`App/ModelModules.swift`](PipelineStudio/App/ModelModules.swift)
6. Open a PR

### Good first issues

- **Image generation Run UI** (SD 2.1, etc.) — Flux and SDXL-Turbo are now done; other
  diffusion architectures still need a prompt → image Run view
- **More music generation models** — MusicGen-small is done; medium and large variants need
  catalog entries and engine config updates
- **Pipeline stages** — wire up existing modules into longer chains (e.g. OCR →
  summarize)
- **Model architecture ports** — PaddleOCR, DeepSeek-OCR, dots.ocr (see
  [`ModelSupport.swift`](PipelineStudio/Core/ModelSupport.swift))
- **Add catalog entries** — pick an architecture from the [To be supported](#to-be-supported--architectures-in-mlx-swift-lm-not-yet-in-browser2json)
  table above. 54 of 68 architectures already have ready-to-download 4-bit quants
  (marked ✅) and 2 more have bf16 uploads (marked 🟡). Find the model on
  [Hugging Face](https://huggingface.co/mlx-community), add it to `browser2.json`,
  and open a PR.

### Project conventions

- SwiftUI + `async/await` — no Combine
- `@Observable` macro for state (macOS 14.0 minimum)
- 4-space indent, PascalCase types, camelCase members
- Keep changes scoped — one model type per PR
- Build must stay green: open the project in Xcode and use Product → Build (⌘B)

---

## Roadmap

### Done
- [x] Browse, filter, and compare 435+ models
- [x] Hardware-aware RAM/bandwidth filtering
- [x] One-click install from Hugging Face with progress tracking
- [x] Gated-repo auth flow (Hugging Face token via Keychain)
- [x] LLM chat Run UI
- [x] ASR Run UI (WhisperKit + MLX Whisper + Voxtral)
- [x] TTS Run UI (Kokoro + MLXAudioTTS — Qwen3-TTS, Chatterbox, Orpheus)
- [x] VLM Run UI (image Q&A — Gemma, Qwen VL, LFM2-VL)
- [x] OCR Run UI (PaddleOCR, DeepSeek-OCR, dots.ocr, olmOCR)
- [x] Embeddings Run UI (including ModernBERT standalone)
- [x] Pipeline runner (audio → transcribe → summarize → speak)
- [x] Command palette (⌘K) and keyboard shortcuts
- [x] Agentic chat — 9 local tools (web fetch, compute, MLX model calls, file access, shell)
- [x] Ternary-Bonsai-27B — 27B 2-bit hybrid attention model in ~8.4 GB RAM
- [x] Dual build targets — App Store (sandboxed) + Direct Distribution (notarized standalone)
- [x] Image generation Run UI — FLUX.1-Lite-8B: text prompt → on-device image via native MLX Flux module
- [x] Fixed cancel-install mid-download leaving stale files in `downloads/`
- [x] SDXL-Turbo Run UI — a second image-generation engine: SDXL UNet + dual CLIP encoders +
      4-step DDIM sampler, native MLX
- [x] MusicGen Run UI — text description → on-device WAV; T5-base encoder + 24-layer causal
      transformer + GPU-accelerated EnCodec decoder, native MLX port

### In progress
- [ ] Visual pipeline builder UI
- [ ] Improved gated-model auth flow

### Up for grabs
- [ ] Image generation Run UI (SD 2.1, etc.)
- [ ] Additional music generation models (MusicGen-medium, MusicGen-large)
- [ ] Model comparison benchmark runner
- [ ] Export pipeline as standalone app
- [ ] Architecture ports (PaddleOCR, DeepSeek-OCR, dots.ocr — flagged in
      [`ModelSupport.swift`](PipelineStudio/Core/ModelSupport.swift))
- [ ] Populate `browser2.json` with entries for the 68 unsupported mlx-swift-lm
      architectures (54 with existing 4-bit quants, 2 bf16-only, 12 need model conversion).
      See the [reconciliation table](#to-be-supported--architectures-in-mlx-swift-lm-not-yet-in-browser2json) above.

---

## License

MIT — see [LICENSE](LICENSE).
