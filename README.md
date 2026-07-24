# MLXUI &middot; Local AI Workbench

*Run MLX models visually — no terminal, no Python, no command-line flags.*

[![Platform](https://img.shields.io/badge/platform-macOS%2014.0+-blue)](https://developer.apple.com/macos/)
[![Swift](https://img.shields.io/badge/Swift-5.9+-orange)](https://swift.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-required-333333)](https://developer.apple.com/macos/)

MLXUI is the **SwiftUI user interface layer for MLX models** — a native macOS app built on
[`mlx-swift`](https://github.com/ml-explore/mlx-swift) that lets you browse, install, and
run local AI models on Apple Silicon. It ships a curated catalog of MLX-optimized models
from Hugging Face's [`mlx-community`](https://huggingface.co/mlx-community), downloads
them with one click, and provides a purpose-built Run UI for each model type.

> **The mission:** become the UI layer for every MLX model on Hugging Face — the place
> where anyone on a Mac can discover, download, and try out local AI without ever opening
> a terminal.

---

## What this is

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
- **Pipeline runner** — chain models together (transcribe → summarize → speak) in a
  single workflow
- **Command palette (⌘K)** — find and run any model instantly
- **100% local** — everything runs on-device. No data leaves your machine.
- **No API keys, no per-token pricing** — models download once and run forever

---

## Supported Models

### browser.json — MVP catalog (28 models)

The app ships `browser.json`, a hand-curated set of 28 models across 7 categories.
Every model listed here is downloadable and has a working Run UI.

| Category | Models | Sizes |
|----------|--------|-------|
| **Chat & Text** (8) | Gemma 3, Ministral 3, Qwen3, Llama 3.1, Gemma 2, Qwen2.5, GPT-OSS, Devstral-Small 2 | 1B – 24B |
| **Vision** (4) | LFM2-VL, Gemma 3, Qwen3-VL | 1.6B – 12B |
| **OCR** (4) | PaddleOCR-VL, DeepSeek-OCR-2, dots.ocr, olmOCR 2 | 255M – 7B |
| **Speech-to-Text** (4) | Whisper tiny/small/large-v3, Voxtral-Mini | 37M – 1.5B |
| **Text-to-Speech** (4) | Kokoro, Qwen3-TTS, Chatterbox, Orpheus | 82M – 3B |
| **Embeddings** (4) | all-MiniLM-L6, embeddinggemma, ModernBERT-embed, bge-m3 | 18M – 568M |

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
| Image | Diffusion (Flux, SDXL, ...) | ✅ | ✅ | ⬜ | — |
| Audio | Music generation | ✅ | ✅ | ⬜ | — |

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

Select the **PipelineStudio** scheme, pick My Mac as the target, and press ⌘R.

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

- **Image generation Run UI** (Flux, SDXL) — browse + install already work, needs a
  prompt → image Run view
- **Music generation Run UI** — same pattern, different output type
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

### In progress
- [ ] Visual pipeline builder UI
- [ ] Improved gated-model auth flow

### Up for grabs
- [ ] Image generation Run UI (Flux, SDXL, etc.)
- [ ] Music generation Run UI
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
