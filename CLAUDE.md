# CLAUDE.md

Guidance for working in this repository.

## What this is

**MLXUI** (product name "AI Browser") is a native macOS SwiftUI app for
browsing, comparing, downloading, and running local AI models optimized for Apple
Silicon via MLX. It ships a bundled catalog of ~435 model entries (4-bit MLX
quantizations) sourced from HuggingFace's `mlx-community`, lets the user filter/sort
by hardware fit (RAM, chip bandwidth), installs models by downloading their files
directly from HuggingFace, and runs them locally with MLX (LLM chat, ASR, TTS, VLM, OCR,
embeddings — with more model types added over time).

- Platform: **macOS only**, deployment target **14.0** (uses `@Observable`,
  `NavigationSplitView`).
- Sandboxed + Hardened Runtime. Entitlements: app-sandbox + network.client only.
- No analytics / no data collection.

### Mission

The app aims to support a **growing list of MLX models** — the user downloads and runs each
model to try it out, so **every model needs a Run user interface** matched to its modality
(chat, ASR, TTS, VLM image-Q&A, OCR, embeddings, …). New models are added continuously; a
model isn't "done" until a user can install it and actually run it from the UI. When a model's
architecture has no MLX runner yet, it's flagged (`Core/ModelSupport.swift`) and queued for an
iterative port (backlog § "Model support gaps") rather than left silently broken. The per-model
Run UI + engine wiring follows the module template in `Design/aisdk-aiui-architecture.md`.

## Design docs

The authoritative specs live in `Design/`:

- `Design/pipedesign5.md` — v5 spec. Establishes the data pipeline, the v5 JSON
  schema (4-bit-only variants, `modelType` field, 1.5× RAM overhead, per-chip speed
  scaling), and the original 18-day phase plan.
- `Design/pipedesign6.md` — **current source of truth**. Consolidates v5 + the actual
  Xcode project. Read this first: it has the feature matrix (what's built vs. v1.1+),
  the layer/file map, data-coverage table, formulas, and storage layout.

When the docs and the code disagree, **the code wins** — see "Doc drift" below.

## Architecture

Layered, with a single `@Observable` `AppState` as the hub. (Avoid Combine; use
async/await per repo style.)

```
Views ── ContentView · Sidebar · Home · Browse (Card/Filter/Comparison)
         · Detail · CommandPalette · Components
State ── AppState (@Observable) · InstallManager · ModelRunner
Models ─ BrowserData · DomainNode · ModelEntry · SystemInfo · SidebarSection
Normalization ─ DataNormalizer (license / architecture / task-tag display strings)
Services ─ ModelRunner (MLX) · KeychainHelper (in InstallManager.swift)
Resources ─ browser.json  (the bundled v5 catalog, 435 entries)
```

### Key files (`MLXUI/`)

| File | Role |
|---|---|
| `MLXUIApp.swift` | `@main`. WindowGroup, loading/error/ready split, ⌘K command, `NavigationSplitView`. |
| `State/AppState.swift` | Hub. Holds catalog, filters, sort, comparison, install/run wiring, filter persistence. Defines `SidebarItem`, `SortOrder`, `InstalledModels`. |
| `State/InstallManager.swift` | HF download engine. File resolution via `/api/models/{id}` siblings, sequential download w/ progress delegate, sharded-model detection, verify, atomic move, `installed.json` registry, Keychain token. |
| `Services/ModelRunner.swift` | Runs models locally. **Real MLX LLM inference** (`MLXLLM`/`MLXLMCommon`). Non-LLM types show an "unsupported" NSAlert. Includes a `SimpleTokenizer` fallback. |
| `Models/ModelEntry.swift` | Core model struct + `ModelType`, `ModelSource`, `ModelVariant`, `ModelBenchmarks`. Has `scaledSpeed(bandwidthGBps:)`, `bestVariant`, `qualityScore`. |
| `Models/BrowserData.swift` | Root Codable: version, sidebarSections, domainIndex, domains. |
| `Models/DomainNode.swift` | Recursive domain tree; `allModels` / `totalModelCount`. Models live at leaf nodes. |
| `Models/SidebarSection.swift` | Section → domainIds mapping + `modelCount(in:)`. |
| `Models/SystemInfo.swift` | `detect()` reads `hw.model` via sysctl → chip name → bandwidth table; RAM/disk. |
| `Normalization/DataNormalizer.swift` | Pure display-string mappers for license/architecture/task tags. |
| `Resources/browser.json` | Bundled v5 catalog. |

## How the data flows

1. `AppState.loadBrowserData()` decodes `browser.json` (in app bundle) into
   `BrowserData`. Decoding errors are surfaced with precise key paths.
2. Sidebar shows only sections with `modelCount > 0` (`visibleSections`) — empty
   sections (Video, Data & Science, Verticals, Infra) are hidden.
3. `filteredModels` applies section + source + capability filters, then sorts by
   `SortOrder` (default `.mostDownloaded`). `groupedModels` regroups by leaf domain.
4. RAM "fit": models with `ramGB > filterRAMLimitGB` are dimmed, not removed.

### Catalog generation (outside this repo)

The bundled `browser.json` is produced by Python scripts (`transform_v4.py`,
`patch_v5.py`) that live in the sibling `brainstorm/` data-pipeline dir, **not** in
the Xcode project. They split family groups into entries, enrich from the HF API,
strip unverified variants to 4-bit-only, add `modelType`, and apply the 1.5× RAM
formula. You normally edit Swift here, not the catalog.

## Install & storage

Files land under `~/Library/Application Support/AI Browser/`:

```
installed.json            registry (version + per-model metadata)
models/{model-id}/        config.json, *.safetensors, tokenizer.json, .installed marker
downloads/{model-id}/     temp; deleted on completion or cancel
```

- `model-id` is the HF id with `/` → `--` (e.g. `mlx-community--Qwen3-4B-4bit`).
- The `.installed` marker file is the atomic "install succeeded" signal; the registry
  is reconciled against it on launch (stale entries dropped).
- Gated models: 401/403 → `InstallError.needsAuth`. HF token stored in Keychain
  (service `com.ai-browser`, account `huggingface-token`), sent as `Bearer`.

## Formulas (see pipedesign6 §10)

- **RAM**: `paramCountB × bitsPerWeight/8 × 1.5` (4-bit ≈ 4.5 bits).
- **Speed**: stored `speedTokensPerSec` is an M2 Max baseline; displayed value is
  `base × (detectedBandwidth / 400)` via `ModelEntry.scaledSpeed(bandwidthGBps:)`.
  Tilde prefix = estimated (no measured data exists).
- **Quality**: only ~7 models have benchmarks; `qualityScore` is `nil` otherwise and
  renders as "—". Quality sort puts null-benchmark models last.

## Doc drift — code is ahead of/diverges from the docs

When reasoning, trust the code over `pipedesign5/6.md` on these points:

- **Model runner is real, not a stub.** `ModelRunner.swift` performs actual MLX LLM
  generation (the docs list it as a v1.1 "stub"). It currently runs a hardcoded
  "Introduce yourself" prompt — there is no chat UI yet.
- **Filter persistence is implemented** via `UserDefaults` (`saveFilters`/
  `loadFilters`), despite being listed as v1.1.
- **`DataNormalizer` runs lazily at display time** inside `ModelDetailView` /
  `ComparisonTableView`, *not* once at load in `loadBrowserData()` as the docs state.
- `ModelEntry.scaledSpeed` takes `bandwidthGBps: Double`, not a `SystemInfo`.

## Dependencies

Swift Package deps (Xcode-managed): `mlx-swift`, `mlx-swift-lm` (provides `MLX`,
`MLXLLM`, `MLXLMCommon`), `swift-numerics`, `swift-syntax`. MLX requires Apple
Silicon to run models.

## Build & validate

This project is driven from inside Xcode. Prefer the `xcode-tools` MCP server:

- `BuildProject` — full build (slow but authoritative).
- `XcodeRefreshCodeIssuesInFile` — fast per-file diagnostics; use after edits.
- `RunCodeSnippet` — quick experiments in a file's context.
- Tests use the **Testing** framework (unit) / **XCUIAutomation** (UI). None present yet.

## Conventions (from project style guide)

- PascalCase types, camelCase members; 4-space indent.
- `@State private var` for view state; `let` for constants; avoid force-unwraps.
- SwiftUI + async/await; **do not** introduce Combine.
- Keep changes scoped to the request; don't refactor unrelated code.
- New SwiftUI / Apple APIs may post-date training data — use `DocumentationSearch`
  (especially for Liquid Glass, FoundationModels, latest SwiftUI).
