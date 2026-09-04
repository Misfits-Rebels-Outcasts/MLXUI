# CLAUDE.md

Guidance for working in this repository.

## What this is

**MLXUI** is a native macOS SwiftUI app for Apple Silicon that does two things:

1. **Browse, compare, install and run local MLX models** — a bundled catalog sourced from
   HuggingFace, filtered/sorted by hardware fit (RAM, chip bandwidth), installed by downloading
   files directly from HF, and run locally via MLX (LLM chat, ASR, TTS, VLM, OCR, embeddings,
   diffusion, segmentation, video, upscale).
2. **Run those models as workflows** — **CAT Flow**, reached at *Automate → AI Workflows*. A
   `.cat` file is a numbered list of `Asset → Transform → Asset` rows that a runtime executes top
   to bottom. `MLXUI/FlowKit/` is a Swift port of the `catflow-mlx` Python runtime.

The second half arrived through the CAT Flow merge (`RSI/DelegateMergeBacklog.md`, phases R1…R16)
and is now the larger surface. A change to model support is usually *also* a change to what flows
can run — see "CAT Flow" below.

- Platform: **macOS only**, deployment target **14.0**, set in `Config/Shared.xcconfig`.
- **Two editions** (`Design/dual-distribution.md`):

  | Target | Product | Distribution | Sandbox | Flag |
  |---|---|---|---|---|
  | `MLXUI` | "AI Browser" | Mac App Store | **yes** — app-sandbox, network.client, user-selected read-write | `APPSTORE_BUILD` |
  | `MLXUI-Direct` | "AI Browser Pro" | Developer ID + notarized | **no** — Hardened Runtime only | `DIRECT_BUILD` |

  The Direct edition compiles in the shell / filesystem / subprocess agent tools; the App Store
  target never has those files in its membership, so the sandboxed binary cannot contain them.
  `FlowKit/CapabilityGate.swift` refuses capability classes by channel at run time.
- Build settings live in `Config/{Shared,AppStore,Direct}.xcconfig`, wired as each target's
  `baseConfigurationReference`. **A target-level setting in Xcode's editor overrides the
  xcconfig** — check there first when a setting doesn't take.
- No analytics / no data collection.

### Mission

The app supports a **growing list of MLX models**, and a model isn't "done" until it can be
installed and actually run. That now means two things, both required:

- **A Run UI** matched to the modality (chat, ASR, TTS, image Q&A, OCR, embeddings, diffusion, …),
  built from the module template in `Design/aisdk-aiui-architecture.md`.
- **Reachability from AI Workflows** — a `CatalogBridge` row, a curated manifest under
  `Resources/CatFlow/models/`, and a task the executor genuinely serves. This is step **AM-W** of
  `RSI/prompts/add-model.md` (the `/add-mlxui` intake), and it is the step most often forgotten:
  a model can install and run from Browse while every `.cat` row naming it refuses.

When an architecture has no MLX runner, it's flagged in `Core/ModelSupport.swift` and queued for an
iterative port rather than left silently broken.

## The RSI process — read this before starting work

Work is driven from `RSI/`, one reviewable cycle at a time. Not ad hoc.

| File | Role |
|---|---|
| `RSI/policies.md` | **Read first, every cycle.** Current autonomy level (**L1** — one task per cycle, every diff reviewed) and the standing guardrails. |
| `RSI/backlog.md` | The owner's backlog. `/add-mlxui` appends `AM-*` intake groups here. |
| `RSI/DelegateMergeBacklog.md` | The CAT Flow merge phases (R1…R16), delegated for external implementation. |
| `RSI/journal/` | One entry per task — what was done, what the gates said, what was found. |
| `RSI/evals/smoke-checklist.md` | Human-run smoke rows. A model or flow isn't verified until one passes. |
| `RSI/prompts/add-model.md` | The add-a-model intake (`/add-mlxui`), Steps A–D. |
| `Design/RSI.md` | The autonomy ladder (§3) and guardrails (§9) that `policies.md` instantiates. |

The guardrails that bite most often:

- **MLX-first.** New model runtimes execute the MLX build via `mlx-swift` / `mlx-swift-lm` /
  `mlx-audio-swift`, loading the files `InstallManager` already downloaded (the "A2 pattern").
  Catalog entries for runnable models should be `source: mlx`, `mlx-community/...` repos. No new
  CoreML runtimes. A module must never download weights behind the app's back.
- **No adding or upgrading Swift Package dependencies** without explicit approval — recorded in
  `policies.md` with a date. `mlx-swift` has shipped breaking API changes in *patch* releases, so
  a package that hard-pins a different version is a real conflict, not a formality. Precedent:
  `Vendor/PaddleOCRVL` (vendored with pins bumped rather than added as a remote dep).
- **No Combine.** SwiftUI + async/await.
- One revertable git commit per task; hard gates are hard stops; keep changes in scope.
- **Never edit a golden or fixture to make a change pass** — that's a `SPEC_QUESTIONS.md` entry.

`RSI/DelegateMergeBacklog.md` also carries a standing rule worth knowing: **an owner ruling is
only an owner ruling when the owner's own words are quoted with a date.** Anything else is
"implementer's call, pending owner confirmation" — proceed under that label, don't attribute the
decision to someone else.

## Design docs

The specs live in `Design/`:

- `Design/pipedesign7.md` — **current source of truth.** Successor to v6; reconciles the spec with
  the actual code and follows the code where they disagree.
- `Design/pipedesign6.md` — the previous consolidation. Still the best layer/file map, data-coverage
  table, formulas and storage layout.
- `Design/pipedesign5.md` — origin of the v5 JSON schema (4-bit-only variants, `modelType`, the
  1.5× RAM overhead, per-chip speed scaling).
- `Design/aisdk-aiui-architecture.md` — **the per-model module template**: engine + `PipelineStage`
  + `ModelSDK`/`claim` + Run UI. Every new model port follows it.
- `Design/dual-distribution.md` — the two editions and their divergence hazards.
- `Design/CatFlowDesign/`, `CATFlow_AIWorkflows/`, `AI_Workflows_as_a_list/`, `CATFlowGuideUI/` —
  the flow surface.

When the docs and the code disagree, **the code wins** — see "Doc drift" below.

## Architecture

Layered, with a single `@Observable` `AppState` as the hub.

```
Views ─── ContentView · Sidebar · Home · Browse · Detail · Run · Flows · Pipeline
          · Search · Settings · Components
State ─── AppState (@Observable) · InstallManager
Core ──── ModelRegistry · ModelStore · Pipeline/PipelineStage · StageFactory · RunnerKind
          · ModelSupport · Media/Asset/Kind/Shape · Chat · Agent · HFTokenizerLoader
Modules ─ one per model family (see below)
FlowKit ─ the CAT Flow runtime: parser, validator, interpreter, executors, cache, bridge, tools
Services ─ ModelRunner (MLX chat + agent sessions) · KeychainHelper (in InstallManager.swift)
Normalization ─ DataNormalizer (license / architecture / task-tag display strings)
Resources ─ browser.json · Gallery/ (bundled .cat flows) · Workspaces/ (bundled multi-flow workspaces) · CatFlow/{frames,models}
Vendor ─── PaddleOCRVL (local SPM package)
```

**Modules** (`MLXUI/Modules/`) — Chat, VLM, OCR, DotsOCR, DeepSeekOCR, PaddleOCR, Whisper,
MLXWhisper, Voxtral, MLXAudioTTS, Kokoro, Embedding, Flux, StableDiffusion, MusicGen,
SegmentAnything, WanVideo, SeedVR2, Tools, Demo. Each registers itself in `App/ModelModules.swift`;
`ModelRegistry.bestModule(for:)` is what resolves a `ModelEntry` to a runner.

### Key files (`MLXUI/`)

| File | Role |
|---|---|
| `MLXUIApp.swift` | `@main`. WindowGroup, loading/error/ready split, ⌘K command, `NavigationSplitView`. |
| `App/ModelModules.swift` | Registers every `ModelModule` at launch. A port isn't wired until it's here. |
| `State/AppState.swift` | Hub. Catalog, filters, sort, comparison, install/run wiring, filter persistence, gallery flows, `claimableModelIDs`. Defines `SidebarItem`, `SortOrder`, `InstalledModels`. |
| `State/InstallManager.swift` | HF download engine. File resolution via `/api/models/{id}` siblings, sequential download w/ progress delegate, sharded-model detection, verify, atomic move, `installed.json` registry, Keychain token. |
| `Services/ModelRunner.swift` | Runs models locally: real `MLXLLM`/`MLXLMCommon` inference, streaming chat, agent sessions with tool-call approval (`PendingApproval`). |
| `Core/ModelRegistry.swift` | Module registry + `bestModule(for:)` — the resolution every run path goes through. |
| `Core/ModelStore.swift` | The **one** place `Application Support/AI Browser/…` paths are built. Never rebuild them by hand. |
| `Core/RunnerKind.swift` | `modelType` → `RunnerKind`, plus the known catalog mislabels. **The only place dispatch decisions are made.** |
| `Core/StageFactory.swift` | `StageConfig` (the per-run settings a stage receives) + install/RAM gating before any engine loads. |
| `Core/ModelSupport.swift` | Architectures with no MLX runner yet, keyed on id fragments. Remove an entry when it's ported. |
| `Models/ModelEntry.swift` | Core model struct + `ModelType`, `ModelSource`, `ModelVariant`, `ModelBenchmarks`; `scaledSpeed(bandwidthGBps:)`, `bestVariant`, `qualityScore`. |
| `Models/BrowserData.swift` · `DomainNode.swift` · `SidebarSection.swift` | Catalog root Codable, recursive domain tree (models at leaves), section → domainIds. |
| `Models/SystemInfo.swift` | `detect()` reads `hw.model` via sysctl → chip name → bandwidth table; RAM/disk. |
| `Normalization/DataNormalizer.swift` | Pure display-string mappers for license/architecture/task tags. |
| `Resources/browser.json` | The bundled catalog (**v6.0-mvp, 35 entries**). |

## CAT Flow (`MLXUI/FlowKit/`)

The Python runtime in the sibling **`catflow-mlx`** repo is the **parity reference**. Its path is
`scripts/sync-catflow-fixtures.sh`'s `SRC` (override with `CATFLOW_MLX_DIR`).

- `Fixtures/CatFlow/` holds the Python's conformance corpus, goldens, traces and registry, synced
  by that script, with a `manifest.json` recording the source commit SHA and a per-file sha256.
  `CatFlowFixtureSyncTests` fails on a **stale corpus** *and* on a **hand-edited fixture**.
- **Four tables decide whether a flow row can run.** When a row refuses — or worse, when it
  shouldn't be offered — check all four, in this order:
  1. `TaskCatalog` — does the task exist? What is its `refName` / `refKind` / shape?
  2. `TaskModels.servedRefNamePrefixes` — does `RealExecutor` genuinely have a path for it? This is
     an **opt-in allow-list**: a newly ported task must be added, and a model existing for the
     task's `RunnerKind` is *not* enough.
  3. `TaskModels.taskKinds` + the derived pool — is there a runnable model of that kind, per the
     live catalog, `claimableModelIDs` and `ModelSupport`?
  4. `CatalogBridge.entries` — does the display name the `.cat` **writes** resolve to an installable
     `browser.json` entry? Equivalence (`.same` / `.requantized` / `.sameFamily` / `.substitute`)
     is explicit and reviewable; the last two surface *what actually loads* on the row. Never guess
     a substitution that isn't in the table.
- The **settings authority** is the curated manifest in `Resources/CatFlow/models/`, ported
  verbatim from `catflow-mlx/models/curated/`. **Never edit one to grant a model a task or a
  setting** — that is inventing catalog semantics; file it as a spec question instead (see below).
- **`SPEC_QUESTIONS.md` lives in the `catflow-mlx` repo, not this one.** It is the shared question
  log for both runtimes, and Swift code cross-references its numbers (`SPEC-Q207`…`Q210` in
  `SegmentAnythingStage`, `SPEC-Q55` in `TaskCatalog`, `SPEC-Q71`/`Q117` in `Shape`). Add entries
  there and mark the code `// SPEC-Q<n>`. Don't start a second one here.
- Bundled gallery flows live in `Resources/Gallery/` with `_metadata.json`;
  `AppState.hiddenFlowNumbers` hides individual ones by number.
- **Bundled workspaces** (R17) — a multi-flow directory the app ships — live in
  `Resources/Workspaces/` (files flat, `<id>--` prefixed) and are enumerated by
  `FlowKit/BundledWorkspaces.swift`, which `prepare`s them into `workspaces/<id>/` on first
  use. Two ship: `uses_example` (`AskYourDocs.cat` calls `RagQuery.cat` via `uses:` —
  `UsesResolver` builds the graph the interpreter runs; `canRun` recognises the call) and
  `ask_your_docs` (`Ingest.cat` builds `library.index` from its own `docs/`, `DocChat.cat`
  queries it — the RAG loop, with no prebuilt index). All reachable only through a `FlowScope`
  whose `locationID` is a workspace id (CFM-R17-1). `FlowKit/WorkspaceKnowledge.swift` derives
  the builder/querier pairing that `WorkspaceListView`'s knowledge-base card renders.

## How the data flows

1. `AppState.loadBrowserData()` decodes `browser.json` (in the app bundle) into `BrowserData`.
   Decoding errors are surfaced with precise key paths.
2. The sidebar shows only sections with `modelCount > 0` (`visibleSections`). The catalog ships
   five: `chat-text`, `vision-ocr`, `images-videos`, `audio`, `infrastructure`.
3. `filteredModels` applies section + source + capability filters, then sorts by `SortOrder`
   (default `.mostDownloaded`). `groupedModels` regroups by leaf domain and sorts each section.
4. RAM "fit": models with `ramGB > filterRAMLimitGB` are **dimmed, not removed**. The slider's max
   scales to the machine's RAM.

### Catalog generation

The catalog is generated **in this repo**, by `scripts/`:

- `regenerate_catalog.py` — rebuilds `browser.json`.
- `enrich_descriptions.py` — fills human-facing fields from the HF API.
- `validate_browser_json.py` — schema/consistency gate. Run it after any catalog edit.

`catalog-audit/` keeps the crawl record — earlier catalog revisions, `keepers.csv`,
`quarantine.{csv,json}`, `crawl-skipped.json` — i.e. *why* an entry is or isn't in the catalog.
You normally edit Swift here, not the catalog; when you do edit it, decode must stay green.

## Install & storage

Files land under `~/Library/Application Support/AI Browser/` (both editions — the Direct build is
un-sandboxed, so the path is the same, not a container). Built **only** via `Core/ModelStore.swift`:

```
installed.json            registry (version + per-model metadata)
models/{model-id}/        config.json, *.safetensors, tokenizer.json, .installed marker
downloads/{model-id}/     temp; deleted on completion or cancel
flows/{flow-id}/          user flows, their .blobs/ and run state — exactly one .cat per folder
workspaces/{workspace-id}/  R17: multi-flow workspaces — sibling .cat files + shared docs/ + index/
agent-index/              the agent's local index
WhisperKit/               legacy CoreML ASR assets (superseded by the MLX path)
```

- `model-id` is the HF id with `/` → `--` (e.g. `mlx-community--Qwen3-4B-4bit`).
- The `.installed` marker is the atomic "install succeeded" signal; the registry is reconciled
  against it on launch (stale entries dropped).
- Gated models: 401/403 → `InstallError.needsAuth`. HF token in the Keychain (service
  `com.ai-browser`, account `huggingface-token`), sent as `Bearer`.
- **Keychain and network-client behavior changes need the owner's approval** (`policies.md`).

## Formulas (see pipedesign6 §10)

- **RAM**: `paramCountB × bitsPerWeight/8 × 1.5` (4-bit ≈ 4.5 bits). This is an **LLM-weights
  heuristic**. It does not describe a diffusion model whose transformer is quantized at load, or one
  that ships a separate text encoder — those entries need measured numbers with their provenance in
  the entry's notes.
- **Speed**: stored `speedTokensPerSec` is an M2 Max baseline; displayed as
  `base × (detectedBandwidth / 400)` via `ModelEntry.scaledSpeed(bandwidthGBps:)`. Tilde prefix =
  estimated. Meaningless for diffusion — leave it null rather than invent one.
- **Quality**: few models have benchmarks; `qualityScore` is `nil` otherwise and renders "—".
  Quality sort puts null-benchmark models last.

## Dependencies

Remote Swift Packages (Xcode-managed):

| Package | Pin | Notes |
|---|---|---|
| `ml-explore/mlx-swift` | 0.31.4 | breaking API changes have shipped in patch releases |
| `ml-explore/mlx-swift-lm` | 3.31.3 | provides `MLX`, `MLXLLM`, `MLXVLM`, `MLXEmbedders` |
| `Blaizzy/mlx-audio-swift` | branch `main` | `MLXAudioSTT`, `MLXAudioTTS` — tags ≤ v0.1.2 pin mlx-swift-lm 2.x |
| `argmaxinc/WhisperKit` | 1.0.0 | CoreML ASR, **superseded** by the MLX path; still linked |

Local package: **`Vendor/PaddleOCRVL`** — MIT, vendored rather than added remotely because every
upstream version hard-pins conflicting `mlx-swift` / `swift-transformers` versions
(`RSI/journal/2026-42`). That is the template for any future package with an incompatible pin.

Transitive pins that matter when something conflicts: `swift-transformers` 1.3.3,
`swift-huggingface` 0.9.0, `swift-numerics`, `swift-syntax`, `swift-collections`.

**Adding or upgrading any of these requires explicit owner approval**, recorded in `policies.md`.
MLX requires Apple Silicon to run models.

## Build & validate

Driven from inside Xcode. Prefer the `xcode-tools` MCP server:

- `BuildProject` — full build (slow but authoritative). Build **both** targets when a change could
  diverge by edition (anything touching capabilities, entitlements, or the agent tools).
- `XcodeRefreshCodeIssuesInFile` — fast per-file diagnostics; use after edits.
- `RunCodeSnippet` — quick experiments in a file's context.
- `XcodeWrite` — use this for new files that must land in the bundle (e.g. a curated manifest under
  `Resources/CatFlow/models/`), so Xcode syncs them into the target.

Tests: **`MLXUITests/`, ~109 files**, using the **Testing** framework (`import Testing`,
`@testable import MLXUI`). Most are CAT Flow parity and golden tests. No XCUIAutomation UI tests
yet. `scripts/sync-catflow-fixtures.sh` refreshes the parity corpus.

## Doc drift — code is ahead of / diverges from the docs

Trust the code over `pipedesign5/6.md` on these:

- **The model runner is real and has grown.** `ModelRunner.swift` does actual MLX LLM generation
  with a streaming chat UI *and* agent sessions with tool-call approval — not the v1.1 "stub", and
  not the old hardcoded "Introduce yourself" prompt.
- **`SimpleTokenizer` is gone** (`journal/2026-37`). Tokenization is `Core/HFTokenizerLoader.swift`
  via swift-transformers; the hand-rolled stub produced garbage tokens and broke VLM vision
  placeholders.
- **Filter persistence is implemented** via `UserDefaults` (`saveFilters` / `loadFilters`).
- **`DataNormalizer` runs lazily at display time** inside `ModelDetailView` /
  `ComparisonTableView`, *not* once at load in `loadBrowserData()`.
- `ModelEntry.scaledSpeed` takes `bandwidthGBps: Double`, not a `SystemInfo`.
- **The catalog is ~35 entries, not 435**, and it is generated in-repo by `scripts/`, not by a
  sibling `brainstorm/` directory.

Known inconsistencies in the tree itself — fix or rule on them, don't propagate them:

- **`MLXUITests` carries a target-level `MACOSX_DEPLOYMENT_TARGET = 26.4`** while both app targets
  take **14.0** from `Config/Shared.xcconfig`. A test target with a higher floor than the app can
  compile against APIs the shipping binary cannot use.
- **`AppState.hiddenFlowNumbers` is `[]`**, though `CFM-R12-FIX-13` records an owner ruling of
  2026-08-25 to keep gallery flows 60–69 hidden. Either a later ruling isn't written down, or it
  drifted back.
- **WhisperKit is still linked** although the MLX-first guardrail superseded it (A2).
- **`StageConfig` carries no `seed` / `width` / `height` / `steps`**, so diffusion rows' settings
  are silently dropped — see `CFM-R16-1` in `RSI/DelegateMergeBacklog.md`.

## Conventions (from the project style guide)

- PascalCase types, camelCase members; 4-space indent.
- `@State private var` for view state; `let` for constants; **no force-unwraps**.
- SwiftUI + async/await; **do not** introduce Combine.
- `nonisolated` + `Sendable` for FlowKit types — the runtime crosses actors and is unit-tested off
  the main actor. `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor` is set project-wide, so opting out is
  explicit.
- Keep changes scoped to the request; don't refactor unrelated code.
- New SwiftUI / Apple APIs may post-date training data — use `DocumentationSearch` (especially for
  Liquid Glass, FoundationModels, latest SwiftUI).
- If the spec is ambiguous, **don't decide it silently**: write it in `catflow-mlx`'s
  `SPEC_QUESTIONS.md` with a recommended answer and a failing-test sketch, mark the code
  `// SPEC-Q<n>`, take the most conservative reading, and continue.
