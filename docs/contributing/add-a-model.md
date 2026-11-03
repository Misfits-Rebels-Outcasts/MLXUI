# Adding an MLX model to MLXUI

This is the public procedure for bringing a new MLX model into MLXUI. It covers support checks, the catalog entry, the
download, the runner module, the Run screen, AI Workflows wiring, the Local Server and the tests. In Claude Code, run it as
**`/add-mlxui <hf-repo>`**. Without Claude Code, follow it by hand; nothing in it depends on the tool.

The maintainers plan their work in private notes that aren't in this repo. Anything you need from them will be in the
GitHub issue for the model. If something here and the issue disagree, **the issue wins**. Ask on the issue if you're unsure.

---

## Contributing with an AI agent

Agents are welcome. Paste this into yours (Claude Code, Codex, Cursor, Copilot, or any other) from your local clone, with
the first two lines filled in. In Claude Code, `/add-mlxui <repo>` follows this same guide, but the prompt below works in
any agent.

```text
I want to contribute a new MLX model to MLXUI (https://github.com/Misfits-Rebels-Outcasts/MLXUI) and open a pull request.

Model: <Hugging Face repo, e.g. mlx-community/Some-Model-4bit>
Issue: <link to the GitHub issue for this model, or "none yet">

Work in my local clone of the repo. Read these first, in this order, and follow them:
1. CLAUDE.md: architecture, the "four tables", conventions. Its references to RSI/, Design/ and scripts/
   are private maintainer material that isn't in the repo. Ignore them; don't look for them.
2. docs/contributing/add-a-model.md: the procedure you'll follow (Steps A–D, house rules, traps,
   testing, PR description).
3. The GitHub issue above, in full. Where the issue and the guide disagree, the issue wins.

Then:
- If there's no issue: draft one for me proposing this model (what it adds that MLXUI can't do today,
  its license, its size) and stop. I'll post it and wait for a maintainer.
- If the issue already contains the intake (decisions, work plan): skip to Step D.
- Otherwise: do Steps A–C. Build the project once in Xcode so the pinned package sources exist under
  DerivedData, and read runners there at the revisions in Package.resolved, never GitHub main. Then
  draft the intake comment in the guide's format and STOP. I'll post it. Don't write code until I tell
  you a maintainer has said go ahead.

When implementing (Step D):
- One commit per task, named as the issue says. Build both schemes (MLXUI and MLXUI-Direct) after every task.
- Never change Swift packages, Package.resolved, project.pbxproj package wiring, entitlements, Keychain
  or network code, existing tests, fixtures, gallery flows or existing manifests.
- Write the real-weights tests the guide describes, gated on TEST_RUNNER_MLXUI_TEST_<NAME>_DIR. Tell me
  the exact commands so I can run them on my Mac. A skipped test doesn't count.
- If you hit anything the issue and guide don't settle, stop and ask me, with the answer you'd recommend.
  Mark any small call you make yourself "implementer's call, pending owner confirmation".
- Finish by drafting the PR description in the guide's six-part shape. Don't post to GitHub (issues,
  comments, PRs) without showing me the text first.
```

**You stay in charge of three things the agent can't do for you:** posting to GitHub (read it first), running the
real-weights tests on your Mac, and waiting for a maintainer's go-ahead before code. Please say in the PR which agent you
used. It helps the review.

---

## How a model gets added

1. **An issue exists for the model.** It names the repo and records any decisions already made (variant, display name,
   which rows it joins, what's out of scope). No issue yet? Open one proposing the model, with a link to its Hugging Face
   page and one paragraph on what it adds that MLXUI can't do today. Wait for a maintainer to accept it.
2. **You run the intake** (Steps A–C below) and post the result as **one comment on the issue**. No code yet.
3. **A maintainer replies** with go-ahead, corrections or rulings.
4. **You implement** (Step D), one commit per task, in a **draft PR** opened early.
5. **A maintainer reviews** the PR. They run a private real-weights smoke and memory suite, set the measured `ramGB`, and
   the owner judges output quality (listening, looking). Then they merge.

Read [`CLAUDE.md`](../../CLAUDE.md) first: architecture, modules, `RunnerKind`, the "four tables", install and storage,
formulas, dependencies and conventions. Its references to `RSI/`, `Design/` and `scripts/` point at private material. You
don't need it, and you can't run those scripts.

**You'll need:**
- an Apple Silicon Mac, with Xcode and the macOS version the project's deployment target requires
- enough RAM to load the model with room to spare (Step A.4)
- disk space for the weights twice: the app's install plus a `/private/tmp` copy for tests

---

## Step A — Check that MLXUI can run it

1. **Architecture.** Open the repo's `config.json` on Hugging Face
   (`https://huggingface.co/<repo>/raw/main/config.json`). Note `model_type`, `architectures`, any `text_config` /
   `vision_config` / `audio_config`, and the quantization block. Note the **modality**: chat LLM, vision, OCR, speech to
   text, text to speech, embedding, reranker, diffusion, video, audio, or something new. **Always read `config.json`.** A
   family name can hide a different `model_type` (e.g. `gemma4_unified` vs `gemma4`).
2. **Runner.** Look for that `model_type` **in the package checkouts at the revisions this project pins**:
   - build once in Xcode, then read the sources under DerivedData's `SourcePackages/checkouts/`;
   - the pins are in `MLXUI.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved`.

   Never go by GitHub `main` of a package: it's often ahead of the pin.

   | Modality | Where to look |
   |---|---|
   | Chat LLM | `mlx-swift-lm` → `Libraries/MLXLLM/LLMModelFactory.swift` |
   | Vision / OCR | `mlx-swift-lm` → `Libraries/MLXVLM/VLMModelFactory.swift` |
   | Embedding | `mlx-swift-lm` → `Libraries/MLXEmbedders/Models/` |
   | Speech, audio | `mlx-audio-swift` → `Sources/MLXAudioSTT`, `MLXAudioTTS`, `MLXAudioSTS`, `MLXAudioVAD`, `MLXAudioCodecs` |
   | Anything | app-side ports in `MLXUI/Modules/*`, registered in `MLXUI/App/ModelModules.swift` |

   Also check which package **products** the app targets link (Xcode → target → Frameworks). A runner in an unlinked
   product needs a maintainer to link it.
3. **Files.** List the repo's files. Note every file the loader needs: weights, tokenizer, processor config, chat
   template, sub-folders, and **anything the library fetches from a second repo** (Trap 1).
4. **Size and fit.** Sum the weight files.
   - The provisional `ramGB` is `paramCountB × bitsPerWeight / 8 × 1.5` for LLMs (4-bit ≈ 4.5 bits), or 1.5 × the weights
     otherwise.
   - If that plus ~6 GB is more than your Mac's RAM, say so. You can still write the code; the maintainers run the
     real-weights tests.
5. **License.** Read the model card's license and note it exactly. Non-commercial licenses (e.g. `cc-by-nc-*`) need an
   explicit maintainer decision: MLXUI is used for business workflows.

## Step B — Classify

| Class | Meaning | What Step D looks like |
|---|---|---|
| **SUPPORTED** | An existing module already claims it (e.g. any `mlx-swift-lm` chat LLM → the chat module) | catalog + verify; no new module |
| **SUPPORTED-in-package** | The pinned package implements it, but no MLXUI module claims it yet | a thin module (engine + stage + SDK) in the shape of a sibling |
| **NEEDS-MODULE** | Parts exist but not the path MLXUI needs (e.g. hidden states from a VLM for embeddings) | a module that carries the missing piece |
| **NEEDS-PORT** | No runner anywhere | a from-reference architecture port. **Needs maintainer approval before you start** |

Say which class and why, citing file and line in the pinned checkout.

## Step C — Post the intake comment

Post one comment on the issue in this shape, then **stop and wait** for a maintainer's reply:

```
### Intake: <repo>

**Architecture:** model_type `…`, architectures `…`, modality …  (config.json @ <commit sha>)
**Runner:** <file:line in the pinned checkout>, product `…` (linked? yes/no)
**Class:** SUPPORTED / SUPPORTED-in-package / NEEDS-MODULE / NEEDS-PORT. Reason.
**Files the loader needs:** …   **Download:** … GB   **Provisional ramGB:** …   **License:** …
**Fits my Mac (… GB)?** yes / no

**Steps that apply:** AM1, AM2, AM3 (or AM3–AM5), AM-W (yes/skip: why), AM-S (yes/skip: why), AM-T
**Task list:** one line per commit, MCn-k / AM-k style
**Traps found:** …
**Questions:** each with the answer I recommend
```

---

## Step D — Implement, one commit per task

Name commits `<prefix>-k: <summary>`, using the prefix from the issue (e.g. `MC1-3: …`). Open a **draft PR** after the
first two tasks so review starts early. Build **both** schemes after every task:

```
xcodebuild -scheme MLXUI        -destination 'platform=macOS' build
xcodebuild -scheme MLXUI-Direct -destination 'platform=macOS' build
```

### AM1 — Catalog entry
- Add the entry to `MLXUI/Resources/browser.json` in the right domain leaf, **appended last**:
  - `source: "mlx"`, the right `modelType`, `hfModelId`, `family`
  - `downloadSizeGB` and the provisional `ramGB`, with a note that `ramGB` is an estimate
  - an **integer** `qualityPercent` (a `null` decodes as an error)
  - `license` + `licenseUrl`
  - `speedTokensPerSec` only for LLMs, else `null`
- A new kind of model also means:
  - a new case in `ModelType` (`MLXUI/Models/ModelEntry.swift`, with its icon) and in `RunnerKind`
    (`MLXUI/Core/RunnerKind.swift`)
  - a new domain in `browser.json` `domains`, listed in the right `sidebarSections` entry
- Adding an entry bumps the count guard in `MLXUITests/CatFlowChatModuleTests.swift`. Bump it.
- Test: `browser.json` still decodes through `AppState.loadBrowserData()`.
- `browser.json` is generated by maintainer scripts you don't have. Hand-edit it carefully, and keep your diff minimal and
  appended.

### AM2 — Description and facts test
- About text: what the model does, its limits, the license notice. **No claim the model card doesn't make**: no
  benchmarks, tool use, thinking or languages that aren't documented.
- `<Name>CatalogTests` asserts every fact a person reads on the detail page and in Compare: display name, leaf, context
  window, sizes, license string, architecture, variant count, and which fields render "—". Precedent:
  `MLXUITests/RIFECatalogTests.swift` (`entryFactsMatchTheRepo`).

### AM3 — Download (and, for SUPPORTED, verify)
- `MLXUI/Core/ModelFileSelector.swift` decides what downloads: `*.safetensors` plus `metadataNames`. Check it against the
  repo's real file list, and **pin that list as a test fixture**.
- Never add names to `metadataNames` speculatively. Prove the loader needs a file first.
- A second repo the model needs (a text encoder, a codec) goes through `ModelFileSelector.companionRepo(for:)`. The
  installer puts its files in a sub-folder of the model's folder, so they're installed and removed together.
- Any other change to `MLXUI/State/InstallManager.swift`: **stop and comment with a design first**.
- **SUPPORTED** models: a resolution test proving `registry.bestModule(for:)` (`MLXUI/Core/ModelRegistry.swift`) picks the
  expected module, and no flag in `MLXUI/Core/ModelSupport.swift`. Then skip to AM-W.

### AM4 — Module (SUPPORTED-in-package, NEEDS-MODULE, NEEDS-PORT)
- `MLXUI/Modules/<Name>/`: `<Name>Engine`, `<Name>Stage`, `<Name>SDK` (with `claim`), `<Name>Module`. Register it in
  `MLXUI/App/ModelModules.swift`. Copy the closest sibling:

  | Shape | Sibling to copy |
  |---|---|
  | Audio in → audio out | `DeepFilterNet` |
  | Speech to text | `Parakeet` / `Qwen3ASR` |
  | Speech in / out | `LFMAudio` |
  | Video | `RIFE` |
  | Diffusion | `Flux` |
  | Decision | `Laya` |

- **Load only installed files** (Trap 1). The engine takes the model directory from `ModelStore`, and the stage takes a
  directory rather than reading `ModelStore.shared` itself, so tests can point it at a copy.
- Put the Run screen's logic in a `@MainActor` model object that the view only binds to (`RIFERunModel`,
  `DeepFilterNetRunModel`). Tests drive it the way the buttons do.
- Take a `MemoryBudget` lease (`MLXUI/Core/MemoryBudget.swift`) for a run. Read library results through
  `MLXErrorGuard.read(_:as:)` (`MLXUI/Core/MLXErrorGuard.swift`), per Trap 3.
- **Claim by family, never by architecture** (Trap 4).

### AM5 — Run screen
- Bind the view only to the run model. Stop must work. Refuse inputs over the stated limits with one plain sentence
  *before* any work starts. Errors read as one plain sentence, never a raw error dump.

### AM-W — AI Workflows wiring (skip only if the issue says so)
This makes `.cat` flow rows able to use the model. `CLAUDE.md` § "Four tables" explains why all four matter.

1. **Display name.** Short (≤ 20 chars) and not already in `CatalogBridge.entries` (`MLXUI/FlowKit/CatalogBridge.swift`).
2. **Pool.** Append the display name **last** in the right pool in `MLXUI/FlowKit/TaskModels.swift`:

   | Modality | Pool |
   |---|---|
   | Chat LLM | `llmModels` (shared by every text task) |
   | Speech to text | `"Transcribe"` |
   | Text to speech | `"Speak"` |
   | Vision | `"Describe Image"` |
   | OCR | `"OCR"` |
   | Embedding | `"Embed"` |
   | Reranker | `"Rerank"` |
   | Image | `"Generate Image"` |
   | Video | `"Generate Video"` |
   | Audio generation | `"Generate Sound"` |

   A task nothing served before also needs its `RunnerKind` in `taskKinds` **and** its refName prefix in
   `servedRefNamePrefixes` (an opt-in allow-list).
3. **`BridgeEntry`** in `CatalogBridge.entries`:
   - `display`, `pinnedID` (the `hfModelId`), `candidates`, `manifestFile`
   - `equivalence`: `.same` unless the issue says otherwise. Never invent a substitution.
4. **Manifest** `MLXUI/Resources/CatFlow/models/<slug>.json`, modelled on an existing one. Its `notes` say
   "MLXUI-authored". **Never edit an existing manifest.** Those are ported verbatim from the CAT Flow reference runtime.
5. **Tests** (`<Name>WorkflowWiringTests`):
   - the bridge lookup;
   - `TaskModels.defaultModel(forTask:)` is **unchanged** for every pool joined (a new model never becomes a default by
     accident);
   - the name is last in each pool;
   - the manifest loads from the bundle.
6. **A task or setting the CAT Flow language doesn't define** isn't added here. Say so in the PR. The maintainers take it
   to the language spec. Never add a `TaskCatalog` row yourself.

### AM-S — Local Server (chat LLMs only; others skip)
The Serve toggle appears for any installed chat model, but behaviour comes from tables keyed on `model_type`. An unknown
family degrades silently.

1. **Tool calls.** Read the model's chat template (`chat_template.jinja` / `tokenizer_config.json`) and cite the deciding
   lines. `MLXUI/Core/Agent/ToolCallFormatTable.swift` (format override, support, system placement) and
   `ToolCallHistoryTable.swift`. Leave support `.unknown` rather than guess.
2. **Thinking.** `ReasoningTemplateTable.swift`: `.promptOpened`, `.modelEmitted` or `.none`.
3. **Memory and context.** `MLXUI/Core/ModelContextWindow.swift` must return the real window, and
   `MLXUI/Services/Server/ServeMemoryAdmission.swift` must produce a non-nil profile from `config.json`. Pin the
   `config.json` and template under `Fixtures/<Name>/` and test from them.
4. **Prompt cache** (`PromptCacheAllowlist`) and **structured output** (`ServeStructuredOutputFamilies`): leave the family
   **off**. Maintainers add it after their own real-model runs.
5. **Tests.** Extend the existing table suites. Never edit an existing row or golden.

### AM-T — Real-weights tests and memory
Unit tests run without weights. These run on real weights and **skip** unless you point them at a copy:

- Gate each suite: `@Test(.enabled(if: …))` on an environment variable `TEST_RUNNER_MLXUI_TEST_<NAME>_DIR` naming a model
  folder. The `TEST_RUNNER_` prefix is how `xcodebuild` passes it to the test host. Precedent:
  `MLXUITests/DeepFilterNetRealLoadTests.swift`.
- The App Store test host is sandboxed and can't read `~/Library/Application Support`. Install the model in the Direct
  app, then copy it:

  ```
  mkdir -p /private/tmp/mlxui-models
  cp -Rc ~/Library/Application\ Support/AI\ Browser/models/<org>--<repo> /private/tmp/mlxui-models/
  TEST_RUNNER_MLXUI_TEST_<NAME>_DIR=/private/tmp/mlxui-models/<org>--<repo> \
    xcodebuild -scheme MLXUI -destination 'platform=macOS' test -parallel-testing-enabled NO \
    -only-testing:MLXUITests/<Suite>
  ```

- Write:
  1. a **real-load test**: load through the same path the app uses, run once, check real output;
  2. a **main-actor probe**: `@MainActor`, inside `MLXErrorGuard.guardedRun`, the way the Run screen calls it (template:
     `AudioEngineMainActorProbeTests` in `MLXUITests/MLXThreadStreamTests.swift`);
  3. a **footprint test** printing peak `phys_footprint` per phase: load + first run, a typical input, and **the worst
     input the Run screen allows** (template: `MLXUITests/LFMAudioFootprintTests.swift`).
- Use bundled inputs from `MLXUI/Resources/Samples/`, or generate them in the test (`say` speech, synthetic images). Never
  commit personal recordings or photos.
- **A skipped test isn't a pass.** Paste the command and its output in the PR.
- Don't change `ramGB` from your numbers. The maintainers measure on their machines and set it.

---

## House rules

- **No new or upgraded Swift packages**, no `Package.resolved` changes, no package wiring in `project.pbxproj`, no
  entitlement, Keychain or network-client changes. These need the owner's approval; propose them on the issue.
- Before pushing, `git diff main --stat -- '*.pbxproj'` should be empty. Builds can bump `CURRENT_PROJECT_VERSION`; revert it.
- **Extend tests; never weaken or edit existing expectations, goldens or fixtures.** Never edit `Fixtures/CatFlow/`,
  `Resources/Gallery/` or existing curated manifests.
- **Shared tables are append-only:** `browser.json`, `RunnerKind`, `ModelType`, `ModelModules.swift`, `CatalogBridge`,
  `TaskModels`. Other work lands in them in parallel.
- Keep the change to the model. Don't refactor unrelated code. Follow `CLAUDE.md` § Conventions (Swift Testing, no
  force-unwraps, no Combine).
- Any decision the issue doesn't cover: make the conservative choice, mark it in code and in the PR as **"implementer's
  call, pending owner confirmation"**, and list it in the PR.

## Traps

1. **Downloads behind the app's back.** `fromPretrained`, `HubApi().snapshot`, `resolveOrDownloadModel`, or a config field
   holding a repo id (a text encoder `name: "t5-base"`, say) all fetch from the network on first use, outside the app's
   store, with no progress UI. Load with `fromDirectory` / `fromModelDirectory` / `loadConvertedWeights` from the installed
   folder. Patch any config field that names a repo to a local path. **Test that a missing file throws instead of
   downloading.**
2. **Files that don't download.** Anything not `*.safetensors` and not in `metadataNames` is silently skipped. Prove what
   the loader reads before adding a name.
3. **Thread-bound MLX arrays (mlx-swift 0.32).** An array built with an `MLXArray` index following another index can only be
   evaluated on the thread that built it, and reading it after a failed `eval` crashes. So:
   - read library results through `MLXErrorGuard.read(_:as:)`;
   - `eval` arrays you build before they cross an `await`, an actor hop or a stored property;
   - keep `MixedSubscriptScanTests` green;
   - add the main-actor probe.
4. **Claiming by architecture.** Several catalog entries can share a `model_type` (a chat model and an embedder, say).
   Claim by `family`, and test that each such entry resolves to exactly one module.
5. **Sample rates.** Speech-to-text runs at 16 kHz, DeepFilterNet at 48 kHz, codecs at their own rates. Read the model's
   rate and never route audio through another model's resampler.
6. **Chat templates under swift-jinja.** Templates can render differently from Python's Jinja, and some expect structured
   content rather than plain text. Render one real request in a test and compare it to the reference in the model card.
7. **Run-on generation.** If `generation_config.json` lacks the right `eos_token_id`, set the stop set in the engine from
   the tokenizer.
8. **License lookalikes.** The same family often ships sizes under different licenses. Assert the exact repo id in the
   catalog test.

## The PR description

1. **Commits:** one line per task commit.
2. **Tests:** the commands, pass/fail counts, and the output of the real-weights tests (skipped doesn't count). New red
   tests by name; tests already red on `main` listed separately.
3. **Memory:** the footprint lines per phase.
4. **Implementer's calls:** every decision not in the issue.
5. **Questions:** each with the answer you recommend.
6. **For the owner to judge:** paths or attachments of outputs that need eyes or ears (audio, images, video, a
   transcript). Generated from bundled or synthetic inputs, never personal data.
