import Foundation

/// How closely a `browser.json` candidate matches the catflow manifest's pinned model.
/// `.same`/`.requantized` run silently; `.sameFamily`/`.substitute` surface a note on the
/// row (R3's inspector). A display name with **no** candidate is not runnable — never guess
/// a substitution that isn't in the table. See `RSI/DelegateMergeBacklog.md` CFM-R2-2.
nonisolated enum Equivalence: String, Sendable, Equatable, CaseIterable {
    case same, requantized, sameFamily, substitute

    /// A one-line note for `.sameFamily`/`.substitute`; `nil` for the silent cases.
    /// Surfaces *what actually loads* so a substitution is never hidden.
    func note(display: String, substitutedID: String) -> String? {
        switch self {
        case .same, .requantized:
            return nil
        case .sameFamily, .substitute:
            return "running \(display) as \(substitutedID)"
        }
    }
}

/// One bridge row: the `.cat`-visible display name → a ranked list of installable
/// `browser.json` `hfModelId`s, with the catflow manifest's pinned id (the settings
/// authority) and the equivalence class. Explicit and reviewable — never derived.
nonisolated struct BridgeEntry: Sendable, Equatable {
    let display: String        // the name a .cat writes, e.g. "Whisper Large v3"
    let pinnedID: String       // catflow's manifest id — the settings authority
    let candidates: [String]   // browser.json hfModelIds, best first
    let equivalence: Equivalence
    /// The curated-manifest filename (under `Resources/CatFlow/models/`) whose settings
    /// (enums, ranges, defaults) govern this display name.
    let manifestFile: String
    /// Disambiguates a candidate hfModelId that names more than one catalog card — MoC-6's
    /// `Qwen3.5-9B` (`llm`) / `Qwen3.5-9B Vision` (`vision`) share one download, so matching by
    /// hfModelId alone in `resolve` is no longer unique. `nil` for every other row, where the
    /// `id == hfModelId`-slug invariant still holds and hfModelId matching is unambiguous.
    let modelType: ModelType?

    init(display: String, pinnedID: String, candidates: [String],
         equivalence: Equivalence, manifestFile: String, modelType: ModelType? = nil) {
        self.display = display
        self.pinnedID = pinnedID
        self.candidates = candidates
        self.equivalence = equivalence
        self.manifestFile = manifestFile
        self.modelType = modelType
    }
}

/// The outcome of resolving a display name against the catalog.
nonisolated enum CatalogBridgeResolution: Sendable, Equatable {
    /// Runnable: the `ModelSlot` to install/run, plus the equivalence and any note. MS-3:
    /// was `ModelEntry` — every existing caller resolved a `browser.json` row, which is now
    /// `.cataloged(ModelEntry)`; `slot.modelEntry` is the unwrap for code that still needs the
    /// raw entry (RealExecutor's MLX load path, the cache key).
    case runnable(ModelSlot, equivalence: Equivalence, note: String?)
    /// Not runnable: a plain sentence naming the model and why (never a silent `nil`), plus
    /// MS-3's typed fix-it action — `nil` here for both fallback sentences below; Phase RM
    /// attaches one for a genuinely remote name (RM-3).
    case notRunnable(display: String, reason: String, action: SetupAction?)
}

/// Maps a `.cat` display name ("Whisper Large v3") to an installable `browser.json`
/// `ModelEntry`. Exact-id matching is **never** used — the two catalogs don't intersect by
/// HF id (7 of 32; Whisper Large v3 and Kokoro 82M both miss). Settings still come from
/// the curated manifest (`Resources/CatFlow/models/`); only the *weights* are substituted.
nonisolated enum CatalogBridge {

    /// The twenty-three display names the bridge runs. **`SAM Base`** joined 2026-08-27 (CFM-R15-1):
    /// hazard H2 / `CFM-R13-6` was **ruled option (1)** — a headless default — and the bridge
    /// maps the reference's own `SAM Base` id onto the one installable segmentation entry,
    /// `sam3-4bit`, as a `.substitute` so the substitution is shown on the row, never hidden.
    /// CFM-R13-9/12: OCR (`olmOCR-2 7B`, `dots.ocr`) and `Describe Image` (`LFM2-VL 1.6B`,
    /// `Gemma 3 4B`) joined 2026-08-26 — all four models were already in `browser.json`, so
    /// this is a table change only, no catalog intake. `olmOCR-2 7B` resolves to the
    /// `-mlx` build of the same 1025 checkpoint (a different repo → `.sameFamily`, so the
    /// substitution is surfaced, never hidden); `LFM2-VL 1.6B` is pinned 8-bit in the
    /// manifest but ships 4-bit in the catalog (`.requantized`, silent).
    /// CFM-R14-1: the three ASR gaps (`Whisper Tiny`, `Whisper Small`, `Voxtral Mini 4B
    /// Realtime`) joined 2026-08-26, so every `browser.json` ASR entry is reachable from
    /// the `Transcribe` row's Model menu.
    static let entries: [BridgeEntry] = [
        BridgeEntry(
            display: "Whisper Large v3",
            pinnedID: "mlx-community/whisper-large-v3-mlx",
            candidates: ["mlx-community/whisper-large-v3-asr-fp16"],
            equivalence: .requantized,
            manifestFile: "whisper-large-v3.json"),
        // CFM-R14-1 — the ASR gaps. whisper-tiny/whisper-small have no catflow manifest for
        // Whisper Small (author one below); Whisper Tiny pins the catflow `whisper-tiny`
        // manifest id but loads the `-asr-fp16` catalog build (same checkpoint, packaged for
        // the app — `.requantized`, silent). Voxtral resolves exactly.
        BridgeEntry(
            display: "Whisper Tiny",
            pinnedID: "mlx-community/whisper-tiny",
            candidates: ["mlx-community/whisper-tiny-asr-fp16"],
            equivalence: .requantized,
            manifestFile: "whisper-tiny.json"),
        BridgeEntry(
            display: "Whisper Small",
            pinnedID: "mlx-community/whisper-small-asr-fp16",
            candidates: ["mlx-community/whisper-small-asr-fp16"],
            equivalence: .same,
            manifestFile: "whisper-small.json"),
        BridgeEntry(
            display: "Voxtral Mini 4B Realtime",
            pinnedID: "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
            candidates: ["mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"],
            equivalence: .same,
            manifestFile: "voxtral-mini-4b-realtime-2602-4bit.json"),
        BridgeEntry(
            display: "Qwen3 8B",
            pinnedID: "mlx-community/Qwen3-8B-4bit",
            candidates: ["mlx-community/Qwen3-8B-4bit"],
            equivalence: .same,
            manifestFile: "qwen3-8b-4bit.json"),
        BridgeEntry(
            display: "Kokoro 82M",
            pinnedID: "mlx-community/Kokoro-82M-4bit",
            candidates: ["mlx-community/Kokoro-82M-bf16"],
            equivalence: .requantized,
            manifestFile: "kokoro-82m-4bit.json"),
        BridgeEntry(
            display: "Ministral 3B",
            pinnedID: "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
            candidates: ["mlx-community/Ministral-3-3B-Instruct-2512-4bit"],
            equivalence: .same,
            manifestFile: "ministral-3b-4bit.json"),
        BridgeEntry(
            display: "BGE-M3",
            pinnedID: "mlx-community/bge-m3-mlx-4bit",
            candidates: ["mlx-community/bge-m3-mlx-fp16"],
            equivalence: .requantized,
            manifestFile: "bge-m3.json"),
        BridgeEntry(
            display: "MusicGen",
            pinnedID: "meta/musicgen-small",
            candidates: ["jasonvassallo/mlx-musicgen-small"],
            equivalence: .substitute,
            manifestFile: "musicgen-small.json"),
        BridgeEntry(
            display: "Llama 3.1 8B",
            pinnedID: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            candidates: ["mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"],
            equivalence: .same,
            manifestFile: "llama-3.1-8b-4bit.json"),
        // CFM-R13-9 — OCR.
        BridgeEntry(
            display: "olmOCR-2 7B",
            pinnedID: "mlx-community/olmOCR-2-7B-1025-4bit",
            candidates: ["mlx-community/olmOCR-2-7B-1025-mlx-4bit"],
            equivalence: .sameFamily,
            manifestFile: "olmocr-2-7b-1025-4bit.json"),
        BridgeEntry(
            display: "dots.ocr",
            pinnedID: "mlx-community/dots.ocr-4bit",
            candidates: ["mlx-community/dots.ocr-4bit"],
            equivalence: .same,
            manifestFile: "dots-ocr-4bit.json"),
        // MoC-1-4 (RSI/DelegateMoCBacklog.md) — GLM-OCR. No reference manifest exists in
        // catflow-mlx, so `glm-ocr-4bit.json` is MLXUI-authored (see its own "notes").
        // **DA-7 (RSI/DelegateDeciderBacklog.md, 2026-09-09): promoted.** Decision C's
        // "do not promote" (2026-09-05) was amended — it is now **first** in
        // `taskModels["OCR"]` and the default seed for a new OCR row, because the two
        // incumbents can't be typed as a row model (`splitModelSettings`, DA-8) and GLM-OCR
        // is the one OCR entry verified on a real receipt scan (smoke 81 / 9b-S2b). The
        // MLXUI-authored manifest means making `catflow-mlx` compatible is two jobs — the
        // gallery flows and an upstream curated manifest (SPEC-Q215).
        BridgeEntry(
            display: "GLM-OCR",
            pinnedID: "mlx-community/GLM-OCR-4bit",
            candidates: ["mlx-community/GLM-OCR-4bit"],
            equivalence: .same,
            manifestFile: "glm-ocr-4bit.json"),
        // OCP-3-1 prerequisite / AM-W (RSI/DelegateOCRPromptBacklog.md §5, journal
        // `2026-235`) — PaddleOCR-VL. Installs and runs from Browse (OCP-0) and is already
        // in the derived OCR pool, but had no bridge row, so `CuratedManifest.load` — keyed
        // on `manifestFile` — had nowhere to hang a settings manifest. `display` is exactly
        // `browser.json`'s `displayName` ("PaddleOCR-VL-1.5"), the name BasicGallery flow 3
        // (gallery number 72) already writes, so `resolve` returns the identical `ModelEntry`
        // it did through the R14-FIX-1 no-bridge fallback — only now a manifest can govern
        // the row. No reference manifest exists: catflow-mlx evaluated PaddleOCR-VL during
        // SPEC-Q131 curation and rejected it, so `paddleocr-vl-1.5-4bit.json` is
        // MLXUI-authored (see its own "notes"), like `glm-ocr-4bit.json`. Not added to
        // `taskModels["OCR"]` — same as GLM-OCR: picker-reachable, never the default seed.
        BridgeEntry(
            display: "PaddleOCR-VL-1.5",
            pinnedID: "mlx-community/PaddleOCR-VL-1.5-4bit",
            candidates: ["mlx-community/PaddleOCR-VL-1.5-4bit"],
            equivalence: .same,
            manifestFile: "paddleocr-vl-1.5-4bit.json"),
        // CFM-R13-12 — Describe Image.
        BridgeEntry(
            display: "LFM2-VL 1.6B",
            pinnedID: "mlx-community/LFM2-VL-1.6B-8bit",
            candidates: ["mlx-community/LFM2-VL-1.6B-4bit"],
            equivalence: .requantized,
            manifestFile: "lfm2-vl-1.6b-8bit.json"),
        BridgeEntry(
            display: "Gemma 3 4B",
            pinnedID: "mlx-community/gemma-3-4b-it-4bit",
            candidates: ["mlx-community/gemma-3-4b-it-4bit"],
            equivalence: .same,
            manifestFile: "gemma-3-4b-it-4bit.json"),
        // SV-AM-W — SeedVR2 3B super-resolution (ported in SV-AM4/AM5, journal/2026-162).
        // Only the int8 build ships in browser.json today; the fp16 variant
        // (`mlx-community/SeedVR2-3B-mlx`) has no catalog entry, so it is **not** listed —
        // the golden `everyCandidateExistsInBrowserCatalog` exists exactly to catch a
        // candidate that names no catalog model (CFM-R15 drive-by fix).
        BridgeEntry(
            display: "SeedVR2 3B",
            pinnedID: "mlx-community/SeedVR2-3B-mlx-int8",
            candidates: ["mlx-community/SeedVR2-3B-mlx-int8"],
            equivalence: .same,
            manifestFile: "seedvr2-3b.json"),
        // CFM-R15-1 — Segment. `6-CutOutSubject` names `SAM Base` (the reference manifest's
        // own id `mlx/sam-base`); the only installable segmentation entry is SAM3. `.substitute`
        // renders "running SAM Base as mlx-community/sam3-4bit" on the row, so the different
        // generation is *shown* — the MusicGen precedent (`meta/musicgen-small` → MusicGen).
        BridgeEntry(
            display: "SAM Base",
            pinnedID: "mlx/sam-base",
            candidates: ["mlx-community/sam3-4bit"],
            equivalence: .substitute,
            manifestFile: "sam-base.json"),
        // MoC-2-4 (RSI/DelegateMoCBacklog.md) — Qwen3.5 9B, the llm-typed (text-tower) entry
        // only (Decision A). No reference manifest exists in catflow-mlx, so
        // `qwen3.5-9b-4bit.json` is MLXUI-authored from `qwen3-8b-4bit.json`'s shape (see its
        // own "notes"). Appended to `taskModels["llmModels"]` last, per Decision D as ruled —
        // reachable from every text row's Model menu, never the default seed.
        // `modelType: .llm` disambiguates from MoC-6's `Qwen3.5-9B Vision` card, which shares
        // this same hfModelId (one download between the two).
        BridgeEntry(
            display: "Qwen3.5 9B",
            pinnedID: "mlx-community/Qwen3.5-9B-MLX-4bit",
            candidates: ["mlx-community/Qwen3.5-9B-MLX-4bit"],
            equivalence: .same,
            manifestFile: "qwen3.5-9b-4bit.json",
            modelType: .llm),
        // MoC-4-4 (RSI/DelegateMoCBacklog.md) — Qwen3 Reranker 0.6B, the model on MoC-3's
        // seam. "Qwen3 Reranker 0.6B" is already the pool string in
        // `taskModels["Rerank"]` — used exactly, no pool edit (Decision D doesn't apply to
        // this phase). The manifest is a byte-identical copy of
        // `catflow-mlx/models/curated/qwen3-reranker-0.6b-4bit.json` (MoC-FIX-3 corrected an
        // earlier reconstruction made under the mistaken belief that repo wasn't reachable).
        BridgeEntry(
            display: "Qwen3 Reranker 0.6B",
            pinnedID: "mlx-community/Qwen3-Reranker-0.6B-4bit",
            candidates: ["mlx-community/Qwen3-Reranker-0.6B-4bit"],
            equivalence: .same,
            manifestFile: "qwen3-reranker-0.6b-4bit.json"),
        // MoC-6-2 (RSI/DelegateMoCBacklog.md) — Qwen3.5 9B's vision half, on top of MoC-5's
        // shared install path. Same hfModelId as "Qwen3.5 9B" above; `modelType: .vision`
        // is what keeps the two displays from resolving to each other. No reference manifest
        // (MLXUI-authored, see qwen3.5-9b-vision-4bit.json's own notes). Appended to
        // `taskModels["Describe Image"]` last — the seed ("LFM2-VL 1.6B") does not move.
        BridgeEntry(
            display: "Qwen3.5 9B Vision",
            pinnedID: "mlx-community/Qwen3.5-9B-MLX-4bit",
            candidates: ["mlx-community/Qwen3.5-9B-MLX-4bit"],
            equivalence: .same,
            manifestFile: "qwen3.5-9b-vision-4bit.json",
            modelType: .vision),
        // LY-8 (RSI/DelegateLayaBacklog.md) — Laya 0.4B, the decision encoder on the LY seam.
        // Resolves exactly: the one catalog card and the one catflow manifest id are the same
        // repo. "Laya 0.4B" is now prepended (not appended) to `taskModels["Classify"/"Gate"/
        // "Score"]`, per the 2026-09-27 gate D amendment — the default for new rows of those
        // three tasks.
        BridgeEntry(
            display: "Laya 0.4B",
            pinnedID: "aac6fef/laya-mlx",
            candidates: ["aac6fef/laya-mlx"],
            equivalence: .same,
            manifestFile: "laya-0.4b.json"),
        // CL-7 (RSI/DelegateCLMBacklog.md) — CLM 8B, the second decision encoder on the CL
        // seam (CL-6's `CLMDecisionEngine`). Resolves exactly: the one catalog card and the
        // one catflow manifest id are the same repo. "CLM 8B" is appended (not prepended) to
        // `taskModels["Classify"/"Gate"/"Score"]`, per gate D — it joins the pool but is never
        // the default; `defaultModel(forTask:)` for those three stays "Laya 0.4B".
        BridgeEntry(
            display: "CLM 8B",
            pinnedID: "RealityCat/CLM-v0.1-8B-MLX-8bit",
            candidates: ["RealityCat/CLM-v0.1-8B-MLX-8bit"],
            equivalence: .same,
            manifestFile: "clm-v0.1-8b-8bit.json"),
        // CF-8 (RSI/DelegateClefBacklog.md) — Clef Flash, the third decision encoder on the CL
        // seam (CF-7's `ClefDecisionEngine`). Resolves exactly: the catalog card and the catflow
        // manifest id are the same repo. "Clef Flash" is appended to
        // `taskModels["Classify"/"Gate"/"Score"]` (gate C1) — it joins the pool last and is never
        // the default; `defaultModel(forTask:)` for those three stays "Laya 0.4B".
        BridgeEntry(
            display: "Clef Flash",
            pinnedID: "mlx-community/clef-flash-4bit",
            candidates: ["mlx-community/clef-flash-4bit"],
            equivalence: .same,
            manifestFile: "clef-flash-4bit.json"),
        // Q38-3 (RSI/DelegateQwen38Backlog.md) -- the 3-bit Qwen3.8 27B card from Q38-2. B1 as
        // ruled by the owner 2026-10-03: its **own** display name ("Qwen3.8 27B 3-bit"), `.same`
        // equivalence, resolving to exactly the one catalog card, which is byte-identical to this
        // manifest's `id` (`lukaskremla/Qwen3.8-27B-3bit-MLX`). It is a separate name from the
        // 4-bit `Qwen3.8 27B` B1 also rules (Q38-6) precisely so a flow says which build it runs
        // and `FlowPreflight` judges each one's `ramGB` honestly: this one is 19.08 GB, the 4-bit
        // is not runnable below 32 GB. `modelType: .llm` is explicit because the 4-bit card will
        // carry the same architecture string and a `nil` here would fall back to the ambiguous
        // hfModelId match (`BridgeEntry.modelType`'s doc comment) once Phase B lands.
        // "Qwen3.8 27B 3-bit" is appended **last** to `TaskModels.llmModels`, so the seed
        // ("Ministral 3B") does not move -- same discipline as Qwen3.5 9B (MoC-2-4 D) and CLM 8B.
        BridgeEntry(
            display: "Qwen3.8 27B 3-bit",
            pinnedID: "lukaskremla/Qwen3.8-27B-3bit-MLX",
            candidates: ["lukaskremla/Qwen3.8-27B-3bit-MLX"],
            equivalence: .same,
            manifestFile: "qwen3.8-27b-3bit.json",
            modelType: .llm),
        // Q38-6 (RSI/DelegateQwen38Backlog.md) -- the official 4-bit Qwen3.8 27B card from Q38-5.
        // Q38-0 B1: the bare display name "Qwen3.8 27B" is the 4-bit build, `.same` onto the one
        // `mlx-community/Qwen3.8-27B-4bit` card. The Q38-6 precondition ("waits for Q38-S5") was
        // waived by the owner 2026-10-05 (no 32 GB+ Mac; the card was turned on in 9d12896,
        // `hideQwen3827B = false`); Q38-S5 / smoke row 107 stays open for the first 32 GB+ report.
        // `modelType: .llm` is explicit, as for the 3-bit, so the shared `qwen3_5` architecture can
        // never resolve through the ambiguous hfModelId match. `ramGB` is 26.0 (ruling C1), so
        // `FlowPreflight` refuses it below 32 GB. Appended after the 3-bit in `llmModels`.
        BridgeEntry(
            display: "Qwen3.8 27B",
            pinnedID: "mlx-community/Qwen3.8-27B-4bit",
            candidates: ["mlx-community/Qwen3.8-27B-4bit"],
            equivalence: .same,
            manifestFile: "qwen3.8-27b-4bit.json",
            modelType: .llm),
        // MM-W (RSI/backlog.md, group "MiMo V2.6 9B") -- MiMo V2.6 9B, the OptiQ mixed 4/8-bit
        // `qwen3_5` text model from MM1. MM-0 A (owner, 2026-10-17): the display name is
        // "MiMo V2.6 9B". `.same` onto the one catalog card; catflow-mlx has no manifest for it, so
        // `mimo-v2.6-9b-optiq-4bit.json` is MLXUI-authored from `qwen3.5-9b-4bit.json`'s shape (see
        // its own "notes"). `modelType: .llm` is explicit, following the shared-`qwen3_5` precedent,
        // so the architecture string can never resolve through the ambiguous hfModelId match.
        // Appended **last** to `TaskModels.llmModels` (MM-0 C), so the seed ("Ministral 3B") does
        // not move. Needs no new task: `servedRefNamePrefixes` / `taskKinds` are untouched.
        BridgeEntry(
            display: "MiMo V2.6 9B",
            pinnedID: "mlx-community/MiMo-V2.6-Distill-Qwen-9B-OptiQ-4bit",
            candidates: ["mlx-community/MiMo-V2.6-Distill-Qwen-9B-OptiQ-4bit"],
            equivalence: .same,
            manifestFile: "mimo-v2.6-9b-optiq-4bit.json",
            modelType: .llm),
        // MP1-W (RSI/DelegateModelPackBacklog.md, MP-0 E ruled 2026-10-21) -- Gemma 4 E4B, the
        // `gemma4` text model, and its vision half: two displays, one install (the Qwen3.5 9B
        // precedent, MoC-5/MoC-6). Both resolve `.same` onto the cards that share the one
        // `mlx-community/gemma-4-e4b-it-4bit` repo, so `modelType` is explicit on both (a nil would
        // fall back to the ambiguous hfModelId match, `BridgeEntry.modelType`'s doc comment). No
        // catflow-mlx manifest exists for either, so both are MLXUI-authored (see their own
        // "notes"). "Gemma 4 E4B" is appended **last** to `TaskModels.llmModels` and "Gemma 4 E4B
        // Vision" last to `"Describe Image"`, so neither seed moves. Image + text only: the audio
        // tower is dropped at load, so there is no audio task.
        BridgeEntry(
            display: "Gemma 4 E4B",
            pinnedID: "mlx-community/gemma-4-e4b-it-4bit",
            candidates: ["mlx-community/gemma-4-e4b-it-4bit"],
            equivalence: .same,
            manifestFile: "gemma-4-e4b-it-4bit.json",
            modelType: .llm),
        BridgeEntry(
            display: "Gemma 4 E4B Vision",
            pinnedID: "mlx-community/gemma-4-e4b-it-4bit",
            candidates: ["mlx-community/gemma-4-e4b-it-4bit"],
            equivalence: .same,
            manifestFile: "gemma-4-e4b-it-4bit-vision.json",
            modelType: .vision),
        // MP2-W (RSI/DelegateModelPackBacklog.md, MP2-W; owner rulings V and AE, 2026-10-24) -- Parakeet TDT
        // 0.6B v3, the `Transcribe` row's fifth model. `.same`: the card's repo is the pinned repo. No catflow-mlx
        // manifest exists for it, so `parakeet-tdt-0.6b-v3.json` is MLXUI-authored (see its "notes"). Appended
        // **last** to the `"Transcribe"` pool, so the seed ("Whisper Tiny") does not move.
        BridgeEntry(
            display: "Parakeet TDT 0.6B v3",
            pinnedID: "mlx-community/parakeet-tdt-0.6b-v3",
            candidates: ["mlx-community/parakeet-tdt-0.6b-v3"],
            equivalence: .same,
            manifestFile: "parakeet-tdt-0.6b-v3.json"),
        // MP3-W (RSI/DelegateModelPackBacklog.md, MP3-W) -- Qwen3-ASR 1.7B (8-bit), the `Transcribe` row's sixth model.
        // `.same`: the card's repo is the pinned repo. No catflow-mlx manifest exists for it, so
        // `qwen3-asr-1.7b-8bit.json` is MLXUI-authored (see its "notes"). Appended **last** to the `"Transcribe"` pool,
        // so the seed ("Whisper Tiny") does not move. Implementer's call, pending owner confirmation, that the row is
        // offered at all: the backlog says to wire it, and no owner ruling is quoted for this model's pool slot.
        BridgeEntry(
            display: "Qwen3-ASR 1.7B",
            pinnedID: "mlx-community/Qwen3-ASR-1.7B-8bit",
            candidates: ["mlx-community/Qwen3-ASR-1.7B-8bit"],
            equivalence: .same,
            manifestFile: "qwen3-asr-1.7b-8bit.json"),
        // MP6-W (RSI/DelegateModelPackBacklog.md, MP6-W; MP-0 I) -- LFM2.5-Audio 1.5B (6-bit), the `Transcribe` row's seventh
        // model and the `Speak` row's fourth. `.same`: the card's repo is the pinned repo. No catflow-mlx manifest exists for it,
        // so `lfm2.5-audio-1.5b-6bit.json` is MLXUI-authored. Appended **last** to both pools; neither seed moves. One bridge
        // row serves both tasks (the manifest lists both). Voice conversation is not a flow task and is not reachable from here.
        BridgeEntry(
            display: "LFM2.5-Audio 1.5B",
            pinnedID: "mlx-community/LFM2.5-Audio-1.5B-6bit",
            candidates: ["mlx-community/LFM2.5-Audio-1.5B-6bit"],
            equivalence: .same,
            manifestFile: "lfm2.5-audio-1.5b-6bit.json"),
        // MP7-W (RSI/DelegateModelPackBacklog.md, MP7-W) -- FLUX.2 Klein 4B, the `Generate Image` row's second model (named in the pool since CFM-R12). The catalog card is the
        // mlx-community bf16 snapshot that the app quantizes at load (flows default to quantize 8), the manifest pins `black-forest-labs/FLUX.2-klein-4B`: a different repo
        // of the same checkpoint, so `.requantized` (silent, as the Whisper Large v3 and LFM2-VL rows). The manifest is a byte-identical copy of
        // `catflow-mlx/models/curated/flux2-klein-4b.json` (never edited; its "FLUX.2 Community License" vs the card's Apache-2.0 is a SPEC-Q, backlog MP7-1 D).
        // Only `Generate Image` is served: `Init/Encode/Decode Latent` stay unserved (no stage exposes them). The manifest's `lora`, `controlnet`, `preview`, `negative`, a
        // non-default `guidance` and a `quantize` outside 4/8 are refused by name in `RealExecutor.stageConfig`, never dropped.
        BridgeEntry(
            display: "FLUX.2 Klein 4B",
            pinnedID: "black-forest-labs/FLUX.2-klein-4B",
            candidates: ["mlx-community/FLUX.2-klein-4B-bf16"],
            equivalence: .requantized,
            manifestFile: "flux2-klein-4b.json"),
        // MP7-W-f (owner, Boo, 2026-11-01: "Gallery 60/66: add a .substitute bridge row for the exact display 'Z-Image Turbo (4-bit)' -> FLUX.2 Klein 4B, as one more MP7-W task.")
        // The bundled gallery flows 60 (Generate a Product Shot) and 66 (Eight Seeds, One Sheet) write this name, and no Z-Image card exists in browser.json, so they were refused.
        // `.substitute` shows "running Z-Image Turbo (4-bit) as mlx-community/FLUX.2-klein-4B-bf16" on the row, never hidden. Only this EXACT display: the bare "Z-Image Turbo"
        // stays unbridged, and no other substitution is guessed. The settings authority is Klein's manifest, so a row's guidance/negative/lora/controlnet/preview/quantize are judged
        // as Klein's. It is listed AFTER the Klein row on purpose: `entry(forCandidate:)` (and so `TaskModels.displayName(for:)`) must keep answering "FLUX.2 Klein 4B" for that card.
        BridgeEntry(
            display: "Z-Image Turbo (4-bit)",
            pinnedID: "black-forest-labs/FLUX.2-klein-4B",
            candidates: ["mlx-community/FLUX.2-klein-4B-bf16"],
            equivalence: .substitute,
            manifestFile: "flux2-klein-4b.json"),
        // MC3-8 (issue #6, owner decision 8): `Estimate Depth` is served by the Depth Anything V2 Small port. The
        // catalog card's own display name resolves it exactly (`.same`); gallery flow 62 writes "Depth Pro", the
        // reference runtime's model, which cannot run here, so a `.substitute` row shows what actually loads — the
        // same pattern as "SAM Base" -> SAM3 and "Z-Image Turbo (4-bit)" -> FLUX.2 Klein. `Resources/Gallery/` is
        // never edited: the Python runtime still serves Depth Pro there. Both rows are appended last so
        // `entry(forCandidate:)`/`TaskModels.displayName(for:)` keep naming the card "Depth Anything V2 Small".
        BridgeEntry(
            display: "Depth Anything V2 Small",
            pinnedID: "depth-anything/Depth-Anything-V2-Small-hf",
            candidates: ["depth-anything/Depth-Anything-V2-Small-hf"],
            equivalence: .same,
            manifestFile: "depth-anything-v2-small.json"),
        BridgeEntry(
            display: "Depth Pro",
            pinnedID: "depth-anything/Depth-Anything-V2-Small-hf",
            candidates: ["depth-anything/Depth-Anything-V2-Small-hf"],
            equivalence: .substitute,
            manifestFile: "depth-anything-v2-small.json"),
        // MC2-6 (owner decision 10) -- TranslateGemma 4B, the `Translate` row's only dedicated translator. `.same`: the
        // card's repo is the pinned repo; `modelType: .llm` is explicit (the catalog card is `.llm`, but the family
        // override routes it as `.translate`). No catflow-mlx manifest exists for it, so
        // `translategemma-4b-it-4bit.json` is MLXUI-authored from MiMo's shape (see its "notes"). Appended **last** to
        // the `Translate` pool, so the seed ("Ministral 3B") does not move. It is offered for `Translate` only: it is a
        // `.translate` model and `TaskModels.taskExtraKinds` names only that task.
        BridgeEntry(
            display: "TranslateGemma 4B",
            pinnedID: "mlx-community/translategemma-4b-it-4bit",
            candidates: ["mlx-community/translategemma-4b-it-4bit"],
            equivalence: .same,
            manifestFile: "translategemma-4b-it-4bit.json",
            modelType: .llm),
    ]

    static func entry(for display: String) -> BridgeEntry? {
        entries.first { $0.display == display }
    }

    /// The bridge entry that lists `candidateID` as one of its candidates — the reverse
    /// lookup the derived model pool uses to name a `.cat` display for a catalog model
    /// (CFM-R14-2). `nil` when the model is unbridged (then its `hfModelId` is the name).
    static func entry(forCandidate candidateID: String) -> BridgeEntry? {
        entries.first { $0.candidates.contains(candidateID) }
    }

    /// MP1-W: the same reverse lookup, but for a card that **shares its repo with another card** (Qwen3.5 9B /
    /// Qwen3.5 9B Vision, Gemma 4 E4B / Gemma 4 E4B Vision): a row that pins a `modelType` only answers for a
    /// card of that type, so the Vision card is named by the Vision row, not by the text row that happens to
    /// list the repo first. A row with no `modelType` answers for any card, exactly as before.
    static func entry(forCandidate candidateID: String, modelType: ModelType) -> BridgeEntry? {
        entries.first { $0.candidates.contains(candidateID) && ($0.modelType == nil || $0.modelType == modelType) }
    }

    /// Resolve a display name to an installable `ModelEntry`. `catalog` is the flat list
    /// of `browser.json` entries. Picks the first candidate present in the catalog
    /// (best-first order is the table's).
    ///
    /// **CFM-R14-FIX-1 — the derived-pool fallback.** R14-2's derived pool offers catalog
    /// models with no bridge row; their display name is either the `hfModelId` itself or (after
    /// FIX-7) the catalog `displayName`. A bridge miss is therefore **not** "can't run" anymore —
    /// fall back to matching the display against the catalog by id or display name, and resolve
    /// `.same` with no curated manifest (a manifest-less model is non-stochastic: `withSeed`
    /// already tolerates the missing file). Without this, 20 of the 34 catalog entries would be
    /// selectable and pinned yet refused at Run.
    static func resolve(_ display: String, catalog: [ModelEntry], ignoringGaps: [String] = []) -> CatalogBridgeResolution {
        if let entry = entry(for: display) {
            for candidate in entry.candidates {
                if let model = catalog.first(where: {
                    $0.hfModelId == candidate && (entry.modelType == nil || $0.modelType == entry.modelType)
                }) {
                    let note = entry.equivalence.note(display: display, substitutedID: candidate)
                    return .runnable(.cataloged(model), equivalence: entry.equivalence, note: note)
                }
            }
            return .notRunnable(display: display,
                                reason: "\(display) isn't installed in the model catalog — add it to the catalog before this flow can run.",
                                action: nil)
        }
        // AFM-1/RM: a system or provider slot names itself directly (`apple-foundation @
        // system`, `claude-sonnet @ anthropic`) — checked before the catalog fallback since
        // neither is a `browser.json` entry at all.
        if let systemRef = TaskModels.systemModelRef(forDisplay: display) {
            return .runnable(.system(systemRef), equivalence: .same, note: nil)
        }
        if let providerRef = TaskModels.providerModelRef(forDisplay: display) {
            return .runnable(.provider(providerRef), equivalence: .same, note: nil)
        }
        // No bridge row: an R14-2/7 derived pick. Match the catalog directly.
        if let model = catalog.first(where: { $0.hfModelId == display || $0.displayName == display }) {
            // MP7-2: a catalogued model with a known "no runner yet" gap (`ModelSupport`) is not runnable — without this the
            // display-name fallback would offer the download and then fail at Run with an unsupported-model error.
            if let gap = ModelSupport.unsupportedReason(for: model, ignoringGaps: ignoringGaps) {
                return .notRunnable(display: display, reason: "\(display): \(gap)", action: nil)
            }
            return .runnable(.cataloged(model), equivalence: .same, note: nil)
        }
        // RM-3: a row naming a remote provider by its own `name @ provider` text, whose
        // specific manifest isn't ported/known (a provider RM hasn't reached yet, or a typo)
        // gets a provider-specific refusal with the fix-it button, not the generic sentence
        // below — the display text alone says what it's asking for, no manifest needed.
        // `apple-foundation @ system` also matches `" @ "` but resolved via
        // `systemModelRef(forDisplay:)` above already; excluded here defensively in case
        // that ever fails to load, so this never tells a system row to "add a key."
        if let range = display.range(of: " @ "), !TaskModels.systemDisplayNames.contains(display) {
            let provider = String(display[range.upperBound...]).capitalized
            return .notRunnable(
                display: display,
                reason: "\(display) runs on \(provider)'s servers. Add your \(provider) key in Settings to use it.",
                action: .openSettings(.providers))
        }
        return .notRunnable(display: display,
                            reason: "\(display) isn't in the runnable-model table — this flow needs a model MLXUI can't run yet.",
                            action: nil)
    }
}

// MARK: - Curated manifest (the settings authority)

/// A value that can be a string, number, or boolean — manifests' setting defaults are
/// mixed-type (`"default": "af_heart"`, `"default": 1.0`, `"default": true`).
nonisolated enum FlexValue: Codable, Sendable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) { self = .string(s); return }
        if let b = try? c.decode(Bool.self) { self = .bool(b); return }
        if let n = try? c.decode(Double.self) { self = .number(n); return }
        throw DecodingError.dataCorruptedError(in: c, debugDescription: "setting default must be a string, number, or bool")
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let s): try c.encode(s)
        case .number(let n): try c.encode(n)
        case .bool(let b):   try c.encode(b)
        }
    }

    /// The value as the string the settings layer carries — matching the reference's
    /// `"true"/"false"/str(default)` (`registry.py::resolve_settings`). An integral number
    /// renders without a `.0` so `Int("2048")` succeeds downstream.
    var settingString: String {
        switch self {
        case .string(let s): return s
        case .bool(let b):   return b ? "true" : "false"
        case .number(let n): return n == n.rounded() ? String(Int(n)) : String(n)
        }
    }
}

/// One curated manifest's setting spec — the enums/ranges/defaults the Python's
/// `resolve_settings` validates against. Ported from `models/curated/*.json` (`settings`).
nonisolated struct SettingSpec: Codable, Sendable, Equatable {
    var type: String
    var values: [String]?
    var min: Double?
    var max: Double?
    var defaultValue: FlexValue?
    var mapsTo: String?
    var positionalOK: Bool?
    var autoOK: Bool?

    enum CodingKeys: String, CodingKey {
        case type, values, min, max
        // SET-1: the JSON key is `maps_to` — before this it decoded from `"mapsTo"` and so was
        // always nil (nothing consumed it until `resolveEngineSettings`).
        case mapsTo = "maps_to"
        case defaultValue = "default"
        case positionalOK = "positional_ok"
        case autoOK = "auto_ok"
    }
}

/// The consumed slice of a manifest's `capabilities` block. Only `rejected_settings` —
/// the `{setting: reason}` map `core/validator.py:2941` turns into a reasoned **E708**
/// naming the model — is read here (SPEC-Q212, owner 2026-09-08, OCP-3-4). The block's
/// free-form keys (`variant`, `supports_guidance`, `latent_space`, …) are ignored, the
/// same way `CuratedManifest` ignores the rest of the file.
///
/// No producer consumes this yet — `FlowValidator` performs no manifest settings
/// resolution and MLXUI emits no E708 anywhere. `VAL-1` (`RSI/backlog.md`) is the item
/// that ports the check; this decode is its prerequisite, split out per SPEC-Q212's
/// "its own item" ruling.
nonisolated struct ManifestCapabilities: Codable, Sendable, Equatable {
    var rejectedSettings: [String: String]?

    enum CodingKeys: String, CodingKey {
        case rejectedSettings = "rejected_settings"
    }
}

/// A manifest resource footprint (`resources`).
nonisolated struct ManifestResources: Codable, Sendable, Equatable {
    var diskGB: Double?
    var ramGB: Double?
    var bits: Int?

    enum CodingKeys: String, CodingKey {
        case diskGB = "disk_gb"
        case ramGB = "ram_gb"
        case bits
    }
}

/// A curated manifest — the settings authority for a bridge display name. Only the fields
/// FlowKit consumes are decoded; the rest of the file is ignored.
///
/// AFM-1: `kind`/`engine`/`tasks`/`credentials` join the decode (MS correctly left this to
/// whichever phase first needed it — every provider manifest on disk has always had them;
/// `Codable` silently dropped them before this). `kind` is what `FlowValidator.isRemoteRow`
/// now consults instead of a `" @ "` string test — see `TaskModels.systemDisplayNames`.
nonisolated struct CuratedManifest: Codable, Sendable, Equatable {
    var id: String
    var display: String
    // SPEC-Q224
    /// `"local"` (a `browser.json` download) · `"system"` (Phase AFM, e.g. Apple Foundation
    /// Models) · `"provider"` (Phase RM, a remote API or LAN endpoint) · `"search"` (Phase
    /// WS, a `.net`-class tool's key — `tavily.json`/`brave.json` — never a model, so
    /// `TaskModels.providerModels` and everything model-picker-facing filters explicitly
    /// on `kind == "provider"` rather than `kind != nil`, and a `"search"` manifest is
    /// structurally inert there). `nil` for a manifest written before this field existed
    /// in the Swift decode — never assumed `"local"`.
    var kind: String?
    /// The dispatch string (`"mlx-embed"`, `"anthropic-api"`, `"apple-foundation-models"`, …).
    /// Not yet consumed by any Swift dispatch — RM-2/AFM's own stage names its engine
    /// directly rather than switching on this string, so it decodes for now only to keep the
    /// manifest's full shape visible, matching the reference.
    var engine: String?
    /// The task names (or `@`-prefixed group references, unexpanded — see AFM's `SPEC-Q220`
    /// entry) this manifest's model serves. Only read by `TaskModels`'s system/provider
    /// registries when they're built from a manifest.
    var tasks: [String]?
    /// The Keychain account name a `.provider` manifest's key lives under (`"anthropic"`),
    /// or `nil` for a `.system` manifest or a credential-less LAN endpoint. Phase KEY reads
    /// this; RM-2 is the first to actually resolve one to a Keychain account.
    var credentials: String?
    /// RM-4b/RM-2: `"internet"` (a hosted API) or `"lan"` (an on-prem/self-hosted endpoint
    /// the user's own network reaches, never a third party). `nil` for every manifest
    /// written before this field existed — never assumed `"internet"`; callers that care
    /// check for `"lan"` explicitly rather than treating absence as either value.
    var egress: String?
    /// RM-2: the `openai-compatible` engine's endpoint (Groq, Together, OpenRouter, a LAN
    /// box). `nil` for `anthropic-api`/`openai-api`/`deepseek-api`, whose base URL is fixed
    /// per engine and lives in `ProviderExecutor`, not the manifest (ported verbatim from
    /// `catflow-mlx/src/catflow/engines/provider.py`'s own adapters).
    var baseURL: String?
    var settings: [String: SettingSpec]
    var capabilities: ManifestCapabilities?
    var resources: ManifestResources?

    enum CodingKeys: String, CodingKey {
        case id, display, kind, engine, tasks, credentials, egress
        case baseURL = "base_url"
        case settings, capabilities, resources
    }

    init(id: String, display: String, kind: String? = nil, engine: String? = nil,
         tasks: [String]? = nil, credentials: String? = nil, egress: String? = nil,
         baseURL: String? = nil, settings: [String: SettingSpec],
         capabilities: ManifestCapabilities? = nil, resources: ManifestResources?) {
        self.id = id
        self.display = display
        self.kind = kind
        self.engine = engine
        self.tasks = tasks
        self.credentials = credentials
        self.egress = egress
        self.baseURL = baseURL
        self.settings = settings
        self.capabilities = capabilities
        self.resources = resources
    }

    static func load(bundle: Bundle = .main, manifestFile: String) -> CuratedManifest? {
        // The `Resources/CatFlow/models/` group is a synchronized folder, and Xcode's
        // resource copy flattens it to the top of `Contents/Resources/` — every curated
        // manifest ships there, not under a `CatFlow/models` subdirectory. Confirmed against
        // the built product for every existing manifest, not just this one.
        guard let url = bundle.url(forResource: (manifestFile as NSString).deletingPathExtension,
                                   withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(CuratedManifest.self, from: data)
    }

    /// SET-1 — Registry §3 steps 1-3, ported verbatim from
    /// `catflow-mlx/src/catflow/catalog/registry.py::resolve_settings` + `translate_settings`:
    /// bind a row's `key=value` (and one `positional_ok` bare token) against this manifest's
    /// schema, fill omitted keys from their declared `default`, then rename each to its
    /// `maps_to` engine kwarg. An empty schema is a pure passthrough — `[:]` (SPEC-Q76).
    ///
    /// **Deliberate divergence from the reference (SPEC-Q218):** the reference returns a
    /// `problems` list for check-time E701/E702/E703 and *silently drops* an unknown or
    /// out-of-range key at run time. MLXUI has no check-time port yet (`VAL-1`, `RSI/backlog.md`),
    /// so it **throws** `FlowError.invalidSettings` here — MoC-FIX-2's "refuse at the seam"; a
    /// wrong sampler nobody notices is the CFM-R16-1 failure mode.
    func resolveEngineSettings(_ raw: String?, rowLabel: String) throws -> [String: String] {
        guard !settings.isEmpty else { return [:] }
        let parsed = FlowSettings(raw)
        var provided = parsed.orderedPairs
        if let positionalKey = settings.first(where: { $0.value.positionalOK == true })?.key,
           !provided.contains(where: { $0.key == positionalKey }),
           let bare = parsed.firstBare() {
            provided.append((positionalKey, bare))
        }

        var resolved: [String: String] = [:]
        for (key, value) in provided {
            guard let spec = settings[key] else {
                throw FlowError.invalidSettings(
                    row: rowLabel, setting: key,
                    detail: "\(display) doesn't take a `\(key)` setting (it takes "
                        + "\(settings.keys.sorted().joined(separator: ", ")))")
            }
            try Self.validateSettingValue(key: key, value: value, spec: spec, display: display, rowLabel: rowLabel)
            resolved[key] = value
        }
        for (key, spec) in settings where resolved[key] == nil {
            guard let fallback = spec.defaultValue?.settingString else { continue }
            resolved[key] = fallback
        }

        var translated: [String: String] = [:]
        for (key, value) in resolved { translated[settings[key]?.mapsTo ?? key] = value }
        return translated
    }

    /// The reference's `registry.py::_validate_value` — enum membership (E702) and number
    /// range (E703). `auto`/`{placeholder}` values defer (the runtime resolves them first).
    private static func validateSettingValue(key: String, value: String, spec: SettingSpec,
                                             display: String, rowLabel: String) throws {
        if spec.autoOK == true, value == "auto" { return }
        if value.contains("{"), value.contains("}") { return }   // `{item}`/`{index}` — deferred
        switch spec.type {
        case "enum":
            let allowed = spec.values ?? []
            guard allowed.contains(value) else {
                throw FlowError.invalidSettings(
                    row: rowLabel, setting: key,
                    detail: "\(display) takes \(allowed.joined(separator: " / ")) for `\(key)`, not \"\(value)\"")
            }
        case "number":
            guard let num = Double(value) else {
                throw FlowError.invalidSettings(
                    row: rowLabel, setting: key, detail: "`\(key)` needs a number, not \"\(value)\"")
            }
            let lo = spec.min, hi = spec.max
            if (lo.map { num < $0 } ?? false) || (hi.map { num > $0 } ?? false) {
                let range = "\(lo.map(Self.numString) ?? "any") to \(hi.map(Self.numString) ?? "any")"
                throw FlowError.invalidSettings(
                    row: rowLabel, setting: key,
                    detail: "\(display) needs `\(key)` in the range \(range) — \(value) is outside it")
            }
        default:
            break
        }
    }

    private static func numString(_ n: Double) -> String {
        n == n.rounded() ? String(Int(n)) : String(n)
    }
}

/// KEY-2 — the "Model & Search Providers" Settings section's row list, scanned from the
/// installed manifests rather than hardcoded, so a manifest RM or WS adds gets a
/// Settings row for free with no change here (that's the whole reason KEY comes before
/// either — Q223 in `catflow-mlx/SPEC_QUESTIONS.md` already settled that a search
/// provider is a `credentials:`-bearing manifest exactly like a model provider, so this
/// one scan covers both). RM-4b/RM-FIX-1 reuses the same bundle scan for `TaskModels
/// .providerEgress(forDisplay:)` — one general "read every installed manifest" primitive,
/// two call sites.
nonisolated extension CuratedManifest {
    /// The testable core: decode every URL as a `CuratedManifest` and keep the
    /// `credentials` name of the ones that have ANY declared `kind` and DO name one.
    /// `CuratedManifest.load`'s own directory (`Resources/CatFlow/models`, flattened to
    /// the bundle root by Xcode's synchronized-folder copy — see `load`'s comment) sits
    /// beside wholly unrelated bundled JSON: `browser.json`, the Gallery's
    /// `_metadata.json`, per-index `*-manifest.json`. None of those share this decode's
    /// required shape (`id`/`display`/`settings`), so they fail to decode and are
    /// silently skipped — the same way any bundle resource this type doesn't recognize
    /// already is. Requiring `kind != nil` on top is belt-and-suspenders against a
    /// decode that happens to succeed by accident on a JSON file that isn't a manifest
    /// at all.
    static func installedManifests(manifestURLs urls: [URL]) -> [CuratedManifest] {
        urls.compactMap { url in
            guard let data = try? Data(contentsOf: url),
                  let manifest = try? JSONDecoder().decode(CuratedManifest.self, from: data),
                  manifest.kind != nil else { return nil }
            return manifest
        }
    }

    /// The production entry point: every top-level `.json` in the app bundle (where
    /// every `Resources/` subdirectory flattens to, per
    /// `PBXFileSystemSynchronizedRootGroup`).
    static func installedManifests(bundle: Bundle = .main) -> [CuratedManifest] {
        let urls = bundle.urls(forResourcesWithExtension: "json", subdirectory: nil) ?? []
        return installedManifests(manifestURLs: urls)
    }

    static func installedCredentialNames(manifestURLs urls: [URL]) -> [String] {
        Set(installedManifests(manifestURLs: urls).compactMap(\.credentials)).sorted()
    }

    /// KC-5.2 — the credential names split by manifest `kind` for the Providers pane:
    /// `search` manifests (Brave, Tavily) vs everything else (`provider`: Anthropic, OpenAI,
    /// DeepSeek). A name two manifests share lands under `search` if any of them is a search one.
    static func credentialGroups(manifests: [CuratedManifest]) -> (search: [String], provider: [String]) {
        var search = Set<String>(), provider = Set<String>()
        for manifest in manifests {
            guard let name = manifest.credentials else { continue }
            if manifest.kind == "search" { search.insert(name) } else { provider.insert(name) }
        }
        return (search.sorted(), provider.subtracting(search).sorted())
    }

    static func credentialGroups(bundle: Bundle = .main) -> (search: [String], provider: [String]) {
        credentialGroups(manifests: installedManifests(bundle: bundle))
    }

    /// Phase KEY's original entry point, now built on `installedManifests(bundle:)` (RM-4b
    /// needed the same bundle scan for `TaskModels.providerEgress(forDisplay:)`, so the
    /// decode is shared rather than duplicated). As of RM, `claude-sonnet-4.json` / `gpt-5.6-
    /// luna.json` / `deepseek-v4-flash.json` each name one (`anthropic` / `openai` /
    /// `deepseek`) — no longer `[]`, per journal `2026-260`.
    static func installedCredentialNames(bundle: Bundle = .main) -> [String] {
        installedCredentialNames(manifestURLs: bundle.urls(forResourcesWithExtension: "json", subdirectory: nil) ?? [])
    }

    /// WA-4 — the curated manifests as a `FlowRegistry`, the input `FlowValidator.checkFlow`
    /// needs to resolve a display name (E104's exemption, E209's canonicalization, the
    /// offdevice/F010 registry paths). Reuses this file's own `installedManifests(bundle:)`
    /// scan rather than a second directory read — one general primitive, three call sites
    /// now (KEY-2, RM-4b, this one). `capabilities` maps to `[:]`: no curated manifest field
    /// carries the `constrained_decoding` capability `FlowManifest.capabilityBool` reads, and
    /// that read is a `FlowLint` (`lintRemoteDeciderUnconstrained`), never a `FlowIssue` —
    /// outside anything `checkFlow` returns.
    static func installedFlowRegistry(bundle: Bundle = .main) -> DirectoryFlowRegistry {
        let manifests = installedManifests(bundle: bundle).map { m in
            FlowManifest(id: m.id, display: m.display, kind: m.kind ?? "", engine: m.engine ?? "",
                        tasks: m.tasks ?? [], capabilities: [:])
        }
        return DirectoryFlowRegistry(manifests: manifests)
    }
}
