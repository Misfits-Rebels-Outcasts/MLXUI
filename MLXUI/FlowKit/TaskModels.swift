import Foundation

/// CFM-R12-FIX-12 + CFM-R14-2 — a model task's *capability declaration* and the derived
/// pool. Since R14-2 the display-name pools are **not the authority** — each model task
/// names the `RunnerKind` it needs (plus an optional family constraint where one is genuinely
/// required), and candidates are derived from the live catalog by asking the things that know:
/// the corrected `runnerKind`, the registry's own claim answer (`AppState.claimableModelIDs`),
/// and `ModelSupport`. The display-name pools remain only as *preference ordering* for the
/// default seed (the first pool-named entry that is in the derived set).
///
/// `defaultModel(forTask:catalog:claimableModelIDs:)` returns the first derived candidate that
/// the pool names, or nil when the build can't run any of them — which is how
/// `Segment`/`Upscale`/`Rerank`/… honestly report that no runnable model exists in this build
/// (CFM-R14-FIX-2: the derived pool requires the executor to genuinely serve the task, so a
/// `.image`-kind model for the latent family does not make them offerable). `Segment` was in
/// that honest-empty set until the owner ruled `CFM-R13-6` option (1) 2026-08-27 — the
/// executor now serves it and `SAM3` derives (CFM-R15-1).
nonisolated enum TaskModels {
    /// The capability declaration — a model task names the `RunnerKind` it needs. This is
    /// what replaced the display-name pools as the *filter*. Tasks with no entry here (e.g.
    /// `Upscale`, `Estimate Depth`) have no runner kind and therefore no candidates — honest
    /// "no model in this build", whatever the display-name pool used to claim. `Rerank` joined
    /// 2026-09-05 (MoC-3-1, `RSI/DelegateMoCBacklog.md`) — it stays honestly empty until MoC-4
    /// adds the first `.rerank` catalog entry; no catalog entry means `derivedModels` still
    /// returns `[]` for it, exactly as before this entry existed.
    private static let taskKinds: [String: RunnerKind] = [
        "Transcribe": .asr,
        "Embed": .embedding,
        "Generate": .llm,
        "Summarize": .llm,
        "Translate": .llm,
        "Answer": .llm,
        "Rewrite": .llm,
        "Draft": .llm,
        "Ask": .llm,
        "Title": .llm,
        "Critique": .llm,
        "Verify": .llm,
        "Revise": .llm,
        "Merge": .llm,
        "Text to Table": .llm,
        // DA-1 (RSI/DelegateDeciderBacklog.md, owner ruling 2026-09-09): the six deciders name
        // the `RunnerKind` `RealExecutor.runDecider` actually builds — a plain LLM stage
        // (`makeModelStage(_, .default)`) for every one, whether it renders a runtime-owned
        // frame (Classify, Gate, Score, Judge, Think) or builds the ask from the asset directly
        // (Decide, `engines.llm.decide`). With no entry here `derivedModels` failed on its
        // first guard clause, the pool was `[]`, and `TaskAvailability` reported
        // `.needsNewerSupport` — an empty Model menu and a false "needs a model" warning on the
        // 21 gallery rows that name these tasks and run fine. `Extract Structured` is
        // deliberately NOT here: it has no `RealExecutor` path at all (DA-3a ports it, DA-3b
        // offers it), and adding it is the mistake `offerableModelTasksAreExecutorServed`
        // warns against.
        "Decide": .llm, "Classify": .llm, "Gate": .llm,
        "Score": .llm, "Judge": .llm, "Think": .llm,
        "Speak": .tts,
        "Describe Image": .vision,
        "OCR": .ocr,
        "Generate Image": .image,
        "Edit Image": .image,
        "Instruct Edit": .image,
        "Inpaint": .image,
        "Init Latent": .image,
        "Encode Latent": .image,
        "Decode Latent": .image,
        "Denoise": .image,
        "Generate Video": .video,
        "Animate": .video,
        "Generate Sound": .music,
        "Segment": .segmentation,
        "Rerank": .rerank,
    ]

    /// Optional family constraints — filled only where a task genuinely needs one (a `.cat`
    /// row for `Transcribe` wants whisper/voxtral-family ASR, but every `.asr` catalog entry
    /// today already is one, so the table stays empty until a case names itself).
    private static let taskFamilies: [String: String] = [:]

    /// CFM-R14-FIX-2 — the refName prefixes `RealExecutor` genuinely serves. Availability must
    /// depend on **the executor having a path for the task**, not only on a model existing for
    /// its kind. A `.image`-kind model exists for `Edit Image`/`Inpaint`/the latent family, but
    /// no stage accepts what those rows hand it, so offering one would be a confidently wrong
    /// answer. `engines.diffusion.segment` was deliberately absent while hazard H2 / `CFM-R13-6`
    /// was the owner's to answer; it was **ruled option (1) 2026-08-27** (a headless default),
    /// so the prefix is served and `Segment` is offerable — CFM-R15-1. Kept as an explicit
    /// allow-list (not a denylist) so a newly ported task must opt in.
    private static let servedRefNamePrefixes: [String] = [
        "engines.llm.",          // plain LLM (Generate, Extract Structured, Text to Table)
        "engines.asr.",          // Transcribe (audio → text)
        "engines.tts.",          // Speak (text → audio)
        "engines.vlm.",          // Describe Image / OCR (image → text)
        "engines.embed.",        // Embed (text → vector)
        "engines.diffusion.generate_image",   // text → image
        "engines.diffusion.generate_video",   // text → video
        "engines.diffusion.generate_sound",   // text → music
        "engines.diffusion.segment",          // Segment (image → binary mask) — CFM-R15-1
        "engines.rerank.",       // Rerank ([text] + query → [text], reordered) — MoC-3-1
    ]

    /// Whether `RealExecutor` genuinely serves `task` (CFM-R14-FIX-2). A **frame-backed** task
    /// (refKind == .frame — Summarize, Rewrite, the deciders, …) is served: `runModel` renders
    /// the runtime-owned frame and runs the LLM stage on it. An engine task is served only when
    /// its refName matches a served prefix. Anything else has no executor path and therefore no
    /// derived pool, whatever the registry could claim.
    static func isServedByExecutor(_ task: String) -> Bool {
        guard let desc = TaskCatalog.get(task) else { return false }
        if desc.refKind == .frame { return true }
        return servedRefNamePrefixes.contains { desc.refName.hasPrefix($0) }
    }

    /// The `RunnerKind` a model task needs, or nil for a task with no model requirement.
    static func runnerKind(forTask task: String) -> RunnerKind? {
        taskKinds[task]
    }

    /// `models_for_task` — a task's usual model pool (display names), ported from
    /// `catalog/models.py::TASK_MODELS`. **Preference ordering only since CFM-R14-2** — the
    /// pool never filters; it orders the default seed. Candidates come from `derivedModels`.
    private static let taskModels: [String: [String]] = [
        "Transcribe": ["Whisper Tiny", "Whisper Small", "Whisper Large v3", "Voxtral Mini 4B Realtime"],
        "Embed": ["BGE-M3"],
        "Generate": llmModels,
        "Summarize": llmModels,
        "Translate": llmModels,
        "Answer": llmModels,
        "Rewrite": llmModels,
        "Draft": llmModels,
        "Ask": llmModels,
        "Title": llmModels,
        "Critique": llmModels,
        "Verify": llmModels,
        "Revise": llmModels,
        "Merge": llmModels,
        "Speak": ["Kokoro 82M", "Qwen3-TTS 1.7B", "Qwen3-TTS 0.6B"],
        "Rerank": ["BGE Reranker", "Qwen3 Reranker 0.6B"],
        "Text to Table": llmModels,
        // MoC-6-2 (RSI/DelegateMoCBacklog.md): appended last, same discipline as MoC-2-4 —
        // the seed ("LFM2-VL 1.6B") does not move.
        "Describe Image": ["LFM2-VL 1.6B", "Gemma 3 4B", "Qwen3.5 9B Vision"],
        "OCR": ["olmOCR-2 7B", "dots.ocr"],
        "Generate Image": ["Z-Image Turbo", "FLUX.2 Klein 4B"],
        "Edit Image": ["Z-Image Turbo"],
        "Instruct Edit": ["FLUX.1 Kontext"],
        "Inpaint": ["FLUX.1 Fill"],
        "Upscale": ["SeedVR2 3B"],
        "Generate Video": ["Wan2.1 1.3B"],
        "Animate": ["Wan2.1 14B"],
        "Generate Sound": ["MusicGen"],
        "Segment": ["SAM Base"],
        "Init Latent": ["Z-Image Turbo", "FLUX.2 Klein 4B", "FLUX.1 Kontext", "FLUX.1 Fill"],
        "Encode Latent": ["Z-Image Turbo", "FLUX.2 Klein 4B", "FLUX.1 Kontext", "FLUX.1 Fill"],
        "Decode Latent": ["Z-Image Turbo", "FLUX.2 Klein 4B", "FLUX.1 Kontext", "FLUX.1 Fill"],
        "Denoise": ["Z-Image Turbo"],
    ]

    // MoC-2-4 (RSI/DelegateMoCBacklog.md): "Qwen3.5 9B" appended last, per Decision D as ruled
    // 2026-09-05 — reachable from every text row's Model menu, but `defaultModel(forTask:)`
    // walks this array in order and returns the first name that resolves, so appending (not
    // prepending) keeps today's seed ("Ministral 3B") unmoved. Do not reorder this array
    // without re-reading that ruling — moving a name toward the front changes what fourteen
    // task rows and three gallery flows actually run.
    private static let llmModels = ["Ministral 3B", "Llama 3.1 8B", "Qwen3 8B",
                                    "Qwen2.5 14B", "Ternary-Bonsai 4B", "Ternary-Bonsai 8B",
                                    "Qwen3.5 9B"]

    /// A task's model pool (display names) — preference ordering for the seed, nothing more.
    static func models(forTask task: String) -> [String] {
        taskModels[task] ?? []
    }

    /// The derived pool for a model task — **the** candidates in this build. Filtered by the
    /// corrected `runnerKind` (never `modelType`), the registry's own claim answer, `ModelSupport`,
    /// **and the executor genuinely serving the task** (CFM-R14-FIX-2 — a model existing for a
    /// kind is not enough; the runtime must have a path for that task). This is the function the
    /// editor, the picker, and availability all read — "check and run can't drift" applied to
    /// the model side.
    static func derivedModels(for task: String,
                              catalog: [ModelEntry],
                              claimableModelIDs: Set<String>) -> [ModelEntry] {
        guard let kind = taskKinds[task], isServedByExecutor(task) else { return [] }
        let family = taskFamilies[task]?.lowercased()
        return catalog.filter { entry in
            guard entry.runnerKind == kind,
                  claimableModelIDs.contains(entry.id),
                  ModelSupport.unsupportedReason(for: entry) == nil else { return false }
            if let family, !(entry.id + " " + entry.hfModelId).lowercased().contains(family) {
                return false
            }
            return true
        }
    }

    /// The display name a `.cat` should write for a derived model: the bridge display name
    /// when the model is a bridge candidate (what the Python's `.cat` files use), else the
    /// catalog's own `displayName` (CFM-R14-FIX-7: a **readable** name, not the raw id — the
    /// `models:` block then pins that name to the `hfModelId`, so display and id stay two
    /// different strings, exactly what the block exists for). A flow naming
    /// `Qwen3-VL-4B-Instruct` with `models: { Qwen3-VL-4B-Instruct = mlx-community/... }`
    /// reads like a flow; the raw id version read like a dump.
    static func displayName(for model: ModelEntry) -> String {
        CatalogBridge.entry(forCandidate: model.hfModelId)?.display ?? model.displayName
    }

    /// The default model for a model-class task — the **first pool-named derived candidate**
    /// (preference ordering over the derived set; CFM-R14-2). Nil = no runnable model exists.
    static func defaultModel(forTask task: String,
                             catalog: [ModelEntry],
                             claimableModelIDs: Set<String>) -> String? {
        let derived = derivedModels(for: task, catalog: catalog, claimableModelIDs: claimableModelIDs)
        for display in models(forTask: task) {
            if derived.contains(where: { displayName(for: $0) == display }) { return display }
        }
        return derived.first.map { displayName(for: $0) }
    }
}
