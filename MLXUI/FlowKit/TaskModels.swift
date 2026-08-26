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
/// `Segment`/`Upscale`/`Generate Image`/… honestly report that no model exists in this build
/// (Upscale/Rerank/Estimate Depth have no `taskKinds` entry and no catalog model, so their
/// pools stay empty).
nonisolated enum TaskModels {
    /// The capability declaration — a model task names the `RunnerKind` it needs. This is
    /// what replaced the display-name pools as the *filter*. Tasks with no entry here (e.g.
    /// `Upscale`, `Rerank`, `Estimate Depth`) have no runner kind and therefore no candidates
    /// — honest "no model in this build", whatever the display-name pool used to claim.
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
    ]

    /// Optional family constraints — filled only where a task genuinely needs one (a `.cat`
    /// row for `Transcribe` wants whisper/voxtral-family ASR, but every `.asr` catalog entry
    /// today already is one, so the table stays empty until a case names itself).
    private static let taskFamilies: [String: String] = [:]

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
        "Describe Image": ["LFM2-VL 1.6B", "Gemma 3 4B"],
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

    private static let llmModels = ["Ministral 3B", "Llama 3.1 8B", "Qwen3 8B",
                                    "Qwen2.5 14B", "Ternary-Bonsai 4B", "Ternary-Bonsai 8B"]

    /// A task's model pool (display names) — preference ordering for the seed, nothing more.
    static func models(forTask task: String) -> [String] {
        taskModels[task] ?? []
    }

    /// The derived pool for a model task — **the** candidates in this build. Filtered by the
    /// corrected `runnerKind` (never `modelType`), the registry's own claim answer, and
    /// `ModelSupport`. This is the function the editor, the picker, and availability all read
    /// — "check and run can't drift" applied to the model side.
    static func derivedModels(for task: String,
                              catalog: [ModelEntry],
                              claimableModelIDs: Set<String>) -> [ModelEntry] {
        guard let kind = taskKinds[task] else { return [] }
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
    /// raw `hfModelId` (what CFM-R14-4 pins into the flow's `models:` block — a name no other
    /// runtime could resolve would be a portability lie).
    static func displayName(for model: ModelEntry) -> String {
        CatalogBridge.entry(forCandidate: model.hfModelId)?.display ?? model.hfModelId
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
