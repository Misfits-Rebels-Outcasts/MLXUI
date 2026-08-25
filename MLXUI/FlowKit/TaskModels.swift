import Foundation

/// CFM-R12-FIX-12 — the per-task model pools (`_TASK_MODELS`) and the bridge-aware default
/// resolution, moved out of `FlowEditorModel` so `TaskAvailability` (FlowKit) can answer "is
/// this model task runnable" from the same authority the editor uses. `defaultModel(forTask:)`
/// returns the first pool entry the `CatalogBridge` can actually run, or nil — which is how
/// `Segment`/`Upscale`/`Generate Image`/… honestly report that no model exists in this build.
nonisolated enum TaskModels {
    private static let llmModels = ["Ministral 3B", "Llama 3.1 8B", "Qwen3 8B",
                                    "Qwen2.5 14B", "Ternary-Bonsai 4B", "Ternary-Bonsai 8B"]

    /// `models_for_task` — a task's usual model pool (display names), ported from
    /// `catalog/models.py::TASK_MODELS`.
    private static let taskModels: [String: [String]] = [
        "Transcribe": ["Whisper Tiny", "Whisper Large v3", "Voxtral Mini 4B Realtime"],
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

    /// A task's model pool (display names).
    static func models(forTask task: String) -> [String] {
        taskModels[task] ?? []
    }

    /// The default model for a model-class task — the **first pool entry the build can
    /// actually run** (`CatalogBridge` is the authority). Nil = no runnable model exists.
    static func defaultModel(forTask task: String) -> String? {
        for display in models(forTask: task) where CatalogBridge.entry(for: display) != nil {
            return display
        }
        return nil
    }

    /// Every task's seeded default model (for the FIX-3 invariant test).
    static var allDefaultModels: [(task: String, model: String?)] {
        taskModels.keys.sorted().map { ($0, defaultModel(forTask: $0)) }
    }
}
