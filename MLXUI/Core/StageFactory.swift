import Foundation

/// Per-stage knobs supplied when a model is turned into a runnable stage. One bag
/// of options covers every runner kind; each stage reads only what it needs.
/// See `Design/pipeline-stage-sketch.md` §factory.
nonisolated struct StageConfig: Sendable, Hashable {
    var voice: String?         // tts
    var speed: Float           // tts
    var systemPrompt: String?  // llm
    var maxTokens: Int         // llm
    var temperature: Float     // llm — sampling temperature; 0 = greedy/deterministic (DA-5)
    var language: String?      // asr
    var prompt: String?        // vlm — the question asked about the image
    var seed: UInt64?          // diffusion — the concrete PRNG seed (CFM-R16-1)
    var width: Int?            // diffusion — requested output width (CFM-R16-1)
    var height: Int?           // diffusion — requested output height (CFM-R16-1)
    var steps: Int?            // diffusion — requested denoise steps (CFM-R16-1)
    /// BG-2 (`RSI/DelegateBasicImageFlowsBacklog.md`): the diffusion **upscale factor**,
    /// already validated by `RealExecutor.stageConfig(for:)` to a whole number inside the
    /// manifest's 1…4 range. `nil` means the row named no `scale=` and the stage's own default
    /// applies. Before BG-2 `SeedVR2SDK.makeStage` hardcoded `scale: 2`, so a row's `scale=`
    /// was silently dropped: flow 64 writes `scale=2.0` and worked only by coincidence, and
    /// any `scale=4` would have produced 2× output with no error — the CFM-R16-1 lesson.
    var scale: Int?            // diffusion — upscale factor (BG-2)
    var query: String?         // rerank — the query every candidate is scored against (MoC-3-2)
    var topK: Int?             // rerank — truncate the sorted result to this many (MoC-3-2)
    /// MP6-W: the flow task that asked for this stage (`"Transcribe"` / `"Speak"`), set only by `RealExecutor.stageConfig`.
    /// One model that serves two tasks (LFM2.5-Audio: speech to text *and* text to speech) can't be told apart by its
    /// `RunnerKind` alone; every other SDK ignores it. Part of the engine-cache key on purpose: the two tasks build two stages.
    var flowTask: String?
    /// MP7-W: the flow row's `quantize=` for a model whose manifest declares one (FLUX.2 Klein: 4 or 8). Set only by `RealExecutor.stageConfig` for such a row, already
    /// validated there; `nil` = the stage's own default. Part of the engine-cache key (a different quantize is a different engine) and so kept by `weightsOnly`.
    /// nil = not set. Klein resolves it to the manifest default (8) at `makeStage`; every other stage ignores it.
    var quantize: Int?
    /// MC2-4: TranslateGemma's two codes from the pinned language table, resolved by `RealExecutor` (the target from the
    /// row's `to=`, the source detected on the input) or by the Translate screen's per-run stage. Every other SDK
    /// ignores them. Per-run settings, not weight settings, so `weightsOnly` drops them.
    var sourceLanguage: String?   // translate — source language code
    var targetLanguage: String?   // translate — target language code

    init(
        voice: String? = nil,
        speed: Float = 1.0,
        systemPrompt: String? = nil,
        maxTokens: Int = 512,
        // Defaults to `LLMEngine.generate`'s own historical default (DA-5): every existing
        // call site keeps its current, sampled behaviour — a `0` default here would silently
        // make every chat / summary / decider row greedy, a product change nobody asked for.
        temperature: Float = 0.7,
        language: String? = nil,
        prompt: String? = nil,
        seed: UInt64? = nil,
        width: Int? = nil,
        height: Int? = nil,
        steps: Int? = nil,
        scale: Int? = nil,
        query: String? = nil,
        topK: Int? = nil,
        flowTask: String? = nil,
        quantize: Int? = nil,
        sourceLanguage: String? = nil,
        targetLanguage: String? = nil
    ) {
        self.voice = voice
        self.speed = speed
        self.systemPrompt = systemPrompt
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.language = language
        self.prompt = prompt
        self.seed = seed
        self.width = width
        self.height = height
        self.steps = steps
        self.scale = scale
        self.query = query
        self.topK = topK
        self.flowTask = flowTask
        self.quantize = quantize
        self.sourceLanguage = sourceLanguage
        self.targetLanguage = targetLanguage
    }

    static let `default` = StageConfig()

    /// SC-5c: this config without the per-run settings that don't change which weights load (what a Run window's
    /// engine-cache key uses, so a new seed / voice / prompt books no second engine).
    var weightsOnly: StageConfig {
        var c = self
        c.voice = nil; c.speed = 1.0; c.prompt = nil
        c.seed = nil; c.width = nil; c.height = nil; c.steps = nil
        c.sourceLanguage = nil; c.targetLanguage = nil
        return c
    }
}

/// Builds a runnable `PipelineStage` for an installed model, gating on install
/// state and RAM *before* any engine loads (a model can fit on disk yet OOM RAM).
///
/// Concrete stages are added in later cycles — LLM (M3), ASR (M8), TTS (M10). Until
/// a kind has a stage, it routes to `StageError.unsupportedModel` so the caller can
/// show a graceful "unsupported" view rather than crash.
enum StageFactory {
    static func make(
        for model: ModelEntry,
        config: StageConfig = .default,
        installedModelIDs: Set<String>,
        availableRAMGB: Double
    ) throws -> any PipelineStage {
        guard installedModelIDs.contains(model.id) else {
            throw StageError.modelNotInstalled(id: model.id)
        }
        // Reuse the catalog's 1.5× RAM estimate (`ModelEntry.ramGB`); block before OOM.
        guard model.ramGB <= availableRAMGB else {
            throw StageError.insufficientRAM(requiredGB: model.ramGB, availableGB: availableRAMGB)
        }

        switch model.runnerKind {
        case .llm, .asr, .tts, .embedding, .vision, .ocr, .image, .music, .segmentation, .video, .upscale, .rerank, .decision, .diarization, .enhance, .voice, .interpolate, .separate, .translate, .unsupported:
            throw StageError.unsupportedModel(id: model.id, kind: model.runnerKind)
        }
    }
}
