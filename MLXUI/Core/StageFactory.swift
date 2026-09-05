import Foundation

/// Per-stage knobs supplied when a model is turned into a runnable stage. One bag
/// of options covers every runner kind; each stage reads only what it needs.
/// See `Design/pipeline-stage-sketch.md` §factory.
nonisolated struct StageConfig: Sendable, Hashable {
    var voice: String?         // tts
    var speed: Float           // tts
    var systemPrompt: String?  // llm
    var maxTokens: Int         // llm
    var language: String?      // asr
    var prompt: String?        // vlm — the question asked about the image
    var seed: UInt64?          // diffusion — the concrete PRNG seed (CFM-R16-1)
    var width: Int?            // diffusion — requested output width (CFM-R16-1)
    var height: Int?           // diffusion — requested output height (CFM-R16-1)
    var steps: Int?            // diffusion — requested denoise steps (CFM-R16-1)
    var query: String?         // rerank — the query every candidate is scored against (MoC-3-2)
    var topK: Int?             // rerank — truncate the sorted result to this many (MoC-3-2)

    init(
        voice: String? = nil,
        speed: Float = 1.0,
        systemPrompt: String? = nil,
        maxTokens: Int = 512,
        language: String? = nil,
        prompt: String? = nil,
        seed: UInt64? = nil,
        width: Int? = nil,
        height: Int? = nil,
        steps: Int? = nil,
        query: String? = nil,
        topK: Int? = nil
    ) {
        self.voice = voice
        self.speed = speed
        self.systemPrompt = systemPrompt
        self.maxTokens = maxTokens
        self.language = language
        self.prompt = prompt
        self.seed = seed
        self.width = width
        self.height = height
        self.steps = steps
        self.query = query
        self.topK = topK
    }

    static let `default` = StageConfig()
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
        case .llm, .asr, .tts, .embedding, .vision, .ocr, .image, .music, .segmentation, .video, .upscale, .rerank, .unsupported:
            throw StageError.unsupportedModel(id: model.id, kind: model.runnerKind)
        }
    }
}
