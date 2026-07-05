import Foundation

/// Per-stage knobs supplied when a model is turned into a runnable stage. One bag
/// of options covers every runner kind; each stage reads only what it needs.
/// See `Design/pipeline-stage-sketch.md` §factory.
nonisolated struct StageConfig: Sendable {
    var voice: String?         // tts
    var speed: Float           // tts
    var systemPrompt: String?  // llm
    var maxTokens: Int         // llm
    var language: String?      // asr
    var prompt: String?        // vlm — the question asked about the image

    init(
        voice: String? = nil,
        speed: Float = 1.0,
        systemPrompt: String? = nil,
        maxTokens: Int = 512,
        language: String? = nil,
        prompt: String? = nil
    ) {
        self.voice = voice
        self.speed = speed
        self.systemPrompt = systemPrompt
        self.maxTokens = maxTokens
        self.language = language
        self.prompt = prompt
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
        case .llm, .asr, .tts, .embedding, .vision, .ocr, .unsupported:
            throw StageError.unsupportedModel(id: model.id, kind: model.runnerKind)
        }
    }
}
