import Foundation

/// The MVP pipeline: an audio file → transcript → summary, saved to disk. Assembles
/// `[ReadAudio → Resample(16k) → Whisper ASR → Template → LLM summarize → SaveText]`.
/// TTS (→ playable WAV) is deferred (backlog M10/M11b) pending a consumable engine.
///
/// Run input is `.text(audioFilePath)` (ReadAudio consumes the path). `asr`/`llm` are
/// injectable so the whole chain is unit-testable with mocks (no model downloads).
enum AudioSummaryPipeline {
    static let defaultTemplate = "Summarize the following transcript in 3 bullet points:\n\n{input}"

    /// Default output location in the app's container.
    nonisolated static func defaultOutputURL() -> URL {
        let base = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return base.appendingPathComponent("AI Browser/audio-summary.txt")
    }

    /// Assemble the chain with injected ASR + LLM stages (tests pass mocks).
    nonisolated static func pipeline(
        saveTo url: URL,
        template: String = defaultTemplate,
        asr: any PipelineStage,
        llm: any PipelineStage
    ) -> Pipeline {
        Pipeline(stages: [
            ReadAudioStage(),
            ResampleStage(targetRate: 16_000),
            asr,
            TemplatePromptStage(template: template),
            llm,
            SaveTextStage(url: url),
        ])
    }

    /// Build the MVP pipeline backed by the real engines: WhisperKit for ASR and the
    /// installed MLX LLM for summarization.
    @MainActor
    static func pipeline(
        for model: ModelEntry,
        saveTo url: URL,
        whisperModel: String = "base",
        template: String = defaultTemplate,
        config: StageConfig = .default
    ) -> Pipeline {
        pipeline(saveTo: url,
                 template: template,
                 asr: ASRStage(model: whisperModel),
                 llm: LLMStage(model: model, config: config))
    }
}
