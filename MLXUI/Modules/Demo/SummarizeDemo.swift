import Foundation

/// The first end-to-end Open Pipeline (no external deps): wrap text in a summarize
/// template → run an LLM stage → save the summary to a file. Proves auto-chaining and
/// the run engine before any audio dependency (backlog M-demo). M11 generalizes this
/// to the audio chain.
enum SummarizeDemo {
    static let defaultTemplate = "Summarize the following in 3 bullet points:\n\n{input}"

    /// Default output location in the app's container.
    nonisolated static func defaultOutputURL() -> URL {
        ModelStore.shared.file(named: "demo-summary.txt")
    }

    /// Assemble `[Template → LLM → Save]`. `llm` is injectable so the chain can run with
    /// a mock in tests; production passes `LLMStage(model:)`.
    nonisolated static func pipeline(
        saveTo url: URL,
        template: String = defaultTemplate,
        llm: any PipelineStage
    ) -> Pipeline {
        Pipeline(stages: [
            TemplatePromptStage(template: template),
            llm,
            SaveTextStage(url: url),
        ])
    }

    /// Build the demo pipeline backed by the real MLX engine for an installed LLM.
    @MainActor
    static func pipeline(
        for model: ModelEntry,
        saveTo url: URL,
        template: String = defaultTemplate,
        config: StageConfig = .default
    ) -> Pipeline {
        pipeline(saveTo: url, template: template, llm: LLMStage(model: model, config: config))
    }
}
