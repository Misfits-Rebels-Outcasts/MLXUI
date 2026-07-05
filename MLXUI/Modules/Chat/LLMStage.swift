import Foundation

/// Pure prompt composition for the chat/summarize stage — the unit-testable core.
enum LLMPrompt {
    /// Combine an optional system prompt with the upstream text. A blank/absent
    /// system prompt yields the user text unchanged.
    static func compose(systemPrompt: String?, userText: String) -> String {
        guard let s = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty else {
            return userText
        }
        return s + "\n\n" + userText
    }
}

/// `text → text` stage backed by MLX LLM generation (the same engine the chat sheet
/// uses, via `LLMEngine`). The generator is injectable so the stage's logic is
/// unit-testable without loading a model. See `Design/pipeline-stage-sketch.md`.
nonisolated struct LLMStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }

    private let systemPrompt: String?
    private let generate: @Sendable (String) async throws -> String

    /// Designated init with an injectable generator (tests pass a mock).
    init(
        id: String,
        name: String,
        systemPrompt: String?,
        generate: @escaping @Sendable (String) async throws -> String
    ) {
        self.id = id
        self.name = name
        self.systemPrompt = systemPrompt
        self.generate = generate
    }

    func run(
        _ input: Media,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> Media {
        try require(input, .text)
        guard case let .text(userText) = input else {
            throw StageError.kindMismatch(expected: .text, got: input.kind)
        }
        let prompt = LLMPrompt.compose(systemPrompt: systemPrompt, userText: userText)
        progress(0.1)
        let output = try await generate(prompt)
        progress(1.0)
        return .text(output)
    }
}

extension LLMStage {
    /// Build a stage backed by the real MLX engine for an installed model. Runs on
    /// the main actor to read `model` (the app target defaults to main-actor isolation);
    /// the captured values are `Sendable`, so the resulting stage still runs off-actor.
    @MainActor
    init(model: ModelEntry, config: StageConfig = .default) {
        let dir = LLMEngine.modelDirectory(for: model.id)
        let maxTokens = config.maxTokens
        self.init(
            id: model.id,
            name: model.displayName,
            systemPrompt: config.systemPrompt,
            generate: { prompt in
                try await LLMEngine.generate(prompt: prompt, modelDir: dir, maxTokens: maxTokens)
            }
        )
    }
}
