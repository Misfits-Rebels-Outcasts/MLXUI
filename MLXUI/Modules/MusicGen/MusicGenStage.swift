import Foundation

/// `text → audio` stage for MusicGen text-to-music (MG-MOD1). Accepts a text prompt and
/// produces a generated 32 kHz mono `AudioBuffer`. The generator is injectable so the stage's
/// contract is unit-testable without a model (the same seam `TTSStage`/`FluxStage` use).
nonisolated struct MusicGenStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .text }
    var produces: MediaKind { .audio }

    private let generate: @Sendable (String, @Sendable (Double) -> Void) async throws -> AudioBuffer

    /// Designated init with an injectable generator (tests pass a mock).
    init(
        id: String,
        name: String,
        generate: @escaping @Sendable (String, @Sendable (Double) -> Void) async throws -> AudioBuffer
    ) {
        self.id = id
        self.name = name
        self.generate = generate
    }

    func run(
        _ input: Media,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> Media {
        try require(input, .text)
        guard case let .text(prompt) = input else {
            throw StageError.kindMismatch(expected: .text, got: input.kind)
        }
        progress(0.1)
        let buffer = try await generate(prompt, progress)
        progress(1.0)
        return .audio(buffer)
    }
}

extension MusicGenStage {
    /// Stage for a catalog model id, backed by the real MusicGen engine (`MusicGenEngine`).
    /// `maxSteps` controls the generated duration (250 ≈ 5 s).
    init(modelID: String, maxSteps: Int = 250) {
        self.init(id: "musicgen.\(modelID)", name: "MusicGen (\(modelID))") { prompt, progress in
            try await MusicGenEngine.generate(prompt: prompt, modelID: modelID, maxSteps: maxSteps, progress: progress)
        }
    }
}
