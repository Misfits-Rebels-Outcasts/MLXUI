import Foundation

/// `text → audio` stage backed by Kokoro (via `KokoroEngine`). The synthesizer is
/// injectable so the stage's logic is unit-testable without downloading a model — the same
/// seam `LLMStage` / `ASRStage` use. Produces a 24 kHz `AudioBuffer` for the M5 WAV writer.
/// See `Design/pipeline-stage-sketch.md` (TTSStage).
nonisolated struct TTSStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .text }
    var produces: MediaKind { .audio }

    /// text → synthesized audio.
    private let synthesize: @Sendable (String) async throws -> AudioBuffer

    /// Designated init with an injectable synthesizer (tests pass a mock).
    init(
        id: String,
        name: String,
        synthesize: @escaping @Sendable (String) async throws -> AudioBuffer
    ) {
        self.id = id
        self.name = name
        self.synthesize = synthesize
    }

    func run(
        _ input: Media,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> Media {
        try require(input, .text)
        guard case let .text(text) = input else {
            throw StageError.kindMismatch(expected: .text, got: input.kind)
        }
        progress(0.1)
        let buffer = try await synthesize(text)
        progress(1.0)
        return .audio(buffer)
    }
}

extension TTSStage {
    /// Stage backed by the real Kokoro engine, loading the **installed** model directory for
    /// `modelID` (config + weights + voices that `InstallManager` downloads). `voice` picks
    /// the speaker; `speed` scales pacing (1.0 = normal).
    init(modelID: String, voice: String = KokoroEngine.defaultVoice, speed: Float = 1.0) {
        let directory = KokoroEngine.installedModelDirectory(id: modelID)
        self.init(id: "kokoro.\(modelID).\(voice)", name: "Kokoro TTS (\(voice))") { text in
            try await KokoroEngine.synthesize(text, voice: voice, speed: speed, modelDirectory: directory)
        }
    }
}
